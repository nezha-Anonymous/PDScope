"""PDScope Algorithm 1 controlled policy, without clocks or tensor execution.

Eq.3/Eq.5 compare an inclusive CPU boundary (counterfactual); Eq.4 excludes
it. Actual placements are always disjoint. The prose's continuous prefetch
window and confidence-tested ceil boundary resolve Algorithm 1's floor typo.
All timing inputs must be measured, or labelled as fixtures.
"""
from __future__ import annotations
import math
import operator


class PDScopePolicyError(ValueError):
    pass


def _number(value, name, *, positive=False):
    if (type(value) not in (int, float) or not math.isfinite(value)
            or value < 0 or (positive and value == 0)):
        raise PDScopePolicyError('invalid '+name)
    return float(value)


def _counts(values, name):
    result = dict(values)
    if any(type(e) is not int or not 0 <= e < 128 or type(n) is not int or n < 1
           for e,n in result.items()):
        raise PDScopePolicyError('invalid '+name)
    return result


def _resident(values):
    values = tuple(values)
    if len(values)!=len(set(values)) or any(type(e) is not int or not 0<=e<128 for e in values):
        raise PDScopePolicyError('invalid resident IDs')
    return set(values)


def confidence_prefetch_count(window, hit_rate, limit):
    """Fully hidden floor copies, plus a profitable partially exposed boundary.

    No prediction or future truth is read here. ``hit_rate`` is an explicit
    causal estimate owned by the caller, not this request's eventual accuracy.
    """
    _number(window, 'prefetch window'); _number(hit_rate, 'hit rate')
    if hit_rate>1 or type(limit) is not int or limit<0:
        raise PDScopePolicyError('invalid prefetch confidence or capacity')
    boundary = math.ceil(window)
    benefit = hit_rate*(window-boundary+1) - (1-hit_rate)*(boundary-window)
    chosen = max(0, boundary if benefit>0 else boundary-1)
    # An integer number of completely hidden copies has no exposed cost,
    # including a zero prior hit rate: retain these as the conservative limit.
    if window == math.floor(window): chosen=int(window)
    return min(limit,chosen), dict(continuous_window=window,ceil_boundary=boundary,
                                   normalized_boundary_benefit=benefit,hit_rate=hit_rate)


def decode_cpu_target(top_k, costs):
    """Paper's optimal CPU-resident count n_c = argmin max((k-n) t_g, n t_c).

    Uses the single-token cost row only, so the runtime can compute the same
    target the planner will use *before* the full plan exists (the overlap path
    needs it to know whether resident experts can still be unloaded).
    """
    if type(top_k) is not int or not 1 <= top_k <= 128:
        raise PDScopePolicyError('invalid top-k for the CPU balance target')
    one = costs.for_load(1)
    gpu, cpu = _number(one['gpu_ms'], 'gpu cost', positive=True), _number(one['cpu_ms'], 'cpu cost', positive=True)
    return min(range(top_k+1), key=lambda n: (max((top_k-n)*gpu, n*cpu), n))


def plan_pdscope(current_counts, next_counts, *, current_layer, phase,
                 current_gpu_ids, next_gpu_ids, costs, attention_ms,
                 hit_rate, prefetch_slots, io_startup_ms=0., top_k=8,
                 cpu_balance_target_override=None, joint_next=True, record_checks=True,
                 compute_formula_target=True, trusted=False):
    """Compute paper load/prefetch/unload sets from currently available data.

    GPU-resident IDs may include submitted copies; their actual completion is
    the executor's responsibility. Current-layer truth cannot be substituted
    for target-layer predictions: returned cross-layer keys retain layer IDs.
    ``unload`` withdraws GPU execution; immutable static cache copies may stay
    within their already charged memory budget, an explicit retention option.
    """
    cur,nxt = (_counts(current_counts,'current demand'),_counts(next_counts,'next prediction')) if not trusted else (dict(current_counts),dict(next_counts))
    if (not cur or phase not in ('prefill','decode') or type(current_layer) is not int
            or not 0<=current_layer<48 or (current_layer==47 and nxt)
            or type(top_k) is not int or not 1<=top_k<=128
            or type(prefetch_slots) is not int or prefetch_slots<0):
        raise PDScopePolicyError('invalid phase, layer, demand or capacity')
    if phase=='decode' and (len(cur)!=top_k or any(n!=1 for n in cur.values())
                            or (nxt and (len(nxt)!=top_k or any(n!=1 for n in nxt.values())))):
        raise PDScopePolicyError('decode needs one top-k assignment per expert')
    attention_ms=_number(attention_ms,'attention cost')
    io_startup_ms=_number(io_startup_ms,'IO startup')
    _number(hit_rate,'hit rate')
    if hit_rate>1:raise PDScopePolicyError('hit rate exceeds one')
    if (cpu_balance_target_override is not None
            and (type(cpu_balance_target_override) is not int
                 or not 0 <= cpu_balance_target_override <= top_k)):
        raise PDScopePolicyError('invalid CPU balance target override')
    if (type(joint_next) is not bool or type(record_checks) is not bool
            or type(compute_formula_target) is not bool or type(trusted) is not bool):
        raise PDScopePolicyError('invalid next-layer coordination mode or check recording')
    resident = _resident(current_gpu_ids)&cur.keys()
    next_resident = _resident(next_gpu_ids)&nxt.keys()
    loads = set(cur.values())|set(nxt.values())|{1}
    table={}
    for n in loads:
        c=costs.for_load(n)
        if c.get('tokens')!=n:raise PDScopePolicyError('cost token load mismatch')
        table[n]={k:_number(c[k],k,positive=True) for k in ('cpu_ms','gpu_ms','h2d_ms')}
    entry=lambda l,e,n: (l,e,n)
    order=operator.itemgetter(2,0,1)
    current=sorted([entry(current_layer,e,n) for e,n in cur.items() if e not in resident],key=order)
    following=sorted([entry(current_layer+1,e,n) for e,n in nxt.items() if e not in next_resident],key=order)
    def prefix(xs):
        """cpu(xs[:m]) for every m, left-to-right, identical summation order."""
        out=[0.]*(len(xs)+1)
        for i,x in enumerate(xs):
            out[i+1]=out[i]+table[x[2]]['cpu_ms']
        return out
    def suffix(xs,initial_compute=0.):
        """gpu(xs[i:]) for every i in one reverse pass, identical recurrence.

        The original evaluated this per candidate boundary, i.e. O(n^2) over up
        to 128 experts per layer.  The recurrence only ever walks the list from
        its tail, so one pass yields every suffix value.
        """
        out=[0.]*(len(xs)+1)
        dma=io_startup_ms;compute=initial_compute
        for i in range(len(xs)-1,-1,-1):
            x=xs[i]
            dma+=table[x[2]]['h2d_ms']
            compute=max(compute,dma)+table[x[2]]['gpu_ms']
            out[i]=compute
        out[len(xs)]=initial_compute if initial_compute else 0.
        return out
    def suffix_max(xs,key):
        out=[0.]*(len(xs)+1)
        for i in range(len(xs)-1,-1,-1):
            out[i]=max(out[i+1],table[xs[i][2]][key])
        return out
    formula_nc=decode_cpu_target(top_k,costs) if compute_formula_target else None
    nc=(cpu_balance_target_override if cpu_balance_target_override is not None else
        (formula_nc if formula_nc is not None else decode_cpu_target(top_k,costs)))
    mode='prefill'
    if phase=='decode':
        high_current,high_next=len(current)>nc,len(following)>nc
        mode={(True,False):'a_load',(False,True):'b_prefetch',
              (False,False):'c_unload',(True,True):'d_cross_layer'}[(high_current,high_next)]
    global_queue=[];global_checks=[];local_checks=[];boundary=len(current)
    if mode in ('prefill','d_cross_layer'):
        # Surgical ablation: only remove next-layer jobs from this joint
        # boundary. Keep the same predictor, phase modes, local placement and
        # prefetch admission rules. This is not a HybriMoE reproduction.
        merged=sorted(current+(following if joint_next else []),key=order)
        merged_gpu=suffix(merged);merged_cpu=prefix(merged)
        for i in range(len(merged)):
            tg,tc=merged_gpu[i],merged_cpu[i+1]+attention_ms
            if record_checks:global_checks.append(dict(boundary=i,gpu_ms=tg,cpu_ms=tc))
            if tg<tc:
                global_queue=merged[i:];break
        eligible={x[:2] for x in global_queue}
    else:
        eligible={x[:2] for x in current} if mode=='a_load' else set()
    if mode in ('prefill','d_cross_layer','a_load'):
        current_gpu=suffix(current);current_cpu=prefix(current)
        for i,candidate in enumerate(current):
            if candidate[:2] not in eligible:continue
            tc=current_cpu[i+(mode=='a_load')];tg=current_gpu[i]
            if record_checks:local_checks.append(dict(boundary=i,expert=candidate[1],gpu_ms=tg,cpu_ms=tc))
            if tg<tc:
                boundary=i;break
    on_demand=current[boundary:]
    unload=[]
    if mode in ('b_prefetch','c_unload'):
        # Lowest token count, then expert ID resolves otherwise unspecified ties.
        unload=sorted(resident,key=lambda e:(cur[e],e))[:max(0,nc-len(current))]
    gpu_ids=(resident-set(unload))|{x[1] for x in on_demand}
    cpu_ids=set(cur)-gpu_ids
    actual_cpu_ms=sum(table[cur[e]]['cpu_ms'] for e in cpu_ids)
    resident_gpu_ms=sum(table[cur[e]]['gpu_ms'] for e in resident-set(unload))
    # Resident and demand kernels share ONE compute stream. DMA can overlap
    # resident work; the two compute queues cannot overlap each other.
    actual_gpu_ms=suffix(on_demand,resident_gpu_ms)[0] if on_demand else resident_gpu_ms
    gap=actual_cpu_ms-actual_gpu_ms
    # Same expert bytes, measured IO service estimate at each selected load;
    # use the largest next cost to avoid claiming a larger hidden window.
    tio=max((table[x[2]]['h2d_ms'] for x in following),default=table[1]['h2d_ms'])
    window=max(0.,(gap+attention_ms)/tio)
    count,confidence=confidence_prefetch_count(window,hit_rate,min(prefetch_slots,len(following)))
    if mode in ('a_load','c_unload') or not following:count=0
    constraints=[]
    if phase=='decode':
        count=min(count,max(0,len(following)-nc))
        if count:
            following_cpu=prefix(following);following_gpu=suffix_max(following,'gpu_ms')
            n=len(following)
            while count:
                # Eq.6 uses only the unhidden tail of the selected transfers.
                exposed=max(0.,count-window)*tio+following_gpu[n-count]
                remaining=following_cpu[n-count]
                constraints.append(dict(count=count,exposed_gpu_ms=exposed,remaining_cpu_ms=remaining))
                if exposed<remaining:break
                count-=1
    prefetch=list(reversed(following[-count:])) if count else []
    if gpu_ids&cpu_ids or gpu_ids|cpu_ids!=set(cur):raise AssertionError('demand partition failed')
    return dict(phase=phase,mode=mode,cpu_balance_target=nc,
        cpu_balance_formula_target=formula_nc,
        cpu_balance_target_override=cpu_balance_target_override,
        cpu_queue=sorted(cpu_ids,key=lambda e:(-cur[e],e)),
        gpu_resident_queue=sorted(resident-set(unload),key=lambda e:(-cur[e],e)),
        on_demand_queue=[x[1] for x in reversed(on_demand)],
        prefetch_queue=[x[1] for x in prefetch],unload_queue=unload,
        placements={e:'gpu' if e in gpu_ids else 'cpu' for e in sorted(cur)},
        current_cpu_candidates=[x[1] for x in current],next_cpu_candidates=[x[1] for x in following],
        global_queue=[list(x) for x in global_queue],global_checks=global_checks,
        local_checks=local_checks,boundary=boundary,decode_prefetch_constraints=constraints,
        estimate=dict(cpu_ms=actual_cpu_ms,gpu_ms=actual_gpu_ms,gap_ms=gap,attention_ms=attention_ms,
                      io_ms=tio,io_startup_ms=io_startup_ms,confidence=confidence),
        joint_next=joint_next,record_checks=record_checks,
        scope='paper Algorithm1 core; explicit index/ceil interpretation; measured costs supplied externally')
