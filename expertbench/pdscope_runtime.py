"""Actual PDScope paper decisions on the common BF16 expert executor.

Static resident bytes and forecast bytes share one explicit cache allowance;
on-demand ping-pong slots are separate. Prediction never replaces true routing.
"""
from __future__ import annotations
import ctypes
from collections import Counter
import hashlib
from pathlib import Path

from expertbench.cache import ExpertCache, ExpertKey, DemandID
from expertbench.pdscope_policy import plan_pdscope


def compact_expert_groups(routing, experts=128):
    """Group true IDs in native slot/token order with one host index upload.

    CPU indices are retained for CPU work; per-expert CUDA nonzero and repeated
    device-to-host index copies are unnecessary. Routing weights are untouched.
    """
    import torch
    n,k=routing.selected.shape
    if n<1 or k<1:raise ValueError('empty expert routing')
    flat=routing.selected.detach().cpu().T.contiguous().view(-1)
    counts=torch.bincount(flat,minlength=experts).tolist()
    if len(counts)!=experts:raise ValueError('expert ID outside configured range')
    order=torch.argsort(flat,stable=True)
    tokens=order.remainder(n);positions=tokens*k+torch.div(order,n,rounding_mode='floor')
    indices=torch.stack((tokens,positions)).to(routing.selected.device)
    weights=routing.weights.reshape(-1)[indices[1]]
    groups={};cpu_indices={};start=0
    for e,count in enumerate(counts):
        if count:
            end=start+count
            groups[e]=(indices[0,start:end],weights[start:end])
            cpu_indices[e]=tokens[start:end]
            start=end
    return groups,cpu_indices


class ResidentGraphCosts:
    """Keep an eager upper bound when only static experts have captured graphs.

    The 'measured' mode charges a static-only measurement to eager speculative
    hits as well, so it is an opt-in diagnostic; 'eager_bound' is the default
    scheduler model for this mixed execution backend.
    """
    def __init__(self,base,gpu_ms,mode):
        if mode not in ('eager_bound','measured') or gpu_ms<=0:
            raise ValueError('invalid resident graph cost mode')
        self.base=base;self.gpu_ms=gpu_ms;self.mode=mode

    def for_load(self,n):
        c=self.base.for_load(n)
        if n==1 and self.mode=='measured':
            c=dict(c,gpu_ms=self.gpu_ms,
                   h2d_ms=max(.000001,c['h2d_ms']+c['gpu_ms']-self.gpu_ms))
        return c


def load_hot_profile(path, masters, slots):
    raw=Path(path).read_bytes()
    rows=[tuple(map(int,line.split(','))) for line in raw.decode().splitlines() if line.strip()]
    if (type(slots) is not int or slots<0 or len(set(rows))!=len(rows)
            or any(len(row)!=2 or not 0<=row[0]<48 or not 0<=row[1]<128 for row in rows)):
        raise ValueError('invalid static hot profile or capacity')
    keys=[ExpertKey(l,e,'bf16') for l,e in rows if ExpertKey(l,e,'bf16') in masters]
    if len(keys)<slots:raise ValueError('hot table cannot fill the declared static capacity')
    return keys[:slots],dict(path=str(Path(path).resolve()),sha256=hashlib.sha256(raw).hexdigest(),
                            entries=len(rows),eligible_entries=len(keys),selected=slots,
                            origin='user supplied approximate ShareGPT profile; no test-future routing')


class PDScopeCudaRuntime:
    def __init__(self,masters,*,hot_keys,speculative_slots,costs,cpu_backend,
                 device=0,streams=None,capture_events=False,diagnostics=True,
                 cache_storage=None,staging_storage=None,source_mode='all_masters',resident_graphs=False,
                 graph_cost_mode='eager_bound',cpu_batch='off',balance_profile=False,
                 dispatch='reference',decode_cpu_target=None,
                 merge='reference',input_pregather=False,graph_output_pool=False,
                 demand_slots=2,native_dispatch=None):
        from expertbench.executor import ExpertExecutor
        if (type(speculative_slots) is not int or not 0<=speculative_slots<=128
                or len(set(hot_keys))!=len(hot_keys) or not set(hot_keys)<=masters.keys()):
            raise ValueError('invalid PDScope cache contract')
        sizes={k:w.nbytes for k,w in masters.items()}
        if len(set(sizes.values()))!=1:raise ValueError('equal-size original BF16 experts required')
        self.size=next(iter(sizes.values()));self.hot=set(hot_keys)
        if cpu_batch not in ('off','decode','both'):raise ValueError('invalid CPU packing mode')
        if dispatch not in ('reference','compact'):raise ValueError('invalid dispatch mode')
        self.dispatch=dispatch
        if merge not in ('reference','fused'):
            raise ValueError('invalid merge mode')
        if type(input_pregather) is not bool:
            raise ValueError('input pre-gather switch must be boolean')
        # fused: one weighted index_add over every expert row instead of one per
        # expert (same math, different accumulation order).
        self.merge=merge
        # Gather every resident expert's rows with ONE kernel and join the
        # default stream once per group instead of once per expert.
        self.input_pregather=input_pregather
        if type(graph_output_pool) is not bool:
            raise ValueError('graph output pool switch must be boolean')
        # Keep a two-deep per-key ring of graph outputs instead of allocating one
        # tensor per replay (allocator churn per expert per layer).
        self.graph_output_pool=graph_output_pool
        if type(demand_slots) is not int or demand_slots not in (2,3,4):
            raise ValueError('invalid demand staging depth')
        # Demand staging slots decide how many expert copies can be in flight
        # while the previous one computes; 2 is the historical ping-pong.
        self.demand_slots=demand_slots
        # Optional module whose ``submit_group``/``finish_and_merge`` pair performs
        # the whole CPU-group tail (submit, join, stage, scatter) natively.
        self.native_dispatch=native_dispatch
        self._dispatch_buffers={}
        if (decode_cpu_target is not None
                and (type(decode_cpu_target) is not int or not 0 <= decode_cpu_target <= 8)):
            raise ValueError('invalid decode CPU balance target')
        self.decode_cpu_target=decode_cpu_target
        self.cpu_batch=cpu_batch;self.balance_profile=balance_profile
        self.diagnostics=diagnostics
        self.layers={k.layer for k in masters}
        self.speculative_slots=speculative_slots;self.costs=costs
        self.cache=ExpertCache(sizes,(len(hot_keys)+speculative_slots)*self.size,
                               [self.size]*demand_slots,policy='LRU',initial_hotset=hot_keys)
        self.executor=ExpertExecutor(self.cache,masters,device,pinned_source_mode=source_mode,
                                     cpu_backend=cpu_backend,streams=streams,capture_events=capture_events,
                                     cache_storage=cache_storage,staging_storage=staging_storage)
        self.pending={};self.transient=set();self.forecasts={}
        self.prediction_counts={phase:dict(hits=0,total=0) for phase in ('prefill','decode')}
        self.initial_stats=None
        self.graphs=None;self.graph_cost_calibration=None
        try:
            for key in hot_keys:
                self.executor.finish_load(self.executor.load(key,origin='initial'),wait=True)
            self.initial_stats=self.cache.stats()
            if resident_graphs:self.prepare_resident_graphs(hot_keys,graph_cost_mode)
            self.decode_costs=self.costs
        except BaseException:
            self.close();raise

    def prepare_resident_graphs(self,keys,cost_mode='eager_bound'):
        """Capture immutable hot weights; all demand/staging paths stay eager.

        Shared graph pool is safe because replays use one compute stream and
        every result is cloned before another replay. Inputs and output copies
        stay in the timed call. By default the scheduler retains the measured
        eager cost, since speculative and demand GPU paths still execute eager.
        Static-only measurements remain recorded separately.
        """
        import torch,time,statistics
        from expertbench.executor import ResidentExpertGraphs
        from expertbench.adapters.qwen import expert_forward
        self.graphs=ResidentExpertGraphs(self.executor,keys,immutable_hotset=True,shared_pool=True,
                                         pooled_output=self.graph_output_pool)
        graph=self.graphs
        self.executor.gpu_backend=lambda key,x,t:graph(key,x,t) if x.shape[0]==1 and key in graph.entries else expert_forward(x,t)
        x=torch.linspace(-.5,.5,2048,device=self.executor.device,dtype=torch.bfloat16).reshape(1,-1)
        samples=[]
        for i in range(48):
            key=keys[i%min(16,len(keys))];identity=DemandID('resident_graph_calibration',i,key.layer)
            self.cache.on_demand(identity,{key:1})
            t=time.perf_counter()
            self.executor.finish_gpu(self.executor.gpu(key,x),wait=True)
            elapsed=(time.perf_counter()-t)*1000
            self.cache.end_demand(identity)
            if i>=16:samples.append(elapsed)
        base=self.costs;gpu_ms=statistics.median(samples);original=base.for_load(1)
        self.graph_cost_calibration=dict(samples_ms=samples,median_ms=gpu_ms,
            previous_gpu_ms=original['gpu_ms'],setup=graph.setup,
            scheduler_cost_mode=cost_mode,
            scheduler_gpu_ms=original['gpu_ms'] if cost_mode=='eager_bound' else gpu_ms,
            nonresident_path='eager; measured mode is historical static-only cost diagnostic')
        self.costs=ResidentGraphCosts(base,gpu_ms,cost_mode)

    def _dispatch_workspace(self,rows,hidden):
        import ctypes
        key=(rows,hidden)
        ws=self._dispatch_buffers.get(key)
        if ws is None:
            import torch
            ws=dict(packed=torch.empty((rows,hidden),dtype=torch.bfloat16),
                    block=torch.empty((rows,hidden),dtype=torch.bfloat16),
                    count=ctypes.c_int32(rows),
                    selected=(ctypes.c_int64*rows)(*([0]*rows)),
                    coeff=(ctypes.c_float*rows)(*([1.0]*rows)),
                    tokens=torch.zeros(rows,dtype=torch.long,device=self.executor.device),
                    scratch=torch.empty((rows,hidden),dtype=torch.bfloat16,device=self.executor.device),
                    )
            self._dispatch_buffers[key]=ws
        return ws

    def poll(self,wait=False):
        for key,ticket in list(self.pending.items()):
            if self.executor.finish_load(ticket,wait=wait):del self.pending[key]

    def _retire_current(self,layer,*,wait):
        unfinished=self.cache.unfinished_demand_keys
        for key in list(self.transient):
            if key.layer!=layer or key in unfinished:continue
            if key in self.pending:
                if not self.executor.finish_load(self.pending[key],wait=wait):continue
                del self.pending[key]
            self.cache.evict(key);self.transient.remove(key)
        self.executor.reclaim()

    def forward(self,inputs,gate,*,layer,step,request='pdscope',next_gate=None,
                top_k=8,normalize=True,attention_ms=0.,prefetch=True):
        import torch
        import time
        layer_started=time.perf_counter() if self.balance_profile else 0.
        from expertbench.adapters.qwen import route,expert_groups,tensor_hash
        if (inputs.ndim!=2 or (step>0 and inputs.shape[0]!=1)
                or (step==0 and inputs.shape[0]<2) or layer not in self.layers):
            raise ValueError('invalid prefill/decode activation or layer')
        phase='prefill' if step==0 else 'decode'
        self.executor.host_stage_phase=phase
        key_of=lambda target,e:ExpertKey(target,e,'bf16')
        routing=route(inputs,gate,top_k,normalize)
        cpu_indices={}
        if self.dispatch=='compact' and phase=='prefill':
            groups,cpu_indices=compact_expert_groups(routing)
        else:
            groups={e:(tokens,coeff) for e,tokens,coeff in expert_groups(routing,
                        mode='single_token' if phase=='decode' else 'generic')}
        counts={e:len(t) for e,(t,_) in groups.items()}
        micro={}
        self.poll()
        # The true demand must be declared before anything is submitted, because
        # the cache refuses to hand out a copy for an expert outside it.
        # Always snapshot: the per-layer completion accounting below is a
        # correctness check and must not be tied to an optional profiling switch.
        before=self.cache.stats()
        identity=DemandID(request,step,layer)
        observations=self.cache.on_demand(identity,{key_of(layer,e):n for e,n in counts.items()})
        forecast_key=(request,step,layer)
        prior=self.forecasts.pop(forecast_key,None)
        if prior:
            self.prediction_counts[phase]['hits']+=len(set(prior)&set(counts))
            self.prediction_counts[phase]['total']+=len(prior)
        observed=self.prediction_counts[phase]
        hit_rate=observed['hits']/observed['total'] if observed['total'] else 0.
        next_routing=None;next_counts={}
        if prefetch and next_gate is not None:
            # Layer-local IDs are generated by the TARGET gate on CURRENT
            # activation. Never copy this layer's IDs into the target layer.
            next_routing=route(inputs,next_gate,top_k,normalize)
            next_counts=dict(Counter(next_routing.selected.cpu().flatten().tolist()))
        key=lambda e:ExpertKey(layer,e,'bf16')
        nkey=lambda e:ExpertKey(layer+1,e,'bf16')
        resident=[e for e in counts if self.cache.lookup(key(e)) is not None]
        parts={};tasks={};gpu_tasks={};slots=[];issued=[];capacity_skipped=[]
        native_calls_before=getattr(self.executor.cpu_backend,'completed_calls',0)
        clock=time.perf_counter if self.balance_profile else lambda:0.
        timing={};started=clock()
        if self.balance_profile:
            micro['group_materialize_ms']=(started-layer_started)*1000
        packed=self.cpu_batch=='both' or (self.cpu_batch=='decode' and phase=='decode')
        gpu_input=lambda e:inputs if self.dispatch=='compact' and phase=='decode' else inputs[groups[e][0]]
        cpu_index=lambda e:cpu_indices[e] if e in cpu_indices else groups[e][0].cpu()
        cpu_inputs=None;output=None
        if self.balance_profile:
            micro['predict_materialize_ms']=(clock()-started)*1000
        next_resident=[e for e in next_counts if self.cache.lookup(nkey(e)) is not None]
        if self.balance_profile:
            micro['cache_lookup_ms']=(clock()-started)*1000
        timing['pre_plan_host_ms']=(clock()-layer_started)*1000
        decision=plan_pdscope(counts,next_counts,current_layer=layer,phase=phase,
            current_gpu_ids=resident,next_gpu_ids=next_resident,costs=self.decode_costs if phase=='decode' else self.costs,
            attention_ms=attention_ms,hit_rate=hit_rate,prefetch_slots=self.speculative_slots,top_k=top_k,
            cpu_balance_target_override=(self.decode_cpu_target if phase=='decode' else None),
            joint_next=True,
            record_checks=False,
            # Counts come straight from this layer's grouping: no defensive copy,
            # and the formula n_c is only diagnostic when an override is set.
            trusted=True,compute_formula_target=(self.decode_cpu_target is None))
        if self.balance_profile:
            micro['plan_pdscope_ms']=(clock()-started)*1000
        timing['route_plan_host_ms']=(clock()-layer_started)*1000
        timing.update(micro)
        with torch.inference_mode():
            native_cpu=None
            if (self.native_dispatch is not None and self.merge=='fused' and phase=='decode'):
                handles=self.executor.cpu_backend.native_group_handles(layer)
                if handles is not None:
                    native_cpu=sorted(decision['cpu_queue'])
                    pool,layer_obj=handles
                    ws=self._dispatch_workspace(len(native_cpu),inputs.shape[1])
                    t=clock()
                    row=inputs.cpu().contiguous()
                    self.native_dispatch.submit_group(
                        pool,layer_obj,row.data_ptr(),len(native_cpu),inputs.shape[1]*2,
                        ws['packed'].data_ptr(),ws['block'].data_ptr(),
                        ctypes.addressof(ws['count']),ctypes.addressof(ws['selected']),
                        ctypes.addressof(ws['coeff']),list(native_cpu))
                    timing['cpu_input_and_submit_ms']=(clock()-started)*1000
                    del handles,pool,layer_obj,row  # never retain the owned native pool
            late_cpu=[] if native_cpu is not None else list(decision['cpu_queue'])
            d2h_start=clock()
            if late_cpu:
                if cpu_inputs is None:cpu_inputs=inputs.cpu()
                if self.balance_profile:
                    timing['cpu_input_copy_ms']=(clock()-d2h_start)*1000
                if packed:
                    if (self.input_pregather and phase=='prefill' and self.dispatch=='compact'
                            and len(late_cpu)>1):
                        # One gather for the whole CPU group instead of one per
                        # expert; the packed submission already wants them flat.
                        indices=[cpu_index(e) for e in late_cpu]
                        rows=cpu_inputs[torch.cat(indices)]
                        chunks=rows.split([index.numel() for index in indices],dim=0)
                        jobs={key(e):chunk for e,chunk in zip(late_cpu,chunks)}
                    else:
                        jobs={key(e):(cpu_inputs if phase=='decode' else cpu_inputs[cpu_index(e)])
                              for e in late_cpu}
                    submitted_jobs=self.executor.submit_cpu_many(jobs)
                    tasks={k.expert:v for k,v in submitted_jobs.items()}
                else:
                    for e in late_cpu:
                        tasks[e]=self.executor.submit_cpu(key(e),cpu_inputs[cpu_index(e)])
            submitted=clock();timing['cpu_input_and_submit_ms']=(submitted-started)*1000
            if self.balance_profile:
                timing['cpu_d2h_and_pack_ms']=(submitted-d2h_start)*1000
            timing['cpu_and_resident_submit_from_layer_start_ms']=(submitted-layer_started)*1000
            resident_list=list(decision['gpu_resident_queue'])
            for e in resident_list:
                view=self.cache.lookup(key(e))
                if view.state=='loading':
                    self.executor.finish_load(view.ticket,wait=True);self.pending.pop(key(e),None)
            resident_keys=[key(e) for e in resident_list]
            grouped=self.input_pregather and len(resident_list)>1
            if grouped and self.dispatch=='compact' and phase=='prefill':
                flat_tokens=torch.cat([groups[e][0] for e in resident_list])
                flat_rows=inputs[flat_tokens]
                resident_inputs=list(flat_rows.split([groups[e][0].numel() for e in resident_list],dim=0))
            else:
                grouped=False
                resident_inputs=[gpu_input(e) for e in resident_list]
            # Decode shares one activation tensor across every expert, so the
            # whole layer's submissions can live inside one stream span; prefill
            # gathers per expert and keeps the per-expert join unless the resident
            # group was pre-gathered (handled above by 'grouped').
            layer_span = (not grouped) and phase=='decode'
            if grouped or layer_span:self.executor.begin_gpu_group()
            try:
                for e,rows in zip(resident_list,resident_inputs):
                    gpu_tasks[e]=self.executor.gpu(key(e),rows)
            finally:
                if grouped and not layer_span:self.executor.end_gpu_group()
            queue=decision['on_demand_queue']
            resident_submitted=clock();timing['resident_gpu_submit_ms']=(resident_submitted-submitted)*1000
            prefetch_set=list(decision['prefetch_queue'])
            demand_load_wait=0.;demand_gpu_wait=0.;demand_count=0
            # Lookahead = slots-1: the slot being computed from must stay valid,
            # so S staging slots keep at most S experts in flight (loading or
            # computing). S=2 reproduces the historical ping-pong exactly.
            # Prefill is the I/O-bound phase and measures a real gain from the
            # deeper pipe; decode keeps one outstanding copy because the deeper
            # pipeline there transferred more demand bytes for no gain (R50).
            depth=max(1,min((self.demand_slots-1) if phase=='prefill' else 1,len(queue)))
            inflight=[];next_issue=0
            for _ in range(depth):
                if next_issue<len(queue):
                    inflight.append((next_issue,self.executor.load(key(queue[next_issue]),admit=False)))
                    next_issue+=1
            for position,e in enumerate(queue):
                _,ticket=inflight.pop(0)
                wait_start=clock();self.executor.finish_load(ticket,wait=True);demand_load_wait+=clock()-wait_start
                compute=self.executor.gpu(key(e),gpu_input(e))
                slots.append(dict(expert=e,slot=ticket.slot,generation=ticket.generation))
                # Keep the pipe full: issuing the next copy now lets it overlap
                # this expert's computation instead of only the next one's.
                if next_issue<len(queue):
                    inflight.append((next_issue,self.executor.load(key(queue[next_issue]),admit=False)))
                    next_issue+=1
                if len({item.slot for _,item in inflight})!=len(inflight):
                    raise AssertionError('an in-flight copy reuses a live consumer slot')
                wait_start=clock();parts[e]=self.executor.finish_gpu(compute,wait=True)
                demand_gpu_wait+=clock()-wait_start
                parts[e].record_stream(torch.cuda.current_stream(inputs.device))
                demand_count+=1
            demand_finished=clock();timing['demand_load_compute_and_wait_ms']=(demand_finished-resident_submitted)*1000
            if layer_span and not grouped:self.executor.end_gpu_group()
            if self.balance_profile:
                timing['demand_load_wait_ms']=demand_load_wait*1000
                timing['demand_gpu_wait_ms']=demand_gpu_wait*1000
                timing['demand_experts']=demand_count
            # Current DMA is fully submitted/consumed before next-layer copies.
            # Retire only non-consumers; native CPU work can still be running.
            self._retire_current(layer,wait=True)
            for e in prefetch_set:
                if len(self.transient)>=self.speculative_slots:
                    capacity_skipped.append(e);continue
                target=nkey(e)
                if self.cache.lookup(target) is not None:continue
                ticket=self.executor.load(target,origin='prefetch')
                self.pending[target]=ticket;self.transient.add(target);issued.append(e)
            if issued:self.forecasts[(request,step,layer+1)]=tuple(issued)
            prefetch_submitted=clock();timing['retire_and_prefetch_submit_ms']=(prefetch_submitted-demand_finished)*1000
            for e,compute in gpu_tasks.items():
                parts[e]=self.executor.finish_gpu(compute,wait=True)
                parts[e].record_stream(torch.cuda.current_stream(inputs.device))
            gpu_finished=clock();timing['resident_gpu_finish_wait_ms']=(gpu_finished-prefetch_submitted)*1000
            cpu_wait=cpu_return=0.
            cpu_values={}
            if native_cpu is not None:
                # Join the queue, stage and scatter: everything after the CPU
                # experts have produced their rows.
                t=clock()
                rows=len(native_cpu)
                ws=self._dispatch_workspace(rows,inputs.shape[1])
                handles=self.executor.cpu_backend.native_group_handles(layer)
                if handles is None:raise AssertionError('native CPU group lost its handles')
                pool,layer_obj=handles
                weight=torch.cat([groups[e][1] for e in native_cpu])
                output=torch.zeros_like(inputs)
                output=self.native_dispatch.finish_and_merge(
                    pool,ws['block'].data_ptr(),rows,inputs.shape[1]*2,ws['scratch'],
                    output,ws['tokens'],weight)
                u=clock();cpu_wait+=u-t
                for e in native_cpu:self.cache.finish_cpu(key(e))
                note=getattr(self.executor.cpu_backend,'note_native_dispatch',None)
                if note is not None:note(rows)
                self.dispatch_calls=getattr(self,'dispatch_calls',0)+1
                del handles,pool,layer_obj
            elif packed and len(tasks)>1:
                # One native queue join for the whole packed group instead of one
                # per expert row.
                t=clock();cpu_values=self.executor.finish_cpu_group(list(tasks.values()),wait=True);u=clock()
                cpu_wait+=u-t
                group=getattr(self.executor,'last_cpu_group',None)
                if group is not None and self.balance_profile:
                    timing['cpu_native_span_ms']=group['span_ms']
                    timing['cpu_native_rows']=group['rows']
                    timing['cpu_native_experts']=group['experts']
            else:
                for e,task in tasks.items():
                    t=clock();value=self.executor.finish_cpu(task,wait=True)[0];u=clock()
                    if packed:cpu_values[e]=value
                    else:parts[e]=value.to(inputs.device)
                    v=clock()
                    cpu_wait+=u-t;cpu_return+=v-u
            if cpu_values:
                t=clock();keys=list(cpu_values);sizes=[cpu_values[e].shape[0] for e in keys]
                returned=torch.cat([cpu_values[e] for e in keys],dim=0).to(inputs.device)
                parts.update(zip(keys,returned.split(sizes,dim=0)));cpu_return+=clock()-t
            timing['cpu_finish_wait_ms']=cpu_wait*1000
            timing['cpu_output_to_gpu_ms']=cpu_return*1000
            merge_start=clock()
            if output is None:output=torch.zeros_like(inputs)
            ordered=sorted(groups)
            merge_build_start=clock()
            if self.merge=='fused' and ordered:
                skip=set(native_cpu or ())
                # One weighted scatter instead of one cast+scale+scatter per
                # expert. Same operands and dtype; the accumulation order of the
                # duplicate-token sums changes, so this stays an explicit variant.
                keep=[e for e in ordered if e not in skip]
                if keep:
                    index=torch.cat([groups[e][0] for e in keep])
                    weight=torch.cat([groups[e][1] for e in keep])[:,None]
                    rows_=torch.cat([parts[e] for e in keep],dim=0)
                    output.index_add_(0,index,rows_.to(output.dtype)*weight)
            else:
                for e in ordered:
                    tokens,coeff=groups[e]
                    output.index_add_(0,tokens,(parts[e]*coeff[:,None]).to(output.dtype))
            timing['merge_submit_ms']=(clock()-merge_start)*1000
            if self.balance_profile:
                timing['merge_build_ms']=(merge_build_start-merge_start)*1000
        self.cache.end_demand(identity)
        self._retire_current(layer,wait=True)
        self.poll()
        if self.diagnostics and any(self.cache.lookup(k) is None for k in self.hot):
            raise AssertionError('speculation evicted a charged static hot copy')
        if before is not None:
            after=self.cache.stats()
            delta={k:v-before.get(k,0) for k,v in after.items()}
        else:
            delta={}
        if delta.get('cpu_completed_experts',0)+delta.get('gpu_completed_experts',0)!=len(counts):
            raise AssertionError('lost or duplicated routed expert')
        cpu_tokens=sum(counts[e] for e in decision['cpu_queue'])
        timing['layer_host_ms']=(clock()-layer_started)*1000
        balance=(dict(cpu_experts=len(tasks),gpu_experts=len(counts)-len(tasks),
            cpu_token_assignments=cpu_tokens,gpu_token_assignments=sum(counts.values())-cpu_tokens,
            native_cpu_calls=getattr(self.executor.cpu_backend,'completed_calls',0)-native_calls_before,
            cpu_packing=packed,cpu_batched_io=packed,host_ms=timing,
            timing_scope='host submission and exposed completion waits; NOT isolated CPU/GPU compute or utilization')
            if self.balance_profile else None)
        return output,routing,dict(phase=phase,step=step,layer=layer,token_counts=counts,
            balance=balance,cpu_packing=packed,cpu_batched_io=packed,dispatch=self.dispatch,decode_submission='packed',
            decision=decision,prefetch_issued=issued,prefetch_capacity_skipped=capacity_skipped,
            next_prediction_counts=next_counts,causal_hit_counts=dict(observed),
            predictor='next_layer_gate_on_current_activation',
            scheduler='cross_layer',predictor_prior_hit_rate=0.,
            selected_sha256=tensor_hash(routing.selected) if self.diagnostics else None,
            next_selected_sha256=(tensor_hash(next_routing.selected)
                if self.diagnostics and next_routing is not None else None),
            ready_at_reveal=sum(o.ready_hit for o in observations),
            loading_at_reveal=sum(o.state_at_reveal=='loading' for o in observations),
            counters=delta,ping_pong_slots=slots,
            static_retention='logical decode unload executes on CPU; charged immutable GPU copy retained',
            # The current executor's grouped eager path is the only path that
            # actually submits a batch.  CUDA-graph mode intentionally falls
            # back to per-expert submission; report the observed flag below and
            # never let a requested-but-inactive batch read as "no benefit".
            merge_mode=self.merge,input_pregather=bool(self.input_pregather),demand_slots=self.demand_slots,
            graph_output_pool=bool(self.graph_output_pool),
            pipeline='CPU async queue; current demand DMA before target-layer speculation',
            measured_performance=False)

    def close(self):
        if self.graphs is not None:self.graphs.close();self.graphs=None
        self.executor.shutdown()
