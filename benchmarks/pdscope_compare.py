"""Measure PDScope's single-request prefill and decode throughput.

The stock Transformers attention, KV cache, router math and greedy selection are
retained; only the MoE expert computation is replaced by the PDScope runtime.

Two modes:

* ``measure`` measures prefill and decode throughput on frozen ShareGPT inputs,
  with two warm-up requests before the measured ones. This produces the numbers
  in results/reference_measurements.md.
* ``two_layer`` runs real layers 0/1 on a synthetic activation and checks routing,
  accuracy and buffer lifetimes without loading the whole model.

Short runs are not paper throughput.
"""
from __future__ import annotations
from collections import Counter
from datetime import datetime, timezone
import argparse
import bisect
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import time
import traceback

from benchmarks.harness import capture_manifest, command
from expertbench.run_guard import experiment_lease


# Per-layer cache capacity ceiling. The real limit is the GPU byte cap, which the
# per-case memory guard enforces; this only rejects nonsense values.
SLOT_CEILING = 56


def load_native_layer_dispatch():
    """Load the native layer dispatch built by native/build_native_layer_dispatch.sh."""
    import importlib.util
    root = Path(__file__).resolve().parents[1]
    candidates = sorted((root / 'expertbench' / 'native-build' / 'layer-dispatch').glob('pdscope_layer_dispatch*.so'))
    if not candidates:
        raise RuntimeError('native layer dispatch is not built; run ./scripts/build_native.sh')
    path = candidates[0]
    spec = importlib.util.spec_from_file_location('pdscope_layer_dispatch', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, 'submit_and_merge'):
        raise ValueError('native dispatch lacks submit_and_merge')
    return module


class FileCosts:
    """Measured per-expert cost table, interpolated only inside its domain.

    The eight measured points ship with the repository, so a fresh checkout can
    run without reproducing the offline calibration first.
    """
    def __init__(self, path):
        path = Path(path)
        raw = path.read_bytes()
        payload = json.loads(raw)
        if payload.get('kind') != 'pdscope_cost_table':
            raise ValueError('cost table file has an unexpected kind')
        self.points = {int(n): {k: float(v) for k, v in point.items()}
                       for n, point in payload['points'].items()}
        self.loads = sorted(self.points)
        if self.loads[0] != 1 or self.loads[-1] != 4096:
            raise ValueError('cost table must cover the measured 1..4096 domain')
        self.sources = [dict(path=str(path.resolve()), sha256=hashlib.sha256(raw).hexdigest(),
                             kind='shipped_cost_table', platform=payload.get('platform'),
                             model=payload.get('model'), scope=payload.get('scope'))]

    def for_load(self, n):
        if not self.loads[0] <= n <= self.loads[-1]:
            raise ValueError('expert load outside the measured domain')
        if n in self.points:
            return dict(tokens=n, **self.points[n])
        upper = bisect.bisect_left(self.loads, n)
        lo, hi = self.loads[upper-1], self.loads[upper]
        ratio = (n-lo)/(hi-lo)
        return dict(tokens=n, **{k: self.points[lo][k] + ratio*(self.points[hi][k]-self.points[lo][k])
                                 for k in self.points[lo]})


def prepare_latency_inputs(tokenizer, dataset, lengths=(256,1024,4096), *, seed=0):
    """Freeze the supplied latency.py B1 workload exactly, without inference.

    Filter nonempty conversations BEFORE shuffling; reset the cursor for each
    requested input length, accept by word count, then tokenize raw text as the
    prototype does. Record actual token count rather than padding short input
    to claim the requested length. No chat template or generation randomness.
    """
    import random
    raw=Path(dataset).read_bytes();records=json.loads(raw)
    candidates=[]
    for row,record in enumerate(records):
        conversations=record.get('conversations',[])
        if conversations:
            text=' '.join(conversations[0]['value'].split())
            candidates.append((row,record.get('id'),text))
    random.Random(seed).shuffle(candidates)
    requests=[]
    for length in lengths:
        if type(length) is not int or length<1:raise ValueError('positive requested length required')
        selected=next(((rank,row,rid,text) for rank,(row,rid,text) in enumerate(candidates)
                       if len(text.split())>=length),None)
        if selected is None:raise ValueError('no latency.py qualifying text for '+str(length))
        rank,row,rid,text=selected
        encoded=tokenizer([text],padding=True,truncation=True,max_length=length)
        ids=[int(n) for n in encoded['input_ids'][0]]
        mask=[int(n) for n in encoded['attention_mask'][0]]
        if not ids or len(ids)>length or len(mask)!=len(ids) or any(n!=1 for n in mask):
            raise ValueError('unexpected B1 raw-tokenizer shape/mask')
        requests.append(dict(requested_input_tokens=length,actual_input_tokens=len(ids),
            row=row,id=rid,shuffled_rank=rank,word_count=len(text.split()),
            text_sha256=hashlib.sha256(text.encode()).hexdigest(),
            input_ids=ids,attention_mask=mask,
            input_ids_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest()))
    return dict(protocol='pdscope_latency_py_raw_B1_seed0_v1',seed=seed,
        dataset=str(Path(dataset).resolve()),dataset_sha256=hashlib.sha256(raw).hexdigest(),
        candidates=len(candidates),chat_template=False,requests=requests,
        generation='greedy; fixed64 outputs; first output from prefill;63 decode forwards',
        warmup='two complete requests on the same input with persistent owned pinned sources; reset KV each time')


def phase_metrics(input_tokens, token_ready_seconds):
    if (type(input_tokens) is not int or input_tokens<1 or len(token_ready_seconds)<2
            or any(not isinstance(t,(int,float)) or not __import__('math').isfinite(t) or t<=0 for t in token_ready_seconds)
            or any(b<=a for a,b in zip(token_ready_seconds,token_ready_seconds[1:]))):
        raise ValueError('positive ordered token-completion timestamps required')
    prefill=token_ready_seconds[0];decode=token_ready_seconds[-1]-prefill
    count=len(token_ready_seconds)-1
    return dict(input_tokens=input_tokens,output_tokens=len(token_ready_seconds),decode_forwards=count,
        prefill_seconds=prefill,decode_seconds=decode,request_output_seconds=token_ready_seconds[-1],
        prefill_tokens_per_second=input_tokens/prefill,decode_tokens_per_second=count/decode,
        end_to_end_output_tokens_per_second=len(token_ready_seconds)/token_ready_seconds[-1],
        decode_tpot_seconds=decode/count,
        scope='first generated token belongs to prefill; decode counts actual subsequent forwards')


class SharedModelSession:
    """One immutable CPU/native model, sequentially reset GPU policy state."""
    def __init__(self,args):
        self.args=args;self.began=time.monotonic();self.stages=[]
        self.pdscope_core=self.backend=self.pool_owner=self.model=None
        self.native_layers={};self.hooks=[];self.attention_events={};self.context={}
        self.traces=[];self.completed=[];self.attention_samples={};self.attention_costs={}
        self.hot_info=None;self.cache_pool=None;self.staging_pool=None;self.pool_contract=None

    def budget(self):
        import torch
        if time.monotonic()-self.began>self.args.deadline_seconds:raise TimeoutError('model session deadline exceeded')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>self.args.host_rss_cap_bytes:raise MemoryError('host RSS cap exceeded')
        if torch.cuda.memory_reserved(0)>self.allocator_cap:raise MemoryError('GPU allocator cap exceeded')

    def sample(self,stage,**extra):
        import torch
        self.budget()
        point=dict(stage=stage,elapsed_s=time.monotonic()-self.began,
            host_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            allocated=torch.cuda.memory_allocated(0),reserved=torch.cuda.memory_reserved(0),**extra)
        self.stages.append(point);print(json.dumps(point),flush=True)
        (self.args.output.parent/'progress.json').write_text(json.dumps(dict(stages=self.stages),indent=2)+'\n')

    def initialize(self,specs):
        import torch
        from transformers import AutoTokenizer,Qwen3MoeForCausalLM
        from expertbench.adapters.qwen import collect_model_experts,pin_cpu_master_storage
        from baselines.ktransformers.runtime import (load_cpu_extension,
            FullPhysicalCpuPoolOwner,LoadedNativeCpuLayer,ChunkedNativeCpuBackend)
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
        self.device=torch.device('cuda',0);torch.cuda.set_device(self.device)
        self.allocator_cap=self.args.gpu_total_bytes-1024**3
        torch.cuda.set_per_process_memory_fraction(self.allocator_cap/torch.cuda.get_device_properties(0).total_memory,0)
        self.costs=FileCosts(self.args.cost_table)
        tokenizer=AutoTokenizer.from_pretrained(self.args.model,local_files_only=True,trust_remote_code=False)
        import random
        random.seed(0);torch.manual_seed(0);torch.cuda.manual_seed_all(0)
        self.frozen_inputs=prepare_latency_inputs(tokenizer,self.args.dataset,sorted({s['input_tokens'] for s in specs}))
        self.sources=self.frozen_inputs['requests']
        (self.args.output.parent/'prompt_sources.json').write_text(json.dumps(self.sources,indent=2)+'\n')
        self.sample('prompts_ready')
        self.model=Qwen3MoeForCausalLM.from_pretrained(self.args.model,torch_dtype=torch.bfloat16,
            device_map={'':'cpu'},low_cpu_mem_usage=True,
            attn_implementation=getattr(self.args,'attn_implementation','eager'),
            local_files_only=True,trust_remote_code=False).eval()
        if (self.model.config.num_hidden_layers,self.model.config.num_experts,self.model.config.num_experts_per_tok)!=(48,128,8):
            raise ValueError('unexpected checkpoint shape')
        self.masters=collect_model_experts(self.model)
        self.sample_keys=[k for k in self.masters if k.expert in (0,127)]
        self.original_hashes={k:self.masters[k].hashes() for k in self.sample_keys}
        session=self
        class ExpertBlock(torch.nn.Module):
            def __init__(self,gate,layer):super().__init__();self.gate,self.layer=gate,layer
            def forward(self,hidden):return session.forward_layer(self.layer,hidden,self.gate.weight)
        for layer,decoder in enumerate(self.model.model.layers):
            decoder.mlp=ExpertBlock(decoder.mlp.gate,layer)
            def start_attention(module,inputs,layer=layer):
                if not session.context.get('capture_attention'):return
                event=torch.cuda.Event(enable_timing=True);event.record()
                session.attention_events[(session.context['step'],layer)]=[event,None]
            def end_attention(module,inputs,output,layer=layer):
                if not session.context.get('capture_attention'):return
                event=torch.cuda.Event(enable_timing=True);event.record()
                session.attention_events[(session.context['step'],layer)][1]=event
            self.hooks.append(decoder.self_attn.register_forward_pre_hook(start_attention))
            self.hooks.append(decoder.self_attn.register_forward_hook(end_attention))
        self.model.to(self.device);self.sample('nonexpert_on_gpu')
        self.pinned=pin_cpu_master_storage(self.masters,block_cap_bytes=256*1024**2,check_budget=self.budget)
        if any(self.masters[k].hashes()!=self.original_hashes[k] for k in self.sample_keys):raise AssertionError('sampled pinned source changed')
        self.identities={k:tuple((n,t.data_ptr(),t._version) for n,t in w.tensors.items()) for k,w in self.masters.items()}
        self.layer_masters={l:{k:w for k,w in self.masters.items() if k.layer==l} for l in range(48)}
        self.sample('pinned_master_ready')
        native,self.native_info=load_cpu_extension(self.args.cpu_extension)
        self.sample('native_ready',path=self.native_info['path'],scope='internal_BufferB_read_ranges')
        self.pool_owner=FullPhysicalCpuPoolOwner(native,
            threads_per_subpool=getattr(self.args,'pdscope_cpu_threads',24))
        for layer in range(48):
            self.native_layers[layer]=LoadedNativeCpuLayer(native,self.pool_owner.pool,self.layer_masters[layer],
                layer=layer,source_owners=(self.masters,),max_tokens=64,numa_parts=2)
            if layer%4==3:self.sample('native_layer_'+str(layer))
        self.backend=ChunkedNativeCpuBackend(self.pool_owner.pool,self.native_layers,self.masters,
            source_owners=(self.masters,self.native_layers),max_tokens=64,record_calls=False)
        expert_size=next(iter({w.nbytes for w in self.masters.values()}))
        pool_slots=max(int(s['cache_slots_per_layer'])*48 for s in specs)
        staging_slots=max(int(s.get('demand_slots',2)) for s in specs)
        self.pool_contract=dict(expert_bytes=expert_size,cache_slots=pool_slots,
                                cache_reserved_bytes=pool_slots*expert_size,
                                staging_reserved_bytes=staging_slots*expert_size,
                                staging_slots=staging_slots,
                                allocation_before_attention=True,shared_across_sequential_strategies=True)
        self.sample('shared_device_pools_ready',**self.pool_contract)
        self.native_info['pool']=self.pool_owner.summary()
        self.streams=(torch.cuda.Stream(device=self.device),torch.cuda.Stream(device=self.device))
        self.control_affinity=None
        cpu=getattr(self.args,'pdscope_control_cpu',None)
        if cpu is not None:
            os.sched_setaffinity(0,{cpu})
            self.control_affinity=dict(mode='single_cpu',cpus=[cpu],after=sorted(os.sched_getaffinity(0)),
                scope='control thread only; native worker affinities are owned by the adapter')
        elif getattr(self.args,'pdscope_control_cores','off')=='idle':
            summary=self.pool_owner.summary()
            workers={c for entry in summary['threads'].values()
                      if entry.get('name')!='python' for c in entry['affinity']}
            idle=sorted(set(summary['main_affinity'])-workers)
            if not idle:raise RuntimeError('no idle logical CPU is free of worker threads')
            os.sched_setaffinity(0,set(idle))
            self.control_affinity=dict(mode='idle_cores',cpus=idle,workers=sorted(workers),
                after=sorted(os.sched_getaffinity(0)),
                scope='control thread only; native worker affinities are owned by the adapter')
        self.sample('shared_model_ready',control_affinity=self.control_affinity)

    def reset_strategy(self,spec):
        import torch
        from expertbench.pdscope_runtime import PDScopeCudaRuntime,load_hot_profile
        self.close_strategy()
        torch.cuda.synchronize(self.device)
        self.context=dict(spec=spec,step=0,capture_attention=spec.get('capture_attention',False))
        self.traces=[];self.attention_events={}
        self.hot_info=None
        before=time.perf_counter()
        self.attention_samples={};self.attention_costs={}
        size=self.pool_contract['expert_bytes']
        self.cache_pool=torch.empty(spec['cache_slots_per_layer']*48*size,dtype=torch.uint8,device=self.device)
        self.staging_pool=tuple(torch.empty(size,dtype=torch.uint8,device=self.device)
                                for _ in range(spec.get('demand_slots',2)))
        hot,self.hot_info=load_hot_profile(self.args.hot_file,self.masters,
            spec['cache_slots_per_layer']*48-spec['speculative_slots'])
        self.pdscope_core=PDScopeCudaRuntime(self.masters,hot_keys=hot,speculative_slots=spec['speculative_slots'],
            costs=self.costs,cpu_backend=self.backend,streams=self.streams,capture_events=False,
            diagnostics=spec['diagnostics'],
            cache_storage=self.cache_pool[:spec['cache_slots_per_layer']*48*self.pool_contract['expert_bytes']],
            staging_storage=self.staging_pool,source_mode='all_masters',
            resident_graphs=spec['resident_graphs'],graph_cost_mode=spec['graph_cost_mode'],
            cpu_batch=spec['cpu_batch'],balance_profile=spec['balance_profile'],
            dispatch=spec['dispatch'],decode_cpu_target=spec['decode_cpu_target'],
            native_dispatch=load_native_layer_dispatch() if spec['native_dispatch'] else None,
            merge=spec['merge'],input_pregather=spec['input_pregather'],
            graph_output_pool=spec['graph_output_pool'],demand_slots=spec['demand_slots'])
        torch.cuda.synchronize(self.device)
        return time.perf_counter()-before

    def forward_layer(self,layer,hidden,gate):
        import torch
        from expertbench.adapters.qwen import route,tensor_hash
        spec=self.context['spec'];step=self.context['step'];shape=hidden.shape;x=hidden.reshape(-1,shape[-1])
        if spec['diagnostics']:self.budget()
        if x.shape[0]!=(spec['input_tokens'] if step==0 else 1):raise AssertionError('native layer token shape changed')
        next_gate=self.model.model.layers[layer+1].mlp.gate.weight if layer<47 else None
        estimate=self.attention_costs.get((spec['input_tokens'],'prefill' if step==0 else 'decode',layer+1),0.)
        out,routing,row=self.pdscope_core.forward(x,gate,layer=layer,step=step,request=spec['case_id'],
            next_gate=next_gate,attention_ms=estimate)
        counts=row['token_counts']
        if sum(counts.values())!=x.shape[0]*8:raise AssertionError('missing routed assignment')
        if spec['diagnostics']:
            if not torch.isfinite(out).all():raise AssertionError('nonfinite expert output')
            row['selected_sha256']=tensor_hash(routing.selected)
        self.traces.append(row)
        return out.reshape(shape),routing.logits

    def close_strategy(self):
        if self.pdscope_core is not None:self.pdscope_core.close();self.pdscope_core=None
        self.cache_pool=None;self.staging_pool=None

    def close(self):
        self.close_strategy()
        for hook in self.hooks:hook.remove()
        self.hooks.clear()
        if self.backend is not None:self.backend.close()
        for native in self.native_layers.values():native.close()
        self.native_layers.clear()
        if self.pool_owner is not None:self.pool_owner.close()
        self.staging_pool=None;self.cache_pool=None

    def execute_case(self,spec):
        import torch
        import statistics
        import gzip
        source=next(s for s in self.sources if s['requested_input_tokens']==spec['input_tokens'])
        prompt=dict(source,ids=source['input_ids'],input_tokens=source['actual_input_tokens'])
        if prompt['input_tokens']!=spec['input_tokens']:raise ValueError('raw input token count differs')
        setup=self.reset_strategy(spec)
        stats_before=Counter();sources_before=Counter()
        if self.pdscope_core is not None:
            stats_before.update(self.pdscope_core.cache.stats())
            sources_before.update(self.pdscope_core.executor.source_metrics)
        torch.cuda.reset_peak_memory_stats(self.device)
        native_before=self.backend.completed_calls
        native_tokens_before=self.backend.completed_tokens
        ids=torch.tensor([prompt['ids']],device=self.device,dtype=torch.long)
        past=None;generated=[];ready=[]
        self.sample('case_ready',case_id=spec['case_id'],method=spec['method'],input_tokens=spec['input_tokens'])
        with torch.inference_mode():
            torch.cuda.current_stream(self.device).synchronize()
            began=time.perf_counter()
            for step in range(spec['output_tokens']):
                self.budget();self.context['step']=step
                mask=torch.ones((1,spec['input_tokens']+step),device=self.device,dtype=torch.long)
                outputs=self.model(input_ids=ids,attention_mask=mask,past_key_values=past,
                    use_cache=True,return_dict=True,logits_to_keep=1,output_router_logits=False)
                ids=outputs.logits[:,-1:].argmax(dim=-1);past=outputs.past_key_values
                # Scalar delivery waits only for this output's stream dependencies;
                # do not drain unrelated speculative copies at every token boundary.
                token=int(ids.item())
                ready.append(time.perf_counter()-began);generated.append(token)
            output_end=time.perf_counter()
        stats=Counter()
        if self.pdscope_core is None:raise AssertionError('no PDScope runtime for this case')
        core=self.pdscope_core;core.poll(wait=True);ex=core.executor
        if (core.pending or core.transient or core.forecasts or ex.loads or ex.computes
                or ex.cpu_tasks or core.cache.active_demand):raise AssertionError('PDScope executor did not drain')
        stats.update(core.cache.stats());stats.subtract(stats_before)
        source_metrics=Counter(core.executor.source_metrics);source_metrics.subtract(sources_before)
        self.backend.wait_all()
        if self.pdscope_core is not None:
            expected_cpu=sum(t['token_counts'][e] for t in self.traces for e in t['decision']['cpu_queue'])
            if self.backend.completed_tokens-native_tokens_before!=expected_cpu:
                raise AssertionError('native CPU packed/unpacked row assignments differ from true dispatch')
        complete_end=time.perf_counter()
        expected=(spec['input_tokens']+spec['output_tokens']-1)*48*8
        if (len(self.traces)!=spec['output_tokens']*48 or past.get_seq_length()!=spec['input_tokens']+spec['output_tokens']-1
                or stats['physical_demand_tokens']!=expected
                or sum(sum(t['token_counts'].values()) for t in self.traces)!=expected
                or stats['h2d_bytes_completed']!=stats['h2d_bytes_submitted']
                or stats['physical_expert_demands']!=stats['cpu_completed_experts']+stats['gpu_completed_experts']
                or not torch.isfinite(outputs.logits).all()):raise AssertionError('request work, KV, bytes or logits failed')
        attention=[]
        for (step,layer),events in sorted(self.attention_events.items()):
            ms=events[0].elapsed_time(events[1]);phase='prefill' if step==0 else 'decode'
            attention.append(dict(step=step,layer=layer,ms=ms))
            if spec['role']!='measure' and spec['case_id'].endswith('_00'):self.attention_samples.setdefault((spec['input_tokens'],phase,layer),[]).append(ms)
        if spec['role']!='measure':
            self.attention_costs={k:statistics.median(v) for k,v in self.attention_samples.items()}
        process=command(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'])
        used=[int(row.split(',')[1].strip())*1024**2 for row in process['stdout'].splitlines()
              if row.split(',')[0].strip()==str(os.getpid())]
        if len(used)!=1:raise RuntimeError('actual process GPU usage unavailable')
        reserved=torch.cuda.memory_reserved(self.device);peak=torch.cuda.max_memory_reserved(self.device)
        external=max(0,used[0]-reserved)
        if used[0]>self.args.gpu_total_bytes or peak+external>self.args.gpu_total_bytes:
            raise MemoryError('GPU total or conservative peak estimate exceeds24GiB')
        self.budget()
        path=self.args.output.parent/'traces'/(spec['case_id']+'.json.gz');path.parent.mkdir(exist_ok=True)
        with gzip.open(path,'wt') as f:json.dump(self.traces,f,separators=(',',':'))
        result=dict(case=dict(spec),prompt=prompt,metrics=phase_metrics(spec['input_tokens'],ready),
            measured_performance=spec['role']=='measure' and not spec['diagnostics'],
            token_ready_seconds=ready,generated_ids=generated,kv_final_length=past.get_seq_length(),
            strategy_initialization_seconds=setup,background_drain_seconds=complete_end-output_end,
            request_complete_seconds=complete_end-began,cache_stats=dict(stats),
            hot_profile=self.hot_info,native_completed_chunks=self.backend.completed_calls-native_before,
            cache_payload_allowance_bytes=spec['cache_slots_per_layer']*48*9*1024**2,
            source_metrics_delta=dict(source_metrics),
            resident_graphs=(dict(calibration=self.pdscope_core.graph_cost_calibration,
                                 replays=self.pdscope_core.graphs.calls)
                             if self.pdscope_core is not None and self.pdscope_core.graphs is not None else None),
            cache_pool_reserved_bytes=(self.cache_pool.numel() if self.cache_pool is not None else 0),
            demand_staging_bytes=spec['demand_slots']*9*1024**2,
            demand_staging_aliases='PDScope declared demand slots; one expert payload per slot',
            gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(self.device),gpu_peak_reserved_bytes=peak,
            gpu_process_end_bytes=used[0],gpu_process_end_external_bytes=external,
            gpu_peak_reserved_plus_end_external_bytes=peak+external,
            gpu_peak_scope='allocator observed peak plus end-process nonallocator bytes; not NVML continuous sampling',
            host_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            trace=dict(path=str(path.relative_to(self.args.output.parent)),sha256=hashlib.sha256(path.read_bytes()).hexdigest()),
            attention_cuda_ms=attention,
            timing_scope='same native eager attention/KV; output completion via default-stream scalar delivery; drain reported separately')
        if self.traces:
            # The configured host path must actually reach the runtime; a silent
            # default would make a measurement describe a different pipeline.
            observed=self.traces[0]
            if (observed.get('demand_slots')!=spec['demand_slots']
                    or observed.get('merge_mode')!=spec['merge']
                    or bool(observed.get('input_pregather'))!=spec['input_pregather']
                    or bool(observed.get('graph_output_pool'))!=spec['graph_output_pool']):
                raise AssertionError('configured host path did not reach the runtime')
        if spec['balance_profile']:
            phase_balance={}
            for phase in ('prefill','decode'):
                rows=[t for t in self.traces if t['phase']==phase]
                work=Counter();host=Counter();events=Counter()
                for row in rows:
                    b=row['balance']
                    work.update({k:b[k] for k in ('cpu_experts','gpu_experts','cpu_token_assignments','gpu_token_assignments','native_cpu_calls')})
                    host.update(b['host_ms']);events.update(row['counters'])
                phase_balance[phase]=dict(work=dict(work),host_ms=dict(host),
                    h2d_bytes=events.get('h2d_bytes_submitted',0),
                    prefetch_loads=events.get('prefetch_loads_submitted',0),fresh_prefetch_hits=events.get('fresh_prefetch_hits',0),
                    scope='host intervals/exposed waits, not isolated compute time or GPU/CPU utilization')
            result['phase_balance']=phase_balance
            self.sample('case_balance',case_id=spec['case_id'],phases=phase_balance)
        with (self.args.output.parent/'requests.jsonl').open('a') as f:f.write(json.dumps(result,separators=(',',':'))+'\n')
        self.completed.append(result)
        self.sample('case_completed',case_id=spec['case_id'],metrics=result['metrics'])
        del outputs,past,ids
        return result

    def finalize(self):
        import torch
        if self.identities!={k:tuple((n,t.data_ptr(),t._version) for n,t in w.tensors.items()) for k,w in self.masters.items()}:
            raise AssertionError('CPU master storage/version changed')
        if any(self.masters[k].hashes()!=self.original_hashes[k] for k in self.sample_keys):raise AssertionError('sampled CPU master changed')
        return dict(schema_version=2,status='completed',kind='single_request_measure',
            measured_performance=any(r['measured_performance'] for r in self.completed),
            requests=self.completed,model=self.args.model,model_initialization_stages=self.stages,
            native_attention_and_kv=True,
            attention_implementation=getattr(self.args,'attn_implementation','eager'),
            native_backend=self.native_info,master_storage=self.pinned,device_buffer_pool=self.pool_contract,
            original_weight_samples_unchanged=True,all_master_storage_versions_unchanged=True,
            calibration_sources=self.costs.sources,
            attention_cost_table=[dict(input_tokens=n,phase=p,layer=l,median_ms=ms,
                                       samples=len(self.attention_samples[(n,p,l)]))
                                  for (n,p,l),ms in sorted(self.attention_costs.items())],
            model_identity=dict(config_sha256=hashlib.sha256((Path(self.args.model)/'config.json').read_bytes()).hexdigest(),
                index_sha256=hashlib.sha256((Path(self.args.model)/'model.safetensors.index.json').read_bytes()).hexdigest(),
                sampled_expert_hashes={str((k.layer,k.expert)):v for k,v in self.original_hashes.items()},
                full_checkpoint_shard_hashes='not computed'),
            torch_environment=dict(version=torch.__version__,file=torch.__file__,cuda=torch.version.cuda,
                alloc_conf=os.environ.get('PYTORCH_ALLOC_CONF') or os.environ.get('PYTORCH_CUDA_ALLOC_CONF'),
                control_threads=torch.get_num_threads(),
                native_compute_workers=2*int(getattr(self.args,'pdscope_cpu_threads',24)),
                allow_tf32=False,
                allow_bf16_reduced_precision_reduction=False),
            total_device_budget_bytes=self.args.gpu_total_bytes,allocator_cap_bytes=self.allocator_cap,
            control_affinity=self.control_affinity,
            host_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)


def build_measure_cases(args):
    """Two warm-up plus ``--repetitions`` measured requests per input length."""
    if not 1 <= args.repetitions <= 10:
        raise ValueError(f'--repetitions must be 1..10, got {args.repetitions}')
    if any(type(n) is not int or not 128 <= n <= 4096 for n in args.lengths):
        raise ValueError('--lengths must be integers in 128..4096, got ' + repr(args.lengths))
    if not 1 <= args.cache_slots <= SLOT_CEILING:
        raise ValueError(f'--cache-slots must be 1..{SLOT_CEILING}, got {args.cache_slots}')
    if not 0 <= args.speculative_slots <= 128:
        raise ValueError(f'--speculative-slots must be 0..128, got {args.speculative_slots}')
    specs = []
    targets = args.pdscope_decode_cpu_targets or [None]
    suffix = 'graph' if args.pdscope_graphs else 'eager'
    if args.pdscope_cpu_batch != 'off':
        suffix += '_cpu_' + args.pdscope_cpu_batch
    if args.pdscope_dispatch != 'reference':
        suffix += '_' + args.pdscope_dispatch
    if args.pdscope_native_dispatch:
        suffix += '_dispatch'
    if args.pdscope_merge != 'reference':
        suffix += '_' + args.pdscope_merge
    for length in args.lengths:
        for target in targets:
            case_suffix = suffix + (f'_target{target}' if target is not None else '')
            for r in range(2 + args.repetitions):
                specs.append(dict(
                    case_id=f'pdscope_{case_suffix}_{length}_{r:02d}', method='pdscope',
                    input_tokens=length, output_tokens=64,
                    cache_slots_per_layer=args.cache_slots, speculative_slots=args.speculative_slots,
                    prompt_index=0, role='warmup' if r < 2 else 'measure',
                    diagnostics=False, capture_attention=r < 2,
                    resident_graphs=args.pdscope_graphs,
                    graph_cost_mode=args.pdscope_graph_cost if args.pdscope_graphs else None,
                    cpu_batch=args.pdscope_cpu_batch, dispatch=args.pdscope_dispatch,
                    balance_profile=args.pdscope_balance, decode_cpu_target=target,
                    merge=args.pdscope_merge, input_pregather=args.pdscope_input_pregather,
                    graph_output_pool=args.pdscope_graph_output_pool,
                    demand_slots=args.pdscope_demand_slots,
                    native_dispatch=bool(args.pdscope_native_dispatch)))
    return specs


def run(args):
    import gzip
    specs=build_measure_cases(args)
    (args.output.parent/'cases.json').write_text(json.dumps(dict(schema_version=1,cases=specs),indent=2)+'\n')
    session=SharedModelSession(args)
    try:
        session.initialize(specs)
        for spec in specs:session.execute_case(spec)
        result=session.finalize()
    except BaseException as error:
        if session.traces:
            with gzip.open(args.output.parent/'partial_trace.json.gz','wt') as f:json.dump(session.traces,f)
        try:session.close()
        except BaseException as cleanup:error.add_note('cleanup failure: '+repr(cleanup))
        raise
    session.close()
    return result


def probe_packed_cpu(backend,masters):
    """Bounded real native packing checks and interleaved component timings."""
    import torch
    import statistics
    import torch.nn.functional as F
    from expertbench.cache import ExpertKey
    generator=torch.Generator(device='cpu').manual_seed(20261011)
    jobs={ExpertKey(0,e,'bf16'):torch.randn((n,2048),generator=generator).to(torch.bfloat16)
          for e,n in ((0,33),(1,35),(2,3))}
    saved={k:x.clone() for k,x in jobs.items()}
    calls_before=backend.completed_calls;tokens_before=backend.completed_tokens
    futures=backend.submit_many(jobs);values={k:v.result()[0] for k,v in futures.items()}
    packed_calls=backend.completed_calls-calls_before
    if packed_calls!=2 or backend.completed_tokens-tokens_before!=71:
        raise AssertionError('packing did not preserve 71 rows across two64-row calls')
    checks=[]
    for key,x in jobs.items():
        original=backend(key,x);w=masters[key].tensors
        ref=F.linear(F.silu(F.linear(x.float(),w['gate_proj'].float()))*
                     F.linear(x.float(),w['up_proj'].float()),w['down_proj'].float())
        actual=values[key];error=float(torch.linalg.vector_norm(actual.float()-ref)/torch.linalg.vector_norm(ref).clamp_min(1e-9))
        if actual.shape!=x.shape or not torch.equal(x,saved[key]) or not torch.isfinite(actual).all() or error>.02:
            raise AssertionError('packed row reconstruction or FP32 reference failed')
        checks.append(dict(expert=key.expert,rows=x.shape[0],unpacked_bitwise=torch.equal(actual,original),fp32_relative_l2=error))
    timings=[]
    with torch.inference_mode():
        for n in (1,8,32):
            for k in (1,2,4,8):
                tasks={ExpertKey(0,e,'bf16'):torch.randn((n,2048),generator=generator).to(torch.bfloat16) for e in range(k)}
                samples={m:[] for m in ('individual','packed')};calls={m:[] for m in samples}
                for repeat in range(9):
                    for method in (('individual','packed') if repeat%2==0 else ('packed','individual')):
                        old=backend.completed_calls;begin=time.perf_counter()
                        pending=(backend.submit_many(tasks) if method=='packed' else {key:backend.submit(key,x) for key,x in tasks.items()})
                        outputs={key:f.result()[0] for key,f in pending.items()}
                        elapsed=(time.perf_counter()-begin)*1000
                        if repeat>=2:samples[method].append(elapsed);calls[method].append(backend.completed_calls-old)
                timings.append(dict(experts=k,rows_per_expert=n,
                    milliseconds={m:dict(median=statistics.median(v),samples=v) for m,v in samples.items()},native_calls=calls))
    return dict(checks=checks,cross_boundary_rows=71,cross_boundary_native_calls=packed_calls,
        timings=timings,scope='CPU-only completed submission/packing/native/row reconstruction; interleaved2warmup+7measure; not model throughput')


def probe_compact_groups(gate):
    """Check native slot/token order and measure completed grouping only."""
    import torch
    import statistics
    from expertbench.adapters.qwen import route,expert_groups
    from expertbench.pdscope_runtime import compact_expert_groups
    generator=torch.Generator(device='cpu').manual_seed(20261012)
    rows=[]
    with torch.inference_mode():
        for n in (1,16,256,1024,4096):
            x=torch.randn((n,gate.shape[1]),generator=generator).to(torch.bfloat16).to(gate.device)
            routing=route(x,gate,8,True)
            expected={e:(t,w) for e,t,w in expert_groups(routing)}
            actual,cpu_indices=compact_expert_groups(routing)
            if set(expected)!=set(actual):raise AssertionError('grouped expert set changed')
            for e,(t,w) in expected.items():
                if (not torch.equal(t,actual[e][0]) or not torch.equal(w,actual[e][1])
                        or not torch.equal(t.cpu(),cpu_indices[e])):
                    raise AssertionError('native slot/token order or weight changed')
            samples={m:[] for m in ('reference','compact')}
            for rep in range(7):
                for mode in (('reference','compact') if rep%2==0 else ('compact','reference')):
                    torch.cuda.synchronize(gate.device);started=time.perf_counter()
                    result=expert_groups(routing) if mode=='reference' else compact_expert_groups(routing)
                    torch.cuda.synchronize(gate.device);elapsed=(time.perf_counter()-started)*1000
                    if rep>=2:samples[mode].append(elapsed)
            rows.append(dict(tokens=n,experts=len(actual),assignments=sum(len(v[0]) for v in actual.values()),
                native_order_and_weights_bitwise=True,
                milliseconds={m:dict(median=statistics.median(v),samples=v) for m,v in samples.items()}))
    return dict(rows=rows,scope='same revealed gate outputs; grouping only,2warmup+5measure; no full model throughput')


def run_two_layer(args):
    """Real 0/1-layer weights, synthetic activation, no whole-model timing."""
    import torch
    import torch.nn.functional as F
    from expertbench.adapters.qwen import load_expert, load_router, pin_cpu_master_storage, tensor_hash
    from expertbench.cache import ExpertKey
    from expertbench.pdscope_runtime import PDScopeCudaRuntime, load_hot_profile
    from baselines.ktransformers.runtime import (load_cpu_extension,FullPhysicalCpuPoolOwner,
        LoadedNativeCpuLayer,ChunkedNativeCpuBackend)
    from benchmarks.harness import paired_prefill_diagnostic
    began=time.monotonic()
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    device=torch.device('cuda',0);torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(1024**3/torch.cuda.get_device_properties(device).total_memory,device)
    torch.cuda.reset_peak_memory_stats(device)
    def budget():
        if time.monotonic()-began>args.deadline_seconds:raise TimeoutError('two-layer deadline')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>args.host_rss_cap_bytes:raise MemoryError('two-layer host cap')
    masters={};gates={};provenance={}
    for layer in (0,1):
        gate,cfg,provenance[layer]=load_router(args.model,layer);gates[layer]=gate.to(device)
        for expert in range(128):
            key=ExpertKey(layer,expert,'bf16');masters[key]=load_expert(args.model,key);budget()
    hashes={k:w.hashes() for k,w in masters.items()}
    pinned=pin_cpu_master_storage(masters,block_cap_bytes=256*1024**2,check_budget=budget)
    if hashes!={k:w.hashes() for k,w in masters.items()}:raise AssertionError('pinned source bytes changed')
    native,info=load_cpu_extension(args.cpu_extension)
    pool=backend=core=None;loaded={};rows=[]
    try:
        pool=FullPhysicalCpuPoolOwner(native,
            threads_per_subpool=getattr(args,'pdscope_cpu_threads',24))
        bylayer={l:{k:w for k,w in masters.items() if k.layer==l} for l in (0,1)}
        for layer in (0,1):loaded[layer]=LoadedNativeCpuLayer(native,pool.pool,bylayer[layer],layer=layer,source_owners=(masters,),max_tokens=64,numa_parts=2)
        backend=ChunkedNativeCpuBackend(pool.pool,loaded,masters,source_owners=(masters,loaded),max_tokens=64,
            record_calls=False)
        packed_probe=probe_packed_cpu(backend,masters) if args.pdscope_cpu_batch!='off' else None
        grouping_probe=probe_compact_groups(gates[0]) if args.pdscope_dispatch=='compact' else None
        hot,hot_info=load_hot_profile(args.hot_file,masters,16)
        costs=FileCosts(args.cost_table)
        expert_size=next(iter({w.nbytes for w in masters.values()}))
        cache_pool=torch.empty(32*expert_size,dtype=torch.uint8,device=device)
        staging_pool=tuple(torch.empty(expert_size,dtype=torch.uint8,device=device)
                           for _ in range(args.pdscope_demand_slots))
        core=PDScopeCudaRuntime(masters,hot_keys=hot,speculative_slots=16,costs=costs,cpu_backend=backend,
                                capture_events=True,cache_storage=cache_pool,staging_storage=staging_pool,
                                resident_graphs=args.pdscope_graphs,
                                graph_cost_mode=args.pdscope_graph_cost,cpu_batch=args.pdscope_cpu_batch,
                                balance_profile=args.pdscope_balance,dispatch=args.pdscope_dispatch,
                                merge=args.pdscope_merge,input_pregather=args.pdscope_input_pregather,
                                graph_output_pool=args.pdscope_graph_output_pool,
                                native_dispatch=load_native_layer_dispatch() if args.pdscope_native_dispatch else None,
                                demand_slots=args.pdscope_demand_slots)
        calibration_tokens=core.cache.stats().get('physical_demand_tokens',0)
        generator=torch.Generator(device='cpu').manual_seed(20261011)
        with torch.inference_mode():
            for step,n in enumerate((16,1,1,1,1)):
                x=torch.randn((n,2048),generator=generator).to(torch.bfloat16).to(device)
                for layer in (0,1):
                    budget()
                    out,routing,row=core.forward(x,gates[layer],layer=layer,step=step,
                        next_gate=gates.get(layer+1),attention_ms=0.)
                    logits=F.linear(x,gates[layer]);coeff,selected=F.softmax(logits,dim=-1,dtype=torch.float32).topk(8,dim=-1)
                    coeff=(coeff/coeff.sum(dim=-1,keepdim=True)).to(x.dtype)
                    if not torch.equal(selected,routing.selected) or not torch.equal(coeff,routing.weights):raise AssertionError('native gate changed')
                    ref=paired_prefill_diagnostic(x,routing,bylayer[layer],layer,row['decision']['placements'],backend,out)
                    batch_ok = (ref['async_vs_serial']['relative_l2'] <= 1e-3
                                and ref['async_vs_serial']['max_abs_over_reference_rms'] <= .05)
                    # A fused merge reorders the BF16 accumulation of the same
                    # operands, so it cannot be bitwise against the per-expert
                    # order; its explicit bound is recorded here and the
                    # authoritative accuracy check (native vs FP32) still applies.
                    # For a differently-ordered sum the serial reference is a
                    # different accumulation, not a more accurate value, so the
                    # fused path is judged by the FP32 contract below and only
                    # required to stay finite and order-sane here.
                    fused_ok = (ref['async_vs_serial']['relative_l2'] <= 1e-2
                                and ref['async_vs_serial']['finite'])
                    if args.pdscope_merge == 'fused':
                        order_ok = fused_ok
                    else:
                        order_ok = ref['async_vs_serial_bitwise']
                    if (not order_ok
                            or not ref['native_queue_vs_fp32']['finite']
                            or ref['native_queue_vs_fp32']['relative_l2']>.02):
                        raise AssertionError('two-layer numerical mismatch: '+str(ref))
                    row.update(input_sha256=tensor_hash(x),output_sha256=tensor_hash(out),reference=ref)
                    rows.append(row)
                    print(json.dumps(dict(step=step,layer=layer,mode=row['decision']['mode'],cpu=len(row['decision']['cpu_queue']),gpu=len(row['decision']['gpu_resident_queue'])+len(row['decision']['on_demand_queue']),prefetch=len(row['prefetch_issued']))),flush=True)
                    x=F.layer_norm(out+x,(2048,)) # declared synthetic next-layer activation
        core.poll(wait=True);stats=core.cache.stats();ex=core.executor
        if (stats['h2d_bytes_completed']!=stats['h2d_bytes_submitted'] or core.cache.active_demand
                or core.transient or core.pending or core.forecasts
                or ex.cpu_tasks or ex.loads or ex.computes
                or stats['physical_demand_tokens']!=320+calibration_tokens):raise AssertionError('two-layer work/lifetime mismatch')
        if hashes!={k:w.hashes() for k,w in masters.items()}:raise AssertionError('original expert bytes changed')
        budget()
        return dict(status='completed',kind='pdscope_two_layer_check',measured_performance=False,
            packed_cpu_probe=packed_probe,compact_grouping_probe=grouping_probe,
            graph_cost_calibration=core.graph_cost_calibration,
            model=args.model,scope='synthetic16-token prefill and4decode; original layer0/1 experts; no attention model',
            routing=provenance,rows=rows,calibration_sources=costs.sources,attention_estimate_ms=0.,
            attention_estimate_scope='conservative zero; no attention in this component check',
            static_profile=hot_info,master_hashes_unchanged=True,master_storage=pinned,
            native_backend=dict(info,pool=pool.summary()),native_completed_chunks=backend.completed_calls,
            initial_stats=core.initial_stats,cache_stats=stats,cache_memory=core.cache.memory(),
            storage=ex.storage(),events=ex.events,
            gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
            gpu_allocator_cap_bytes=1024**3,
            host_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    finally:
        if core is not None:core.close()
        if 'staging_pool' in locals():staging_pool=None
        if 'cache_pool' in locals():cache_pool=None
        if backend is not None:backend.close()
        for obj in loaded.values():obj.close()
        if pool is not None:pool.close()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    # Measurement
    parser.add_argument('--mode',choices=('measure','two_layer'),default='measure',
                        help='measure = prefill/decode throughput protocol; '
                             'two_layer = numerical and CPU-service check')
    parser.add_argument('--lengths',nargs='+',type=int,default=[512,1024,2048,4096],
                        help='input lengths to measure, 128..4096 (default: 512 1024 2048 4096)')
    parser.add_argument('--repetitions',type=int,default=2,
                        help='measured requests per length after two warm-ups (default: 2)')
    parser.add_argument('--output',type=Path,
                        help='result.json path (default: runs/<UTC timestamp>/result.json)')
    parser.add_argument('--model',default=os.environ.get('PDSCOPE_MODEL',''),
                        help='checkpoint directory (or set PDSCOPE_MODEL)')
    parser.add_argument('--dataset',default=os.environ.get('PDSCOPE_DATASET',''),
                        help='ShareGPT json (or set PDSCOPE_DATASET)')
    parser.add_argument('--cost-table',type=Path,default=None,
                        help='measured per-expert cost table (default: costs/cost_table.json)')
    parser.add_argument('--hot-file',type=Path,default=Path('hot/qwen.txt'),
                        help='static hot-expert table (default: hot/qwen.txt)')
    parser.add_argument('--gpu-total-bytes',type=int,default=24*1024**3,
                        help='hard GPU memory budget for the process (default: 24 GiB)')
    # Execution configuration (the defaults are the measured reference setup)
    parser.add_argument('--cache-slots',type=int,default=40,
                        help='resident experts per layer (default: 40)')
    parser.add_argument('--speculative-slots',type=int,default=128,
                        help='slots the prefetcher may fill beyond the static hot set (default: 128)')
    parser.add_argument('--pdscope-cpu-threads',type=int,default=18,
                        help='worker threads per NUMA subpool (default: 18; 36 in total)')
    parser.add_argument('--pdscope-decode-cpu-targets',nargs='+',type=int,
                        help='decode CPU/GPU balance target(s) to try instead of the scheduler rule')
    parser.add_argument('--pdscope-graphs',action='store_true',default=True,
                        help=argparse.SUPPRESS)
    parser.add_argument('--no-pdscope-graphs',dest='pdscope_graphs',action='store_false',
                        help='execute the resident hot set eagerly instead of via CUDA graphs')
    parser.add_argument('--pdscope-graph-cost',choices=('eager_bound','measured'),default='eager_bound',
                        help='cost charged to a graph-replayed layer (default: eager_bound)')
    parser.add_argument('--pdscope-cpu-batch',choices=('off','decode','both'),default='both',
                        help='batch the per-expert CPU submissions of a layer (default: both)')
    parser.add_argument('--pdscope-dispatch',choices=('reference','compact'),default='compact',
                        help='pack the CPU expert group into one native submission (default: compact)')
    parser.add_argument('--pdscope-native-dispatch',action='store_true',default=True,
                        help=argparse.SUPPRESS)
    parser.add_argument('--no-pdscope-native-dispatch',dest='pdscope_native_dispatch',
                        action='store_false',
                        help='submit the CPU expert group through ATen instead of the native call')
    parser.add_argument('--pdscope-merge',choices=('reference','fused'),default='fused',
                        help='fused = one weighted index_add over all experts (default: fused)')
    parser.add_argument('--pdscope-input-pregather',action='store_true',default=True,
                        help=argparse.SUPPRESS)
    parser.add_argument('--no-pdscope-input-pregather',dest='pdscope_input_pregather',
                        action='store_false',
                        help='gather each resident expert input separately')
    parser.add_argument('--pdscope-graph-output-pool',action='store_true',default=True,
                        help=argparse.SUPPRESS)
    parser.add_argument('--no-pdscope-graph-output-pool',dest='pdscope_graph_output_pool',
                        action='store_false',
                        help='allocate a fresh graph output per replay instead of reusing a ring')
    parser.add_argument('--pdscope-demand-slots',type=int,choices=(2,3,4),default=4,
                        help='demand staging depth (default: 4)')
    parser.add_argument('--attn-implementation',choices=('eager','sdpa','flash_attention_2'),
                        default='sdpa',
                        help='attention backend for the unchanged Transformers path (default: sdpa)')
    parser.add_argument('--torch-expandable-segments',action='store_true',default=True,
                        help=argparse.SUPPRESS)
    parser.add_argument('--no-torch-expandable-segments',dest='torch_expandable_segments',
                        action='store_false',
                        help='do not set PYTORCH_ALLOC_CONF=expandable_segments:True before CUDA init')
    parser.add_argument('--pdscope-balance',action='store_true',
                        help='record per-phase work and exposed host waits in the result')
    parser.add_argument('--pdscope-control-cpu',type=int,
                        help='pin the control thread to one logical CPU after the native pool is built')
    parser.add_argument('--pdscope-control-cores',choices=('off','idle'),default='off',
                        help='idle = pin the control thread to logical CPUs no worker claims')
    parser.add_argument('--cpu-extension',type=Path,
                        help='locally built CPU expert kernel (default: the one scripts/build_native.sh produced)')
    # Internal caps; keep them overridable but out of the way.
    parser.add_argument('--output-tokens',type=int,default=64,help=argparse.SUPPRESS)
    parser.add_argument('--host-rss-cap-bytes',type=int,default=256*1024**3,help=argparse.SUPPRESS)
    parser.add_argument('--deadline-seconds',type=int,default=900,help=argparse.SUPPRESS)
    args=parser.parse_args(argv)
    alloc_conf=None
    if getattr(args,'torch_expandable_segments',False):
        alloc_conf='expandable_segments:True'
        os.environ['PYTORCH_ALLOC_CONF']=alloc_conf
        os.environ['PYTORCH_CUDA_ALLOC_CONF']=alloc_conf
    if not args.model or not Path(args.model).is_dir():
        raise SystemExit('set PDSCOPE_MODEL (or --model) to the Qwen3-30B-A3B checkpoint directory')
    if not args.dataset or not Path(args.dataset).is_file():
        raise SystemExit('set PDSCOPE_DATASET (or --dataset) to the ShareGPT json file')
    root=Path(__file__).resolve().parents[1]
    # Defaults live in the checkout, so the tool works from any working directory.
    if args.cost_table is None:
        shipped=root/'costs'/'cost_table.json'
        args.cost_table=shipped if shipped.is_file() else None
    if not args.hot_file.is_absolute() and not args.hot_file.is_file():
        args.hot_file=root/args.hot_file
    if args.output is None:
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        args.output=Path('runs')/stamp/'result.json'
    args.output=args.output.resolve()
    if args.output.exists():
        raise SystemExit(f'refusing to overwrite an existing result: {args.output}')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    # One run per output parent: the lease serializes cooperating runs on this host
    # so two experiments never share the GPU by accident.
    lease_root=args.output.parent.parent if args.output.parent.parent.is_dir() else args.output.parent
    manifest={};started=time.monotonic()
    try:
        with experiment_lease(lease_root,run_id=args.output.parent.name,kind='pdscope_compare') as lease:
            manifest=capture_manifest(args,root)
            manifest.update(entry='benchmarks/pdscope_compare.py',lease=lease)
            (args.output.parent/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
            from benchmarks.harness import capture_native_stdio
            with capture_native_stdio(args.output.parent/'run.log'):
                   result=(run_two_layer(args) if args.mode=='two_layer' else run(args))
            args.output.write_text(json.dumps(result,indent=2)+'\n')
            manifest.update(status='completed',result_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest())
    except BaseException as exc:
        manifest.update(status='failed',error_type=type(exc).__name__,error=str(exc))
        (args.output.parent/'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        manifest.update(finished_at=datetime.now(timezone.utc).isoformat(),runner_seconds=time.monotonic()-started,
                        host_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        (args.output.parent/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(status=result['status'],output=str(args.output))),flush=True)


if __name__=='__main__':main()
