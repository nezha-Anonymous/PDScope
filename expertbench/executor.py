"""Minimal expert executor: CPU master copies and event-confirmed CUDA copies.

Single control thread, no scheduler or full-model inference. Its cache ledger
belongs to this executor lifetime and must not be reused after shutdown/errors.
"""
from dataclasses import dataclass
import contextlib
import time

import torch

from expertbench.adapters.qwen import PROJECTIONS, expert_forward, expert_groups, route
from expertbench.cache import LayerPartitionCache, DecodeScoreLayerCache, PrefillStaticCache, ExpertKey


def complete_if_ready(event, callback, *, wait=False):
    """Never advance cache lifetime merely because work has been enqueued."""
    if wait:
        event.synchronize()
    if not event.query():
        return False
    callback()
    return True


@dataclass
class PendingCompute:
    handle: object
    output: object
    event: object
    input_ref: object
    timing_start: object = None


@dataclass
class PendingCpuCompute:
    key: object
    future: object
    input_ref: object


class ResidentExpertGraphs:
    """Explicit n1 submission probe, bound to immutable static cache entries.

    Each expert owns a private graph pool. Replay, input copy and output clone
    use the executor's single compute stream; returned outputs never alias a
    buffer that a subsequent replay overwrites. Routing/merge remain outside.
    """
    def __init__(self, executor, keys, *, immutable_hotset=False, shared_pool=False,
                 pooled_output=False):
        keys = list(keys)
        allowed=(executor.cache.policy=='static' or
                 (immutable_hotset and set(keys)<=executor.cache.initial_hotset))
        if not allowed or executor.closed or executor.computes or executor.loads:
            raise ValueError('resident graphs require an idle static executor')
        self.executor, self.stream = executor, executor.compute_stream
        self.entries, self.calls, self.closed = {}, 0, False
        # A captured replay writes into one fixed buffer, so preserving the
        # result needs a copy.  Allocating a fresh tensor per replay makes the
        # allocator retire a block per expert per layer; a two-deep per-key ring
        # of permanent buffers removes that churn (one token per replay, so the
        # ring is tiny) while keeping the previous result alive until the next
        # replay of the same expert.
        # False: clone per replay. True: hand out one of two permanent buffers
        # per key, so the allocator does not retire a block per expert per layer.
        self.pooled_output = pooled_output
        self.output_ring, self.output_slot = {}, {}
        if not keys or len(set(keys)) != len(keys):
            raise ValueError('resident graphs require distinct resident keys')
        torch.cuda.synchronize(executor.device)
        started = time.perf_counter_ns()
        before = self._memory()
        pool=torch.cuda.graph_pool_handle() if shared_pool else None
        try:
            with torch.inference_mode(), torch.cuda.stream(self.stream):
                for key in keys:
                    view = executor.cache.lookup(key)
                    if view is None or view.state != 'ready' or view.ticket.pool != 'cache':
                        raise ValueError('resident graph weights must already occupy a ready cache entry')
                    tensors = executor.copies[view.ticket][1]
                    hidden = torch.linspace(-.5,.5,tensors['gate_proj'].shape[1],
                        dtype=torch.bfloat16,device=executor.device).reshape(1,-1)
                    for _ in range(3):
                        expected = expert_forward(hidden, tensors)
                    self.stream.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=self.stream, pool=pool):
                        output = expert_forward(hidden, tensors)
                    self.entries[key] = (view.ticket, dict(tensors), hidden, output, graph)
                    if self.pooled_output is True:
                        self.output_ring[key] = [output.clone(), output.clone()]
                        self.output_slot[key] = 0
                    graph.replay()
                    self.stream.synchronize()
                    if not torch.equal(output, expected):
                        raise ValueError('resident graph setup differs from eager expert math')
            self.setup = dict(graph_count=len(keys), pooled_output=self.pooled_output,
                preparation_ns=time.perf_counter_ns()-started,
                eager_warmup_calls=3*len(keys), validation_replays=len(keys),
                private_pools=not shared_pool, immutable_hotset=immutable_hotset,
                input_shape=[1, hidden.shape[1]],
                input_copy_and_output_clone_inside_clock=True,
                allocator_before=before, allocator_after=self._memory(),
                scope='same expert_forward; fixed static resident weights, single compute stream; no route or merge capture')
        except BaseException:
            self.close()
            raise

    def _memory(self):
        return dict(allocated_bytes=torch.cuda.memory_allocated(self.executor.device),
                    reserved_bytes=torch.cuda.memory_reserved(self.executor.device))

    def __call__(self, key, inputs, tensors):
        if self.closed or self.executor.closed or key not in self.entries:
            raise ValueError('closed or unknown resident graph')
        ticket, bound, hidden, output, graph = self.entries[key]
        view = self.executor.cache.lookup(key)
        # The weight dict handed in by the executor is the same object for a given
        # ticket, so the element-wise identity walk is only needed when it changes.
        if tensors is not bound and (set(tensors) != set(bound)
                                    or any(tensors[n] is not bound[n] for n in bound)):
            raise ValueError('resident graph weight binding changed')
        if (view is None or view.ticket != ticket or view.state != 'in_use'
                or inputs.shape != hidden.shape or inputs.dtype != hidden.dtype or inputs.device != hidden.device
                or torch.cuda.current_stream(inputs.device) != self.stream):
            raise ValueError('resident graph binding, input or stream changed')
        hidden.copy_(inputs)
        graph.replay()
        if getattr(self, 'pooled_output', False):
            ring = self.output_ring[key]
            slot = 1 - self.output_slot[key]
            self.output_slot[key] = slot
            ring[slot].copy_(output)
            result = ring[slot]
        else:
            result = output.clone()
        self.calls += 1
        return result

    def close(self):
        if not self.closed:
            self.stream.synchronize()
            for _, _, _, _, graph in self.entries.values():
                graph.reset()
            self.entries.clear()
            self.closed = True


def initialize_cpu_thread(expected_threads):
    """Complete Torch's thread-local setup before the first BLAS/ATen call."""
    import os
    import threading
    started = time.perf_counter_ns()
    # In the installed OpenMP backend, this calls lazy_init_num_threads and
    # applies the owner's configured count to this thread's OpenMP/MKL state.
    actual = torch.get_num_threads()
    if actual != expected_threads:
        raise RuntimeError('CPU worker initialization differs from owner thread count')
    return {'method': 'torch_get_num_threads_before_first_compute',
        'expected_threads': expected_threads, 'torch_threads': actual,
        'native_thread_id': threading.get_native_id(), 'affinity': sorted(os.sched_getaffinity(0)),
        'started_host_ns': started, 'completed_host_ns': time.perf_counter_ns()}


class ExpertExecutor:
    def __init__(self, cache, masters, device=0, *, pinned_source_mode="all_masters", profile_stages=False, capture_events=True, streams=None, initialize_cpu_worker=True, cpu_backend=None, cpu_staging_copy="torch", cache_storage=None, staging_storage=None):
        if pinned_source_mode not in ("all_masters", "lazy_pinned", "single_staging"):
            raise ValueError("unknown pinned source mode")
        if cpu_staging_copy not in ("torch", "memcpy"):
            raise ValueError("unknown CPU staging copy implementation")
        self.cpu_staging_copy = cpu_staging_copy
        if type(initialize_cpu_worker) is not bool:
            raise ValueError('CPU worker initialization must be explicitly boolean')
        self.initialize_cpu_worker = initialize_cpu_worker
        if cpu_backend is not None and not callable(cpu_backend):
            raise ValueError('CPU backend must be an explicit synchronous callable')
        self.cpu_backend = cpu_backend
        self.gpu_backend = None  # Optional explicitly prepared resident submission probe.
        # One default-stream join per submission group instead of one per expert.
        # Off by default: opening a group after materialising every input row on
        # the default stream is only a win when the whole group is submitted at once.
        self.group_stream_sync = False
        # Host-only stage counters for the per-expert submission path.  Off by
        # default (no perf_counter calls at all); enabled by an explicit probe.
        self.cpu_worker_init = None
        self.pinned_source_mode = pinned_source_mode
        self.profile_stages = profile_stages
        self.capture_events = capture_events
        self.stage_timings, self.load_timing_starts = {}, {}
        self.source_event = None
        self.host_staging = None
        self.last_source_key, self.staged_source = None, {}
        self.cache, self.masters = cache, dict(masters)
        if set(cache.expert_bytes) != set(self.masters):
            raise ValueError("cache and master weight registries differ")
        for key, weights in self.masters.items():
            if weights.nbytes != cache.expert_bytes[key] or any(t.device.type != "cpu" or t.dtype != torch.bfloat16
                                                                for t in weights.tensors.values()):
                raise ValueError("master bytes/device/dtype do not match the cache contract")
        self.device = torch.device("cuda", device)
        # Caller-owned streams are for sequential executor lifetimes only.
        # Their BLAS/runtime workspaces can then persist once per model run,
        # instead of allocating another workspace on every new compute stream.
        if streams is None:
            self.load_stream = torch.cuda.Stream(device=self.device)
            self.compute_stream = torch.cuda.Stream(device=self.device)
        else:
            if (not isinstance(streams, tuple) or len(streams) != 2
                    or any(not isinstance(stream, torch.cuda.Stream) or stream.device != self.device for stream in streams)
                    or streams[0].cuda_stream == streams[1].cuda_stream
                    or any(not stream.query() for stream in streams)):
                raise ValueError('shared executor streams must be a distinct idle pair on the same device')
            self.load_stream, self.compute_stream = streams
        self.copies, self.loads, self.computes = {}, {}, {}
        self.closed = False
        self.cpu_pool, self.cpu_tasks = None, {}
        self.cache_storage = cache_storage
        self._cache_pool_slots, self._cache_free_slots = {}, []
        if cache_storage is not None:
            sizes = set(cache.expert_bytes.values())
            if (len(sizes) != 1 or not isinstance(cache_storage, torch.Tensor)
                    or cache_storage.device != self.device or cache_storage.dtype != torch.uint8
                    or cache_storage.ndim != 1 or not cache_storage.is_contiguous()
                    or cache_storage.numel() != cache.cache_bytes or cache.cache_bytes % next(iter(sizes))):
                raise ValueError('borrowed cache storage must exactly cover whole equal-size GPU slots')
            self._cache_slot_bytes = next(iter(sizes))
            self._cache_free_slots = list(reversed(range(cache.cache_bytes // self._cache_slot_bytes)))
        self.shared_staging = staging_storage is not None
        # Staging is really preallocated: payload smaller than a slot does not
        # reduce its reserved bytes. GPU cache copies are allocated as needed.
        with torch.cuda.device(self.device), torch.cuda.stream(self.load_stream):
            if staging_storage is None:
                self.staging = [torch.empty(size, dtype=torch.uint8, device=self.device)
                                for size in cache.staging_slot_bytes]
            else:
                if (not isinstance(staging_storage, tuple) or len(staging_storage) != len(cache.staging_slot_bytes)
                        or any(not isinstance(t, torch.Tensor) or t.device != self.device or t.dtype != torch.uint8
                               or t.ndim != 1 or not t.is_contiguous() or t.numel() != size
                               for t, size in zip(staging_storage, cache.staging_slot_bytes))):
                    raise ValueError('borrowed staging must match every declared GPU slot')
                spans = sorted((t.data_ptr(), t.data_ptr()+t.numel()) for t in staging_storage)
                if any(a[1] > b[0] for a,b in zip(spans, spans[1:])):
                    raise ValueError('borrowed demand buffers must not overlap')
                if cache_storage is not None:
                    cache_span = (cache_storage.data_ptr(), cache_storage.data_ptr()+cache_storage.numel())
                    if any(cache_span[0] < end and start < cache_span[1]
                           for start, end in spans):
                        raise ValueError('borrowed cache and staging buffers must not overlap')
                self.staging = list(staging_storage)
        self.pinned = ({key: {name: tensor.pin_memory() for name, tensor in weights.tensors.items()}
                        for key, weights in self.masters.items()} if pinned_source_mode == "all_masters" else {})
        if pinned_source_mode == "single_staging":
            self.host_staging = torch.empty(max(w.nbytes for w in self.masters.values()),
                                            dtype=torch.uint8, pin_memory=True)
        self.events = []
        self._free_events = []
        self.source_metrics = dict(pin_experts=0, pin_bytes=0, pin_seconds=0.,
                                   staging_copies=0, staging_bytes=0)

    def prepare_source(self, key):
        """Do not overwrite a pinned source until its preceding DMA completes."""
        self._open()
        if self.pinned_source_mode == "all_masters":
            return self.pinned[key]
        if self.pinned_source_mode == "lazy_pinned":
            if key not in self.pinned:
                started = time.perf_counter()
                self.pinned[key] = {name: tensor.pin_memory() for name, tensor in self.masters[key].tensors.items()}
                # Some lightweight lifecycle tests construct an executor without
                # CUDA initialization. Counters belong to real owned instances.
                if hasattr(self, 'source_metrics'):
                    self.source_metrics['pin_experts'] += 1
                    self.source_metrics['pin_bytes'] += self.masters[key].nbytes
                    self.source_metrics['pin_seconds'] += time.perf_counter()-started
            return self.pinned[key]
        profile = getattr(self, "profile_stages", False)
        wait_start = time.perf_counter_ns() if profile else None
        if self.source_event is not None:
            self.source_event.synchronize()
        copy_start = time.perf_counter_ns() if profile else None
        views, offset = {}, 0
        for name in PROJECTIONS:
            tensor = self.masters[key].tensors[name]
            size = tensor.numel() * tensor.element_size()
            view = self.host_staging[offset:offset + size].view(tensor.dtype).view(tensor.shape)
            if getattr(self, 'cpu_staging_copy', 'torch') == 'memcpy':
                import ctypes
                if not tensor.is_contiguous() or not view.is_contiguous():
                    raise ValueError('byte staging requires contiguous identical BF16 layouts')
                ctypes.memmove(view.data_ptr(), tensor.data_ptr(), size)
            else:
                view.copy_(tensor)
            views[name] = view
            offset += size
        self.last_source_key, self.staged_source = key, views
        if hasattr(self, 'source_metrics'):
            self.source_metrics['staging_copies'] += 1
            self.source_metrics['staging_bytes'] += offset
        if profile:
            self.stage_timings.update(source_wait_ms=(copy_start - wait_start) / 1e6,
                                      cpu_weight_stage_ms=(time.perf_counter_ns() - copy_start) / 1e6)
        return views

    def _open(self):
        if self.closed:
            raise RuntimeError("executor has been shut down; its ledger cannot be reused")

    def reclaim(self):
        self._open()
        for ticket in self.cache.take_retired():
            if ticket in self.loads or any(
                    (ticket in getattr(p, 'handles', ())) or
                    (getattr(p, 'handle', None) is not None and p.handle.ticket == ticket)
                    for p in self.computes.values()):
                raise RuntimeError("cache attempted to retire a copy with pending backend work")
            self.copies.pop(ticket, None)
            pooled = getattr(self, '_cache_pool_slots', {})
            if ticket in pooled:
                self._cache_free_slots.append(pooled.pop(ticket))
            if getattr(self, "capture_events", True):
                self.events.append({"event": "retired", "generation": ticket.generation, "pool": ticket.pool})

    def _cache_flat(self, ticket, nbytes):
        if getattr(self, 'cache_storage', None) is None:
            return torch.empty(nbytes, dtype=torch.uint8, device=self.device)
        if (ticket.pool != 'cache' or nbytes != self._cache_slot_bytes
                or ticket in self._cache_pool_slots or not self._cache_free_slots):
            raise RuntimeError('cache pool slot ownership or size mismatch')
        slot = self._cache_free_slots.pop()
        self._cache_pool_slots[ticket] = slot
        return self.cache_storage[slot*nbytes:(slot+1)*nbytes]

    def load(self, key, *, origin="demand", admit=True):
        self._open()
        ticket = self.cache.begin_load(key, origin=origin, admit=admit)
        self.reclaim()  # Old physical copies released before allocating new ones.
        if ticket in self.copies:
            return ticket
        weights = self.masters[key]
        source = self.prepare_source(key)
        with torch.cuda.device(self.device), torch.cuda.stream(self.load_stream):
            flat = (self.staging[ticket.slot] if ticket.pool == "demand_staging" else
                    self._cache_flat(ticket, weights.nbytes))
            views, offset = {}, 0
            for name in PROJECTIONS:
                tensor = weights.tensors[name]
                size = tensor.numel() * tensor.element_size()
                views[name] = flat[offset:offset + size].view(tensor.dtype).view(tensor.shape)
                offset += size
            self.copies[ticket] = (flat, views)
            self.cache.submit(ticket)
            if self.profile_stages:
                start = torch.cuda.Event(enable_timing=True)
                start.record(self.load_stream)
                self.load_timing_starts[ticket] = start
            for name in PROJECTIONS:
                views[name].copy_(source[name], non_blocking=True)
            # One event after all three projections: it is the completion signal
            # for the whole expert, and it is what keeps the staging slot (and
            # the pinned source) alive until the copy has actually landed.
            event = torch.cuda.Event(enable_timing=self.profile_stages)
            event.record(self.load_stream)
            self.loads[ticket] = event
            if self.pinned_source_mode == "single_staging":
                self.source_event = event
        if getattr(self, "capture_events", True):
            self.events.append({"event": "load_submitted", "generation": ticket.generation, "pool": ticket.pool,
                                "expert_bytes": weights.nbytes, "source_pinned": all(t.is_pinned() for t in source.values()),
                                "pinned_source_mode": self.pinned_source_mode,
                                "cpu_staging_copy": self.cpu_staging_copy,
                                "ledger_state": self.cache.lookup(key).state})
        return ticket

    def finish_load(self, ticket, *, wait=False):
        self._open()
        event = self.loads[ticket]
        first_query = event.query()
        if wait:
            event.synchronize()
        if not event.query():
            return False
        self.cache.complete_load(ticket)
        del self.loads[ticket]
        if self.profile_stages:
            self.stage_timings["weight_dma_gpu_span_ms"] = self.load_timing_starts.pop(ticket).elapsed_time(event)
        if getattr(self, "capture_events", True):
            self.events.append({"event": "load_completed", "generation": ticket.generation,
                                "query_before_wait": first_query, "query_after_wait": event.query(),
                                "ledger_state": self.cache.lookup(ticket.key).state})
        return True

    def refresh_prefill(self,identity,counts,*,valid_counts=None):
        """Execute one already revealed layer's refresh before its true demand.

        This correctness bridge waits for real DMA and accounts its full host
        cost. The two host intervals are not isolated GPU kernel timings.
        """
        self._open()
        if (not isinstance(self.cache,PrefillStaticCache) or self.loads or self.computes or self.cpu_tasks):
            raise ValueError('prefill refresh requires its dedicated cache and no outstanding executor work')
        self.reclaim()
        before=self.cache.stats()
        started=time.perf_counter_ns()
        plan=self.cache.prepare_prefill(identity,counts,valid_token_counts=valid_counts)
        planned=time.perf_counter_ns()
        for ticket in plan['tickets']:
            actual=self.load(ticket.key,origin='refresh')
            if actual!=ticket or actual not in self.loads:
                raise AssertionError('executor did not submit the reserved prefill refresh generation')
            self.finish_load(actual,wait=True)
        self.reclaim()
        self.cache.finish_prefill_refresh(identity)
        finished=time.perf_counter_ns()
        stats=self.cache.stats()
        submitted=stats.get('refresh_h2d_bytes_submitted',0)-before.get('refresh_h2d_bytes_submitted',0)
        completed=stats.get('refresh_h2d_bytes_completed',0)-before.get('refresh_h2d_bytes_completed',0)
        expected=sum(self.masters[t.key].nbytes for t in plan['tickets'])
        if (submitted!=completed or completed!=expected or self.loads or self.computes
                or self.storage()['cache_tensor_bytes']!=self.cache.memory()['cache_payload_bytes']):
            raise AssertionError('physical refresh bytes/storage differ from the completed ledger')
        return {k:v for k,v in plan.items() if k!='tickets'} | dict(
            request=identity.request,step=identity.step,refresh_h2d_bytes_submitted=submitted,
            physical_token_counts=[counts.get(ExpertKey(identity.layer,e,'bf16'),0) for e in range(self.cache.experts)],
            valid_token_counts=[(counts if valid_counts is None else valid_counts).get(ExpertKey(identity.layer,e,'bf16'),0)
                                for e in range(self.cache.experts)],
            refresh_h2d_bytes_completed=completed,planning_host_ms=(planned-started)/1e6,
            copy_and_wait_host_ms=(finished-planned)/1e6,
            timing_scope='synchronous host completion intervals; not isolated DMA/kernel cost')

    def begin_gpu_group(self):
        """Join the default stream once for a span of already-materialised inputs.

        Every input of the span must already exist on the default stream when the
        group opens (pre-gathered rows, or the shared activation of a decode step);
        the device/stream contexts are then entered once instead of per expert.
        """
        self._open()
        self.group_stream_sync = True
        self._group_device = torch.cuda.device(self.device)
        self._group_device.__enter__()
        self._group_stream = torch.cuda.stream(self.compute_stream)
        self._group_stream.__enter__()
        self.compute_stream.wait_stream(torch.cuda.current_stream(self.device))

    def end_gpu_group(self):
        if getattr(self, '_group_stream', None) is not None:
            self._group_stream.__exit__(None, None, None)
            self._group_stream = None
        if getattr(self, '_group_device', None) is not None:
            self._group_device.__exit__(None, None, None)
            self._group_device = None
        self.group_stream_sync = False

    def gpu(self, key, inputs):
        self._open()
        if inputs.device != self.device or inputs.dtype != torch.bfloat16:
            raise ValueError("GPU input must already be BF16 on the configured device")
        if inputs.ndim != 2 or inputs.shape[1] != self.masters[key].tensors["gate_proj"].shape[1]:
            raise ValueError("invalid expert input shape")
        handle = self.cache.acquire(key)
        flat, tensors = self.copies[handle.ticket]
        input_stream = torch.cuda.current_stream(self.device)
        span = self.group_stream_sync
        with (contextlib.nullcontext() if span else torch.cuda.device(self.device)), \
             (contextlib.nullcontext() if span else torch.cuda.stream(self.compute_stream)), \
             torch.inference_mode():
            if not span:
                self.compute_stream.wait_stream(input_stream)
            flat.record_stream(self.compute_stream)
            inputs.record_stream(self.compute_stream)
            start = None
            if self.profile_stages:
                start = torch.cuda.Event(enable_timing=True)
                start.record(self.compute_stream)
                submit_start = time.perf_counter_ns()
            backend = self.gpu_backend
            output = expert_forward(inputs, tensors) if backend is None else backend(key, inputs, tensors)
            if self.profile_stages:
                self.stage_timings["gpu_forward_host_submit_ms"] = (time.perf_counter_ns() - submit_start) / 1e6
            event = (self._free_events.pop() if self._free_events
                     else torch.cuda.Event(enable_timing=self.profile_stages))
            event.record(self.compute_stream)
        pending = PendingCompute(handle, output, event, inputs, start)
        self.computes[handle] = pending
        if getattr(self, "capture_events", True):
            self.events.append({"event": "compute_submitted", "generation": handle.ticket.generation,
                                "ledger_state": self.cache.lookup(key).state})
        return pending

    def _recycle_event(self, event):
        """Reuse recording events; only called once the event is completed."""
        if not getattr(self, 'profile_stages', False):
            self._free_events.append(event)

    def finish_gpu(self, pending, *, wait=False):
        self._open()
        if self.computes.get(pending.handle) is not pending:
            raise ValueError("unknown or already completed GPU computation")
        if not complete_if_ready(pending.event, lambda: self.cache.release(pending.handle), wait=wait):
            return None
        del self.computes[pending.handle]
        if pending.timing_start is not None:
            self.stage_timings["gpu_forward_device_span_ms"] = pending.timing_start.elapsed_time(pending.event)
        pending.input_ref = None
        if getattr(self, "capture_events", True):
            self.events.append({"event": "compute_completed", "generation": pending.handle.ticket.generation,
                                "query_after_wait": pending.event.query()})
        completed, pending.event = pending.event, None
        self._recycle_event(completed)
        self.reclaim()
        return pending.output

    def cpu(self, key, inputs):
        self._open()
        if inputs.device.type != "cpu" or inputs.dtype != torch.bfloat16:
            raise ValueError("CPU input must be BF16")
        with torch.inference_mode():
            start = time.perf_counter_ns() if self.profile_stages else None
            backend = getattr(self, 'cpu_backend', None)
            if key not in self.masters:
                raise ValueError('CPU backend key is outside the master registry')
            output = (expert_forward(inputs, self.masters[key].tensors) if backend is None
                      else backend(key, inputs))
            if (not isinstance(output,torch.Tensor) or output.device.type!='cpu'
                    or output.dtype!=torch.bfloat16 or output.shape!=inputs.shape):
                raise ValueError('CPU backend must return completed unweighted CPU BF16 input-shaped output')
            if self.profile_stages:
                self.stage_timings["cpu_forward_ms"] = (time.perf_counter_ns() - start) / 1e6
        self.cache.finish_cpu(key)
        return output

    def submit_cpu(self, key, inputs):
        """CPU math only in the worker; the owner retains all ledger authority."""
        from concurrent.futures import ThreadPoolExecutor
        import threading
        import os
        self._open()
        backend = getattr(self, 'cpu_backend', None)
        if backend is not None and not callable(getattr(backend, 'submit', None)):
            raise ValueError('external CPU backends are only qualified for synchronous execution')
        if (key in self.cpu_tasks or key not in self.cache.unfinished_demand_keys
                or inputs.device.type != 'cpu' or inputs.dtype != torch.bfloat16
                or inputs.ndim != 2 or inputs.shape[1] != self.masters[key].tensors['gate_proj'].shape[1]):
            raise ValueError('CPU submission requires unique active demand and CPU BF16 input')
        if backend is not None:
            pending = PendingCpuCompute(key, backend.submit(key, inputs), inputs)
            self.cpu_tasks[key] = pending
            return pending
        if self.cpu_pool is None:
            expected_threads = torch.get_num_threads()
            def initialize():
                self.cpu_worker_init = initialize_cpu_thread(expected_threads)
            self.cpu_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='expertbench-cpu',
                initializer=initialize if getattr(self, 'initialize_cpu_worker', True) else None)
        tensors = self.masters[key].tensors
        def work():
            started = time.perf_counter_ns()
            with torch.inference_mode():
                output = expert_forward(inputs, tensors)
            ended = time.perf_counter_ns()
            return output, {'start_host_ns': started, 'end_host_ns': ended,
                            'native_thread_id': threading.get_native_id(), 'torch_threads': torch.get_num_threads(),
                            'affinity': sorted(os.sched_getaffinity(0))}
        pending = PendingCpuCompute(key, self.cpu_pool.submit(work), inputs)
        self.cpu_tasks[key] = pending
        return pending

    def finish_cpu(self, pending, *, wait=False):
        self._open()
        if self.cpu_tasks.get(pending.key) is not pending:
            raise ValueError('CPU completion does not own an active task')
        if not wait and not pending.future.done():
            return None
        output, timing = pending.future.result()  # Failure cannot mark demand complete.
        self.cache.finish_cpu(pending.key)
        del self.cpu_tasks[pending.key]
        pending.input_ref = None
        return output, timing

    def submit_cpu_many(self,jobs):
        """Keep per-expert ownership while the native backend packs row work."""
        self._open()
        backend=getattr(self,'cpu_backend',None)
        if not jobs or not callable(getattr(backend,'submit_many',None)):
            raise ValueError('packed CPU execution needs the qualified native backend')
        active=self.cache.unfinished_demand_keys  # one sorted() for the whole group
        for key,inputs in jobs.items():
            if (key in self.cpu_tasks or key not in active
                    or inputs.device.type!='cpu' or inputs.dtype!=torch.bfloat16
                    or inputs.ndim!=2 or inputs.shape[1]!=self.masters[key].tensors['gate_proj'].shape[1]):
                raise ValueError('packed CPU submission requires unique active BF16 demands')
        futures=backend.submit_many(jobs)
        if set(futures)!=set(jobs):raise AssertionError('packed backend lost expert ownership')
        pending={key:PendingCpuCompute(key,futures[key],inputs) for key,inputs in jobs.items()}
        self.cpu_tasks.update(pending)
        return pending

    def finish_cpu_group(self, pendings, *, wait=True):
        """Complete a packed CPU submission with ONE native queue sync.

        A packed call produces one native future per chunk plus per-expert row
        view; asking each row for its result re-syncs the whole queue once per
        expert.  The rows of one parent are all ready at the same instant, so
        the queue is joined once and the row intervals are sliced directly.
        ``finish_cpu`` keeps its per-task semantics for every other caller.
        """
        self._open()
        pendings = list(pendings)
        if not pendings:
            return {}
        for pending in pendings:
            if self.cpu_tasks.get(pending.key) is not pending:
                raise ValueError('CPU completion does not own an active task')
        parents = {id(getattr(p.future, 'parent', None)) for p in pendings}
        parent = getattr(pendings[0].future, 'parent', None) if len(parents) == 1 else None
        self.last_cpu_group = None
        if parent is not None:
            output, timing = parent.result()  # single queue join for every row
            # submit->sync span of the native queue: the actual CPU expert
            # service time of this group, independent of when the host asked.
            span = (timing.get('sync_completed_host_ns', 0) - timing.get('submit_host_ns', 0)) / 1e6
            self.last_cpu_group = dict(span_ms=span, rows=int(parent.hidden.shape[0]) if parent.hidden is not None else 0,
                                       experts=len({p.key.expert for p in pendings}))
        results = {}
        for pending in pendings:
            future = pending.future
            if parent is not None and future.parent is parent and parent.output is not None:
                output = parent.output[future.begin:future.end]
                timing = dict(parent.timing, packed_row_interval=[future.begin, future.end])
            else:
                output, timing = future.result()
            self.cache.finish_cpu(pending.key)
            del self.cpu_tasks[pending.key]
            pending.input_ref = None
            results[pending.key.expert] = output
            block = parent.output if parent is not None else None
        # The contiguous backend block lets a native tail stage every CPU expert
        # output with one copy; it is None for mixed/individual submissions.
        self.last_cpu_group_block = block if block is not None and len(pendings) > 1 else None
        return results

    def cache_tensor_bytes(self):
        """Live resident device payload without scanning immutable CPU masters."""
        return sum(flat.numel() * flat.element_size() for ticket, (flat, _) in self.copies.items()
                   if ticket.pool == "cache")

    def storage(self):
        return {"cache_tensor_bytes": self.cache_tensor_bytes(),
                "cache_storage_mode": ('borrowed_contiguous' if getattr(self, 'cache_storage', None) is not None else 'per_expert'),
                "cache_reserved_tensor_bytes": (self.cache_storage.numel() if getattr(self, 'cache_storage', None) is not None else self.cache_tensor_bytes()),
                "staging_tensor_bytes": sum(t.numel() * t.element_size() for t in self.staging),
                "staging_storage_mode": 'borrowed_shared' if getattr(self, 'shared_staging', False) else 'owned',
                "cpu_master_bytes": sum(w.nbytes for w in self.masters.values()),
                "pinned_source_bytes": (self.host_staging.numel() if self.host_staging is not None else
                                         sum(t.numel() * t.element_size() for ts in self.pinned.values() for t in ts.values()))}

    def shutdown(self):
        """Drain streams even on failure; discard this executor and ledger."""
        if not self.closed:
            backend = getattr(self, 'cpu_backend', None)
            if self.cpu_tasks and callable(getattr(backend, 'wait_all', None)):
                backend.wait_all()
            if self.cpu_pool is not None:
                self.cpu_pool.shutdown(wait=True)
                self.cpu_pool = None
            self.cpu_tasks.clear()
            self.load_stream.synchronize()
            self.compute_stream.synchronize()
            self.copies.clear()
            self.loads.clear()
            self.load_timing_starts.clear()
            self.computes.clear()
            self.staging.clear()
            self.cache_storage = None
            getattr(self, '_cache_pool_slots', {}).clear()
            getattr(self, '_cache_free_slots', []).clear()
            self.pinned.clear()
            self.staged_source.clear()
            self.host_staging, self.source_event = None, None
            self.closed = True
