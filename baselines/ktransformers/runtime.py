"""Synchronous platform interface for explicitly borrowed, loaded KT CPU layers.

This module neither loads an extension nor creates a worker pool. The caller
owns their construction, runtime qualification and final destruction. It must
keep the original weight buffers alive and declare the object's token capacity.
"""
import ctypes
import hashlib
import importlib.util
import os
from pathlib import Path
import threading

# Variants this repository can build from the pinned KT source. The build records
# its own tag (__cpu_variant__); the loader accepts the family and reports what it got.
PD_SCOPE_VARIANTS = ('avx2','avx512_vbmi','avx512_vnni','avx512_base','avx512_bf16')


def load_cpu_extension(path=None):
    """Load the locally built ktransformers BF16 expert kernel.

    The path defaults to the binary ``scripts/build_native.sh`` produced.
    """
    root = Path(__file__).resolve().parents[2]
    if path is None:
        built = root / 'expertbench' / 'native-build' / 'kt-kernel'
        candidates = sorted(built.glob('kt_kernel_ext*.so'))
        if not candidates:
            raise RuntimeError('the CPU expert kernel is not built yet; '
                               'run ./scripts/build_native.sh')
        path = candidates[0]
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location('kt_kernel_ext', path)
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    variant = getattr(native, '__cpu_variant__', None)
    if variant not in PD_SCOPE_VARIANTS:
        raise ValueError('CPU expert kernel must be a supported x86 BF16 build, got ' + repr(variant))
    try:
        native.moe.AVX2BF16_MOE
    except AttributeError as error:
        raise ValueError('CPU expert kernel has no AVX2 BF16 MoE binding') from error
    return native, dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                        variant=variant, operator='AVX2BF16_MOE')


class FullPhysicalCpuPoolOwner:
    """Own one CPUInfer pool with a declared thread count per NUMA subpool.

    It does not change main/Torch affinities. Native threads bind themselves
    through the kernel's own hwloc topology; their actual placement is validated.
    """
    def __init__(self, native, *, threads_per_subpool=24):
        self.owner_thread = threading.get_native_id()
        if type(threads_per_subpool) is not int or not 1 <= threads_per_subpool <= 24:
            raise ValueError('subpool thread count must be within the declared 24-core socket')
        allowed = os.sched_getaffinity(0)
        core_cpus = {}
        for cpu in sorted(allowed):
            base = Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
            identity = (int((base / 'physical_package_id').read_text()), int((base / 'core_id').read_text()))
            core_cpus.setdefault(identity, []).append(cpu)
        if len(core_cpus) != 48 or {p for p, _ in core_cpus} != {0, 1}:
            raise ValueError('full-CPU probe requires declared two-socket 48-physical-core cn18 topology')
        self.core_cpus = core_cpus
        self.closed = False
        self.pool = None
        self.thread_record = {}
        before = {p.name for p in Path('/proc/self/task').iterdir()}
        config = native.WorkerPoolConfig()
        config.subpool_count, config.subpool_numa_map = 2, [0, 1]
        config.subpool_thread_count = [threads_per_subpool, threads_per_subpool]
        self.threads_per_subpool = threads_per_subpool
        self.pool = native.CPUInfer(config)
        self.pool.sync()
        try:
            self.thread_record = {p.name: dict(name=(p / 'comm').read_text().strip(),
                affinity=sorted(os.sched_getaffinity(int(p.name))))
                for p in Path('/proc/self/task').iterdir() if p.name not in before}
            compute = [r for r in self.thread_record.values() if r['name'].startswith('numa_')]
            expected = 2 * threads_per_subpool
            if (len(compute) != expected or any(len(r['affinity']) != 1 for r in compute)
                    or any(r['affinity'][0] not in {c for v in core_cpus.values() for c in v}
                           for r in compute)):
                raise ValueError('native workers do not bind one declared physical core each')
        except BaseException:
            self.close()
            raise

    def summary(self):
        n=getattr(self,'threads_per_subpool',24)
        return dict(implementation='full_physical_cpu_pool', workers=2*n, subpool_threads=[n,n],
                    subpool_numa=[0,1], threads=self.thread_record, closed=self.closed,
                    main_affinity=sorted(os.sched_getaffinity(0)))

    def close(self):
        if threading.get_native_id() != self.owner_thread:
            raise RuntimeError('native pool must close on owner thread')
        if not self.closed:
            if self.pool is not None:
                self.pool.sync()
                self.pool = None
            self.closed = True
            if any((Path('/proc/self/task') / tid).exists() for tid in self.thread_record):
                raise RuntimeError('native threads remain after pool release')


PROJECTION_SHAPES = {'gate_proj': (768, 2048), 'up_proj': (768, 2048), 'down_proj': (2048, 768)}


class NativeCpuPoolOwner:
    """Own the verified cn18 NUMA0 pool: fifteen workers plus control CPU15."""
    @staticmethod
    def affinities():
        return {p.name: sorted(os.sched_getaffinity(int(p.name))) for p in Path('/proc/self/task').iterdir()}

    def __init__(self, native, *, preexisting_affinities):
        self.owner_thread = threading.get_native_id()
        observed = self.affinities()
        if (self.owner_thread != os.getpid() or observed != preexisting_affinities
                or str(self.owner_thread) not in observed
                or any(cpus != list(range(16)) for cpus in observed.values())):
            raise ValueError('native pool requires the unchanged declared runtime threads on CPU0-15')
        self.preexisting_affinities = observed
        self.thread_control = None
        self.closed = self.thread_cleanup_verified = False
        worker = native.WorkerPoolConfig()
        worker.subpool_count, worker.subpool_numa_map, worker.subpool_thread_count = 1, [0], [15]
        self.pool = native.CPUInfer(worker)

    def _owner(self):
        if threading.get_native_id() != self.owner_thread:
            raise RuntimeError('native pool must run on its creating control thread')

    def summary(self):
        return dict(implementation='native_cpu_pool_owner_v1', owner_thread=self.owner_thread,
            closed=self.closed, workers=15, numa_node=0, control_cpu=15,
            preexisting_threads=len(self.preexisting_affinities), bound=self.thread_control is not None,
            pool_reference_released=self.pool is None, thread_cleanup_verified=self.thread_cleanup_verified)

    def close(self):
        self._owner()
        if self.thread_cleanup_verified:
            return
        if not self.closed:
            self.pool.sync()
            self.pool = None
            self.closed = True
        expected = ({tid: [15] for tid in self.preexisting_affinities}
                    if self.thread_control is not None else self.preexisting_affinities)
        if self.affinities() != expected:
            raise RuntimeError('native pool threads remain or runtime affinities changed after owner release')
        self.thread_cleanup_verified = True


def cpu_expert_buffers(hidden, expert, *, max_tokens=1):
    """Validate and retain all pointers needed by one native queued call."""
    import torch
    if (type(max_tokens) is not int or not 1 <= max_tokens <= 64
            or type(expert) is not int or not 0 <= expert < 128 or hidden.device.type != 'cpu'
            or hidden.dtype != torch.bfloat16 or hidden.ndim != 2 or hidden.shape[1] != 2048
            or not 1 <= hidden.shape[0] <= max_tokens
            or not hidden.is_contiguous() or hidden.data_ptr() % 64):
        raise ValueError('native CPU interface requires aligned contiguous BF16 within the declared token capacity')
    tokens = hidden.shape[0]
    selected = (ctypes.c_int64 * tokens)(*([expert] * tokens))
    coefficient = (ctypes.c_float * tokens)(*([1.0] * tokens))
    count = ctypes.c_int32(tokens)
    output = torch.empty_like(hidden)
    if output.data_ptr() % 64:
        raise ValueError('CPU output is not 64-byte aligned')
    return selected, coefficient, count, output


def cpu_expert_forward(pool, moe, hidden, expert, *, max_tokens=1):
    """Return a completed, unweighted CPU BF16 expert result; no fallback."""
    selected, coefficient, count, output = cpu_expert_buffers(hidden, expert, max_tokens=max_tokens)
    pool.submit(moe.forward_task(ctypes.addressof(count), 1, ctypes.addressof(selected),
        ctypes.addressof(coefficient), hidden.data_ptr(), output.data_ptr(), False))
    pool.sync()
    return output


def _weight_identity(weights):
    import torch
    if set(weights.tensors) != set(PROJECTION_SHAPES):
        raise ValueError('native backend requires all three original projections')
    identity = []
    for name, shape in PROJECTION_SHAPES.items():
        tensor = weights.tensors[name]
        if (not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape
                or tensor.dtype != torch.bfloat16 or tensor.device.type != 'cpu'
                or not tensor.is_contiguous() or tensor.data_ptr() % 64):
            raise ValueError('native backend master layout differs from its loaded BF16 source')
        identity.append((name, id(tensor), tensor.data_ptr(), tensor._version))
    return tuple(identity)


class LoadedNativeCpuLayer:
    """Own one loaded kernel layer and its pointer lifetimes; borrow the pool."""
    def __init__(self, native, pool, masters, *, layer, source_owners, max_tokens=1, numa_parts=1):
        from expertbench.cache import ExpertKey
        if (type(layer) is not int or not 0 <= layer < 48
                or type(max_tokens) is not int or not 1 <= max_tokens <= 64
                or not isinstance(source_owners, tuple) or not source_owners
                or type(numa_parts) is not int or numa_parts not in (1, 2)):
            raise ValueError('native layer requires an explicit layer, capacity and source owners')
        expected = {ExpertKey(layer, expert, 'bf16') for expert in range(128)}
        if set(masters) != expected or any(w.key != k for k, w in masters.items()):
            raise ValueError('native layer requires exactly its 128 original expert slots')
        for weights in masters.values():
            _weight_identity(weights)
        self.owner_thread = threading.get_native_id()
        self.layer, self.max_tokens = layer, max_tokens
        self.numa_parts = numa_parts
        self.closed, self.loaded = False, False
        self._pool, self._masters, self._source_owners = pool, dict(masters), source_owners
        self._native = native
        self._mask = (ctypes.c_bool * 128)(*([False] * 128))
        self._mapping = (ctypes.c_int64 * 128)(*range(128))
        self._moe = self._config = None
        try:
            config = native.moe.MOEConfig(128, 8, 2048, 768, ctypes.addressof(self._mask))
            config.layer_idx, config.pool, config.max_len = layer, pool.backend_, max_tokens
            config.swiglu_limit = config.swiglu_alpha = 0.0
            # The *distributed wheel's NativeMoEWrapper* passes whole BF16
            # expert tensors in one outer pointer list even with multiple
            # NUMA pools. Its AVX TP wrapper owns partitioning. The older
            # uploaded AMX header's pre-sharded interface does not apply.
            for name, field in (('gate_proj', 'gate'), ('up_proj', 'up'), ('down_proj', 'down')):
                setattr(config, field + '_projs', [[masters[ExpertKey(layer, e, 'bf16')].tensors[name].data_ptr()
                                                   for e in range(128)]])
                setattr(config, field + '_scales', [[0] * 128])
            self._config = config
            self._moe = native.moe.AVX2BF16_MOE(config)
            pool.submit(self._moe.load_weights_task(ctypes.addressof(self._mapping)))
            pool.sync()
            self.loaded = True
        except BaseException as error:
            # A traceback may retain this partially initialized object. Release
            # its pool/weight references explicitly before its owner unwinds.
            try:
                self.close()
            except BaseException as cleanup:
                error.add_note('native layer cleanup failed: '+repr(cleanup))
            # The exception traceback also retains constructor arguments and
            # locals, even after the object's own references are cleared.
            pool = config = None
            raise

    def _owner(self):
        if threading.get_native_id() != self.owner_thread:
            raise RuntimeError('native layer must run on its creating control thread')

    def forward_task(self, *args):
        self._owner()
        if self.closed or not self.loaded:
            raise RuntimeError('native layer is closed or not loaded')
        return self._moe.forward_task(*args)

    def summary(self):
        return dict(implementation='loaded_native_cpu_layer_v1', layer=self.layer, max_tokens=self.max_tokens,
            numa_parts=self.numa_parts,
            loaded=self.loaded, closed=self.closed, borrowed_pool=True,
            retained_experts=len(self._masters), source_owner_groups=len(self._source_owners),
            weight_load_submissions=1 if self.loaded else 0, live_native_objects=int(self._moe is not None))

    def close(self):
        self._owner()
        if self.closed:
            return
        self._pool.sync()
        self._moe = None
        self._config = self._mask = self._mapping = None
        self._masters.clear()
        self._source_owners = ()
        self._pool = self._native = None
        self.closed = True


class NativeCpuModelOwner:
    """Own loaded layers and one pool; borrow aligned immutable CPU masters.

    Extension provenance, process runtime and before/after raw weight checks
    belong to the caller. No model, weight copying, per-call hashes or fallback
    are hidden here. The exposed backend executes completed n1 BF16 calls.
    """
    def __init__(self, native, masters, *, layers, source_owners, sample=None):
        from expertbench.cache import ExpertKey
        if (not isinstance(layers, (list,tuple)) or not layers
                or any(type(layer) is not int for layer in layers)
                or list(layers)!=list(range(len(layers))) or len(layers)>48
                or not isinstance(source_owners,tuple) or not source_owners
                or sample is not None and not callable(sample)):
            raise ValueError('native model owner requires consecutive layers from zero and explicit sources')
        expected={ExpertKey(layer,expert,'bf16') for layer in layers for expert in range(128)}
        if set(masters)!=expected or any(w.key!=k for k,w in masters.items()):
            raise ValueError('native model owner requires all 128 experts of each declared layer')
        for weights in masters.values():
            _weight_identity(weights)
        self.owner_thread=threading.get_native_id()
        self.registered_layers=list(layers)
        self.masters=masters
        self.source_owners=source_owners
        self.layers={}
        self.pool_owner=self.backend=None
        self.closed=False
        self.backend_after_close=None
        self.layers_after_close={}
        try:
            self.pool_owner=NativeCpuPoolOwner(native,preexisting_affinities=NativeCpuPoolOwner.affinities())
            for layer in layers:
                selected={k:masters[k] for k in expected if k.layer==layer}
                self.layers[layer]=LoadedNativeCpuLayer(native,self.pool_owner.pool,selected,
                    layer=layer,source_owners=(masters,*source_owners),max_tokens=1)
                if sample is not None:sample('native_loaded_layer_'+str(layer))
            self.backend=BorrowedNativeCpuBackend(self.pool_owner.pool,self.layers,masters,
                source_owners=(masters,*source_owners),max_tokens=1,record_calls=False)
        except BaseException as error:
            try:
                self.close()
            except BaseException as cleanup:
                error.add_note('native model owner cleanup failed: '+repr(cleanup))
            raise

    def _owner(self):
        if threading.get_native_id()!=self.owner_thread:
            raise RuntimeError('native model owner must run on its creating control thread')

    def summary(self):
        return dict(implementation='native_cpu_model_owner_v1',closed=self.closed,
            registered_layers=self.registered_layers,source_experts=len(self.masters) if self.masters is not None else 0,
            extra_master_copy_bytes=0,record_calls=False,
            backend=self.backend.summary() if self.backend is not None else self.backend_after_close,
            layers={str(k):v.summary() for k,v in self.layers.items()},layers_after_close=self.layers_after_close,
            pool=self.pool_owner.summary() if self.pool_owner is not None else None)

    def close(self):
        self._owner()
        if self.closed:return
        # All calls are synchronous, but native layer teardown still drains the
        # shared pool. Do not release it while a layer/backend retains it.
        if self.backend is not None:
            self.backend.close()
            self.backend_after_close=self.backend.summary()
            self.backend=None
        for layer in reversed(list(self.layers)):
            loaded=self.layers[layer]
            loaded.close()
            self.layers_after_close[str(layer)]=loaded.summary()
            del self.layers[layer]
        if self.pool_owner is not None:self.pool_owner.close()
        self.masters=None
        self.source_owners=()
        self.closed=True


class BorrowedNativeCpuBackend:
    """Callable ExpertExecutor backend with explicit single-thread ownership.

    All 128 experts must be registered for each borrowed layer. Identity checks
    detect Python/Torch storage replacement or mutation; the caller's source
    checksum audit remains responsible for writes through raw native pointers.
    Closing releases only this object's references, never the external pool.
    """
    def __init__(self, pool, layers, masters, *, source_owners, max_tokens=1, record_calls=True):
        from expertbench.cache import ExpertKey
        if (pool is None or not isinstance(layers, dict) or not layers
                or any(type(layer) is not int or not 0 <= layer < 48 or obj is None
                       for layer, obj in layers.items())
                or type(max_tokens) is not int or not 1 <= max_tokens <= 64
                or type(record_calls) is not bool or not isinstance(source_owners, tuple) or not source_owners):
            raise ValueError('native backend requires explicit loaded layers, owners and bounded capacity')
        expected = {ExpertKey(layer, expert, 'bf16') for layer in layers for expert in range(128)}
        if set(masters) != expected or any(weights.key != key for key, weights in masters.items()):
            raise ValueError('native backend registry must cover all 128 experts of each loaded layer')
        for layer, obj in layers.items():
            if isinstance(obj, LoadedNativeCpuLayer) and (obj.closed or not obj.loaded or obj.layer != layer
                    or obj._pool is not pool or obj.max_tokens < max_tokens
                    or any(masters[key] is not weights for key, weights in obj._masters.items())):
                raise ValueError('loaded native layer binding, capacity or pool differs from the backend')
        self._identities = {key: _weight_identity(weights) for key, weights in masters.items()}
        # Experts whose master identity has already been verified in this backend
        # lifetime; the masters are immutable after load and the driver re-hashes
        # them per request, so the per-row re-check only cost host time.
        self._validated_masters = set()
        self._masters = dict(masters)
        self._registry = masters
        self._layers = dict(layers)
        self._pool = pool
        self._source_owners = source_owners
        self._registered_layers = sorted(layers)
        self.max_tokens = max_tokens
        self.record_calls = record_calls
        self.owner_thread = threading.get_native_id()
        self.closed = False
        self._in_flight = False
        self.completed_calls = self.completed_tokens = 0
        self.calls = []

    def _owner(self):
        if threading.get_native_id() != self.owner_thread:
            raise RuntimeError('native backend must run on its creating control thread')

    def __call__(self, key, hidden):
        self._owner()
        if self.closed or self._in_flight:
            raise RuntimeError('native backend is closed or already executing')
        if key not in self._masters:
            raise ValueError('native backend requested an unregistered expert')
        weights = self._masters[key]
        if (len(self._registry) != len(self._masters) or self._registry.get(key) is not weights
                or _weight_identity(weights) != self._identities[key]):
            raise ValueError('native backend master storage changed after layer loading')
        self._in_flight = True
        try:
            output = cpu_expert_forward(self._pool, self._layers[key.layer], hidden, key.expert,
                                        max_tokens=self.max_tokens)
            self.completed_calls += 1
            self.completed_tokens += hidden.shape[0]
            if self.record_calls:
                raw = ctypes.string_at(output.data_ptr(), output.numel() * output.element_size())
                self.calls.append(dict(layer=key.layer, expert=key.expert, tokens=hidden.shape[0],
                                       output_sha256=hashlib.sha256(raw).hexdigest()))
            return output
        finally:
            self._in_flight = False

    def summary(self):
        return dict(implementation='borrowed_native_cpu_v1', closed=self.closed,
            borrowed_pool=True, registered_layers=self._registered_layers, max_tokens=self.max_tokens,
            owner_thread=self.owner_thread, active_experts=len(self._masters),
            source_owner_groups=len(self._source_owners), completed_calls=self.completed_calls,
            completed_tokens=self.completed_tokens, recorded_calls=len(self.calls),
            in_flight=self._in_flight)

    def close(self):
        self._owner()
        if self._in_flight:
            raise RuntimeError('cannot close a running native backend')
        self.closed = True
        self._layers.clear()
        self._masters.clear()
        self._identities.clear()
        self._registry = None
        self._pool = None
        self._source_owners = ()


class NativeCpuFuture:
    """Completion acknowledged only by the native owner's explicit sync."""
    def __init__(self, owner, key, hidden, buffers):
        import time
        self.owner, self.key, self.hidden, self.buffers = owner, key, hidden, buffers
        self.ready = False
        self.submitted_ns = time.perf_counter_ns()
        self.output, self.timing = None, None

    def done(self):
        return self.ready

    def result(self):
        self.owner.wait_all()
        if not self.ready:
            raise RuntimeError('native CPU task has no acknowledged completion')
        return self.output, self.timing


class QueuedNativeCpuBackend(BorrowedNativeCpuBackend):
    """Owner-thread submissions to the serial asynchronous CPU queue.

    No Python math worker is used. Input/IDs/count/coefficients/output survive
    until pool.sync, and the pool's native workers perform the actual math.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pending = []

    def submit(self, key, hidden):
        self._owner()
        if self.closed or key not in self._masters:
            raise ValueError('native queue closed or unknown expert')
        weights = self._masters[key]
        if (self._registry.get(key) is not weights or _weight_identity(weights) != self._identities[key]):
            raise ValueError('native queue master storage changed')
        buffers = cpu_expert_buffers(hidden, key.expert, max_tokens=self.max_tokens)
        future = NativeCpuFuture(self, key, hidden, buffers)
        selected, coefficient, count, output = buffers
        try:
            self._pool.submit(self._layers[key.layer].forward_task(ctypes.addressof(count), 1,
                ctypes.addressof(selected), ctypes.addressof(coefficient), hidden.data_ptr(), output.data_ptr(), False))
        except BaseException:
            self._pool.sync()  # retain this call's pointer buffers until the queue drains
            raise
        self.pending.append(future)
        self._in_flight = True
        return future

    def wait_all(self):
        import time
        self._owner()
        if not self.pending:
            return
        self._pool.sync()
        completed_ns = time.perf_counter_ns()
        for future in self.pending:
            future.output = future.buffers[-1]
            future.timing = dict(submit_host_ns=future.submitted_ns, sync_completed_host_ns=completed_ns,
                                 scope='queue submission through common sync; not isolated expert compute',
                                 owner_thread=self.owner_thread)
            self.completed_calls += 1
            self.completed_tokens += future.hidden.shape[0]
            if self.record_calls:
                output = future.output
                raw = ctypes.string_at(output.data_ptr(), output.numel() * output.element_size())
                self.calls.append(dict(layer=future.key.layer, expert=future.key.expert,
                                       tokens=future.hidden.shape[0], output_sha256=hashlib.sha256(raw).hexdigest(),
                                       packed_expert_rows=getattr(future,'packed_experts',None)))
            future.hidden, future.buffers = None, None
            future.ready = True
        self.pending.clear()
        self._in_flight = False

    def close(self):
        self.wait_all()
        super().close()


class NativeCpuChunkFuture:
    """An ordered row partition of one expert invocation, with no skipped rows."""
    def __init__(self, children):
        self.children = children
        self.output, self.timing = None, None

    def done(self):
        return all(child.done() for child in self.children)

    def result(self):
        import torch
        if self.output is None:
            results = [child.result() for child in self.children]
            self.output = torch.cat([r[0] for r in results], dim=0)
            self.timing = dict(submit_host_ns=results[0][1]['submit_host_ns'],
                sync_completed_host_ns=results[-1][1]['sync_completed_host_ns'],
                native_chunks=len(results), scope='native queue sync plus ordered row concatenation; not isolated compute')
        return self.output, self.timing


class NativeCpuRowFuture:
    """One expert's original row interval inside a packed native invocation."""
    def __init__(self,parent,begin,end):
        self.parent,self.begin,self.end=parent,begin,end

    def done(self):return self.parent.done()

    def result(self):
        output,timing=self.parent.result()
        return output[self.begin:self.end],dict(timing,packed_row_interval=[self.begin,self.end])


class ChunkedNativeCpuBackend(QueuedNativeCpuBackend):
    """Full input up to 4096 rows through bounded original-BF16 native calls.

    Chunking changes submission/compute cost and is recorded. Each row keeps
    its original inputs, weights and position; no routing/batch approximation.
    """
    def _row_bound(self, hidden):
        if hidden.ndim != 2 or not 1 <= hidden.shape[0] <= 4096:
            raise ValueError('chunked native expert requires 1..4096 token rows')

    def submit(self, key, hidden):
        self._row_bound(hidden)
        if hidden.shape[0] <= self.max_tokens:
            return super().submit(key, hidden)
        children = [super(ChunkedNativeCpuBackend, self).submit(key, part)
                    for part in hidden.split(self.max_tokens, dim=0)]
        return NativeCpuChunkFuture(children)

    def submit_many(self,jobs):
        """Pack independent expert rows, top-k=1 and unit coefficient per row.

        Each output remains unweighted; callers retain the original merge order.
        Master identity is validated once per expert per backend lifetime: the
        masters are immutable after load and the driver re-hashes them every
        request, so re-deriving the identity tuple per row per layer only burned
        host time. The packed input and output buffers are recycled instead of
        allocated per layer.
        """
        import torch
        self._owner()
        items=list(jobs.items())
        if self.closed or not items or len({k.layer for k,_ in items})!=1:
            raise ValueError('packed CPU submission requires one nonempty active layer')
        validated=self._validated_masters
        offsets=[];ids=[];rows=[];position=0
        for key,hidden in items:
            if key not in self._masters:raise ValueError('packed CPU expert is not registered')
            self._row_bound(hidden)
            if (hidden.device.type!='cpu' or hidden.dtype!=torch.bfloat16 or hidden.shape[1]!=2048
                    or not hidden.is_contiguous() or hidden.data_ptr()%64):
                raise ValueError('packed CPU input must be aligned contiguous CPU BF16')
            weights=self._masters[key]
            if self._registry.get(key) is not weights:
                raise ValueError('packed CPU master changed')
            if key not in validated:
                if _weight_identity(weights)!=self._identities[key]:
                    raise ValueError('packed CPU master changed')
                validated.add(key)
            rows.append(hidden);ids.extend([key.expert]*hidden.shape[0])
            offsets.append((key,position,position+hidden.shape[0]));position+=hidden.shape[0]
        ordinary_calls=sum((x.shape[0]+self.max_tokens-1)//self.max_tokens for _,x in items)
        if (position+self.max_tokens-1)//self.max_tokens>=ordinary_calls:
            return {key:self.submit(key,x) for key,x in items}
        if len({row.data_ptr() for row in rows})==1:
            # Decode: every row is the same activation, so one expanded write into
            # a reused buffer replaces the per-layer allocation plus cat.
            packed=self._packed_buffer(position, rows[0].shape[1])
            packed.copy_(rows[0].expand_as(packed))
        else:
            # Prefill packs a different number of rows every layer; a persistent
            # buffer would only add a second copy, so keep the single cat.
            packed=torch.cat(rows,dim=0)
        children={k:[] for k,_ in items};layer=items[0][0].layer
        for start in range(0,position,self.max_tokens):
            end=min(start+self.max_tokens,position);hidden=packed[start:end]
            _,coefficient,count,output=cpu_expert_buffers(hidden,ids[start],max_tokens=self.max_tokens)
            selected=(ctypes.c_int64*(end-start))(*ids[start:end])
            parent=NativeCpuFuture(self,items[0][0],hidden,(selected,coefficient,count,output))
            parent.packed_experts=tuple(ids[start:end])
            try:
                self._pool.submit(self._layers[layer].forward_task(ctypes.addressof(count),1,
                    ctypes.addressof(selected),ctypes.addressof(coefficient),hidden.data_ptr(),output.data_ptr(),False))
            except BaseException:
                self._pool.sync()
                raise
            self.pending.append(parent);self._in_flight=True
            for key,lo,hi in offsets:
                a,b=max(start,lo),min(end,hi)
                if a<b:children[key].append(NativeCpuRowFuture(parent,a-start,b-start))
        return {key:parts[0] if len(parts)==1 else NativeCpuChunkFuture(parts)
                for key,parts in children.items()}

    def __call__(self, key, hidden):
        self._row_bound(hidden)
        if hidden.shape[0] <= self.max_tokens:
            return super().__call__(key, hidden)
        if self._in_flight:
            raise RuntimeError('synchronous chunked call cannot bypass an outstanding CPU queue')
        return self.submit(key, hidden).result()[0]

    def _packed_buffer(self, rows, cols):
        import torch
        buffer=getattr(self,'_packed',None)
        if buffer is None or buffer.shape!=(rows,cols) or buffer.dtype!=torch.bfloat16:
            buffer=torch.empty((rows,cols),dtype=torch.bfloat16)
            self._packed=buffer
        return buffer

    def note_native_dispatch(self, rows):
        """Account for a packed group executed by the native decode dispatch.

        The dispatch path joins the queue itself, so the counters that
        ``wait_all`` normally advances must be advanced here with the same
        meaning: one completed native call and its row count.
        """
        if type(rows) is not int or rows < 1:
            raise ValueError('invalid native dispatch row count')
        self.completed_calls += 1
        self.completed_tokens += rows

    def native_group_handles(self, layer):
        """(native queue, loaded layer) for the native packed-group dispatch."""
        obj=self._layers.get(layer)
        if not isinstance(obj, LoadedNativeCpuLayer) or obj.closed or not obj.loaded:
            return None
        return self._pool, obj

    def summary(self):
        return dict(super().summary(), native_chunk_rows=self.max_tokens, input_row_limit=4096,
                    completed_calls_count_native_chunks=True,
                    master_identity_validation='once_per_expert_per_backend')
