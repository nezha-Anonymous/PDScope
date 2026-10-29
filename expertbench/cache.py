"""Expert residency bookkeeping, owned by one scheduler/control thread.

No tensors, DMA, clocks or performance simulation live here. An executor must
call complete_load/release ONLY after the corresponding backend event completes.
The byte map is supplied from actual immutable expert representations. Cache
capacity and demand-only staging reservations are disjoint and pre-budgeted.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
import heapq
from types import MappingProxyType
from uuid import uuid4


class CacheError(ValueError):
    """Illegal configuration, identity or lifecycle transition."""


class CapacityBlocked(CacheError):
    """Wait for consumers/transfers to finish, or decline speculative work."""


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise CacheError(f"{name} must be an integer >= {minimum}")


@dataclass(frozen=True, order=True)
class ExpertKey:
    layer: int
    expert: int
    representation: str

    def __post_init__(self):
        _integer(self.layer, "layer")
        _integer(self.expert, "expert")
        if not isinstance(self.representation, str) or not self.representation:
            raise CacheError("representation must be explicit")


@dataclass(frozen=True)
class DemandID:
    # request identifies one generate call (possibly batched). step identifies
    # a forward, including separate prefill chunks; (step, layer) increases.
    request: str
    step: int
    layer: int

    def __post_init__(self):
        if not isinstance(self.request, str) or not self.request:
            raise CacheError("request must be a nonempty string")
        _integer(self.step, "step")
        _integer(self.layer, "layer")


@dataclass(frozen=True)
class RetentionHint:
    source: DemandID
    target: DemandID
    keys: frozenset


@dataclass(frozen=True)
class LoadTicket:
    owner: str
    generation: int
    key: ExpertKey
    pool: str
    slot: int | None


@dataclass(frozen=True)
class UseHandle:
    ticket: LoadTicket
    serial: int


@dataclass(frozen=True)
class EntryView:
    key: ExpertKey
    nbytes: int
    pool: str
    state: str
    ticket: LoadTicket
    submitted: bool
    users: int
    origin: str
    used: bool
    expired: bool


@dataclass(frozen=True)
class DemandObservation:
    key: ExpertKey
    tokens: int
    valid_tokens: int
    nbytes: int
    state_at_reveal: str
    pool_at_reveal: str | None
    hit_source: str | None  # history / fresh_prefetch / fresh_refresh / None

    @property
    def ready_hit(self):
        return self.hit_source is not None


@dataclass
class _Entry:
    ticket: LoadTicket
    nbytes: int
    origin: str
    submitted: bool = False
    ready: bool = False
    used: bool = False
    expired: bool = False
    users: set = field(default_factory=set)
    eviction_revision: int = 0

    @property
    def state(self):
        return "in_use" if self.users else "ready" if self.ready else "loading"


@dataclass
class _Demand:
    identity: DemandID
    counts: tuple
    valid_counts: dict
    observations: tuple
    remaining: set


class ExpertCache:
    """Static, global demand-LRU or lifetime demand-LFU with staging fallback.

    Reservation starts a loading state, submit marks an irrevocable transfer,
    and complete_load makes it usable. A blocked admission has no evictions or
    partial reservation side effects. Immutable views cannot mutate the ledger.
    """

    def __init__(self, expert_bytes, cache_bytes, staging_slot_bytes, *, policy="LRU", initial_hotset=()):
        _integer(cache_bytes, "cache_bytes")
        sizes = dict(expert_bytes)
        if not sizes or any(not isinstance(key, ExpertKey) for key in sizes):
            raise CacheError("expert_bytes must map ExpertKey to actual representation bytes")
        for size in sizes.values():
            _integer(size, "expert bytes", 1)
        slots = tuple(staging_slot_bytes)
        if not slots:
            raise CacheError("at least one demand-only staging slot is required")
        for size in slots:
            _integer(size, "staging slot bytes", 1)
        if max(slots) < max(sizes.values()):
            raise CacheError("demand staging cannot fit the largest expert; forward progress is infeasible")
        hotset = tuple(initial_hotset)
        if len(set(hotset)) != len(hotset) or any(key not in sizes for key in hotset):
            raise CacheError("initial hotset must contain unique registered keys")
        if sum(sizes[key] for key in hotset) > cache_bytes:
            raise CacheError("initial hotset exceeds cache byte budget")
        if policy not in ("static", "LRU", "LFU"):
            raise CacheError("only static, LRU and LFU are implemented")
        self.expert_bytes = MappingProxyType(sizes)
        self.cache_bytes = cache_bytes
        self.staging_slot_bytes = slots
        self.policy = policy
        self.initial_hotset = frozenset(hotset)
        self._owner = uuid4().hex
        self._serial = 0  # Never reset: stale completions cannot match a new run.
        self._entries = {}
        self._retired = []
        self._active = None
        self._last_event = {}
        self._recency = Counter()
        self._frequency = Counter()
        self._stats = Counter()
        self._retention_hint = None
        self._cache_payload_bytes = 0
        self._staging_occupied = set()
        self._eviction_heap = []

    def _key(self, key):
        if key not in self.expert_bytes:
            raise CacheError(f"unknown expert: {key!r}")

    def _ticket(self, ticket):
        entry = self._entries.get(ticket.key)
        if entry is None or entry.ticket != ticket:
            raise CacheError("stale or foreign load ticket")
        return entry

    def _remaining(self, key):
        return self._active is not None and key in self._active.remaining

    def _demand_key(self, key):
        self._key(key)
        if not self._remaining(key):
            raise CacheError("expert is not an unfinished member of the active true demand")

    def lookup(self, key):
        """Side-effect-free observation, not a demand touch or a ready hit."""
        self._key(key)
        entry = self._entries.get(key)
        if entry is None:
            return None
        return EntryView(key, entry.nbytes, entry.ticket.pool, entry.state, entry.ticket,
                         entry.submitted, len(entry.users), entry.origin, entry.used, entry.expired)

    @property
    def resident_payload_bytes(self):
        """Live/loading resident payload already charged by lifecycle events."""
        return self._cache_payload_bytes

    def memory(self):
        """Reservations differ from live payload. Loading bytes already count."""
        return {
            "cache_capacity_bytes": self.cache_bytes,
            "cache_payload_bytes": self._cache_payload_bytes,
            "staging_reserved_bytes": sum(self.staging_slot_bytes),
            "staging_payload_bytes": sum(e.nbytes for e in self._entries.values() if e.ticket.pool == "demand_staging"),
            "staging_occupied_slots": sum(e.ticket.pool == "demand_staging" for e in self._entries.values()),
            "expert_reserved_bytes": self.cache_bytes + sum(self.staging_slot_bytes),
            "physical_allocation": "executor_responsibility_not_measured_here",
        }

    def stats(self):
        """Counts of reported semantic events; no measured times or throughput."""
        return dict(self._stats)

    def quiescent_state(self, request):
        """Independent logical state for comparing completed request prefixes.

        This observes rather than restores state. Physical request/owner IDs
        are omitted; generations, replacement history (including absent keys)
        and heap revisions are retained. Actual executor/GPU completion and
        weight contents require separate proof from the caller.
        """
        if (not isinstance(request, str) or not request or set(self._last_event) != {request}
                or self._active is not None or self._retention_hint is not None or self._staging_occupied
                or self._retired or any(not e.ready or not e.submitted or e.users
                                       or e.ticket.pool != 'cache' for e in self._entries.values())):
            raise CacheError('prefix snapshot requires one completed request and drained copies/uses/hints/retirements')
        encode = lambda key: [key.layer, key.expert, key.representation]
        return dict(schema_version=1, scope='cache_state_only; executor and model bytes require separate proof',
            policy=self.policy, cache_bytes=self.cache_bytes, staging_slot_bytes=list(self.staging_slot_bytes),
            registry=[[encode(k), n] for k, n in sorted(self.expert_bytes.items())],
            initial_hotset=[encode(k) for k in sorted(self.initial_hotset)],
            last_event=list(self._last_event[request]), serial=self._serial,
            entries=[dict(key=encode(k), nbytes=e.nbytes, pool=e.ticket.pool, generation=e.ticket.generation,
                          origin=e.origin, used=e.used, expired=e.expired, eviction_revision=e.eviction_revision)
                     for k, e in sorted(self._entries.items())],
            history=[dict(key=encode(k), recency=self._recency.get(k, 0), frequency=self._frequency.get(k, 0))
                     for k in sorted(set(self._recency) | set(self._frequency))
                     if self._recency.get(k, 0) or self._frequency.get(k, 0)],
            eviction_heap=[[list(priority[:-1]) + [encode(priority[-1])], generation, revision, encode(k)]
                           for priority, generation, revision, k in self._eviction_heap],
            counters={k: v for k, v in sorted(self._stats.items()) if v}, memory=self.memory())

    def prefetch_inventory(self):
        """Current speculative copies; 'used' means completed valid GPU use.

        Expired-unused is a subset of ready/loading-unused, not an additional
        partition. Payload bytes are not DRAM traffic or allocator reservations.
        """
        entries = [e for e in self._entries.values() if e.origin == 'prefetch']
        groups = {'unused_ready': [e for e in entries if e.ready and not e.used],
                  'unused_loading': [e for e in entries if not e.ready and not e.used],
                  'unused_expired': [e for e in entries if e.expired and not e.used],
                  'used_resident': [e for e in entries if e.used]}
        return {name + suffix: (len(values) if suffix == '_copies' else sum(e.nbytes for e in values))
                for name, values in groups.items() for suffix in ('_copies', '_bytes')}

    def take_retired(self):
        """Executor must free these copies before allocating new reservations.

        Retirement is safe only because no submitted load or use still refers
        to these generations. This queue is consumed, not an unbounded trace.
        """
        retired = tuple(self._retired)
        self._retired.clear()
        return retired

    def demand_frequency(self, key):
        self._key(key)
        return self._frequency[key]

    @property
    def active_demand(self):
        return self._active.identity if self._active is not None else None

    @property
    def unfinished_demand_keys(self):
        return tuple(sorted(self._active.remaining)) if self._active is not None else ()

    @property
    def retention_hint(self):
        """Immutable soft eviction preference; never a reservation or touch."""
        return self._retention_hint

    def set_retention_hint(self, source, target, keys):
        """M01/M11: only an active actual layer may protect next-layer keys.

        The caller must supply causally available predictions. This ledger
        validates identities, never consumes future target routing or scores.
        """
        if (self.policy != 'LRU' or self._active is None or self._active.identity != source
                or not isinstance(source, DemandID) or not isinstance(target, DemandID)
                or target != DemandID(source.request, source.step, source.layer + 1)
                or not any(key.layer == target.layer for key in self.expert_bytes)):
            raise CacheError('retention requires active true source and same-step next-layer target on LRU')
        keys = tuple(keys)
        if len(keys) != len(set(keys)):
            raise CacheError('retention candidates must be unique')
        for key in keys:
            self._key(key)
            if key.layer != target.layer:
                raise CacheError('retention key is outside target layer')
        hint = RetentionHint(source, target, frozenset(keys))
        if self._retention_hint is not None:
            if self._retention_hint == hint:
                return False
            raise CacheError('conflicting retention hint before target reveal')
        self._retention_hint = hint
        for key in hint.keys:
            self._queue_eviction_candidate(key)
        self._stats['retention_hint_updates'] += 1
        self._stats['retention_hint_candidates'] += len(keys)
        return True

    def _expire_retention_hint(self):
        if self._retention_hint is None:
            return False
        keys = self._retention_hint.keys
        self._retention_hint = None
        for key in keys:
            self._queue_eviction_candidate(key)
        self._stats['retention_hint_expirations'] += 1
        return True

    def clear_retention_hint(self, request):
        """Explicit request-end cleanup without resetting cache residency."""
        if self._active is not None or request not in self._last_event:
            raise CacheError('request hint cleanup requires a completed true demand')
        if self._retention_hint is not None and self._retention_hint.source.request != request:
            raise CacheError('cannot clear another request retention hint')
        return self._expire_retention_hint()

    def _first_demand_residency(self):
        return self.initial_hotset

    def on_demand(self, identity, token_counts, *, role="true", valid_token_counts=None):
        """Record one full layer-step union, before any admission or execution.

        Repeated callbacks for the same active event are idempotent only if
        token counts match exactly. Past events or a second active event fail.
        All keys receive the same recency tick; ties use stable ExpertKey order.
        """
        if role != "true" or not isinstance(identity, DemandID):
            raise CacheError("only true routing can emit DemandID events")
        if not isinstance(token_counts, Mapping):
            raise CacheError("token_counts must be an explicitly aggregated mapping of unique experts")
        counts = tuple(sorted(dict(token_counts).items()))
        if not counts:
            raise CacheError("empty/padding-only demands must not enter the denominator")
        for key, tokens in counts:
            self._key(key)
            if key.layer != identity.layer:
                raise CacheError("demand keys must belong to the revealed layer")
            _integer(tokens, "effective tokens per expert", 1)
        physical = dict(counts)
        if valid_token_counts is not None and not isinstance(valid_token_counts, Mapping):
            raise CacheError("valid_token_counts must be an aggregated mapping")
        valid = dict(physical if valid_token_counts is None else valid_token_counts)
        for key, tokens in valid.items():
            if key not in physical:
                raise CacheError("valid demand has no physical execution")
            _integer(tokens, "valid tokens per expert", 1)
            if tokens > physical[key]:
                raise CacheError("valid tokens exceed physical tokens")
        if self._active:
            if self._active.identity == identity and self._active.counts == counts and self._active.valid_counts == valid:
                return self._active.observations
            raise CacheError("another or conflicting layer-step demand is active")
        previous = self._last_event.get(identity.request)
        if previous is not None and (identity.step, identity.layer) <= previous:
            raise CacheError("duplicate or out-of-order closed demand event")
        if not self._stats["physical_layer_events"] and any(
                key not in self._entries or not self._entries[key].ready for key in self._first_demand_residency()):
            raise CacheError("initial hotset must finish loading before the first demand")
        # The next true reveal consumes the one-layer horizon. This also
        # expires a skipped target or a request/step boundary; invalid calls
        # above cannot alter an existing hint.
        self._expire_retention_hint()
        observations = []
        for key, tokens in counts:
            entry = self._entries.get(key)
            source = None
            if entry and entry.ready:
                source = ('fresh_'+entry.origin) if entry.origin in ('prefetch','refresh') and not entry.used else 'history'
            observations.append(DemandObservation(key, tokens, valid.get(key, 0), self.expert_bytes[key],
                                                 entry.state if entry else "absent",
                                                 entry.ticket.pool if entry else None, source))
        self._stats["physical_layer_events"] += 1
        self._stats["demand_events"] += int(bool(valid))
        tick = self._stats["physical_layer_events"]
        for obs in observations:
            self._stats["physical_expert_demands"] += 1
            self._stats["physical_demand_tokens"] += obs.tokens
            self._stats["padding_token_assignments"] += obs.tokens - obs.valid_tokens
            self._stats["physical_ready_hits"] += int(obs.ready_hit)
            if not obs.valid_tokens:
                self._stats["padding_only_expert_demands"] += 1
                continue
            self._frequency[obs.key] += 1
            self._recency[obs.key] = tick
            self._stats["expert_demands"] += 1
            self._stats["demand_tokens"] += obs.valid_tokens
            self._stats["demand_bytes"] += obs.nbytes
            if obs.ready_hit:
                self._stats["ready_hits"] += 1
                self._stats[{'history':'history_ready_hits','fresh_prefetch':'fresh_prefetch_hits',
                             'fresh_refresh':'fresh_refresh_hits'}[obs.hit_source]] += 1
                self._stats["ready_hit_tokens"] += obs.valid_tokens
                self._stats["ready_hit_bytes"] += obs.nbytes
                self._stats["cache_resident_hits" if obs.pool_at_reveal == "cache" else "staging_ready_hits"] += 1
            else:
                self._stats["loading_misses" if obs.state_at_reveal == "loading" else "absent_misses"] += 1
        self._active = _Demand(identity, counts, valid, tuple(observations), {key for key, _ in counts})
        return self._active.observations

    def _evictable(self, entry):
        return (self.policy in ("LRU", "LFU", "MRS_decode", "LRU_score_observed") and entry.ticket.pool == "cache" and entry.ready
                and not entry.users and not self._remaining(entry.ticket.key))

    def _eviction_priority(self, entry):
        key = entry.ticket.key
        retained = self._retention_hint is not None and key in self._retention_hint.keys
        order = ((self._frequency[key], self._recency[key], key) if self.policy == "LFU"
                 else (self._recency[key], key))
        return (retained, *order)

    def _compact_eviction_heap(self):
        # Demand touches and hint changes invalidate older revisions. Bound
        # retained metadata even when no admission ever pops the stale nodes.
        if len(self._eviction_heap) > max(64, 4 * len(self._entries)):
            self._eviction_heap = [(self._eviction_priority(e), e.ticket.generation,
                                   e.eviction_revision, e.ticket.key)
                                  for e in self._entries.values() if self._evictable(e)]
            heapq.heapify(self._eviction_heap)

    def _queue_eviction_candidate(self, key):
        entry = self._entries.get(key)
        if entry is not None and self._evictable(entry):
            entry.eviction_revision += 1
            heapq.heappush(self._eviction_heap, (self._eviction_priority(entry), entry.ticket.generation,
                                               entry.eviction_revision, key))
            self._compact_eviction_heap()

    def _victims(self, size, *, layer=None):
        needed = self._cache_payload_bytes + size - self.cache_bytes
        if needed <= 0:
            return []
        selected = []
        while self._eviction_heap and needed > 0:
            node = heapq.heappop(self._eviction_heap)
            priority, generation, revision, key = node
            entry = self._entries.get(key)
            if (entry is None or entry.ticket.generation != generation or entry.eviction_revision != revision
                    or not self._evictable(entry)):
                continue
            if priority != self._eviction_priority(entry):
                # Refresh without compacting while selected nodes are outside
                # the heap; compaction would otherwise reinsert them twice.
                entry.eviction_revision += 1
                heapq.heappush(self._eviction_heap, (self._eviction_priority(entry), generation,
                                                   entry.eviction_revision, key))
                continue
            selected.append(node)
            needed -= entry.nbytes
        # Planning never consumes a live reservation: if total bytes cannot
        # fit, staging fallback or CapacityBlocked leaves every entry intact.
        for node in selected:
            heapq.heappush(self._eviction_heap, node)
        self._compact_eviction_heap()
        return [node[-1] for node in selected] if needed <= 0 else None

    def _evict(self, key):
        entry = self._entries.pop(key)
        self._cache_payload_bytes -= entry.nbytes
        self._retired.append(entry.ticket)
        self._stats["evictions"] += 1
        self._stats["evicted_bytes"] += entry.nbytes
        if entry.origin == "prefetch" and not entry.used:
            self._stats["unused_prefetch_evictions"] += 1

    def evict(self, key):
        self._key(key)
        entry = self._entries.get(key)
        if entry is None or not self._evictable(entry):
            raise CacheError("cannot evict absent/static/loading/referenced/current-demand weights")
        self._evict(key)

    def begin_load(self, key, *, origin="demand", admit=True):
        """Reserve a single copy. Demand falls back to dedicated staging.

        CPU misses never call this method. Repeated reservation returns the
        same ticket without creating another copy or charging transfer bytes.
        An explicit demand bypass uses only staging, even with cache slack.
        """
        self._key(key)
        if origin not in ("initial", "demand", "prefetch"):
            raise CacheError("unknown load origin")
        if type(admit) is not bool or (not admit and origin != "demand"):
            raise CacheError("non-admission is an explicit boolean demand-only option")
        if origin == "demand":
            self._demand_key(key)
        elif origin == "initial":
            if self._stats["physical_layer_events"] or self._active or key not in self.initial_hotset:
                raise CacheError("initial loading is restricted to the declared pre-run hotset")
        elif self.policy == "static" or self._remaining(key):
            raise CacheError("prefetch is not a static-cache or revealed-demand load")
        if key in self._entries:
            return self._entries[key].ticket
        size = self.expert_bytes[key]
        victims = self._victims(size, layer=key.layer) if admit and (self.policy in ("LRU", "LFU", "MRS_decode", "LRU_score_observed") or origin == "initial") else None
        slot = None
        if victims is not None:
            pool = "cache"
        elif origin == "demand":
            available = [(capacity, i) for i, capacity in enumerate(self.staging_slot_bytes)
                         if i not in self._staging_occupied and capacity >= size]
            if not available:
                raise CapacityBlocked("demand staging busy: finish existing transfers/uses before retrying")
            _, slot = min(available)
            pool = "demand_staging"
        else:
            raise CapacityBlocked("no evictable cache bytes; speculative work cannot borrow demand staging")
        # Commit only after a full feasible reservation has been found.
        for victim in victims or ():
            self._evict(victim)
        self._serial += 1
        ticket = LoadTicket(self._owner, self._serial, key, pool, slot)
        self._entries[key] = _Entry(ticket, size, origin)
        if pool == 'cache':
            self._cache_payload_bytes += size
        else:
            self._staging_occupied.add(slot)
        return ticket

    def submit(self, ticket):
        entry = self._ticket(ticket)
        if entry.submitted or entry.ready:
            raise CacheError("transfer already submitted/completed")
        if entry.origin == "prefetch" and self._remaining(ticket.key):
            # Reservation time is not DMA issue time. Once true demand is
            # revealed, a previously queued forecast is ordinary demand work.
            entry.origin = "demand"
            entry.expired = True
        entry.submitted = True
        self._stats[f"{entry.origin}_loads_submitted"] += 1
        self._stats["h2d_bytes_submitted"] += entry.nbytes
        if entry.origin == "demand" and self._active and ticket.key not in self._active.valid_counts:
            self._stats["padding_only_h2d_bytes_submitted"] += entry.nbytes

    def complete_load(self, ticket):
        """Executor reports completion of this exact generation's DMA."""
        entry = self._ticket(ticket)
        if not entry.submitted or entry.ready:
            raise CacheError("completion requires one submitted, incomplete transfer")
        entry.ready = True
        self._stats["h2d_bytes_completed"] += entry.nbytes
        self._queue_eviction_candidate(ticket.key)

    def expire_prefetch(self, ticket):
        """Drop unsubmitted speculation; never cancel a submitted DMA."""
        entry = self._ticket(ticket)
        if entry.expired:
            return False
        if entry.origin != "prefetch":
            raise CacheError("only speculative loads can expire")
        entry.expired = True
        self._stats["prefetch_expirations"] += 1
        if not entry.submitted and not self._remaining(ticket.key):
            del self._entries[ticket.key]
            self._cache_payload_bytes -= entry.nbytes
            self._retired.append(ticket)
            return True
        if not entry.submitted:
            # The forecast expired, but true demand has adopted its reservation.
            # Its eventual transfer is demand work, not a speculative DMA.
            entry.origin = "demand"
        return False

    def acquire(self, key):
        """Protect an immutable ready copy until GPU computation completes.

        Acquire all planned consumers before releasing the last handle; that
        last release completes this expert's active demand.
        """
        self._demand_key(key)
        entry = self._entries.get(key)
        if entry is None or not entry.ready:
            raise CacheError("weights are not ready; a loading object is not a hit")
        self._serial += 1
        entry.users.add(self._serial)
        return UseHandle(entry.ticket, self._serial)

    def release(self, handle):
        """Executor reports completion, not merely kernel launch or enqueue."""
        entry = self._ticket(handle.ticket)
        if handle.serial not in entry.users:
            raise CacheError("stale, foreign or already released use handle")
        entry.users.remove(handle.serial)
        entry.used = entry.used or handle.ticket.key in self._active.valid_counts
        if not entry.users:
            self._active.remaining.remove(handle.ticket.key)
            self._stats["gpu_completed_experts"] += 1
            self._stats["gpu_completed_valid_experts"] += int(handle.ticket.key in self._active.valid_counts)
            if handle.ticket.pool == "demand_staging":
                del self._entries[handle.ticket.key]  # S1 never retains a miss.
                self._staging_occupied.remove(handle.ticket.slot)
                self._retired.append(handle.ticket)
            else:
                self._queue_eviction_candidate(handle.ticket.key)

    def finish_cpu(self, key):
        """CPU consumed its immutable master copy; do not admit GPU weights."""
        self._demand_key(key)
        entry = self._entries.get(key)
        if entry and (entry.users or (entry.origin == "demand" and not entry.ready)):
            raise CacheError("cannot silently switch an active GPU use/load to CPU completion")
        if entry and entry.ticket.pool == "demand_staging":
            raise CacheError("demand staging must be consumed and released through GPU completion")
        self._active.remaining.remove(key)
        self._queue_eviction_candidate(key)
        self._stats["cpu_completed_experts"] += 1
        self._stats["cpu_completed_valid_experts"] += int(key in self._active.valid_counts)

    def end_demand(self, identity):
        if self._active is None or self._active.identity != identity:
            raise CacheError("no matching active demand")
        if self._active.remaining:
            raise CacheError("cannot finish layer: some true experts have not completed")
        self._last_event[identity.request] = (identity.step, identity.layer)
        self._active = None

    def reset(self):
        """Reset a run only after all real work is quiescent; re-load hotset."""
        if self._active or any(not e.ready or e.users for e in self._entries.values()):
            raise CacheError("cannot reset active demands, loads or references")
        self._retired.extend(e.ticket for e in self._entries.values())
        self._entries.clear()
        self._cache_payload_bytes = 0
        self._staging_occupied.clear()
        self._eviction_heap.clear()
        self._last_event.clear()
        self._recency.clear()
        self._frequency.clear()
        self._stats.clear()
        self._retention_hint = None


class PrefillStaticCache(ExpertCache):
    """One request: causal per-layer refresh before prefill, then fixed residency.

    prepare/submit/complete/finish are distinct. Reservation is not a copy or
    a hit; physical retirements must drain before the refreshed demand starts.
    Ordinary static/LRU policies do not acquire this mutation capability.
    """
    def __init__(self, expert_bytes, cache_bytes, staging_slot_bytes, *, initial_hotset=()):
        super().__init__(expert_bytes,cache_bytes,staging_slot_bytes,policy='static',initial_hotset=initial_hotset)
        self.layers=1+max(k.layer for k in self.expert_bytes)
        self.experts=1+max(k.expert for k in self.expert_bytes)
        expected={ExpertKey(l,e,'bf16') for l in range(self.layers) for e in range(self.experts)}
        if (set(self.expert_bytes)!=expected or len(set(self.expert_bytes.values()))!=1
                or sum(self.expert_bytes[k] for k in self.initial_hotset)!=cache_bytes):
            raise CacheError('prefill refresh requires dense BF16 IDs, uniform bytes and full initial cache')
        self._resident_keys=set(self.initial_hotset)
        self._refresh_pending=None;self._refresh_ready=None;self._refresh_request=None;self._prefill_next_layer=0

    def _first_demand_residency(self):
        return self._resident_keys

    def prepare_prefill(self,identity,token_counts,*,valid_token_counts=None):
        from .routing import prefill_layer_resident_plan
        if (not isinstance(identity,DemandID) or identity.step!=0 or identity.layer!=self._prefill_next_layer
                or identity.layer>=self.layers or self._refresh_request not in (None,identity.request)
                or self._active or self._refresh_pending is not None or self._refresh_ready is not None
                or self._retired or self._staging_occupied or set(self._entries)!=self._resident_keys
                or any(not e.ready or not e.submitted or e.users or e.ticket.pool!='cache' for e in self._entries.values())
                or self._cache_payload_bytes!=self.cache_bytes
                or identity.layer>0 and self._last_event.get(identity.request)!=(0,identity.layer-1)):
            raise CacheError('prefill refresh requires the next layer of one request and quiescent full residency')
        if not isinstance(token_counts,Mapping) or valid_token_counts is not None and not isinstance(valid_token_counts,Mapping):
            raise CacheError('prefill refresh requires explicit true count mappings')
        physical=dict(token_counts);valid=dict(physical if valid_token_counts is None else valid_token_counts)
        if not physical or not valid:raise CacheError('prefill refresh requires at least one valid token demand')
        for key,n in physical.items():
            self._key(key);_integer(n,'physical expert tokens',1)
            if key.layer!=identity.layer:raise CacheError('prefill refresh counts belong to a different layer')
        for key,n in valid.items():
            _integer(n,'valid expert tokens',1)
            if key not in physical or n>physical[key]:raise CacheError('prefill valid counts exceed actual demands')
        counts=[valid.get(ExpertKey(identity.layer,e,'bf16'),0) for e in range(self.experts)]
        old=[k.expert for k in self._resident_keys if k.layer==identity.layer]
        plan=prefill_layer_resident_plan(counts,old)
        swaps=[(ExpertKey(identity.layer,s['outgoing'],'bf16'),ExpertKey(identity.layer,s['incoming'],'bf16'))
               for s in plan['swaps']]
        # Everything above is validation; only complete plans retire any generation.
        tickets=[]
        for outgoing,incoming in swaps:
            self._evict(outgoing);self._resident_keys.remove(outgoing);self._resident_keys.add(incoming)
            self._serial+=1
            ticket=LoadTicket(self._owner,self._serial,incoming,'cache',None)
            self._entries[incoming]=_Entry(ticket,self.expert_bytes[incoming],'refresh')
            self._cache_payload_bytes+=self.expert_bytes[incoming];tickets.append(ticket)
        self._refresh_request=identity.request
        self._refresh_pending=dict(identity=identity,physical=physical,valid=valid,tickets=tuple(tickets),plan=plan)
        self._stats['prefill_refresh_plans']+=1;self._stats['prefill_refresh_swaps']+=len(swaps)
        return dict(layer=identity.layer,**plan,tickets=tuple(tickets))

    def begin_load(self,key,*,origin='demand',admit=True):
        if origin=='refresh':
            if type(admit) is not bool or not admit or self._refresh_pending is None:
                raise CacheError('refresh load requires its explicit pending reservation')
            for ticket in self._refresh_pending['tickets']:
                if ticket.key==key:return ticket
            raise CacheError('expert is not an incoming prefill refresh reservation')
        if self._refresh_pending is not None or self._refresh_ready is not None:
            raise CacheError('ordinary loading cannot bypass the prefill refresh transaction')
        return super().begin_load(key,origin=origin,admit=admit)

    def submit(self,ticket):
        entry=self._ticket(ticket);refresh=entry.origin=='refresh'
        super().submit(ticket)
        if refresh:self._stats['refresh_h2d_bytes_submitted']+=entry.nbytes

    def complete_load(self,ticket):
        entry=self._ticket(ticket);refresh=entry.origin=='refresh'
        super().complete_load(ticket)
        if refresh:self._stats['refresh_h2d_bytes_completed']+=entry.nbytes

    def finish_prefill_refresh(self,identity):
        pending=self._refresh_pending
        if (pending is None or pending['identity']!=identity or self._retired
                or any(not self._ticket(t).ready for t in pending['tickets'])):
            raise CacheError('prefill refresh requires all DMA completion and retired physical copies drained')
        self._refresh_ready=pending;self._refresh_pending=None

    def on_demand(self,identity,token_counts,*,role='true',valid_token_counts=None):
        if (not isinstance(identity,DemandID) or role!='true' or identity.request!=self._refresh_request
                or self._refresh_pending is not None or not isinstance(token_counts,Mapping)
                or valid_token_counts is not None and not isinstance(valid_token_counts,Mapping)):
            raise CacheError('prefill static demand requires completed refresh of this actual request')
        if self._active is not None:
            return super().on_demand(identity,token_counts,role=role,valid_token_counts=valid_token_counts)
        if identity.step==0:
            ready=self._refresh_ready
            if (ready is None or ready['identity']!=identity or dict(token_counts)!=ready['physical']
                    or dict(token_counts if valid_token_counts is None else valid_token_counts)!=ready['valid']):
                raise CacheError('prefill demand differs from the counts used for refresh')
        elif self._refresh_ready is not None or self._prefill_next_layer!=self.layers:
            raise CacheError('decode cannot start before all prefill layers complete')
        observations=super().on_demand(identity,token_counts,role=role,valid_token_counts=valid_token_counts)
        if identity.step==0:
            incoming={t.key for t in self._refresh_ready['tickets']}
            self._stats['prefill_newly_refreshed_valid_ready_demands']+=sum(o.ready_hit and o.valid_tokens>0 and o.key in incoming for o in observations)
            self._stats['prefill_existing_valid_ready_demands']+=sum(o.ready_hit and o.valid_tokens>0 and o.key not in incoming for o in observations)
            self._refresh_ready=None;self._prefill_next_layer+=1
        return observations

    def reset(self):
        if self._refresh_pending is not None or self._refresh_ready is not None:
            raise CacheError('cannot reset an unconsumed prefill refresh transaction')
        super().reset()
        self._resident_keys=set(self.initial_hotset)
        self._refresh_request=None;self._prefill_next_layer=0


class LayerPartitionCache(ExpertCache):
    """P0 cache with layer byte quotas fixed by the fully populated initial set.

    The executor still owns one global allocation pool and shared staging.
    Candidates are sorted within the small layer set using the base priority;
    base lifecycle checks protect loading, referenced and unfinished demands.
    """

    def __init__(self, expert_bytes, cache_bytes, staging_slot_bytes, *, policy='LRU', initial_hotset=()):
        initial_hotset = tuple(initial_hotset)
        super().__init__(expert_bytes, cache_bytes, staging_slot_bytes, policy=policy, initial_hotset=initial_hotset)
        if policy not in ('LRU', 'LFU') or sum(self.expert_bytes[k] for k in initial_hotset) != cache_bytes:
            raise CacheError('layer quotas require LRU/LFU and a full declared initial cache')
        layers = sorted({key.layer for key in self.expert_bytes})
        self.layer_cache_bytes = MappingProxyType({layer: sum(self.expert_bytes[k] for k in initial_hotset
                                                            if k.layer == layer) for layer in layers})
        self._layer_residents = {layer: set() for layer in layers}
        self._layer_payload_bytes = Counter()

    def memory(self):
        value = super().memory()
        value.update(cache_scope='fixed_initial_layer_quotas',
            layer_cache_capacity_bytes={str(k): v for k, v in self.layer_cache_bytes.items()},
            layer_cache_payload_bytes={str(k): self._layer_payload_bytes[k] for k in self.layer_cache_bytes})
        return value

    def _queue_eviction_candidate(self, key):
        # This policy recomputes priorities within one layer; no global heap.
        pass

    def _victims(self, size, *, layer=None):
        if layer not in self.layer_cache_bytes:
            raise CacheError('layer-local victim selection requires an explicit layer')
        needed = self._layer_payload_bytes[layer] + size - self.layer_cache_bytes[layer]
        if needed <= 0:
            return []
        entries = (self._entries[k] for k in self._layer_residents[layer])
        candidates = sorted((e for e in entries if self._evictable(e)), key=self._eviction_priority)
        victims = []
        for entry in candidates:
            victims.append(entry.ticket.key)
            needed -= entry.nbytes
            if needed <= 0:
                return victims
        return None

    def begin_load(self, key, *, origin='demand', admit=True):
        if origin == 'prefetch':
            raise CacheError('layer partition currently supports P0 demand execution only')
        ticket = super().begin_load(key, origin=origin, admit=admit)
        if ticket.pool == 'cache' and key not in self._layer_residents[key.layer]:
            self._layer_residents[key.layer].add(key)
            self._layer_payload_bytes[key.layer] += self.expert_bytes[key]
        return ticket

    def _evict(self, key):
        super()._evict(key)
        self._layer_residents[key.layer].remove(key)
        self._layer_payload_bytes[key.layer] -= self.expert_bytes[key]

    def set_retention_hint(self, source, target, keys):
        raise CacheError('layer partition currently supports P0 demand execution only')

    def reset(self):
        super().reset()
        for resident in self._layer_residents.values():
            resident.clear()
        self._layer_payload_bytes.clear()


class DecodeScoreLayerCache(LayerPartitionCache):
    """Controlled decode-only form of HybriMoE v1 Eq.3, not its full system.

    Same initial quotas, LRU prefill, then zero-initialized score history.
    Full native probabilities enter only after the current true demand reveal.
    Priorities use binary64; top-p and eviction ties use ExpertKey order.
    The observe_only diagnostic pays the same score updates but evicts by LRU.
    """

    def __init__(self, expert_bytes, cache_bytes, staging_slot_bytes, *, top_k, initial_hotset=(), observe_only=False):
        from array import array
        _integer(top_k, 'native top_k', 1)
        if type(observe_only) is not bool:
            raise CacheError('score observation mode must be an explicit boolean')
        super().__init__(expert_bytes, cache_bytes, staging_slot_bytes,
                         policy='LRU', initial_hotset=initial_hotset)
        self.observe_only = observe_only
        self.policy = 'LRU_score_observed' if observe_only else 'MRS_decode'
        self.top_k = top_k
        self._score_keys = {layer: tuple(sorted(k for k in self.expert_bytes if k.layer == layer))
                            for layer in self.layer_cache_bytes}
        if any(len(keys) < top_k or [k.expert for k in keys] != list(range(len(keys)))
               for keys in self._score_keys.values()):
            raise CacheError('score rule requires complete contiguous expert IDs per layer')
        self._scores = {layer: array('d', [0.0]) * len(keys) for layer, keys in self._score_keys.items()}
        self._score_request = None
        self._score_identity = None

    def memory(self):
        value = super().memory()
        value.update(replacement_rule=('LRU_with_native_gate_top2k_ema_observation_v1' if self.observe_only
                                       else 'native_gate_top2k_ema_decode_v1'),
            score_alpha=0.5, score_top_k=self.top_k, score_dtype='binary64',
            score_payload_bytes=sum(s.itemsize * len(s) for s in self._scores.values()),
            score_ties='expert_id', score_initialization='zero_per_request', prefill_rule='LRU')
        return value

    def score_values(self, layer):
        return tuple(self._scores[layer])

    def quiescent_state(self, request):
        value = super().quiescent_state(request)
        value['score_history'] = {str(layer): list(s) for layer, s in self._scores.items()}
        return value

    def on_demand(self, identity, token_counts, *, role='true', valid_token_counts=None):
        if (not isinstance(identity, DemandID) or not isinstance(token_counts, Mapping)
                or self._score_request not in (None, identity.request)
                or (identity.step > 0 and (len(token_counts) != self.top_k
                    or any(n != 1 for n in token_counts.values())
                    or (valid_token_counts is not None and dict(valid_token_counts) != dict(token_counts))))):
            raise CacheError('decode score rule requires one request and a single valid token; reset between requests')
        observations = super().on_demand(identity, token_counts, role=role,
                                         valid_token_counts=valid_token_counts)
        self._score_request = identity.request
        return observations

    def update_decode_scores(self, identity, probabilities):
        from array import array
        from math import isfinite, fsum
        if (self._active is None or identity != self._active.identity or identity.step == 0
                or self._score_identity == identity):
            raise CacheError('score update requires the current true decode demand exactly once')
        keys = self._score_keys[identity.layer]
        values = tuple(probabilities)
        if (len(values) != len(keys) or any(type(v) not in (float, int) or not isfinite(v) or not 0 <= v <= 1 for v in values)
                or abs(fsum(values) - 1.0) > 1e-5):
            raise CacheError('score update requires all native softmax probabilities, not normalized top-k')
        order = sorted(range(len(values)), key=lambda i: (-values[i], keys[i]))
        threshold = values[order[self.top_k - 1]]
        if any(values[key.expert] < threshold for key, _ in self._active.counts):
            raise CacheError('score vector disagrees with the native selected experts')
        if self._active.remaining != {key for key, _ in self._active.counts}:
            raise CacheError('score update must precede expert execution')
        chosen = set(order[:min(2 * self.top_k, len(values))])
        previous = self._scores[identity.layer]
        updated = array('d', (0.5 * previous[i] + (0.5 * values[i] if i in chosen else 0.0)
                              for i in range(len(values))))
        self._scores[identity.layer] = updated
        self._score_identity = identity
        self._stats['score_updates'] += 1
        self._stats['score_values_observed'] += len(values)

    def _require_scores(self):
        if (self._active is not None and self._active.identity.step > 0
                and self._score_identity != self._active.identity):
            raise CacheError('current decode scores must be updated before load/use/completion')

    def begin_load(self, key, *, origin='demand', admit=True):
        self._require_scores()
        return super().begin_load(key, origin=origin, admit=admit)

    def acquire(self, key):
        self._require_scores()
        return super().acquire(key)

    def finish_cpu(self, key):
        raise CacheError('decode score rule is a GPU-only controlled path')

    def end_demand(self, identity):
        self._require_scores()
        return super().end_demand(identity)

    def _eviction_priority(self, entry):
        if self.observe_only or self._active is None or self._active.identity.step == 0:
            return super()._eviction_priority(entry)
        key = entry.ticket.key
        return (self._scores[key.layer][key.expert], key)

    def reset(self):
        super().reset()
        for scores in self._scores.values():
            for i in range(len(scores)):
                scores[i] = 0.0
        self._score_request = self._score_identity = None
