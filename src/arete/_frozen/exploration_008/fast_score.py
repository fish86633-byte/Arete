"""Invocation-local compact scoring with unchanged all-link slot utility.

No planner, Replay, queue advance or algorithm decision is executed here. The
caller owns an immutable channel/arrived queue and supplies a finite collection
of complete arrangements before comparing them. Every actually active beam is
included in the physical sum; the cache never defines interference eligibility.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import math
import time

import numpy as np

from arete._frozen.pcqr.search import value
from arete._frozen.exploration_007.local_score import _queue_array, _weighted_age_sum


@dataclass
class CompactSlotScore:
    V: Fraction
    delivered: int
    weighted_served_rt: int
    critical_RT_served: int
    service_rt: np.ndarray
    service_nrt: np.ndarray
    # Full diagnostic arrays are produced only when requested for a fixed gate.
    actual: np.ndarray | None = None
    capacities: np.ndarray | None = None
    interference: np.ndarray | None = None
    cell_capacity: np.ndarray | None = None
    served_age: np.ndarray | None = None
    residual_rt: np.ndarray | None = None
    expired: np.ndarray | None = None
    weighted_wait: int | None = None
    # Raw, ordered active-by-active watts, before diagonal removal. This is
    # diagnostic data, not an additional channel or a physical clipping rule.
    power_matrix: np.ndarray | None = None
    active_edges: np.ndarray | None = None


class SlotScorer:
    """Own one reference slot's queue and finite-pool power matrix.

    API: SlotScorer(rt, nrt, channel, h, arrangements, *, meter=None).
    `arrangements` includes the reference and the already fixed finite
    neighborhood, including crossed edges from swaps. It is consumed once.
    score(arrangement, *, full=False) returns CompactSlotScore; stats() returns
    counters and known numeric payload, not an inferred RSS measurement.

    Construct a fresh object for each reference/slot. There is no module/global
    cache or cross-method reuse. The Channel geometry, physical configuration
    and capacity policy must remain fixed for the lifetime of this object.
    """

    def __init__(self, rt, nrt, channel, h, arrangements, *, meter=None):
        started, cpu_started = time.perf_counter(), time.process_time()
        self.channel, self.meter = channel, meter
        if meter is not None:
            meter.check()
        if isinstance(h, (bool, np.bool_)) or int(h) != h or not 0 <= int(h) < channel.H:
            raise ValueError('invalid planning slot index')
        self.h = int(h)
        self.D = int(channel.config['ttl'])
        self.rt = _queue_array(rt, (channel.M, self.D), 'RT queue').copy()
        self.nrt = _queue_array(nrt, (channel.M,), 'NRT totals').copy()
        self.rt.flags.writeable = False
        self.nrt.flags.writeable = False
        # Match the original active-queue calculation exactly. Valid frozen
        # inputs obey the original integer-range assumptions.
        self.nonempty = (self.rt.sum(axis=1) + self.nrt) > 0
        self.nonempty.flags.writeable = False
        self._stats = {
            'lifetime': 'ONE_REFERENCE_SLOT_OBJECT_WITHIN_ONE_PAID_INVOCATION',
            'prebuild_wall_s': None, 'prebuild_cpu_s': None,
            'power_prebuild_wall_s': 0.0, 'power_prebuild_cpu_s': 0.0,
            'arrangements_registered': 0, 'union_edge_count': 0,
            'power_prebuild_calls': 0, 'score_calls': 0,
            'score_completed': 0, 'score_interrupted': 0,
            'score_wall_s': 0.0, 'score_cpu_s': 0.0,
            'power_for_calls': 0, 'cache_hits': 0, 'empty_active_scores': 0,
            'membership_fallbacks': 0, 'fallback_power_calls': 0,
            'subset_payload_peak_bytes': 0,
            'matrix_payload_bytes': 0, 'edge_payload_bytes': 0,
            'owned_queue_payload_bytes': int(self.rt.nbytes + self.nrt.nbytes),
            'nonempty_payload_bytes': int(self.nonempty.nbytes),
            'known_retained_payload_bytes': 0,
            'python_container_bytes': None, 'rss_peak_bytes': None,
            'payload_scope': 'Owned RT/NRT, nonempty mask, ordered edge array and retained float64 pool matrix; excludes original Channel cache and temporary arrays.',
            'time_scope': 'Constructor and score timers are children of caller Meter phases; never add them again to complete planning wall.',
        }
        union = np.zeros((channel.S, channel.M), dtype=bool)
        for arrangement in arrangements:
            if meter is not None:
                meter.check()
            plan_h = self._checked_arrangement(arrangement)
            union |= plan_h & self.nonempty[None, :]
            self._stats['arrangements_registered'] += 1
        self.edges = np.argwhere(union).astype(np.int64, copy=False)
        self.edges.flags.writeable = False
        self._index = {tuple(int(v) for v in edge): index
                       for index, edge in enumerate(self.edges)}
        self._stats['union_edge_count'] = len(self.edges)
        self._stats['edge_payload_bytes'] = int(self.edges.nbytes)
        self._stats['ordered_union_sha256'] = hashlib.sha256(self.edges.tobytes(order='C')).hexdigest()
        if meter is not None:
            meter.check()
        if len(self.edges):
            tick, cpu_tick = time.perf_counter(), time.process_time()
            self._power = channel.power(self.h, self.edges, self.edges)
            self._stats['power_prebuild_calls'] += 1
            self._stats['power_prebuild_wall_s'] = time.perf_counter() - tick
            self._stats['power_prebuild_cpu_s'] = time.process_time() - cpu_tick
        else:
            self._power = np.zeros((0, 0), dtype=float)
        # channel.power returns an owned array even with the original cache.
        # Preserve its self-signal diagonal; each score owns the zeroed subset.
        self._power.flags.writeable = False
        self._stats['matrix_payload_bytes'] = int(self._power.nbytes)
        self._stats['known_retained_payload_bytes'] = sum(
            self._stats[key] for key in ('matrix_payload_bytes', 'edge_payload_bytes',
                                        'owned_queue_payload_bytes', 'nonempty_payload_bytes'))
        self._stats['prebuild_wall_s'] = time.perf_counter() - started
        self._stats['prebuild_cpu_s'] = time.process_time() - cpu_started
        if meter is not None:
            meter.check()

    def _checked_arrangement(self, arrangement):
        plan_h = np.asarray(arrangement)
        channel = self.channel
        if plan_h.shape != (channel.S, channel.M) or not np.all((plan_h == 0) | (plan_h == 1)):
            raise ValueError('slot arrangement must be binary with shape S by M')
        plan_h = plan_h.astype(bool)
        if (np.any(plan_h & ~channel.visible[self.h])
                or np.any(plan_h.sum(axis=1) > int(channel.config['K']))
                or np.any(plan_h.sum(axis=0) > 1)):
            raise ValueError('infeasible single-slot arrangement')
        return plan_h

    def stats(self):
        """Return a detached scalar counter summary, with N/A for unmeasured RSS."""
        return dict(self._stats)

    def power_for(self, edges):
        """Return owned raw watts in the caller's exact edge order.

        This also exposes a read-only component-gate interface. Membership
        misses recompute the entire ordered matrix with original Channel.power;
        diagonal self-signal is retained. It does not decide active membership.
        Counts include diagnostic requests if a caller invokes this separately.
        """
        if self.meter is not None:
            self.meter.check()
        self._stats['power_for_calls'] += 1
        edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
        if not len(edges):
            result = np.zeros((0, 0), dtype=float)
        else:
            indices = [self._index.get(tuple(int(v) for v in edge)) for edge in edges]
            if all(index is not None for index in indices):
                indices = np.asarray(indices, dtype=np.int64)
                result = self._power[np.ix_(indices, indices)]
                self._stats['cache_hits'] += 1
            else:
                self._stats['membership_fallbacks'] += 1
                self._stats['fallback_power_calls'] += 1
                result = self.channel.power(self.h, edges, edges)
        self._stats['subset_payload_peak_bytes'] = max(
            self._stats['subset_payload_peak_bytes'], int(result.nbytes))
        if self.meter is not None:
            self.meter.check()
        return result

    def score(self, arrangement, *, full=False):
        """Score all active links at fixed Q+, with compact age-bucket service.

        Missing union membership invokes original Channel.power for the entire
        actual active set. It does not omit crossed swap edges or partially sum
        cached and uncached interference. Full mode materializes original-size
        diagnostic arrays, while normal scoring visits RT buckets only for the
        actually served cells after the fixed nonempty mask has been formed.
        """
        started, cpu_started = time.perf_counter(), time.process_time()
        self._stats['score_calls'] += 1
        completed = False
        try:
            if self.meter is not None:
                self.meter.check()
            plan_h = self._checked_arrangement(arrangement)
            actual = plan_h & self.nonempty[None, :]
            edges = np.argwhere(actual)
            cell_capacity = np.zeros(self.channel.M, dtype=np.int64)
            caps = np.zeros(len(edges), dtype=np.int64)
            incoming = np.zeros(len(edges), dtype=float)
            raw_power = np.zeros((0, 0), dtype=float) if full else None
            if len(edges):
                powers = self.power_for(edges)
                if full:
                    raw_power = powers.copy()
                np.fill_diagonal(powers, 0.0)
                incoming = np.asarray([
                    math.fsum(powers[:, j]) + float(self.channel.config.get('I_ext', 0.0))
                    for j in range(len(edges))
                ])
                caps = self.channel.cap(
                    self.channel.signal[self.h, edges[:, 0], edges[:, 1]], incoming)
                cell_capacity[edges[:, 1]] = caps
            else:
                self._stats['empty_active_scores'] += 1

            cells = edges[:, 1]
            # Cell-slot uniqueness guarantees this is a set of different cells.
            active_rt = self.rt[cells]
            oldest = active_rt[:, ::-1]
            older = np.cumsum(oldest, axis=1) - oldest
            served = np.minimum(oldest, np.maximum(caps[:, None] - older, 0))[:, ::-1]
            service_rt = np.zeros(self.channel.M, dtype=np.int64)
            service_nrt = np.zeros(self.channel.M, dtype=np.int64)
            service_rt[cells] = served.sum(axis=1)
            service_nrt[cells] = np.minimum(self.nrt[cells], np.maximum(caps - service_rt[cells], 0))
            delivered = sum(int(v) for v in service_rt[cells]) + sum(int(v) for v in service_nrt[cells])
            weighted_served_rt = _weighted_age_sum(served)
            critical = sum(int(v) for row in served[:, max(0, self.D - 12):] for v in row)
            result = CompactSlotScore(
                value(delivered, weighted_served_rt, self.channel.config), delivered,
                weighted_served_rt, critical, service_rt, service_nrt)

            if full:
                capacities = np.zeros_like(plan_h, dtype=np.int64)
                interference = np.zeros_like(plan_h, dtype=float)
                capacities[edges[:, 0], edges[:, 1]] = caps
                interference[edges[:, 0], edges[:, 1]] = incoming
                served_age = np.zeros_like(self.rt)
                served_age[cells] = served
                residual = self.rt - served_age
                result.actual = actual
                result.capacities = capacities
                result.interference = interference
                result.cell_capacity = cell_capacity
                result.served_age = served_age
                result.residual_rt = residual
                result.expired = residual[:, -1].copy()
                result.weighted_wait = _weighted_age_sum(residual)
                result.power_matrix = raw_power
                result.active_edges = edges.copy()
            if self.meter is not None:
                self.meter.check()
            completed = True
            self._stats['score_completed'] += 1
            return result
        finally:
            if not completed:
                self._stats['score_interrupted'] += 1
            self._stats['score_wall_s'] += time.perf_counter() - started
            self._stats['score_cpu_s'] += time.process_time() - cpu_started
