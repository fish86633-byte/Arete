"""Frozen M1/M2 forward construction with paid all-link marginal scoring.

The only persistent proposal is a complete Boolean plan.  A tentative slot is
committed to that proposal only after the original engine.slot agrees with its
integer service.  No intermediate prefix is replayed through the horizon.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import math
import time

import numpy as np

from arete._frozen.pcqr.engine import slot
from arete._frozen.pcqr.search import value
from arete._frozen.pcqr.support import BudgetExceeded, state_hash
from arete._frozen.exploration_007.local_score import _queue_array, _weighted_age_sum
from arete._frozen.exploration_008.fast_score import CompactSlotScore


RESERVE_S = .10
POOL_PER_SATELLITE_PER_STREAM = 8
CRITICAL_RT_BUCKETS = 12


class _ReserveStop(Exception):
    pass


def _check(meter, *, reserve=True):
    if meter is None:
        return
    meter.check()
    if reserve and meter.remaining <= RESERVE_S:
        raise _ReserveStop('COMMIT_RESERVE_STOP')


def fixed_demand(Q0, A):
    nrt = Q0.nrt
    return tuple(sum(int(x) for x in Q0.rt[m]) + int(nrt[m])
                 + sum(int(x) for x in A[:, m].flat)
                 for m in range(len(Q0.rt)))


def _weights(denominator, prefix_served):
    return tuple(Fraction(max(0, demand - int(prefix_served[m])), demand)
                 if demand else Fraction(0)
                 for m, demand in enumerate(denominator))


def fixed_pool(rt, nrt, channel, h, tpg_slot, denominator, prefix_served):
    """Same finite pool for both policies; numbering uses global satellite IDs."""
    total = rt.sum(axis=1) + nrt
    isolated = channel.cap(channel.signal[h], channel.config.get('I_ext', 0.0))
    debt = _weights(denominator, prefix_served)
    chosen = set()
    streams = []
    for s, gid in enumerate(channel.sat_ids):
        rows = []
        for m in np.flatnonzero(channel.visible[h, s] & (total > 0)):
            m = int(m)
            useful = min(int(total[m]), int(isolated[s, m]))
            if useful <= 0:
                continue
            normalized = debt[m] * Fraction(useful, denominator[m])
            rows.append((m, useful, int(isolated[s, m]), normalized))
        order_v = sorted(rows, key=lambda r: (-r[1], -r[2], int(gid), r[0]))
        order_d = sorted(rows, key=lambda r: (-r[3], -r[1], -r[2], int(gid), r[0]))
        head = [(s, r[0]) for r in order_v[:POOL_PER_SATELLITE_PER_STREAM]]
        debt_edges = [(s, r[0]) for r in order_d[:POOL_PER_SATELLITE_PER_STREAM]]
        chosen.update(head)
        chosen.update(debt_edges)
        streams.append({'global_satellite_id': int(gid),
                        'isolated_cells': [m for _, m in head],
                        'debt_cells': [m for _, m in debt_edges]})
    inherited = np.asarray(tpg_slot, dtype=bool) & channel.visible[h] & (total > 0)[None, :]
    chosen.update((int(s), int(m)) for s, m in np.argwhere(inherited))
    edges = np.asarray(sorted(chosen, key=lambda e: (int(channel.sat_ids[e[0]]), e[1])),
                       dtype=np.int64).reshape(-1, 2)
    return edges, {'per_stream_per_satellite': POOL_PER_SATELLITE_PER_STREAM,
                   'streams': streams, 'edge_count': len(edges),
                   'edges': [(int(channel.sat_ids[s]), int(m)) for s, m in edges],
                   'valid_nonempty_TPG_edges': int(inherited.sum()),
                   'ordering': 'global_satellite_id_then_cell'}


@dataclass
class _LinkScore:
    indices: np.ndarray
    V: Fraction
    delivered: int
    weighted_served_rt: int
    critical_RT_served: int
    service_rt: np.ndarray
    service_nrt: np.ndarray
    critical_by_cell: np.ndarray
    capacities: np.ndarray
    interference: np.ndarray


class BatchSlotScorer:
    """One paid slot-local pool matrix, with exact fsum and integer services.

    score_additions(selected_indices, candidate_indices) evaluates every link
    after each candidate addition in one capacity call.  score(arrangement,
    full=True) exposes the original compact diagnostic fields for fixed gates.
    Every supplied arrangement must be feasible and contained in the pool.
    """

    def __init__(self, rt, nrt, channel, h, edges, *, meter=None, reserve=False):
        started, cpu_started = time.perf_counter(), time.process_time()
        self.channel, self.h, self.meter = channel, int(h), meter
        self.reserve = bool(reserve)
        if isinstance(h, (bool, np.bool_)) or self.h != h or not 0 <= self.h < channel.H:
            raise ValueError('invalid planning slot')
        self.D = int(channel.config['ttl'])
        self.rt = _queue_array(rt, (channel.M, self.D), 'RT queue').copy()
        self.nrt = _queue_array(nrt, (channel.M,), 'NRT totals').copy()
        self.rt.flags.writeable = self.nrt.flags.writeable = False
        self.total_rt = self.rt.sum(axis=1)
        self.nonempty = (self.total_rt + self.nrt) > 0
        self.critical = self.rt[:, max(0, self.D - CRITICAL_RT_BUCKETS):].sum(axis=1)
        self.oldest_cumulative = np.cumsum(self.rt[:, ::-1], axis=1)
        self._weighted_total = _weighted_age_sum(self.rt)
        self._safe_weighted = self._weighted_total <= np.iinfo(np.int64).max
        if self._safe_weighted:
            self.weighted_cumulative = np.cumsum(
                self.rt[:, ::-1] * np.arange(self.D, 0, -1, dtype=np.int64), axis=1)
        else:
            self.weighted_cumulative = None
        edges = np.asarray(edges)
        if edges.size == 0:
            edges = np.empty((0, 2), dtype=np.int64)
        if edges.ndim != 2 or edges.shape[1] != 2 or edges.dtype.kind not in 'iu':
            raise ValueError('pool edges must be an integer N by 2 array')
        edges = np.array(edges, dtype=np.int64, copy=True)
        if (np.any(edges < 0) or np.any(edges[:, 0] >= channel.S)
                or np.any(edges[:, 1] >= channel.M)):
            raise ValueError('pool edge outside the channel')
        if len({tuple(e) for e in edges.tolist()}) != len(edges):
            raise ValueError('duplicate pool edge')
        if len(edges) and np.any(~channel.visible[self.h, edges[:, 0], edges[:, 1]]):
            raise ValueError('invisible pool edge')
        # Empty queues are silent in the original engine, including in gates.
        edges = edges[self.nonempty[edges[:, 1]]]
        self.edges = edges
        self.edges.flags.writeable = False
        self._index = {tuple(int(v) for v in edge): i for i, edge in enumerate(edges)}
        self._stats = {'union_edge_count': len(edges), 'power_prebuild_calls': 0,
                       'batch_calls': 0, 'batch_completed': 0, 'batch_interrupted': 0,
                       'candidate_rows_attempted': 0, 'candidate_rows_completed': 0,
                       'score_wall_s': 0.0, 'score_cpu_s': 0.0,
                       'power_for_calls': 0, 'membership_fallbacks': 0,
                       'cache_lifetime': 'ONE_PAID_INVOCATION_ONE_SLOT',
                       'time_accounting': 'NESTED_IN_DIRECT_GENERATION'}
        _check(meter, reserve=self.reserve)
        if len(edges):
            self._power = channel.power(self.h, edges, edges)
            self._stats['power_prebuild_calls'] = 1
        else:
            self._power = np.zeros((0, 0), dtype=float)
        self._power.flags.writeable = False
        self._stats['matrix_payload_bytes'] = int(self._power.nbytes)
        self._stats['pool_sha256'] = hashlib.sha256(edges.tobytes()).hexdigest()
        self._stats['prebuild_wall_s'] = time.perf_counter() - started
        self._stats['prebuild_cpu_s'] = time.process_time() - cpu_started
        _check(meter, reserve=self.reserve)

    def stats(self):
        return dict(self._stats)

    def power_for(self, edges):
        """Owned raw watts in caller order; missing membership uses full power."""
        _check(self.meter, reserve=self.reserve)
        edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
        self._stats['power_for_calls'] += 1
        indices = [self._index.get(tuple(int(v) for v in edge)) for edge in edges]
        if all(index is not None for index in indices):
            indices = np.asarray(indices, dtype=np.int64)
            return self._power[np.ix_(indices, indices)].copy()
        self._stats['membership_fallbacks'] += 1
        return self.channel.power(self.h, edges, edges)

    def _weighted_for_cell(self, m, counts):
        counts = np.asarray(counts, dtype=np.int64)
        indices = np.searchsorted(self.oldest_cumulative[m], counts, side='left')
        previous_counts = np.where(indices > 0,
                                   self.oldest_cumulative[m, np.maximum(indices - 1, 0)], 0)
        if self._safe_weighted:
            previous_weight = np.where(indices > 0,
                self.weighted_cumulative[m, np.maximum(indices - 1, 0)], 0)
            return previous_weight + (counts - previous_counts) * (self.D - indices)
        out = []
        for count in counts.flat:
            remaining, weighted = int(count), 0
            for age in range(self.D - 1, -1, -1):
                take = min(remaining, int(self.rt[m, age]))
                weighted += take * (age + 1)
                remaining -= take
                if not remaining:
                    break
            out.append(weighted)
        return np.asarray(out, dtype=object).reshape(counts.shape)

    def empty_score(self):
        zeros = np.zeros(self.channel.M, dtype=np.int64)
        return _LinkScore(np.empty(0, dtype=np.int64), value(0, 0, self.channel.config),
                          0, 0, 0, zeros, zeros.copy(), zeros.copy(),
                          np.empty(0, dtype=np.int64), np.empty(0, dtype=float))

    def score_additions(self, selected_indices, candidate_indices, *, full=False):
        """Batch the physical capacity call; include losses on all old beams."""
        started, cpu_started = time.perf_counter(), time.process_time()
        self._stats['batch_calls'] += 1
        completed = False
        candidates = np.asarray(candidate_indices, dtype=np.int64)
        selected = np.asarray(sorted((int(i) for i in selected_indices),
                                     key=lambda i: tuple(self.edges[i])), dtype=np.int64)
        self._stats['candidate_rows_attempted'] += len(candidates)
        try:
            _check(self.meter, reserve=self.reserve)
            if not len(candidates):
                completed = True
                self._stats['batch_completed'] += 1
                return []
            if (np.any(candidates < 0) or np.any(candidates >= len(self.edges))
                    or len(set(selected.tolist())) != len(selected)
                    or np.any(selected < 0) or np.any(selected >= len(self.edges))):
                raise ValueError('invalid pool indices')
            existing = self.edges[selected]
            counts = np.bincount(existing[:, 0], minlength=self.channel.S)
            cells = set(int(m) for m in existing[:, 1])
            if len(cells) != len(existing) or np.any(counts > int(self.channel.config['K'])):
                raise ValueError('infeasible selected arrangement')
            for index in candidates:
                s, m = self.edges[index]
                if int(m) in cells or counts[s] >= int(self.channel.config['K']):
                    raise ValueError('infeasible candidate addition')
            C, L = len(candidates), len(selected)
            indices = np.empty((C, L + 1), dtype=np.int64)
            indices[:, :L] = selected
            indices[:, L] = candidates
            incoming = np.empty((C, L + 1), dtype=float)
            external = float(self.channel.config.get('I_ext', 0.0))
            # fsum is the authoritative reduction.  A partial sum plus a new
            # watt or np.sum would alter the floor-capacity boundary.
            parts = [self._power[selected[selected != receiver], receiver].tolist()
                     for receiver in selected]
            for row, candidate in enumerate(candidates):
                if row % 16 == 0:
                    _check(self.meter, reserve=self.reserve)
                for j, receiver in enumerate(selected):
                    incoming[row, j] = math.fsum(parts[j] + [float(self._power[candidate, receiver])]) + external
                incoming[row, L] = math.fsum(self._power[selected, candidate]) + external
            signal = self.channel.signal[self.h, self.edges[indices, 0], self.edges[indices, 1]]
            caps = self.channel.cap(signal, incoming)
            if caps.shape != incoming.shape or caps.dtype.kind not in 'iu' or np.any(caps < 0):
                raise AssertionError('batch capacity must be nonnegative integer counts')
            served_cells = self.edges[indices, 1]
            rt = np.minimum(caps, self.total_rt[served_cells])
            nrt = np.minimum(self.nrt[served_cells], np.maximum(caps - rt, 0))
            critical = np.minimum(rt, self.critical[served_cells])
            weighted = np.empty(rt.shape, dtype=np.int64 if self._safe_weighted else object)
            for j, m in enumerate(existing[:, 1]):
                weighted[:, j] = self._weighted_for_cell(int(m), rt[:, j])
            for row, m in enumerate(served_cells[:, L]):
                weighted[row, L] = self._weighted_for_cell(int(m), rt[row, L:L + 1])[0]
            out = []
            for row in range(C):
                if row % 16 == 0:
                    _check(self.meter, reserve=self.reserve)
                delivered = sum(int(r) + int(n) for r, n in zip(rt[row], nrt[row]))
                weighted_total = sum(int(x) for x in weighted[row])
                critical_total = sum(int(x) for x in critical[row])
                full_rt = np.zeros(self.channel.M, dtype=np.int64)
                full_nrt = np.zeros(self.channel.M, dtype=np.int64)
                full_critical = np.zeros(self.channel.M, dtype=np.int64)
                full_rt[served_cells[row]] = rt[row]
                full_nrt[served_cells[row]] = nrt[row]
                full_critical[served_cells[row]] = critical[row]
                out.append(_LinkScore(indices[row].copy(),
                    value(delivered, weighted_total, self.channel.config), delivered,
                    weighted_total, critical_total, full_rt, full_nrt, full_critical,
                    caps[row].copy(), incoming[row].copy()))
            _check(self.meter, reserve=self.reserve)
            self._stats['candidate_rows_completed'] += C
            self._stats['batch_completed'] += 1
            completed = True
            return [self._compact(score, full=True) for score in out] if full else out
        finally:
            if not completed:
                self._stats['batch_interrupted'] += 1
            self._stats['score_wall_s'] += time.perf_counter() - started
            self._stats['score_cpu_s'] += time.process_time() - cpu_started

    def _compact(self, score, *, full):
        result = CompactSlotScore(score.V, score.delivered, score.weighted_served_rt,
                                  score.critical_RT_served, score.service_rt.copy(),
                                  score.service_nrt.copy())
        if not full:
            return result
        edges = self.edges[score.indices]
        actual = np.zeros((self.channel.S, self.channel.M), dtype=bool)
        capacities = np.zeros_like(actual, dtype=np.int64)
        interference = np.zeros_like(actual, dtype=float)
        cell_capacity = np.zeros(self.channel.M, dtype=np.int64)
        if len(edges):
            actual[edges[:, 0], edges[:, 1]] = True
            capacities[edges[:, 0], edges[:, 1]] = score.capacities
            interference[edges[:, 0], edges[:, 1]] = score.interference
            cell_capacity[edges[:, 1]] = score.capacities
        oldest = self.rt[:, ::-1]
        older = np.cumsum(oldest, axis=1) - oldest
        served_age = np.minimum(oldest, np.maximum(cell_capacity[:, None] - older, 0))[:, ::-1]
        residual = self.rt - served_age
        result.actual, result.capacities = actual, capacities
        result.interference, result.cell_capacity = interference, cell_capacity
        result.served_age, result.residual_rt = served_age, residual
        result.expired, result.weighted_wait = residual[:, -1].copy(), _weighted_age_sum(residual)
        order = sorted(range(len(edges)), key=lambda i: tuple(edges[i]))
        result.active_edges = edges[order].copy()
        original_indices = score.indices[order]
        result.power_matrix = self._power[np.ix_(original_indices, original_indices)].copy()
        return result

    def score(self, arrangement, *, full=True):
        plan_h = np.asarray(arrangement)
        if plan_h.shape != (self.channel.S, self.channel.M) or not np.all((plan_h == 0) | (plan_h == 1)):
            raise ValueError('invalid slot arrangement')
        plan_h = plan_h.astype(bool)
        if (np.any(plan_h & ~self.channel.visible[self.h])
                or np.any(plan_h.sum(axis=1) > int(self.channel.config['K']))
                or np.any(plan_h.sum(axis=0) > 1)):
            raise ValueError('infeasible slot arrangement')
        active = np.argwhere(plan_h & self.nonempty[None, :])
        try:
            indices = [self._index[tuple(int(v) for v in edge)] for edge in active]
        except KeyError as error:
            raise ValueError('arrangement edge not in the registered finite pool') from error
        score = (self.score_additions(indices[:-1], indices[-1:])[0]
                 if indices else self.empty_score())
        return self._compact(score, full=full)


def score_arrangements(rt, nrt, channel, h, arrangements, *, meter=None):
    """Finite gate API: return full CompactSlotScore for every fixed arrangement.

    It builds one paid union from the supplied arrangements and invokes the same
    addition batch primitive used by construction.  It advances no queue.
    """
    arrangements = list(arrangements)
    union = np.zeros((channel.S, channel.M), dtype=bool)
    for arrangement in arrangements:
        array = np.asarray(arrangement)
        if array.shape != union.shape or not np.all((array == 0) | (array == 1)):
            raise ValueError('invalid fixed arrangement')
        union |= array.astype(bool)
    scorer = BatchSlotScorer(rt, nrt, channel, h, np.argwhere(union), meter=meter)
    return [scorer.score(arrangement, full=True) for arrangement in arrangements]


def _proxy(score, variant, fairness_weights, critical_weights, mean_demand, a, b, ttl):
    if variant == 'M1':
        return score.V
    normal = sum((fairness_weights[m] * (int(rt) + int(nrt))
                  for m, (rt, nrt) in enumerate(zip(score.service_rt, score.service_nrt))
                  if int(rt) + int(nrt)), Fraction(0))
    critical = sum((critical_weights[m] * int(count)
                    for m, count in enumerate(score.critical_by_cell) if int(count)), Fraction(0))
    return score.V + a * mean_demand * normal + b * ttl * mean_demand * critical


def _confirm(score, scorer, item):
    expected = scorer._compact(score, full=True)
    comparisons = {
        'active': np.array_equal(expected.actual, item['active']),
        'capacities': np.array_equal(expected.capacities, item['capacities']),
        'interference': np.array_equal(expected.interference, item['interference']),
        'service_rt': np.array_equal(expected.service_rt, item['service_rt']),
        'service_nrt': np.array_equal(expected.service_nrt, item['service_nrt']),
        'served_age': np.array_equal(expected.served_age, item['served_age']),
        'weighted_served_rt': score.weighted_served_rt == _weighted_age_sum(item['served_age']),
        'critical_RT_served': score.critical_RT_served == sum(
            int(x) for x in item['served_age'][:, max(0, scorer.D - CRITICAL_RT_BUCKETS):].flat),
    }
    if not all(comparisons.values()):
        error = AssertionError('batch scoring differs from original engine.slot')
        error.details = {'h': scorer.h, 'comparisons': comparisons,
                         'expected_capacities': expected.capacities.tolist(),
                         'actual_capacities': item['capacities'].tolist(),
                         'expected_service_rt': expected.service_rt.tolist(),
                         'actual_service_rt': item['service_rt'].tolist(),
                         'expected_service_nrt': expected.service_nrt.tolist(),
                         'actual_service_nrt': item['service_nrt'].tolist()}
        raise error


def construct_proposal(Q0, A, channel, tpg_plan, meter, *, variant, abs_start=0):
    """At most one forward reconstruction, retaining the original TPG suffix."""
    if variant not in ('M1', 'M2'):
        raise ValueError('M1 or M2 required')
    proposal = np.asarray(tpg_plan, dtype=bool).copy()
    state = Q0.copy()
    denominator = fixed_demand(Q0, A)
    prefix_served = [0] * channel.M
    positive = [demand for demand in denominator if demand]
    mean_demand = Fraction(sum(positive), len(positive)) if positive else Fraction(0)
    a, b = value(1, 0, channel.config), value(0, 1, channel.config)
    events = []
    record = {'variant': variant, 'status': 'RUNNING', 'stop': None,
              'prefix_complete_slots': 0, 'slots_started': 0, 'slots': events,
              'fixed_planning_demand': denominator, 'mean_positive_demand': mean_demand,
              'M2_extra_coefficients': [1, 1], 'critical_RT_buckets': CRITICAL_RT_BUCKETS,
              'a': a, 'b': b, 'commit_reserve_s': RESERVE_S,
              'prefix_served': prefix_served, 'candidate_rows_attempted': 0,
              'candidate_rows_completed': 0, 'batch_calls': 0,
              'tie_rule': 'largest_exact_delta_then_global_satellite_id_then_cell',
              'proxy_is_not_full_F': True, 'intermediate_H_replays': 0}
    try:
        for h in range(channel.H):
            _check(meter)
            event = {'h': h, 'status': 'RUNNING', 'additions': [], 'slot_confirmed': False}
            events.append(event)
            record['slots_started'] += 1
            event['absolute_slot'] = abs_start + h
            event['input_state_hash'] = state_hash(state)
            state.add(A[h], abs_start + h)
            nrt = state.nrt
            event.update(arrival_add_calls=1, arrival_applied_before_service=True,
                         qplus_rt_sha256=hashlib.sha256(state.rt.tobytes()).hexdigest(),
                         qplus_nrt_sha256=hashlib.sha256(nrt.tobytes()).hexdigest())
            edges, pool_record = fixed_pool(state.rt, nrt, channel, h, tpg_plan[h],
                                            denominator, prefix_served)
            event['pool'] = pool_record
            scorer = None
            reserve_stop = False
            try:
                scorer = BatchSlotScorer(state.rt, nrt, channel, h, edges,
                                         meter=meter, reserve=True)
                debt = _weights(denominator, prefix_served)
                fairness_weights = tuple(debt[m] / denominator[m] if denominator[m] else Fraction(0)
                                         for m in range(channel.M))
                critical_weights = tuple(Fraction(1, demand) if demand else Fraction(0)
                                         for demand in denominator)
                selected = []
                arrangement = np.zeros((channel.S, channel.M), dtype=bool)
                current_score = scorer.empty_score()
                current_proxy = _proxy(current_score, variant, fairness_weights, critical_weights,
                                       mean_demand, a, b, int(channel.config['ttl']))
                while True:
                    _check(meter)
                    satellite_counts = arrangement.sum(axis=1)
                    occupied = arrangement.any(axis=0)
                    candidates = [i for i, (s, m) in enumerate(scorer.edges)
                                  if not occupied[m] and satellite_counts[s] < int(channel.config['K'])]
                    if not candidates:
                        event['fill_stop'] = 'NO_FEASIBLE_POOL_EDGE'
                        break
                    scores = scorer.score_additions(selected, candidates)
                    ranked = []
                    for index, score in zip(candidates, scores):
                        proxy = _proxy(score, variant, fairness_weights, critical_weights,
                                       mean_demand, a, b, int(channel.config['ttl']))
                        delta = proxy - current_proxy
                        s, m = scorer.edges[index]
                        ranked.append((-delta, int(channel.sat_ids[s]), int(m), index, score, proxy))
                    best = min(ranked, key=lambda row: row[:3])
                    if -best[0] <= 0:
                        event['fill_stop'] = 'NO_POSITIVE_EXACT_MARGINAL'
                        break
                    _, gid, m, index, current_score, current_proxy = best
                    s = int(scorer.edges[index, 0])
                    selected.append(index)
                    arrangement[s, m] = True
                    event['additions'].append({'global_satellite_id': gid, 'cell': m,
                                               'delta_proxy': -best[0], 'proxy': current_proxy,
                                               'actual_V': current_score.V})
            except _ReserveStop:
                reserve_stop = True
                event['fill_stop'] = 'COMMIT_RESERVE_STOP'
                # If scorer construction itself was interrupted, this slot
                # has no completed selected score and cannot join the prefix.
                if scorer is None:
                    event['status'] = 'RESERVE_BEFORE_SCORER_COMPLETE'
                    raise
            finally:
                if scorer is not None:
                    event['fast_score_stats'] = scorer.stats()
                    for key in ('candidate_rows_attempted', 'candidate_rows_completed', 'batch_calls'):
                        record[key] += event['fast_score_stats'][key]
            # This single original slot call is also the authoritative queue
            # advance.  It is paid even if its cooperative completion overruns.
            meter.check()
            item = slot(state, arrangement, channel, h, abs_start + h)
            meter.replay('009_construction_slot', 1)
            _confirm(current_score, scorer, item)
            proposal[h] = arrangement
            for m, (rt, nonrt) in enumerate(zip(item['service_rt'], item['service_nrt'])):
                prefix_served[m] += int(rt) + int(nonrt)
            event.update(status='COMPLETED', slot_confirmed=True,
                         delivered=current_score.delivered, actual_V=current_score.V,
                         weighted_served_rt=current_score.weighted_served_rt,
                         critical_RT_served=current_score.critical_RT_served,
                         output_state_hash=state_hash(state),
                         service_rt=item['service_rt'].tolist(),
                         service_nrt=item['service_nrt'].tolist(),
                         expired=item['expired'].tolist(),
                         service_rt_sha256=hashlib.sha256(item['service_rt'].tobytes()).hexdigest(),
                         service_nrt_sha256=hashlib.sha256(item['service_nrt'].tobytes()).hexdigest(),
                         expired_sha256=hashlib.sha256(item['expired'].tobytes()).hexdigest())
            record['prefix_complete_slots'] += 1
            if reserve_stop:
                record.update(status='PARTIAL_PREFIX', stop='COMMIT_RESERVE_STOP')
                break
        else:
            record.update(status='COMPLETED', stop='HORIZON_COMPLETED')
    except _ReserveStop:
        record.update(status='PARTIAL_PREFIX', stop='COMMIT_RESERVE_STOP')
    except BudgetExceeded:
        record.update(status='PARTIAL_PREFIX', stop='TIME_LIMIT')
        if events and not events[-1]['slot_confirmed']:
            events[-1]['status'] = 'TIME_LIMIT_SLOT_DISCARDED'
    except Exception as error:
        record.update(status='ERROR', stop='CORRECTNESS_FAILURE', error=repr(error),
                      error_details=getattr(error, 'details', None))
        error.construction_record = record
        error.proposal_plan = proposal
        raise
    return proposal, record
