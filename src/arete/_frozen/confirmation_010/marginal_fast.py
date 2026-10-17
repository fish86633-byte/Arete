"""Invocation-local exact M1 representation optimization and explicit NX ablation.

M1-E ranks precisely the original M1 all-link integer service utility. M1-NX
changes only the construction marginal: it ignores loss on previously selected
links, while evaluating the new receiver with all selected interferers. Both
retain authoritative original slot propagation and full-plan verification.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import math
import time

import numpy as np

from arete._frozen.pcqr.engine import slot
from arete._frozen.pcqr.search import value, _utility_coefficients
from arete._frozen.pcqr.support import BudgetExceeded, state_hash
from arete._frozen.exploration_009.marginal import (
    BatchSlotScorer, _LinkScore, _ReserveStop, _check, _confirm,
    fixed_demand, fixed_pool, RESERVE_S, CRITICAL_RT_BUCKETS,
)


@dataclass(frozen=True)
class SparseLinkScore:
    """Only active-receiver fields; N is an unbounded Python integer."""
    indices: np.ndarray
    numerator: int
    delivered: int
    weighted_served_rt: int
    service_rt_active: np.ndarray
    service_nrt_active: np.ndarray
    capacities: np.ndarray
    interference: np.ndarray
    new_link_numerator: int


class FastM1Scorer(BatchSlotScorer):
    """Old dense public scorer remains available; construction uses sparse rows.

``score_additions`` and ``score(..., full=True)`` are inherited unchanged, so a
finite component gate can compare dense and sparse outputs on the same pool.
Every sparse row can be expanded with ``materialize_sparse`` for all old fields.
No coefficient, physical cache, or mutable state survives this scorer/call.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        started, cpu_started = time.perf_counter(), time.process_time()
        coefficients = _utility_coefficients(self.channel.config)
        if coefficients is None:
            raise ValueError('M1-E requires the frozen exact Fraction utility')
        delivery, age, denominator = coefficients
        self.utility_denominator = int(denominator)
        self.utility_delivery = int(delivery)
        self.utility_age = int(age)
        self._stats.update(sparse_candidate_rows=0, dense_materializations=0,
                           utility_coefficient_reductions=1,
                           utility_coefficient_wall_s=time.perf_counter() - started,
                           utility_coefficient_cpu_s=time.process_time() - cpu_started,
                           rank_arithmetic='UNBOUNDED_PYTHON_INTEGER_COMMON_DENOMINATOR')

    def utility_numerator(self, delivered, weighted):
        return self.utility_delivery * int(delivered) + self.utility_age * int(weighted)

    def exact_value(self, numerator):
        return Fraction(int(numerator), self.utility_denominator)

    def empty_sparse(self):
        empty_i = np.empty(0, dtype=np.int64)
        empty_f = np.empty(0, dtype=float)
        return SparseLinkScore(empty_i, 0, 0, 0, empty_i, empty_i,
                               empty_i, empty_f, 0)

    def materialize_sparse(self, score):
        """Construct the old complete score only for a winner or explicit gate."""
        self._stats['dense_materializations'] += 1
        cells = self.edges[score.indices, 1]
        rt = np.zeros(self.channel.M, dtype=np.int64)
        nrt = np.zeros(self.channel.M, dtype=np.int64)
        critical = np.zeros(self.channel.M, dtype=np.int64)
        rt[cells] = score.service_rt_active
        nrt[cells] = score.service_nrt_active
        critical[cells] = np.minimum(score.service_rt_active, self.critical[cells])
        return _LinkScore(score.indices.copy(), self.exact_value(score.numerator),
            score.delivered, score.weighted_served_rt,
            sum(int(v) for v in critical), rt, nrt, critical,
            score.capacities.copy(), score.interference.copy())

    def score_additions_sparse(self, selected_indices, candidate_indices):
        """Same physical arithmetic and iteration as the original dense batch.

The only elided per-row work is Fraction reduction, dense M-cell arrays and
critical-RT fields not used by M1 ranking. Retained views own immutable batch
arrays through NumPy bases and cannot inherit data from another invocation.
        """
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
            weighted = np.empty(rt.shape, dtype=np.int64 if self._safe_weighted else object)
            for j, m in enumerate(existing[:, 1]):
                weighted[:, j] = self._weighted_for_cell(int(m), rt[:, j])
            for row, m in enumerate(served_cells[:, L]):
                weighted[row, L] = self._weighted_for_cell(int(m), rt[row, L:L + 1])[0]
            # Views have one invocation-local immutable batch owner. Only the
            # eventual winner is copied/expanded; no all-M array per candidate.
            for array in (indices, incoming, caps, rt, nrt, weighted):
                array.flags.writeable = False
            out = []
            for row in range(C):
                if row % 16 == 0:
                    _check(self.meter, reserve=self.reserve)
                delivered = sum(int(r) + int(n) for r, n in zip(rt[row], nrt[row]))
                weighted_total = sum(int(x) for x in weighted[row])
                new_delivery = int(rt[row, L]) + int(nrt[row, L])
                out.append(SparseLinkScore(indices[row],
                    self.utility_numerator(delivered, weighted_total), delivered,
                    weighted_total, rt[row], nrt[row], caps[row], incoming[row],
                    self.utility_numerator(new_delivery, weighted[row, L])))
            _check(self.meter, reserve=self.reserve)
            self._stats['candidate_rows_completed'] += C
            self._stats['sparse_candidate_rows'] += C
            self._stats['batch_completed'] += 1
            completed = True
            return out
        finally:
            if not completed:
                self._stats['batch_interrupted'] += 1
            self._stats['score_wall_s'] += time.perf_counter() - started
            self._stats['score_cpu_s'] += time.process_time() - cpu_started


def construct_proposal(Q0, A, channel, tpg_plan, meter, *, variant, abs_start=0):
    """Original forward reconstruction with explicit exact/NX marginal modes."""
    if variant not in ('M1-E', 'M1-NX'):
        raise ValueError('M1-E or M1-NX required')
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
              'proxy_is_not_full_F': True, 'intermediate_H_replays': 0,
              'optimization': 'SPARSE_ACTIVE_ROWS_INTEGER_EXACT_UTILITY',
              'externality_loss_in_marginal': variant == 'M1-E'}
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
                scorer = FastM1Scorer(state.rt, nrt, channel, h, edges,
                                      meter=meter, reserve=True)
                selected = []
                arrangement = np.zeros((channel.S, channel.M), dtype=bool)
                current_sparse = scorer.empty_sparse()
                current_score = scorer.materialize_sparse(current_sparse)
                current_proxy_numerator = 0
                current_proxy = scorer.exact_value(current_proxy_numerator)
                while True:
                    _check(meter)
                    satellite_counts = arrangement.sum(axis=1)
                    occupied = arrangement.any(axis=0)
                    candidates = [i for i, (s, m) in enumerate(scorer.edges)
                                  if not occupied[m] and satellite_counts[s] < int(channel.config['K'])]
                    if not candidates:
                        event['fill_stop'] = 'NO_FEASIBLE_POOL_EDGE'
                        break
                    scores = scorer.score_additions_sparse(selected, candidates)
                    ranked = []
                    for index, score in zip(candidates, scores):
                        delta = (score.numerator - current_sparse.numerator
                                 if variant == 'M1-E' else score.new_link_numerator)
                        s, m = scorer.edges[index]
                        ranked.append((-delta, int(channel.sat_ids[s]), int(m), index, score))
                    best = min(ranked, key=lambda row: row[:3])
                    if -best[0] <= 0:
                        event['fill_stop'] = 'NO_POSITIVE_EXACT_MARGINAL'
                        break
                    _, gid, m, index, current_sparse = best
                    current_score = scorer.materialize_sparse(current_sparse)
                    current_proxy_numerator = (current_sparse.numerator if variant == 'M1-E'
                                               else current_proxy_numerator - best[0])
                    current_proxy = scorer.exact_value(current_proxy_numerator)
                    s = int(scorer.edges[index, 0])
                    selected.append(index)
                    arrangement[s, m] = True
                    event['additions'].append({'global_satellite_id': gid, 'cell': m,
                        'delta_proxy': scorer.exact_value(-best[0]), 'proxy': current_proxy,
                        'actual_V': current_score.V})
            except _ReserveStop:
                reserve_stop = True
                event['fill_stop'] = 'COMMIT_RESERVE_STOP'
                if scorer is None:
                    event['status'] = 'RESERVE_BEFORE_SCORER_COMPLETE'
                    raise
            finally:
                if scorer is not None:
                    event['fast_score_stats'] = scorer.stats()
                    for key in ('candidate_rows_attempted', 'candidate_rows_completed', 'batch_calls'):
                        record[key] += event['fast_score_stats'][key]
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
