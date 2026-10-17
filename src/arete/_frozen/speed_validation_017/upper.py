"""Private M1-E winner screening with certified, integer utility upper bounds.

This module does not change the registered pool, positive threshold, tie rule,
capacity policy, reference plan or queue propagation.  It is a computation-order
variant of confirmation_010.marginal_fast.FastM1Scorer, never a default planner.

Sufficient conditions: frozen nonnegative powers and external interference,
fixed receiver pointing / per-beam signal, fixed slot queue, the original
capacity function, and nonnegative exact service-utility coefficients. Adding
an interferer cannot increase an old link's integer capacity. RT oldest-first
service and then NRT service have nondecreasing delivery and served-age utility
as capacity increases. Thus old-link utility losses are nonnegative, and the
new link's exact utility G bounds the full marginal from above. The first
implementation uses G only; it does NOT implement partial-loss early exits.

All reductions for G are original math.fsum reductions, followed by the original
channel.cap certification. Bounds, best values and ties are Python integers in
the original exact common denominator. No epsilon or clipped loss is used.
"""
from __future__ import annotations

import math
import time

import numpy as np

from arete._frozen.confirmation_010.marginal_fast import FastM1Scorer
from arete._frozen.exploration_009.marginal import _check
from arete._frozen.pcqr.physics import NumericUnresolved


EXACT_BATCH_SIZE = 8  # preregistered computation grouping, never search quota


class UpperBoundMixin:
    """Compose before FastM1Scorer or the private ReuseScorer in the MRO."""

    def __init__(self, *args, oracle_callback=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.oracle_callback = oracle_callback
        self._last_selection_certificate = None
        self._stats.update(
            upper_bound_mode=True, upper_exact_batch_size=EXACT_BATCH_SIZE,
            upper_selection_calls=0, upper_candidate_rows_considered=0,
            upper_bound_rows=0, upper_full_scoring_calls=0,
            upper_full_candidate_rows=0, upper_old_receiver_evaluations=0,
            upper_pruned_rows=0, upper_nonpositive_stop_rows=0,
            partial_loss_early_exits=0, upper_numeric_fallbacks=0,
            upper_bound_sum_wall_s=0.0, upper_bound_sum_cpu_s=0.0,
            upper_bound_capacity_wall_s=0.0, upper_bound_capacity_cpu_s=0.0,
            upper_bound_service_wall_s=0.0, upper_bound_service_cpu_s=0.0,
            upper_bound_sort_wall_s=0.0, upper_bound_sort_cpu_s=0.0,
            upper_screen_extra_wall_s=0.0, upper_screen_extra_cpu_s=0.0,
            oracle_wall_s=0.0, oracle_cpu_s=0.0,
            upper_bound_cache='NONE; G RECOMPUTED FOR EVERY SELECTED SET',
            upper_bound_policy='EXACT_NEW_RECEIVER_UTILITY_ONLY',
        )
        # Whole-slot pool validation is paid once, and is NOT a new power cache.
        self._upper_safe = (
            self.utility_delivery >= 0 and self.utility_age >= 0
            and np.all(np.isfinite(self._power)) and np.all(self._power >= 0)
            and math.isfinite(float(self.channel.config.get('I_ext', 0.0)))
            and float(self.channel.config.get('I_ext', 0.0)) >= 0
            # Retain original NumPy integer semantics, but do not use a
            # monotone-service proof when a queue-total reduction may overflow.
            and int(self.rt.max(initial=0)) * self.D <= np.iinfo(np.int64).max
            and int(self.total_rt.max(initial=0)) + int(self.nrt.max(initial=0))
                <= np.iinfo(np.int64).max
        )

    def last_selection_certificate(self):
        return self._last_selection_certificate

    def _full_batch(self, selected, candidates):
        self._stats['upper_full_scoring_calls'] += 1
        self._stats['upper_full_candidate_rows'] += len(candidates)
        self._stats['upper_old_receiver_evaluations'] += len(selected) * len(candidates)
        scores = super().score_additions_sparse(selected, candidates)
        if self.oracle_callback is not None:
            self._last_full_rows_for_oracle.update(
                (int(index), score) for index, score in zip(candidates, scores))
        return scores

    def _exact_new_utilities(self, selected, candidates):
        started, cpu = time.perf_counter(), time.process_time()
        incoming = np.empty(len(candidates), dtype=float)
        external = float(self.channel.config.get('I_ext', 0.0))
        for row, candidate in enumerate(candidates):
            if row % 16 == 0:
                _check(self.meter, reserve=self.reserve)
            incoming[row] = math.fsum(self._power[selected, candidate]) + external
        self._stats['upper_bound_sum_wall_s'] += time.perf_counter() - started
        self._stats['upper_bound_sum_cpu_s'] += time.process_time() - cpu
        started, cpu = time.perf_counter(), time.process_time()
        edges = self.edges[candidates]
        signal = self.channel.signal[self.h, edges[:, 0], edges[:, 1]]
        caps = self.channel.cap(signal, incoming)
        if caps.shape != incoming.shape or caps.dtype.kind not in 'iu' or np.any(caps < 0):
            raise AssertionError('upper new-link capacity is not a nonnegative integer')
        self._stats['upper_bound_capacity_wall_s'] += time.perf_counter() - started
        self._stats['upper_bound_capacity_cpu_s'] += time.process_time() - cpu
        started, cpu = time.perf_counter(), time.process_time()
        cells = edges[:, 1]
        rt = np.minimum(caps, self.total_rt[cells])
        nrt = np.minimum(self.nrt[cells], np.maximum(caps - rt, 0))
        bounds = []
        for row, m in enumerate(cells):
            if row % 16 == 0:
                _check(self.meter, reserve=self.reserve)
            age = int(self._weighted_for_cell(int(m), rt[row:row + 1])[0])
            bounds.append(self.utility_numerator(int(rt[row]) + int(nrt[row]), age))
        self._stats['upper_bound_service_wall_s'] += time.perf_counter() - started
        self._stats['upper_bound_service_cpu_s'] += time.process_time() - cpu
        self._stats['upper_bound_rows'] += len(candidates)
        return bounds

    def select_addition(self, selected_indices, candidate_indices, current_sparse):
        """Return the original exact positive winner, or None for no positive edge.

The optional correctness oracle runs only AFTER this method has determined its
winner and certificate. It cannot supply bounds, priorities or cached scores.
The normal timed and planning paths must use oracle_callback=None.
        """
        started, cpu = time.perf_counter(), time.process_time()
        child_wall = self._stats['score_wall_s']
        child_cpu = self._stats['score_cpu_s']
        oracle_wall_before = self._stats['oracle_wall_s']
        oracle_cpu_before = self._stats['oracle_cpu_s']
        selected = np.asarray(sorted((int(i) for i in selected_indices),
                                     key=lambda i: tuple(self.edges[i])), dtype=np.int64)
        candidates = np.asarray(candidate_indices, dtype=np.int64)
        if set(map(int, current_sparse.indices)) != set(map(int, selected)):
            raise AssertionError('upper-bound reference score does not match selected set')
        self._stats['upper_selection_calls'] += 1
        self._stats['upper_candidate_rows_considered'] += len(candidates)
        if self.oracle_callback is not None:
            # Correctness evidence only, absent from normal timed/planning work.
            self._last_full_rows_for_oracle = {}
        certificate = {
            'selected_indices': selected.tolist(), 'candidate_indices': candidates.tolist(),
            'current_numerator': int(current_sparse.numerator),
            'utility_denominator': self.utility_denominator,
            'bounds': [], 'winner_index': None, 'winner_delta_numerator': None,
            'fallback_reason': None,
            'policy': 'G_ONLY_STRICT_BOUND_LESS_THAN_EXACT_POSITIVE_BEST',
            'partial_loss_early_exits': 0,
        }
        winner = None

        def rank(index, score):
            delta = int(score.numerator) - int(current_sparse.numerator)
            s, m = self.edges[index]
            return (-delta, int(self.channel.sat_ids[s]), int(m), int(index), score)

        def all_exact(reason):
            self._stats['upper_numeric_fallbacks'] += 1
            certificate['fallback_reason'] = str(reason)
            scores = self._full_batch(selected, candidates)
            rows = [rank(index, score) for index, score in zip(candidates, scores)]
            best = min(rows, key=lambda row: row[:3]) if rows else None
            # Previously pruned labels no longer describe the returned decision.
            for entry in certificate['bounds']:
                entry.update(pruned=False, completed=True, fallback_recomputed=True)
            return (best[3], best[4]) if best is not None and -best[0] > 0 else None

        try:
            _check(self.meter, reserve=self.reserve)
            if (np.any(candidates < 0) or np.any(candidates >= len(self.edges))
                    or len(set(selected.tolist())) != len(selected)
                    or np.any(selected < 0) or np.any(selected >= len(self.edges))):
                raise ValueError('invalid pool indices')
            existing = self.edges[selected]
            counts = np.bincount(existing[:, 0], minlength=self.channel.S)
            occupied = set(int(m) for m in existing[:, 1])
            if len(occupied) != len(existing) or np.any(counts > int(self.channel.config['K'])):
                raise ValueError('infeasible selected arrangement')
            for index in candidates:
                s, m = self.edges[index]
                if int(m) in occupied or counts[s] >= int(self.channel.config['K']):
                    raise ValueError('infeasible candidate addition')
            if not len(candidates):
                winner = None
            elif not self._upper_safe:
                winner = all_exact('NONNEGATIVE_FINITE_PHYSICS_OR_UTILITY_CONDITION_FAILED')
            else:
                try:
                    bounds = self._exact_new_utilities(selected, candidates)
                except NumericUnresolved:
                    # An uncertified bound cannot authorize pruning. Re-enter
                    # the original whole-row path; any real failure is retained.
                    winner = all_exact('NEW_LINK_BOUND_NUMERIC_UNRESOLVED')
                    bounds = None
                if bounds is not None:
                    entries = []
                    for index, bound in zip(candidates, bounds):
                        s, m = self.edges[index]
                        entries.append({'index': int(index), 'gid': int(self.channel.sat_ids[s]),
                                        'cell': int(m), 'G_numerator': int(bound),
                                        'completed': False, 'pruned': False})
                    certificate['bounds'] = entries
                    start_sort, sort_cpu = time.perf_counter(), time.process_time()
                    order = sorted(range(len(entries)), key=lambda pos: (
                        -entries[pos]['G_numerator'], entries[pos]['gid'], entries[pos]['cell']))
                    self._stats['upper_bound_sort_wall_s'] += time.perf_counter() - start_sort
                    self._stats['upper_bound_sort_cpu_s'] += time.process_time() - sort_cpu
                    if all(entry['G_numerator'] <= 0 for entry in entries):
                        for entry in entries:
                            entry.update(pruned=True, prune_kind='NONPOSITIVE_BOUND',
                                         prune_best_positive_numerator=0)
                        self._stats['upper_nonpositive_stop_rows'] += len(entries)
                        self._stats['upper_pruned_rows'] += len(entries)
                    else:
                        baseline_caps = {int(i): int(c) for i, c in
                                         zip(current_sparse.indices, current_sparse.capacities)}
                        best = None
                        cursor = 0
                        unsafe_reason = None
                        while cursor < len(order):
                            _check(self.meter, reserve=self.reserve)
                            best_positive = -best[0] if best is not None and -best[0] > 0 else None
                            active_positions = []
                            while cursor < len(order) and len(active_positions) < EXACT_BATCH_SIZE:
                                pos = order[cursor]
                                cursor += 1
                                entry = entries[pos]
                                if best_positive is not None and entry['G_numerator'] < best_positive:
                                    entry.update(pruned=True, prune_kind='STRICT_UPPER_BOUND',
                                                 prune_best_positive_numerator=int(best_positive))
                                    self._stats['upper_pruned_rows'] += 1
                                else:
                                    active_positions.append(pos)
                            if not active_positions:
                                continue
                            batch = [entries[pos]['index'] for pos in active_positions]
                            scores = self._full_batch(selected, batch)
                            for pos, index, score in zip(active_positions, batch, scores):
                                entry = entries[pos]
                                row = rank(index, score)
                                entry.update(completed=True, delta_numerator=int(-row[0]))
                                if (int(-row[0]) > entry['G_numerator'] or any(
                                        int(c) > baseline_caps[int(i)] for i, c in
                                        zip(score.indices[:-1], score.capacities[:-1]))):
                                    unsafe_reason = 'NEGATIVE_OLD_LINK_LOSS_OR_BOUND_VIOLATION'
                                    break
                                if best is None or row[:3] < best[:3]:
                                    best = row
                            if unsafe_reason:
                                break
                        if unsafe_reason:
                            winner = all_exact(unsafe_reason)
                        elif best is not None and -best[0] > 0:
                            winner = (best[3], best[4])
            if winner is not None:
                certificate['winner_index'] = int(winner[0])
                certificate['winner_delta_numerator'] = int(winner[1].numerator - current_sparse.numerator)
            self._last_selection_certificate = certificate
            if self.oracle_callback is not None:
                oracle_start, oracle_cpu = time.perf_counter(), time.process_time()
                self.oracle_callback(self, selected.copy(), candidates.copy(), current_sparse,
                                     winner, certificate)
                self._stats['oracle_wall_s'] += time.perf_counter() - oracle_start
                self._stats['oracle_cpu_s'] += time.process_time() - oracle_cpu
            return winner
        finally:
            # Nonoverlapping screening overhead; complete scoring stays in its
            # original score_wall/CPU fields. All are nested in direct_generation.
            self._stats['upper_screen_extra_wall_s'] += max(0.0, time.perf_counter() - started
                - (self._stats['score_wall_s'] - child_wall)
                - (self._stats['oracle_wall_s'] - oracle_wall_before))
            self._stats['upper_screen_extra_cpu_s'] += max(0.0, time.process_time() - cpu
                - (self._stats['score_cpu_s'] - child_cpu)
                - (self._stats['oracle_cpu_s'] - oracle_cpu_before))


class UpperScorer(UpperBoundMixin, FastM1Scorer):
    pass
