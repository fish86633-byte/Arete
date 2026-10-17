"""Private U implementation: exact, slot-local RT age batching and reuse.

Only the new-link bound's RT age helper work is reorganized.  This module does
not cache capacities, incoming interference, bounds or complete candidate rows.
The original physical arithmetic, exact scorer, pruning and tie rules remain in
speed_validation_017.upper.  A cache belongs to one scorer's immutable RT copy.
"""
from __future__ import annotations

import math
import time

import numpy as np

from arete._frozen.exploration_009.marginal import _check
from arete._frozen.speed_validation_017.upper import UpperScorer


SERVICE_CHECK_BLOCK = 16  # original service loop's row % 16 check boundary


class BatchedUpperScorer(UpperScorer):
    """Use original per-cell age arithmetic for unique RT service counts.

The public ``rt_age_values(cells, counts)`` returns unbounded Python integers in
the supplied order.  It has exactly one original reserve check at each 16-row
boundary.  The normal bound path calls it for consecutive blocks of at most 16
rows so utility construction remains before the next original check boundary.
"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        started, cpu = time.perf_counter(), time.process_time()
        self._rt_age_cache = {}
        self._stats.update(
            rt_age_batch_mode=True,
            rt_age_cache_lifetime='ONE_SCORER_IMMUTABLE_QPLUS_RT_COPY',
            rt_age_cache_key='CELL_AND_INTEGER_RT_SERVICE_COUNT',
            rt_age_check_block_rows=SERVICE_CHECK_BLOCK,
            rt_age_value_calls=0,
            rt_age_queries=0,
            rt_age_cache_hits=0,
            rt_age_cache_misses=0,
            rt_age_within_call_deduplicated=0,
            rt_age_helper_calls=0,
            rt_age_helper_counts=0,
            rt_age_helper_max_batch=0,
            rt_age_cache_entries=0,
            rt_age_reserve_check_calls=0,
            rt_age_total_wall_s=0.0,
            rt_age_total_cpu_s=0.0,
            rt_age_helper_wall_s=0.0,
            rt_age_helper_cpu_s=0.0,
            rt_age_cache_and_group_wall_s=0.0,
            rt_age_cache_and_group_cpu_s=0.0,
        )
        self._stats['rt_age_cache_init_wall_s'] = time.perf_counter() - started
        self._stats['rt_age_cache_init_cpu_s'] = time.process_time() - cpu

    def rt_age_values(self, cells, counts):
        """Batch uncached counts per cell, preserving original integer results.

Only successful original-helper results enter the private cache.  ``counts``
are serviced RT counts, already min(capacity, total RT) in the normal path;
invalid public diagnostic input is rejected rather than truncated.  A Python
int is retained even when weighted ages exceed NumPy's signed 64-bit range.
        """
        started, cpu = time.perf_counter(), time.process_time()
        helper_wall_before = self._stats['rt_age_helper_wall_s']
        helper_cpu_before = self._stats['rt_age_helper_cpu_s']
        self._stats['rt_age_value_calls'] += 1
        try:
            cell_array = np.asarray(cells)
            count_array = np.asarray(counts)
            if (cell_array.ndim != 1 or count_array.ndim != 1
                    or cell_array.shape != count_array.shape):
                raise ValueError('RT age cells/counts must be equal-length vectors')
            if len(cell_array) and (cell_array.dtype.kind not in 'iu'
                                   or count_array.dtype.kind not in 'iuO'):
                raise ValueError('RT age cells/counts must contain integers')
            self._stats['rt_age_queries'] += len(cell_array)
            out = [None] * len(cell_array)
            for start in range(0, len(cell_array), SERVICE_CHECK_BLOCK):
                # This is the original service row % 16 check, not a new
                # deadline check in grouping, helper calls or cache filling.
                self._stats['rt_age_reserve_check_calls'] += 1
                _check(self.meter, reserve=self.reserve)
                stop = min(start + SERVICE_CHECK_BLOCK, len(cell_array))
                pending = {}
                for row in range(start, stop):
                    m, raw_count = int(cell_array[row]), count_array[row]
                    if (isinstance(raw_count, (bool, np.bool_))
                            or not isinstance(raw_count, (int, np.integer))):
                        raise ValueError('RT age service count must be an integer')
                    count = int(raw_count)
                    if not 0 <= m < self.channel.M or not 0 <= count <= int(self.total_rt[m]):
                        raise ValueError('RT age service count is outside the frozen queue')
                    key = (m, count)
                    cached = self._rt_age_cache.get(key)
                    if cached is not None:
                        out[row] = cached
                        self._stats['rt_age_cache_hits'] += 1
                        continue
                    cell_pending = pending.setdefault(m, {})
                    if count in cell_pending:
                        cell_pending[count].append(row)
                        self._stats['rt_age_within_call_deduplicated'] += 1
                    else:
                        cell_pending[count] = [row]
                        self._stats['rt_age_cache_misses'] += 1
                for m, cell_pending in pending.items():
                    unique_counts = list(cell_pending)
                    helper_start, helper_cpu = time.perf_counter(), time.process_time()
                    self._stats['rt_age_helper_calls'] += 1
                    self._stats['rt_age_helper_counts'] += len(unique_counts)
                    self._stats['rt_age_helper_max_batch'] = max(
                        self._stats['rt_age_helper_max_batch'], len(unique_counts))
                    try:
                        values = self._weighted_for_cell(
                            m, np.asarray(unique_counts, dtype=np.int64))
                    finally:
                        self._stats['rt_age_helper_wall_s'] += time.perf_counter() - helper_start
                        self._stats['rt_age_helper_cpu_s'] += time.process_time() - helper_cpu
                    if values.shape != (len(unique_counts),):
                        raise AssertionError('original RT age helper changed output shape')
                    for count, value in zip(unique_counts, values):
                        exact = int(value)
                        self._rt_age_cache[(m, count)] = exact
                        for row in cell_pending[count]:
                            out[row] = exact
                self._stats['rt_age_cache_entries'] = len(self._rt_age_cache)
            if any(value is None for value in out):
                raise AssertionError('RT age batching failed to fill a requested value')
            return out
        finally:
            wall = time.perf_counter() - started
            cpu_cost = time.process_time() - cpu
            self._stats['rt_age_total_wall_s'] += wall
            self._stats['rt_age_total_cpu_s'] += cpu_cost
            # These two children partition rt_age_total; all are nested in
            # upper_bound_service and constructor wall time, never additive
            # with their parents in the total cost ledger.
            self._stats['rt_age_cache_and_group_wall_s'] += max(
                0.0, wall - (self._stats['rt_age_helper_wall_s'] - helper_wall_before))
            self._stats['rt_age_cache_and_group_cpu_s'] += max(
                0.0, cpu_cost - (self._stats['rt_age_helper_cpu_s'] - helper_cpu_before))

    def _exact_new_utilities(self, selected, candidates):
        # Physical sum and capacity statements intentionally match U verbatim.
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
        for start in range(0, len(cells), SERVICE_CHECK_BLOCK):
            stop = min(start + SERVICE_CHECK_BLOCK, len(cells))
            ages = self.rt_age_values(cells[start:stop], rt[start:stop])
            for row, age in zip(range(start, stop), ages):
                bounds.append(self.utility_numerator(int(rt[row]) + int(nrt[row]), age))
        self._stats['upper_bound_service_wall_s'] += time.perf_counter() - started
        self._stats['upper_bound_service_cpu_s'] += time.process_time() - cpu
        self._stats['upper_bound_rows'] += len(candidates)
        return bounds
