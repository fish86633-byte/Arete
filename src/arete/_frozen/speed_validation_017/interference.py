"""Private switch I: exact dyadic reuse of old-receiver sums.

This is not I_base + p.  Frozen binary64 terms are integer multiples of one
power of two.  The common selected prefix is added as unbounded integers once
per U, and each candidate is then added before a *single* binary64 rounding.
Only CPython 3.14.6 / Darwin ARM64 / ordinary IEEE half-even builds are enabled.
Unsupported platforms, unsafe inputs, overflow and subnormal output revert to
the original math.fsum list in its original order.  New receivers always use
the original math.fsum.  No capacity policy or physical term is changed.

Proof sources matching the actual venv (pyvenv.cfg: 3.14.6):
https://github.com/python/cpython/blob/v3.14.6/Modules/mathmodule.c
https://github.com/python/cpython/blob/v3.14.6/Objects/longobject.c
The first describes correct half-even final rounding under IEEE arithmetic;
the second implements correctly rounded PyLong_AsDouble.  Normal ldexp changes
only the exponent.  Those two rounded values therefore equal the same exact
dyadic sum.  The final + I_ext remains the original binary64 operation.
All setup, conversion and cache preparation is paid by the caller's scorer.
"""
from __future__ import annotations

import ctypes
import math
import platform
import sys
import sysconfig
import time

import numpy as np

from arete._frozen.confirmation_010.marginal_fast import FastM1Scorer, SparseLinkScore
from arete._frozen.exploration_009.marginal import _check


def platform_contract():
    """Conservative, invocation-local gate, not a numerical benchmark."""
    flags = ' '.join(str(sysconfig.get_config_var(key) or '') for key in
                     ('CFLAGS', 'PY_CFLAGS', 'PY_CFLAGS_NODIST',
                      'PY_STDMODULE_CFLAGS', 'CONFIG_ARGS'))
    rounding = None
    try:
        library = ctypes.CDLL(None)
        library.fegetround.restype = ctypes.c_int
        library.fegetround.argtypes = []
        rounding = int(library.fegetround())
    except (OSError, AttributeError):
        pass
    facts = {
        'implementation': sys.implementation.name,
        'version': tuple(sys.version_info[:3]),
        'system': platform.system(), 'machine': platform.machine(),
        'float_format': float.__getformat__('double'),
        'radix': sys.float_info.radix, 'mant_dig': sys.float_info.mant_dig,
        'rounds': sys.float_info.rounds, 'fegetround': rounding,
        'sizeof_double': sysconfig.get_config_var('SIZEOF_DOUBLE'),
        'sizeof_long_double': sysconfig.get_config_var('SIZEOF_LONG_DOUBLE'),
        'unsafe_compiler_flags_present': any(flag in flags for flag in
            ('-ffast-math', '-fassociative-math', '-funsafe-math-optimizations',
             '-Ofast')),
        'proof_scope': 'CPython_3.14.6_DARWIN_ARM64_IEEE_NORMAL_FINITE_NONNEGATIVE',
    }
    facts['enabled'] = (
        facts['implementation'] == 'cpython'
        and facts['version'] == (3, 14, 6)
        and facts['system'] == 'Darwin'
        and facts['machine'] in ('arm64', 'aarch64')
        and facts['float_format'].startswith('IEEE')
        and facts['radix'] == 2 and facts['mant_dig'] == 53
        and facts['rounds'] == 1 and facts['fegetround'] == 0
        and facts['sizeof_double'] == 8 and facts['sizeof_long_double'] == 8
        and not facts['unsafe_compiler_flags_present'])
    return facts


class DyadicInterferenceCache:
    """One scorer/input/slot/power matrix, keyed by complete ordered U.

    Columns contain exact integer terms in a shared dyadic unit.  A changed U
    invalidates all shared prefixes, not columns belonging to immutable power.
    Tentative candidates cannot mutate either prefixes or other candidates.
    """
    def __init__(self, power, external, *, check=None):
        started, cpu_started = time.perf_counter(), time.process_time()
        power = np.asarray(power, dtype=float)
        self.power = power if not power.flags.writeable else power.copy()
        self.power.flags.writeable = False
        self.external = float(external)
        self.check = check
        self.contract = platform_contract()
        self.columns = {}
        self._key = None
        self.prefixes = {}
        self.parts = {}
        self._stats = {
            'I_contract': self.contract,
            'I_lifetime': 'ONE_SCORER_INPUT_SLOT_POWER_AND_ORDERED_U',
            'I_setup_wall_s': time.perf_counter() - started,
            'I_setup_cpu_s': time.process_time() - cpu_started,
            'I_total_wall_s': 0.0, 'I_total_cpu_s': 0.0,
            'I_prefix_wall_s': 0.0, 'I_prefix_cpu_s': 0.0,
            'I_old_sum_wall_s': 0.0, 'I_old_sum_cpu_s': 0.0,
            'I_new_sum_wall_s': 0.0, 'I_new_sum_cpu_s': 0.0,
            'I_U_builds': 0, 'I_U_hits': 0, 'I_U_invalidations': 0,
            'I_column_builds': 0, 'I_column_hits': 0,
            'I_column_float_conversions': 0, 'I_old_receiver_evaluations': 0,
            'I_old_prefix_terms': 0, 'I_reused_old_sum_terms': 0,
            'I_dyadic_rounds': 0, 'I_original_old_fsum_calls': 0,
            'I_original_new_fsum_calls': 0, 'I_numerical_fallbacks': 0,
            'I_fallback_reasons': {},
            'I_oracle_bit_comparisons': 0, 'I_oracle_bit_differences': 0,
            'I_oracle_bit_diff_examples': [],
            'I_cache_update_policy': 'PREFIX_ONLY_AFTER_ORDERED_U_CHANGE',
            'I_float_policy': 'EXACT_INTEGER_SUM_ONE_ROUND_NORMAL_ELSE_ORIGINAL_FSUM',
        }

    def stats(self):
        return dict(self._stats)

    def _column(self, receiver):
        if receiver in self.columns:
            self._stats['I_column_hits'] += 1
            return self.columns[receiver]
        self._stats['I_column_builds'] += 1
        values = self.power[:, receiver].tolist()
        # Include same-satellite powers; no threshold/graph mask is applied.
        if (not self.contract['enabled']
                or any(not math.isfinite(value) or value < 0
                       or 0 < value < sys.float_info.min for value in values)):
            result = None
        else:
            ratios = [value.as_integer_ratio() for value in values]
            self._stats['I_column_float_conversions'] += len(ratios)
            exponent = max((den.bit_length() - 1 for _, den in ratios), default=0)
            units = tuple(num << (exponent - (den.bit_length() - 1))
                          for num, den in ratios)
            result = exponent, units
        self.columns[receiver] = result
        return result

    def _prepare(self, selected):
        key = tuple(int(index) for index in selected)
        if key == self._key:
            self._stats['I_U_hits'] += 1
            return
        started, cpu_started = time.perf_counter(), time.process_time()
        if self._key is not None:
            self._stats['I_U_invalidations'] += 1
        self._stats['I_U_builds'] += 1
        # Do not publish a partially prepared prefix on interruption.
        prefixes, parts = {}, {}
        for j, receiver in enumerate(key):
            if self.check is not None and j % 16 == 0:
                self.check()
            old = tuple(index for index in key if index != receiver)
            parts[receiver] = self.power[list(old), receiver].tolist()
            column = self._column(receiver)
            if column is not None:
                exponent, units = column
                prefixes[receiver] = (exponent, units,
                                      sum(units[index] for index in old))
                self._stats['I_old_prefix_terms'] += len(old)
            else:
                prefixes[receiver] = None
        self._key, self.prefixes, self.parts = key, prefixes, parts
        self._stats['I_prefix_wall_s'] += time.perf_counter() - started
        self._stats['I_prefix_cpu_s'] += time.process_time() - cpu_started

    def _fallback(self, receiver, candidate, reason):
        self._stats['I_numerical_fallbacks'] += 1
        reasons = self._stats['I_fallback_reasons']
        reasons[reason] = reasons.get(reason, 0) + 1
        self._stats['I_original_old_fsum_calls'] += 1
        return math.fsum(self.parts[receiver]
                         + [float(self.power[candidate, receiver])])

    def _old(self, receiver, candidate):
        self._stats['I_old_receiver_evaluations'] += 1
        prefix = self.prefixes[receiver]
        if prefix is None:
            return self._fallback(receiver, candidate, 'PLATFORM_OR_UNSAFE_COLUMN')
        exponent, units, total = prefix
        total += units[candidate]
        if total == 0:
            self._stats['I_dyadic_rounds'] += 1
            self._stats['I_reused_old_sum_terms'] += len(self.parts[receiver])
            return 0.0
        # float(int) can overflow before scaling.  No risky conversion is tried.
        if total.bit_length() > 1023:
            return self._fallback(receiver, candidate, 'INTEGER_CONVERSION_RANGE')
        if total.bit_length() - 1 - exponent < sys.float_info.min_exp - 1:
            return self._fallback(receiver, candidate, 'EXACT_SUM_BELOW_NORMAL_RANGE')
        rounded_integer = float(total)
        try:
            result = math.ldexp(rounded_integer, -exponent)
        except OverflowError:
            return self._fallback(receiver, candidate, 'SCALED_OVERFLOW')
        if not math.isfinite(result) or result < sys.float_info.min:
            # Normal-to-subnormal scaling can double-round: original path wins.
            return self._fallback(receiver, candidate, 'NONNORMAL_SCALED_RESULT')
        self._stats['I_dyadic_rounds'] += 1
        self._stats['I_reused_old_sum_terms'] += len(self.parts[receiver])
        return result

    def incoming(self, selected_indices, candidate_indices):
        started, cpu_started = time.perf_counter(), time.process_time()
        selected = np.asarray(selected_indices, dtype=np.int64)
        candidates = np.asarray(candidate_indices, dtype=np.int64)
        incoming = np.empty((len(candidates), len(selected) + 1), dtype=float)
        try:
            self._prepare(selected)
            for row, candidate in enumerate(candidates):
                if self.check is not None and row % 16 == 0:
                    self.check()
                old_started, old_cpu = time.perf_counter(), time.process_time()
                for j, receiver in enumerate(selected):
                    incoming[row, j] = self._old(int(receiver), int(candidate)) + self.external
                self._stats['I_old_sum_wall_s'] += time.perf_counter() - old_started
                self._stats['I_old_sum_cpu_s'] += time.process_time() - old_cpu
                new_started, new_cpu = time.perf_counter(), time.process_time()
                incoming[row, len(selected)] = (
                    math.fsum(self.power[selected, candidate]) + self.external)
                self._stats['I_original_new_fsum_calls'] += 1
                self._stats['I_new_sum_wall_s'] += time.perf_counter() - new_started
                self._stats['I_new_sum_cpu_s'] += time.process_time() - new_cpu
            return incoming
        finally:
            self._stats['I_total_wall_s'] += time.perf_counter() - started
            self._stats['I_total_cpu_s'] += time.process_time() - cpu_started

    def record_oracle_bits(self, expected, actual):
        """Diagnostic only; never invoked to supply a fast-path decision."""
        expected = np.ascontiguousarray(expected, dtype=float)
        actual = np.ascontiguousarray(actual, dtype=float)
        unequal = expected.view(np.uint64) != actual.view(np.uint64)
        self._stats['I_oracle_bit_comparisons'] += expected.size
        self._stats['I_oracle_bit_differences'] += int(unequal.sum())
        if np.any(unequal) and len(self._stats['I_oracle_bit_diff_examples']) < 8:
            for index in np.argwhere(unequal):
                where = tuple(int(v) for v in index)
                self._stats['I_oracle_bit_diff_examples'].append({
                    'index': where, 'expected': float(expected[where]).hex(),
                    'actual': float(actual[where]).hex()})
                if len(self._stats['I_oracle_bit_diff_examples']) >= 8:
                    break
        return not bool(np.any(unequal))


class ReuseScorer(FastM1Scorer):
    """Original sparse batched scoring; only old-receiver reductions change."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.oracle_callback = None
        self._last_selection_certificate = None
        self._interference_reuse = DyadicInterferenceCache(
            self._power, self.channel.config.get('I_ext', 0.0),
            check=lambda: _check(self.meter, reserve=self.reserve))
        self._stats.update(I_capacity_wall_s=0.0, I_capacity_cpu_s=0.0,
                           I_service_wall_s=0.0, I_service_cpu_s=0.0)

    def stats(self):
        return {**super().stats(), **self._interference_reuse.stats()}

    def score_additions_sparse(self, selected_indices, candidate_indices):
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
            indices[:, :L], indices[:, L] = selected, candidates
            incoming = self._interference_reuse.incoming(selected, candidates)
            signal = self.channel.signal[self.h, self.edges[indices, 0], self.edges[indices, 1]]
            cap_started, cap_cpu = time.perf_counter(), time.process_time()
            caps = self.channel.cap(signal, incoming)
            self._stats['I_capacity_wall_s'] += time.perf_counter() - cap_started
            self._stats['I_capacity_cpu_s'] += time.process_time() - cap_cpu
            if caps.shape != incoming.shape or caps.dtype.kind not in 'iu' or np.any(caps < 0):
                raise AssertionError('batch capacity must be nonnegative integer counts')
            service_started, service_cpu = time.perf_counter(), time.process_time()
            served_cells = self.edges[indices, 1]
            rt = np.minimum(caps, self.total_rt[served_cells])
            nrt = np.minimum(self.nrt[served_cells], np.maximum(caps - rt, 0))
            weighted = np.empty(rt.shape, dtype=np.int64 if self._safe_weighted else object)
            for j, m in enumerate(existing[:, 1]):
                weighted[:, j] = self._weighted_for_cell(int(m), rt[:, j])
            for row, m in enumerate(served_cells[:, L]):
                weighted[row, L] = self._weighted_for_cell(int(m), rt[row, L:L + 1])[0]
            self._stats['I_service_wall_s'] += time.perf_counter() - service_started
            self._stats['I_service_cpu_s'] += time.process_time() - service_cpu
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

    def select_addition(self, selected, candidates, current_sparse):
        scores = self.score_additions_sparse(selected, candidates)
        ranked = [(-(score.numerator - current_sparse.numerator),
                   int(self.channel.sat_ids[self.edges[index, 0]]),
                   int(self.edges[index, 1]), index, score)
                  for index, score in zip(candidates, scores)]
        best = min(ranked, key=lambda row: row[:3]) if ranked else None
        winner = None if best is None or -best[0] <= 0 else (best[3], best[4])
        certificate = {'mode': 'I', 'bounds': [], 'completed_full_rows': len(scores),
                       'best_positive_numerator': 0 if winner is None else -best[0],
                       'pruned': 0, 'ordered_U': list(selected)}
        self._last_selection_certificate = certificate
        callback = getattr(self, 'oracle_callback', None)
        if callback is not None:
            self._last_full_rows_for_oracle = dict(zip(candidates, scores))
            callback(self, selected, candidates, current_sparse, winner, certificate)
        return winner

    def last_selection_certificate(self):
        return self._last_selection_certificate
