"""R2 physical channel and frozen, conservative integer-capacity arithmetic.

Provenance: ``geometry``, ``airy_gain`` and the directional power kernel are
extracted from sparse_matching_bhtp_three_fixes/scenario/model.py (geometry and
physical primitives only). Their operation order, LOS mask and analytical
boresight identities are retained. No old planner is imported. The coordination
graph is NEW: R2 section 4 requires a distinct graph for every slot. Capacity is
NEW: exact rational power-of-two identities and directed Decimal intervals at
80 and 120 digits replace the old one-precision boundary fallback.
"""
from __future__ import annotations

from decimal import Decimal, localcontext, ROUND_FLOOR, ROUND_CEILING
from fractions import Fraction
from collections import OrderedDict
import math

import numpy as np
from scipy.special import j1


class NumericUnresolved(RuntimeError):
    """Uncertified arithmetic, with the full finite input retained for audit."""

    def __init__(self, reason, inputs=None):
        self.reason = str(reason)
        self.inputs = inputs or {}
        super().__init__(f'NUMERIC_UNRESOLVED: {self.reason}')


def decimal_fraction(value):
    """Configuration numbers mean their declared decimal representation."""
    return Fraction(str(value))


def _dec_fraction(value, precision, rounding):
    with localcontext() as ctx:
        ctx.prec = precision
        ctx.rounding = rounding
        return Decimal(value.numerator) / Decimal(value.denominator)


def _ln_bounds(value, precision):
    # Decimal.ln is correctly rounded with ROUND_HALF_EVEN. One ulp, rather
    # than half an ulp, is a conservative absolute bound for this primitive.
    with localcontext() as ctx:
        ctx.prec = precision
        centre = value.ln()
        ulp = Decimal(1).scaleb(centre.adjusted() - precision + 1)
        ctx.rounding = ROUND_FLOOR
        lo = centre - ulp
        ctx.rounding = ROUND_CEILING
        hi = centre + ulp
    return lo, hi


def _capacity_interval(ratio, scale, precision):
    """Outward interval for scale * ln(ratio) / ln(2), all inputs rational."""
    rlo = _dec_fraction(ratio, precision, ROUND_FLOOR)
    rhi = _dec_fraction(ratio, precision, ROUND_CEILING)
    slo = _dec_fraction(scale, precision, ROUND_FLOOR)
    shi = _dec_fraction(scale, precision, ROUND_CEILING)
    llo, _ = _ln_bounds(rlo, precision)
    # ratio >= 1 is an exact input fact, so log cannot be negative even when
    # the outward interval around ln(1) crosses zero.
    llo = max(Decimal(0), llo)
    _, lhi = _ln_bounds(rhi, precision)
    two_lo, two_hi = _ln_bounds(Decimal(2), precision)
    with localcontext() as ctx:
        ctx.prec = precision
        ctx.rounding = ROUND_FLOOR
        lower = slo * llo / two_hi
        ctx.rounding = ROUND_CEILING
        upper = shi * lhi / two_lo
    return lower, upper


def capacity(signal, interference, noise, config):
    """Capacity with no floor epsilon and explicit unresolved-boundary status.

    Physical power inputs are the exact rational values of the frozen binary64
    channel outputs; B, effective slot duration and packet bits are the declared
    decimal constants. Away from an integer the binary64 route uses a fixed
    conservative 128-ulp arithmetic guard. Inside the guard, exact power-of-two
    identities are resolved rationally; otherwise *both* independent precision
    intervals must enclose the same single integer bin. An unproven result is
    never rounded in a method's favour.
    """
    signal, interference = np.broadcast_arrays(np.asarray(signal, dtype=float),
                                              np.asarray(interference, dtype=float))
    noise = float(noise)
    diagnostic = {'signal': signal.tolist(), 'interference': interference.tolist(),
                  'noise': noise, 'interference_includes_I_ext': True,
                  'bandwidth_hz': config['bandwidth_hz'],
                  'dt_data': config.get('dt_data', config['dt']),
                  'packet_bits': config['packet_bits']}
    if (not np.all(np.isfinite(signal)) or not np.all(np.isfinite(interference))
            or not math.isfinite(noise)
            or np.any(signal < 0) or np.any(interference < 0)
            or noise <= 0):
        raise NumericUnresolved('invalid signal, noise or interference', diagnostic)
    scale = (decimal_fraction(config['bandwidth_hz'])
             * decimal_fraction(config.get('dt_data', config['dt']))
             / int(config['packet_bits']))
    if scale <= 0:
        raise NumericUnresolved('nonpositive effective capacity scale', diagnostic)
    value = np.asarray(float(scale) * np.log1p(signal / (noise + interference)) / np.log(2.0))
    if not np.all(np.isfinite(value)) or np.any(value >= np.iinfo(np.int64).max):
        raise NumericUnresolved('nonfinite or overflowing capacity', diagnostic)
    result = np.asarray(np.floor(value), dtype=np.int64)
    guard = 128 * np.finfo(float).eps * np.maximum(1.0, np.abs(value))
    near = (signal > 0) & (np.abs(value - np.rint(value)) <= guard)
    for index in np.flatnonzero(np.ravel(near)):
        s = Fraction.from_float(float(signal.flat[index]))
        denominator = (Fraction.from_float(noise)
                       + Fraction.from_float(float(interference.flat[index])))
        ratio = 1 + s / denominator
        numerator, divisor = ratio.numerator, ratio.denominator
        if (numerator > 0 and numerator & (numerator - 1) == 0
                and divisor > 0 and divisor & (divisor - 1) == 0):
            exact = scale * (numerator.bit_length() - divisor.bit_length())
            result.flat[index] = exact.numerator // exact.denominator
            continue
        intervals = [_capacity_interval(ratio, scale, precision) for precision in (80, 120)]
        bins = [(int(lo.to_integral_value(rounding=ROUND_FLOOR)),
                 int(hi.to_integral_value(rounding=ROUND_FLOOR))) for lo, hi in intervals]
        if bins[0][0] == bins[0][1] == bins[1][0] == bins[1][1]:
            result.flat[index] = bins[0][0]
        else:
            raise NumericUnresolved('capacity floor lacks a certified integer bin',
                                    {**diagnostic, 'flat_index': int(index),
                                     'intervals': [[str(lo), str(hi)] for lo, hi in intervals]})
    return result


def geometry(sat_positions, cells, config, constants):
    """Unchanged spherical geometry, separating LOS and service eligibility."""
    delta = cells[None, None, :, :] - sat_positions[:, :, None, :]
    distance = np.linalg.norm(delta, axis=-1)
    down = delta / distance[..., None]
    cell_up = cells / np.linalg.norm(cells, axis=1)[:, None]
    elev_sine = -np.einsum('hsmc,mc->hsm', down, cell_up)
    sat_nadir = -sat_positions / np.linalg.norm(sat_positions, axis=-1)[..., None]
    scan_cosine = np.einsum('hsmc,hsc->hsm', down, sat_nadir)
    los = elev_sine >= 0.0
    visible = (los & (elev_sine >= np.sin(np.deg2rad(config['min_elevation_deg'])))
               & (scan_cosine >= np.cos(np.deg2rad(config['max_scan_deg']))))
    return down, distance, los, visible


def airy_gain(cos_angle, diameter_m, wavelength_m, peak_gain):
    """Frozen original forward-hemisphere Airy power pattern."""
    cos_angle = np.clip(np.asarray(cos_angle, dtype=float), -1.0, 1.0)
    x = np.pi * diameter_m / wavelength_m * np.sqrt(np.maximum(0.0, 1.0-cos_angle*cos_angle))
    amplitude = np.ones_like(x)
    np.divide(2*j1(x), x, out=amplitude, where=x != 0)
    return peak_gain * amplitude**2 * (cos_angle >= 0)


class Channel:
    """One method's immutable horizon geometry; graph caches are method-local."""

    def __init__(self, sat_positions, cells, sat_ids, config, constants):
        self.sat_positions = np.array(sat_positions, dtype=float, copy=True)
        self.cells = np.array(cells, dtype=float, copy=True)
        self.sat_ids = np.array(sat_ids, dtype=np.int64, copy=True)
        self.config, self.constants = dict(config), dict(constants)
        self.H, self.S = self.sat_positions.shape[:2]
        self.M = len(self.cells)
        if len(self.sat_ids) != self.S or len(set(self.sat_ids.tolist())) != self.S:
            raise ValueError('satellite global IDs must be unique and match geometry')
        self.down, self.distance, self.los, self.visible = geometry(
            self.sat_positions, self.cells, self.config, self.constants)
        self.wavelength = constants['speed_of_light_m_s'] / config['frequency_hz']
        self.tx_peak = config['antenna_efficiency'] * (np.pi*config['tx_diameter_m']/self.wavelength)**2
        self.rx_peak = 10.0**(config['rx_gain_dbi']/10.0)
        self.rx_diameter = self.wavelength/np.pi * np.sqrt(self.rx_peak/config['antenna_efficiency'])
        self.noise = constants['boltzmann_j_k']*config['noise_temperature_k']*config['bandwidth_hz']
        self.path_power = config['power_w'] * (self.wavelength/(4*np.pi*self.distance))**2 * self.los
        self.signal = self.path_power * self.tx_peak * self.rx_peak
        if config['K'] * config['power_w'] > config['satellite_power_limit_w']:
            raise ValueError('fixed per-beam power violates satellite total power limit')
        self._graphs = None
        # A channel may be constructed before the complete planning timer. The
        # optional cache therefore remains cold until the planner explicitly
        # begins a measured call, and is cleared before that timer is finished.
        self._engineering_power_cache = None
        self._engineering_power_cache_stats = None
        for array in (self.sat_positions, self.cells, self.sat_ids, self.down,
                      self.distance, self.los, self.visible, self.path_power, self.signal):
            array.flags.writeable = False

    def begin_engineering_cache(self):
        """Open a cold, bounded power cache for one measured planning call.

        The R2 path is the default. ``r3_engineering`` only permits activation;
        the planner owns activation, cleanup, and their budget accounting.
        Bounds cover retained edge-key bytes and matrix payload, with a second
        entry-count bound for Python container overhead. They are not RSS caps.
        """
        if self._engineering_power_cache is not None:
            raise RuntimeError('power cache already belongs to an active planning call')
        enabled = bool(self.config.get('r3_engineering', False))
        self._engineering_power_cache_stats = {
            'enabled': enabled,
            'max_payload_bytes': 8 * 1024 * 1024,
            'max_entries': 512,
            'requests': 0,
            'hits': 0,
            'misses': 0,
            'builds': 0,
            'evictions': 0,
            'oversize_bypasses': 0,
            'retained_payload_bytes': 0,
            'retained_entries': 0,
            'peak_payload_bytes': 0,
            'peak_entries': 0,
            'payload_scope': 'ordered int64 edge key bytes plus cached float64 matrix bytes',
            'python_container_bytes': None,
            'rss_cap_bytes': None,
            'lifetime': 'one explicit planning call; no global or cross-call cache',
        }
        self._engineering_power_cache = OrderedDict() if enabled else None
        return dict(self._engineering_power_cache_stats)

    def end_engineering_cache(self):
        """Release all cached arrays inside the caller's measured lifetime."""
        if self._engineering_power_cache is not None:
            self._engineering_power_cache.clear()
            self._engineering_power_cache = None
        stats = self._engineering_power_cache_stats
        if stats is None:
            return {'enabled': False, 'lifetime': 'cache was not activated'}
        stats['retained_payload_bytes'] = 0
        stats['retained_entries'] = 0
        return dict(stats)

    def _remember_power(self, key, result):
        cache = self._engineering_power_cache
        stats = self._engineering_power_cache_stats
        # h is a fixed-size slot index; bytes below describe retained numeric
        # payload only. Python key/container overhead is bounded by max_entries
        # and explicitly not represented as measured RSS.
        size = len(key[1]) + len(key[2]) + int(result.nbytes)
        if size > stats['max_payload_bytes']:
            stats['oversize_bypasses'] += 1
            return
        while cache and (len(cache) >= stats['max_entries']
                         or stats['retained_payload_bytes'] + size > stats['max_payload_bytes']):
            old_key, old_value = cache.popitem(last=False)
            stats['retained_payload_bytes'] -= len(old_key[1]) + len(old_key[2]) + int(old_value.nbytes)
            stats['evictions'] += 1
        # engine.slot and background scoring zero diagonals in-place. Never
        # expose the stored value: both misses and hits return owned matrices.
        saved = result.copy()
        saved.flags.writeable = False
        cache[key] = saved
        stats['builds'] += 1
        stats['retained_payload_bytes'] += size
        stats['retained_entries'] = len(cache)
        stats['peak_payload_bytes'] = max(stats['peak_payload_bytes'], stats['retained_payload_bytes'])
        stats['peak_entries'] = max(stats['peak_entries'], len(cache))

    def cap(self, signal, interference):
        return capacity(signal, interference, self.noise, self.config)

    def power(self, h, tx_edges, rx_edges):
        """Directional tx-by-rx watts, including self signal; no graph truncation."""
        tx = np.asarray(tx_edges, dtype=np.int64).reshape(-1, 2)
        rx = np.asarray(rx_edges, dtype=np.int64).reshape(-1, 2)
        if not len(tx) or not len(rx):
            return np.zeros((len(tx), len(rx)), dtype=float)
        cache = self._engineering_power_cache
        key = None
        if cache is not None:
            # Geometry belongs to this immutable Channel. Full ordered bytes,
            # rather than a set/hash, retain duplicates, shape and operation
            # order; swapped/permuted tx/rx requests remain distinct entries.
            key = (h, tx.tobytes(order='C'), rx.tobytes(order='C'))
            stats = self._engineering_power_cache_stats
            stats['requests'] += 1
            saved = cache.get(key)
            if saved is not None:
                cache.move_to_end(key)
                stats['hits'] += 1
                return saved.copy()
            stats['misses'] += 1
        direction = self.down[h, tx[:, 0, None], rx[None, :, 1], :]
        tx_bore = self.down[h, tx[:, 0], tx[:, 1], :]
        rx_bore = self.down[h, rx[:, 0], rx[:, 1], :]
        tx_cos = np.einsum('nvc,nc->nv', direction, tx_bore)
        rx_cos = np.einsum('nvc,vc->nv', direction, rx_bore)
        tx_cos = np.where(tx[:, 1, None] == rx[None, :, 1], 1.0, tx_cos)
        rx_cos = np.where(tx[:, 0, None] == rx[None, :, 0], 1.0, rx_cos)
        tx_gain = airy_gain(tx_cos, self.config['tx_diameter_m'], self.wavelength, self.tx_peak)
        rx_gain = airy_gain(rx_cos, self.rx_diameter, self.wavelength, self.rx_peak)
        result = self.path_power[h, tx[:, 0, None], rx[None, :, 1]] * tx_gain * rx_gain
        own = (tx[:, 0, None] == rx[None, :, 0]) & (tx[:, 1, None] == rx[None, :, 1])
        result = np.where(own, self.signal[h, rx[:, 0], rx[:, 1]][None, :], result)
        if cache is not None:
            self._remember_power(key, result)
        return result

    @property
    def graphs(self):
        if self._graphs is None:
            self._graphs = self._build_graphs()
            self._graphs.flags.writeable = False
        return self._graphs

    def _build_graphs(self, block_size=96):
        """Exact R2 per-slot shared-or-potential-interference coordination graph."""
        graphs = np.zeros((self.H, self.S, self.S), dtype=bool)
        threshold = self.config['zeta_I'] * self.noise
        for h in range(self.H):
            mask = self.visible[h].astype(np.int64)
            graph = mask @ mask.T > 0
            np.fill_diagonal(graph, False)
            for s in range(self.S):
                for t in range(s+1, self.S):
                    if graph[s, t]:
                        continue
                    sm, tm = np.flatnonzero(mask[s]), np.flatnonzero(mask[t])
                    found = False
                    for source, target, am, bm in ((s, t, sm, tm), (t, s, tm, sm)):
                        for a in range(0, len(am), block_size):
                            ac = am[a:a+block_size]
                            tx = np.column_stack((np.full(len(ac), source), ac))
                            for b in range(0, len(bm), block_size):
                                bc = bm[b:b+block_size]
                                rx = np.column_stack((np.full(len(bc), target), bc))
                                if np.any((self.power(h, tx, rx) >= threshold)
                                          & (tx[:, 1, None] != rx[None, :, 1])):
                                    found = True
                                    break
                            if found:
                                break
                        if found:
                            break
                    graph[s, t] = graph[t, s] = found
            graphs[h] = graph
        return graphs
