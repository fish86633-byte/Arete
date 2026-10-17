"""Independent R2 scalar channel and FIFO audit.

This module never calls a planner, Channel.power, or the production queue/Replay
engine.  It accepts their immutable inputs and may compare an expected trace.
Only the declared packet-capacity arithmetic policy is shared with physics.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
import math
from typing import Any

import numpy as np
from scipy.special import j1


class VerificationFailure(RuntimeError):
    """A structural, physical integer, queue, or saved-objective mismatch."""

    def __init__(self, invariant: str, details: dict[str, Any]):
        self.invariant = invariant
        self.details = details
        super().__init__(f"{invariant}: {details}")


@dataclass
class VerifiedState:
    rt: np.ndarray
    nrt_batches: list[list[tuple[int, int]]]

    @property
    def nrt(self) -> np.ndarray:
        return np.asarray([sum(c for _, c in row) for row in self.nrt_batches], dtype=np.int64)


@dataclass
class VerificationTrace:
    initial_state: VerifiedState
    final_state: VerifiedState
    capacities: np.ndarray
    active: np.ndarray
    served_age: np.ndarray
    served_nrt: np.ndarray
    expired: np.ndarray
    signals: np.ndarray
    interference: np.ndarray
    sinr: np.ndarray
    bits: int
    weighted_wait: int
    F: Fraction
    waiting: Fraction
    start_slot: int
    rt_success_batches: list[tuple[int, int, int, int]] = field(default_factory=list)
    rt_expired_batches: list[tuple[int, int, int, int]] = field(default_factory=list)
    nrt_success_batches: list[tuple[int, int, int, int]] = field(default_factory=list)
    compared: list[str] = field(default_factory=list)
    status: str = "PASS"
    physical_input_mode: str = "scalar_geometry_airy"

    @property
    def delivered_packets(self) -> int:
        return int(self.served_age.sum()) + int(self.served_nrt.sum())

    @property
    def end_state(self) -> VerifiedState:
        return self.final_state

    @property
    def W(self) -> Fraction:
        return self.waiting

    @property
    def F_float(self) -> float:
        return float(self.F)

    def summary(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "bits": self.bits,
            "weighted_wait": self.weighted_wait,
            "F_numerator": self.F.numerator,
            "F_denominator": self.F.denominator,
            "F": float(self.F),
            "W_RT": float(self.waiting),
            "delivered_packets": self.delivered_packets,
            "expired_packets": int(self.expired.sum()),
            "final_rt_packets": int(self.final_state.rt.sum()),
            "final_nrt_packets": int(self.final_state.nrt.sum()),
            "verified_slots": int(self.active.shape[0]),
            "start_slot": self.start_slot,
            "compared": list(self.compared),
            "physical_input_mode": self.physical_input_mode,
            "independence": "independent full-power summation and FIFO queue; no planner or production replay",
        }


def _get(config: dict, *names: str) -> Any:
    for name in names:
        if name in config:
            return config[name]
    raise KeyError(names[0])


def _fraction(value: Any) -> Fraction:
    return value if isinstance(value, Fraction) else Fraction(str(value))


def _integer_array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    arr = np.asarray(value)
    if arr.shape != shape or arr.dtype.kind not in "iu" or np.any(arr < 0):
        raise VerificationFailure(name, {"shape": list(arr.shape), "required_shape": list(shape), "dtype": str(arr.dtype)})
    return arr.astype(np.int64, copy=True)


def _snapshot(state: Any, cells: int, ttl: int, start_slot: int) -> VerifiedState:
    if getattr(state, "_arrival_slot", None) is not None:
        raise VerificationFailure("Q0 must precede first arrivals", {"already_added_slot": int(state._arrival_slot)})
    rt = _integer_array(state.rt, (cells, ttl), "initial RT state")
    source = state.nrt_batches
    if len(source) != cells:
        raise VerificationFailure("NRT cell count", {"actual": len(source), "required": cells})
    batches: list[list[tuple[int, int]]] = []
    for m, row in enumerate(source):
        out: list[tuple[int, int]] = []
        previous_birth = None
        for raw_birth, raw_count in row:
            birth, count = int(raw_birth), int(raw_count)
            if raw_birth != birth or raw_count != count or count <= 0 or birth >= start_slot:
                raise VerificationFailure("NRT entrance batch", {"cell": m, "birth": birth, "count": count, "start_slot": start_slot})
            if previous_birth is not None and birth < previous_birth:
                raise VerificationFailure("NRT FIFO order", {"cell": m, "birth": birth, "previous_birth": previous_birth})
            previous_birth = birth
            out.append((birth, count))
        batches.append(out)
    return VerifiedState(rt, batches)


def _unit(vector: Any) -> tuple[tuple[float, float, float], float]:
    v = tuple(float(x) for x in vector)
    norm = math.sqrt(math.fsum(x*x for x in v))
    if not math.isfinite(norm) or norm <= 0:
        raise VerificationFailure("physical vector norm", {"norm": norm})
    return tuple(x/norm for x in v), norm


def _dot(a: Any, b: Any) -> float:
    return math.fsum(float(x)*float(y) for x, y in zip(a, b, strict=True))


def _airy(cos_angle: float, diameter: float, wavelength: float, peak: float) -> float:
    cosine = min(1.0, max(-1.0, float(cos_angle)))
    if cosine < 0:
        return 0.0
    x = math.pi*diameter/wavelength*math.sqrt(max(0.0, 1.0-cosine*cosine))
    amplitude = 1.0 if x == 0 else 2.0*float(j1(x))/x
    return peak*amplitude*amplitude


class ScalarChannel:
    """Independently reconstruct the frozen spherical-LOS/Airy input contract."""

    def __init__(self, channel: Any):
        self.positions = np.asarray(channel.sat_positions, dtype=float)
        self.cells = np.asarray(channel.cells, dtype=float)
        self.sat_ids = np.asarray(channel.sat_ids, dtype=np.int64)
        self.config, self.constants = channel.config, channel.constants
        self.H, self.S = self.positions.shape[:2]
        self.M = len(self.cells)
        c, k = self.config, self.constants
        self.wavelength = float(k["speed_of_light_m_s"])/float(c["frequency_hz"])
        self.tx_diameter = float(c["tx_diameter_m"])
        self.tx_peak = float(c["antenna_efficiency"])*(math.pi*self.tx_diameter/self.wavelength)**2
        self.rx_peak = 10.0**(float(c["rx_gain_dbi"])/10.0)
        self.rx_diameter = self.wavelength/math.pi*math.sqrt(self.rx_peak/float(c["antenna_efficiency"]))
        self.noise = float(k["boltzmann_j_k"])*float(c["noise_temperature_k"])*float(c["bandwidth_hz"])
        self.power_w = float(_get(c, "power_w", "beam_power_w"))
        self.minimum_elev = math.sin(math.radians(float(c["min_elevation_deg"])))
        self.minimum_scan = math.cos(math.radians(float(c["max_scan_deg"])))
        self._directions: dict[tuple[int, int, int], tuple[tuple[float, float, float], float, bool, bool]] = {}
        self._up = [_unit(point)[0] for point in self.cells]

    def geometry(self, h: int, s: int, m: int):
        key = h, s, m
        if key not in self._directions:
            down, distance = _unit(self.cells[m]-self.positions[h, s])
            nadir, _ = _unit(-self.positions[h, s])
            elev = -_dot(down, self._up[m])
            scan = _dot(down, nadir)
            los = elev >= 0.0
            visible = los and elev >= self.minimum_elev and scan >= self.minimum_scan
            self._directions[key] = down, distance, los, visible
        return self._directions[key]

    def received_power(self, h: int, tx: tuple[int, int], rx: tuple[int, int]) -> float:
        s, n = tx
        j, m = rx
        direction, distance, los, _ = self.geometry(h, s, m)
        if not los:
            return 0.0
        base = self.power_w*(self.wavelength/(4.0*math.pi*distance))**2
        if s == j and n == m:
            return base*self.tx_peak*self.rx_peak
        tx_bore = self.geometry(h, s, n)[0]
        rx_bore = self.geometry(h, j, m)[0]
        tx_cos = 1.0 if n == m else _dot(direction, tx_bore)
        rx_cos = 1.0 if s == j else _dot(direction, rx_bore)
        return base*_airy(tx_cos, self.tx_diameter, self.wavelength, self.tx_peak)*_airy(rx_cos, self.rx_diameter, self.wavelength, self.rx_peak)


class TableAuditChannel:
    """Frozen tiny-instance input; table entries are already received watts.

    This mode audits the physical equations and queues given a declared power
    table. It makes no claim to independently validate geometric Airy inputs.
    """

    def __init__(self, channel: Any):
        self.H, self.S, self.M = channel.H, channel.S, channel.M
        self.sat_ids = np.asarray(channel.sat_ids, dtype=np.int64)
        self.visible = np.asarray(channel.visible, dtype=bool)
        self.table = np.asarray(channel.power_table, dtype=float)
        self.noise = float(channel.noise)
        if self.table.shape != (self.H, self.S, self.M, self.S, self.M):
            raise VerificationFailure("frozen power table shape", {"shape": list(self.table.shape)})
        if np.any(~np.isfinite(self.table)) or np.any(self.table < 0) or self.noise <= 0:
            raise VerificationFailure("frozen power table values", {"noise": self.noise})

    def geometry(self, h: int, s: int, m: int):
        return None, None, None, bool(self.visible[h, s, m])

    def received_power(self, h: int, tx: tuple[int, int], rx: tuple[int, int]) -> float:
        return float(self.table[h, tx[0], tx[1], rx[0], rx[1]])


def _same_integer(actual: np.ndarray, expected: Any, invariant: str) -> None:
    other = np.asarray(expected)
    if actual.shape != other.shape:
        raise VerificationFailure(invariant, {"actual_shape": list(actual.shape), "expected_shape": list(other.shape)})
    mismatch = np.argwhere(actual != other)
    if len(mismatch):
        at = tuple(int(x) for x in mismatch[0])
        raise VerificationFailure(invariant, {"index": list(at), "actual": int(actual[at]), "expected": int(other[at]), "mismatches": len(mismatch)})


def compare_states(actual: Any, expected: Any) -> None:
    """Strict RT-age and original-birth FIFO continuity across cycle boundaries."""
    _same_integer(np.asarray(actual.rt), expected.rt, "cross-cycle RT state")
    left = [[(int(b), int(n)) for b, n in row] for row in actual.nrt_batches]
    right = [[(int(b), int(n)) for b, n in row] for row in expected.nrt_batches]
    if left != right:
        raise VerificationFailure("cross-cycle NRT FIFO batches", {"different_cells": [m for m, (a, b) in enumerate(zip(left, right, strict=True)) if a != b]})


def verify_plan(plan: Any, state: Any, arrivals: Any, channel: Any, *, start_slot: int = 0,
                expected: Any | None = None, previous_end: Any | None = None) -> VerificationTrace:
    """Recompute one complete plan; all queue/count comparisons are exact.

    `start_slot` is the absolute index of the first arrival, whose packets are
    absent from `state`.  Optional `previous_end` enforces physical continuity.
    Returned batch rows are (cell, arrival_slot, terminal_slot, packet_count).
    """
    from .physics import capacity

    config = channel.config
    physical = TableAuditChannel(channel) if hasattr(channel, "power_table") else ScalarChannel(channel)
    H, S, M = physical.H, physical.S, physical.M
    ttl = int(_get(config, "ttl", "ttl_slots", "D_RT", "rt_ttl_slots"))
    K = int(_get(config, "K", "K_s", "beams_per_satellite"))
    xraw = np.asarray(plan)
    if xraw.shape != (H, S, M) or not np.all((xraw == 0) | (xraw == 1)):
        raise VerificationFailure("binary plan shape", {"shape": list(xraw.shape), "required_shape": [H, S, M]})
    x = xraw.astype(bool, copy=False)
    if np.any(x.sum(axis=2) > K) or np.any(x.sum(axis=1) > 1):
        raise VerificationFailure("beam capacity or cell uniqueness", {"max_beams": int(x.sum(axis=2).max(initial=0)), "max_cell_occupancy": int(x.sum(axis=1).max(initial=0))})
    A = _integer_array(arrivals, (H, M, 2), "arrivals")
    initial = _snapshot(state, M, ttl, start_slot)
    if previous_end is not None:
        compare_states(initial, previous_end)
    if expected is not None and hasattr(expected, "states") and len(expected.states) != H+1:
        raise VerificationFailure("H+1 saved queue states", {"actual": len(expected.states), "required": H+1})
    if expected is not None and hasattr(expected, "slots") and len(expected.slots) != H:
        raise VerificationFailure("H saved slots", {"actual": len(expected.slots), "required": H})
    rt = initial.rt.copy()
    batches = [list(row) for row in initial.nrt_batches]
    caps = np.zeros((H, S, M), dtype=np.int64)
    active = np.zeros_like(x)
    served_age = np.zeros((H, M, ttl), dtype=np.int64)
    served_nrt = np.zeros((H, M), dtype=np.int64)
    expired = np.zeros((H, M), dtype=np.int64)
    signals = np.zeros(x.shape, dtype=float)
    interference = np.zeros(x.shape, dtype=float)
    sinr = np.zeros(x.shape, dtype=float)
    weighted_wait = 0
    rt_success: list[tuple[int, int, int, int]] = []
    rt_expiry: list[tuple[int, int, int, int]] = []
    nrt_success: list[tuple[int, int, int, int]] = []
    external = np.asarray(config.get("I_ext", 0.0), dtype=float)
    if np.any(external < 0) or np.any(~np.isfinite(external)):
        raise VerificationFailure("external interference", {"shape": list(external.shape)})
    for h in range(H):
        u = start_slot+h
        if expected is not None and hasattr(expected, "states"):
            compare_states(VerifiedState(rt, batches), expected.states[h])
        rt[:, 0] += A[h, :, 0]
        for m in range(M):
            if A[h, m, 1]:
                batches[m].append((u, int(A[h, m, 1])))
        before_rt = rt.sum(axis=1)
        before_nrt = np.asarray([sum(n for _, n in row) for row in batches], dtype=np.int64)
        if expected is not None and hasattr(expected, "slots"):
            saved = expected.slots[h]
            _same_integer(rt, saved["qplus_rt"], "saved service-before RT ages")
            _same_integer(before_nrt, saved["qplus_nrt"], "saved service-before NRT totals")
            if int(saved["absolute_slot"]) != u:
                raise VerificationFailure("saved absolute slot", {"actual": u, "expected": int(saved["absolute_slot"])})
        edges = []
        for s_raw, m_raw in np.argwhere(x[h]):
            s, m = int(s_raw), int(m_raw)
            if not physical.geometry(h, s, m)[3]:
                raise VerificationFailure("service visibility", {"slot": u, "satellite": int(physical.sat_ids[s]), "cell": m})
            if int(before_rt[m])+int(before_nrt[m]) > 0:
                edges.append((s, m))
                active[h, s, m] = True
        for edge in edges:
            s, m = edge
            sig = physical.received_power(h, edge, edge)
            ext = float(external) if external.ndim == 0 else float(np.broadcast_to(external, (H, S, M))[h, s, m])
            interf = math.fsum([ext]+[physical.received_power(h, tx, edge) for tx in edges if tx != edge])
            cap = int(np.asarray(capacity(sig, interf, physical.noise, config)).item())
            caps[h, s, m] = cap
            signals[h, s, m], interference[h, s, m] = sig, interf
            sinr[h, s, m] = sig/(physical.noise+interf)
            left = cap
            for age in range(ttl-1, -1, -1):
                taken = min(int(rt[m, age]), left)
                if taken:
                    rt[m, age] -= taken
                    served_age[h, m, age] = taken
                    rt_success.append((m, u-age, u, taken))
                    left -= taken
            while left and batches[m]:
                birth, available = batches[m][0]
                taken = min(available, left)
                served_nrt[h, m] += taken
                nrt_success.append((m, birth, u, taken))
                left -= taken
                if available == taken:
                    batches[m].pop(0)
                else:
                    batches[m][0] = birth, available-taken
        weighted_wait += sum((age+1)*int(rt[:, age].sum()) for age in range(ttl))
        expired[h] = rt[:, -1]
        if expected is not None and hasattr(expected, "slots"):
            _same_integer(rt, expected.slots[h]["residual"], "saved post-service pre-expiry RT")
        for m in range(M):
            if expired[h, m]:
                rt_expiry.append((m, u-(ttl-1), u, int(expired[h, m])))
        next_rt = np.zeros_like(rt)
        next_rt[:, 1:] = rt[:, :-1]
        next_nrt = np.asarray([sum(n for _, n in row) for row in batches], dtype=np.int64)
        _same_integer(before_rt, served_age[h].sum(axis=1)+expired[h]+next_rt.sum(axis=1), "per-cell RT conservation")
        _same_integer(before_nrt, served_nrt[h]+next_nrt, "per-cell NRT conservation")
        rt = next_rt
    final = VerifiedState(rt.copy(), [list(row) for row in batches])
    dt = _fraction(_get(config, "dt", "dt_s", "delta_t_s", "slot_duration_s"))
    bits = (int(served_age.sum())+int(served_nrt.sum()))*int(_get(config, "packet_bits", "L_pkt"))
    waiting = dt*weighted_wait/ttl
    alpha = _fraction(config["alpha"])
    score = alpha*bits/_fraction(config["B_ref"])-(1-alpha)*waiting/_fraction(config["W_ref"])
    trace = VerificationTrace(initial, final, caps, active, served_age, served_nrt, expired,
                              signals, interference, sinr, bits, weighted_wait, score, waiting,
                              start_slot, rt_success, rt_expiry, nrt_success)
    if hasattr(channel, "power_table"):
        trace.physical_input_mode = "frozen_received_watt_table_no_geometry_claim"
    if expected is not None:
        compare_trace(trace, expected)
    return trace


def compare_trace(actual: VerificationTrace, expected: Any) -> None:
    """Compare independent integer outcomes; saved physical floats are diagnostic."""
    fields = {
        "capacities": ("capacities", "capacity"),
        "active": ("active", "activated"),
        "served_age": ("served_age", "served_rt_age"),
        "served_nrt": ("served_nrt",),
        "expired": ("expired", "expired_rt"),
    }
    for ours, aliases in fields.items():
        name = next((name for name in aliases if hasattr(expected, name)), None)
        if name is not None:
            _same_integer(getattr(actual, ours), getattr(expected, name), f"saved {ours}")
            actual.compared.append(ours)
    if hasattr(expected, "slots"):
        for ours, slot_name in (("capacities", "capacities"), ("active", "active"),
                               ("served_age", "served_age"), ("served_nrt", "service_nrt"),
                               ("expired", "expired")):
            _same_integer(getattr(actual, ours), np.stack([row[slot_name] for row in expected.slots]), f"saved slot {ours}")
            actual.compared.append(f"slots.{slot_name}")
        _same_integer(actual.served_age.sum(axis=2), np.stack([row["service_rt"] for row in expected.slots]), "saved slot RT aggregate service")
        actual.compared.append("slots.service_rt")
    for name in ("bits", "weighted_wait"):
        if hasattr(expected, name):
            if int(getattr(actual, name)) != int(getattr(expected, name)):
                raise VerificationFailure(f"saved {name}", {"actual": int(getattr(actual, name)), "expected": int(getattr(expected, name))})
            actual.compared.append(name)
    if hasattr(expected, "F"):
        target = expected.F
        if isinstance(target, Fraction):
            equal = actual.F == target
        else:
            equal = float(actual.F) == float(target)
        if not equal:
            raise VerificationFailure("saved F", {"actual": str(actual.F), "expected": str(target)})
        actual.compared.append("F")
    if hasattr(expected, "W"):
        if actual.waiting != expected.W:
            raise VerificationFailure("saved W", {"actual": str(actual.waiting), "expected": str(expected.W)})
        actual.compared.append("W")
    end_name = next((name for name in ("final_state", "end_state", "end", "state") if hasattr(expected, name)), None)
    if end_name:
        compare_states(actual.final_state, getattr(expected, end_name))
        actual.compared.append("final_state")
