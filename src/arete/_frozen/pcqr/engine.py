"""R2 shared authoritative queues and exact-integer complete trajectory Replay.

New implementation of R2 equations (1)--(13). The oldest-first vectorization is
equivalent to the audited physical queue primitive in the former
environment/queues.py, but FIFO timestamps are now always retained, arrival
addition has an explicit absolute slot, objectives are exact Fractions, and
suffix reuse verifies every prefix input. No former planning module is imported.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
from contextlib import nullcontext

import numpy as np

from .physics import decimal_fraction
from .fifo import CompactFIFO


def _integers(value, label):
    value = np.asarray(value)
    if value.dtype.kind not in 'iu' or np.any(value < 0):
        raise ValueError(f'{label} must contain nonnegative integers')
    if value.dtype.kind == 'u' and np.any(value > np.iinfo(np.int64).max):
        raise ValueError(f'{label} exceeds int64')
    return np.array(value, dtype=np.int64, copy=True)


class State:
    """Arrival-before-service is explicit; this object otherwise represents Q-."""

    def __init__(self, rt, nrt_batches):
        self.rt = _integers(rt, 'RT queue')
        if self.rt.ndim != 2 or self.rt.shape[1] < 1:
            raise ValueError('RT queue shape must be M by positive TTL')
        if len(nrt_batches) != len(self.rt):
            raise ValueError('one FIFO batch list is required for each cell')
        self.nrt_batches = CompactFIFO(nrt_batches)
        self._arrival_slot = None

    @classmethod
    def from_decoded(cls, rt, nrt_batches, telemetry=None):
        """Own decoded arrays once, retaining all former constructor checks.

        Telemetry records disjoint whole passes rather than per-batch timer
        calls. Known copy bytes are numeric ndarray payloads (object fallback
        pointer payload is separately identified). No physical/model arithmetic
        or message normalization changes here.
        """
        def part(name):
            return telemetry.part(name) if telemetry is not None else nullcontext()
        with part('array_conversion'):
            owned_rt = np.array(rt, copy=True)
        if telemetry is not None:
            telemetry.copied(int(owned_rt.nbytes), 'decoded_rt_array')
        with part('item_validation'):
            if owned_rt.dtype.kind not in 'iu' or np.any(owned_rt < 0):
                raise ValueError('RT queue must contain nonnegative integers')
            if owned_rt.dtype.kind == 'u' and np.any(owned_rt > np.iinfo(np.int64).max):
                raise ValueError('RT queue exceeds int64')
            if owned_rt.ndim != 2 or owned_rt.shape[1] < 1:
                raise ValueError('RT queue shape must be M by positive TTL')
            if len(nrt_batches) != len(owned_rt):
                raise ValueError('one FIFO batch list is required for each cell')
        if owned_rt.dtype != np.dtype(np.int64):
            with part('array_conversion'):
                owned_rt = np.array(owned_rt, dtype=np.int64, copy=True)
            if telemetry is not None:
                telemetry.copied(int(owned_rt.nbytes), 'decoded_rt_dtype_conversion')
        with part('rt_state_construction'):
            result = object.__new__(cls)
            result.rt = owned_rt
            result._arrival_slot = None
        result.nrt_batches = CompactFIFO(nrt_batches, telemetry=telemetry)
        return result

    @classmethod
    def empty(cls, M, ttl):
        if isinstance(ttl, dict):
            ttl = ttl['ttl']
        return cls(np.zeros((int(M), int(ttl)), dtype=np.int64), [[] for _ in range(int(M))])

    @property
    def nrt(self):
        return self.nrt_batches.totals()

    def copy(self):
        result = object.__new__(State)
        result.rt = self.rt.copy()
        result.nrt_batches = self.nrt_batches.copy()
        result._arrival_slot = self._arrival_slot
        return result

    def add(self, arrivals, absolute_slot, *, fast_fifo_metadata=False,
            fast_fifo_batch=False):
        arrivals = _integers(arrivals, 'arrivals')
        if arrivals.shape != (len(self.rt), 2):
            raise ValueError('arrivals must have shape M by 2 (RT,NRT)')
        if self._arrival_slot is not None:
            raise ValueError('arrivals already added; advance the slot before adding again')
        if np.any(self.rt[:, 0] > np.iinfo(np.int64).max - arrivals[:, 0]):
            raise OverflowError('RT arrival count exceeds int64')
        self.rt[:, 0] += arrivals[:, 0]
        if fast_fifo_batch:
            # _integers, shape and overflow checks above still apply. Only
            # already-validated int64 counts use the shared FIFO batch path.
            self.nrt_batches._append_verified_arrivals(arrivals[:, 1], absolute_slot)
        else:
            for m in np.flatnonzero(arrivals[:, 1]):
                self.nrt_batches[int(m)].append((int(absolute_slot), int(arrivals[m, 1])),
                                               exact_metadata=fast_fifo_metadata)
        self._arrival_slot = int(absolute_slot)


def validate(plan, channel):
    plan = np.asarray(plan)
    if plan.shape != (channel.H, channel.S, channel.M):
        raise ValueError('plan dimensions disagree with the channel horizon')
    if not np.all((plan == 0) | (plan == 1)):
        raise ValueError('plan must be binary')
    if np.any(plan.astype(bool) & ~channel.visible):
        raise ValueError('plan contains an ineligible service edge')
    if np.any(plan.sum(axis=2) > int(channel.config['K'])):
        raise ValueError('plan exceeds per-satellite beam limit')
    if np.any(plan.sum(axis=1) > 1):
        raise ValueError('plan serves a cell from more than one satellite')


def slot(state, plan_h, channel, h, absolute_slot):
    """Advance already-arrived Q+ by full SINR, service, expiry and ageing."""
    plan_h = np.asarray(plan_h)
    if not np.all((plan_h == 0) | (plan_h == 1)):
        raise ValueError('slot plan must be binary')
    plan_h = plan_h.astype(bool)
    if plan_h.shape != (channel.S, channel.M):
        raise ValueError('slot plan dimensions disagree with channel')
    if (np.any(plan_h & ~channel.visible[h])
            or np.any(plan_h.sum(axis=1) > int(channel.config['K']))
            or np.any(plan_h.sum(axis=0) > 1)):
        raise ValueError('invalid slot plan')
    if state._arrival_slot is not None and state._arrival_slot != int(absolute_slot):
        raise ValueError('queue arrivals belong to a different absolute slot')
    before_rt = state.rt.copy()
    before_nrt = state.nrt
    active = plan_h & ((before_rt.sum(axis=1) + before_nrt) > 0)[None, :]
    edges = np.argwhere(active)
    capacities = np.zeros_like(plan_h, dtype=np.int64)
    interference = np.zeros_like(plan_h, dtype=float)
    cell_cap = np.zeros(channel.M, dtype=np.int64)
    if len(edges):
        powers = channel.power(h, edges, edges)
        np.fill_diagonal(powers, 0.0)
        # fsum includes every simultaneously active beam and avoids cancellation
        # from subtracting a large self-signal after summing the matrix.
        incoming = np.asarray([math.fsum(powers[:, j]) + float(channel.config.get('I_ext', 0.0))
                               for j in range(len(edges))])
        caps = channel.cap(channel.signal[h, edges[:, 0], edges[:, 1]], incoming)
        capacities[edges[:, 0], edges[:, 1]] = caps
        interference[edges[:, 0], edges[:, 1]] = incoming
        cell_cap[edges[:, 1]] = caps
    oldest = before_rt[:, ::-1]
    older = np.cumsum(oldest, axis=1) - oldest
    served_age = np.minimum(oldest, np.maximum(cell_cap[:, None] - older, 0))[:, ::-1]
    service_rt = served_age.sum(axis=1)
    service_nrt = np.minimum(before_nrt, np.maximum(cell_cap - service_rt, 0))
    residual = before_rt - served_age
    expired = residual[:, -1].copy()
    D = state.rt.shape[1]
    bound = int(residual.max(initial=0)) * len(residual) * D * (D+1) // 2
    if bound <= np.iinfo(np.int64).max:
        weighted_wait = int(np.sum(residual * np.arange(1, D+1, dtype=np.int64), dtype=np.int64))
    else:
        weighted_wait = sum(int(count)*(age+1) for row in residual for age, count in enumerate(row))
    events = []
    for m, age in zip(*np.nonzero(served_age)):
        birth = int(absolute_slot) - int(age)
        events.append((int(m), 0, birth, int(absolute_slot), int(served_age[m, age]), birth+D-1))
    for m in np.flatnonzero(service_nrt):
        remaining = int(service_nrt[m])
        batches = state.nrt_batches[int(m)]
        consumed = 0
        while remaining:
            birth, count = batches[consumed]
            take = min(remaining, count)
            events.append((int(m), 1, birth, int(absolute_slot), take, -1))
            remaining -= take
            if take == count:
                consumed += 1
            else:
                batches[consumed] = (birth, count-take)
                break
        if consumed:
            del batches[:consumed]
    state.rt[:, 1:] = residual[:, :-1]
    state.rt[:, 0] = 0
    state._arrival_slot = None
    if not np.array_equal(before_rt.sum(axis=1), service_rt + expired + state.rt.sum(axis=1)):
        raise AssertionError('per-cell RT conservation failed')
    if not np.array_equal(before_nrt, service_nrt + state.nrt):
        raise AssertionError('per-cell NRT conservation failed')
    return {'qplus_rt': before_rt, 'qplus_nrt': before_nrt,
            'active': active, 'capacities': capacities, 'interference': interference,
            'service_rt': service_rt, 'service_nrt': service_nrt,
            'served_age': served_age, 'expired': expired, 'residual': residual,
            'weighted_wait': weighted_wait, 'events': events,
            'absolute_slot': int(absolute_slot)}


@dataclass
class Trace:
    F: Fraction
    bits: int
    W: Fraction
    weighted_wait: int
    end_state: State
    slots: list
    states: list
    plan: np.ndarray
    arrivals: np.ndarray
    abs_start: int
    channel: object

    @property
    def F_float(self):
        return float(self.F)


def _same_state(a, b):
    return np.array_equal(a.rt, b.rt) and a.nrt_batches == b.nrt_batches


def replay(plan, Q0, A, channel, abs_start=0, start=0, prefix=None):
    """Full-horizon score; safe suffix reuse keeps H+1 arrival-before states."""
    plan = np.asarray(plan)
    A = _integers(A, 'planning arrivals')
    validate(plan, channel)
    plan = plan.astype(bool)
    fast_storage = bool(channel.config.get('r32_replay_storage', False))
    if A.shape != (channel.H, channel.M, 2):
        raise ValueError('planning arrivals must have shape H by M by 2')
    if Q0.rt.shape != (channel.M, int(channel.config['ttl'])) or Q0._arrival_slot is not None:
        raise ValueError('Q0 must be complete arrival-before state with the configured TTL')
    if not 0 <= int(start) <= channel.H or int(start) != start:
        raise ValueError('invalid replay suffix start')
    start = int(start)
    if start:
        if prefix is None or prefix.channel is not channel or prefix.abs_start != int(abs_start):
            raise ValueError('suffix requires the same immutable channel and absolute horizon')
        if (not _same_state(Q0, prefix.states[0]) or not np.array_equal(A, prefix.arrivals)
                or not np.array_equal(plan[:start], prefix.plan[:start])):
            raise ValueError('suffix prefix inputs differ from the reference trajectory')
        slots = list(prefix.slots[:start])
        states = [state.copy() for state in prefix.states[:start+1]]
        state = states[-1].copy()
    else:
        slots, states = [], [Q0.copy()]
        state = Q0.copy()
    for h in range(start, channel.H):
        state.add(A[h], int(abs_start)+h, fast_fifo_metadata=fast_storage,
                  fast_fifo_batch=bool(channel.config.get('r36_batch_arrivals', False)))
        slots.append(slot(state, plan[h], channel, h, int(abs_start)+h))
        states.append(state.copy())
    packets = sum(int(item['service_rt'].sum()) + int(item['service_nrt'].sum()) for item in slots)
    weighted_wait = sum(item['weighted_wait'] for item in slots)
    bits = packets * int(channel.config['packet_bits'])
    W = decimal_fraction(channel.config['dt']) * Fraction(weighted_wait, int(channel.config['ttl']))
    alpha = decimal_fraction(channel.config['alpha'])
    objective = alpha*bits/decimal_fraction(channel.config['B_ref']) - (1-alpha)*W/decimal_fraction(channel.config['W_ref'])
    # astype(bool) and _integers(A) above already allocated independent arrays.
    # Their private ownership can transfer to Trace; all queue snapshots retain
    # their former copies and no caller or prefix array is shared.
    trace_plan = plan if fast_storage else plan.copy()
    trace_arrivals = A if fast_storage else A.copy()
    return Trace(objective, bits, W, weighted_wait, state.copy(), slots, states,
                 trace_plan, trace_arrivals, int(abs_start), channel)


def replay_suffix(plan, Q0, A, channel, reference_trace, first_changed, abs_start=0):
    return replay(plan, Q0, A, channel, abs_start=abs_start,
                  start=first_changed, prefix=reference_trace)
