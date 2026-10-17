"""Independent exact objective evaluator retaining RT ages and NRT totals only.

This module intentionally does not replace pcqr.engine.  NRT birth times are
irrelevant to the frozen objective/activation/service priority, but are needed
for NRT delay reporting and FIFO reconstruction; neither is claimed here.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math

import numpy as np

from arete._frozen.pcqr.engine import _integers, validate
from arete._frozen.pcqr.physics import decimal_fraction


class LightState:
    def __init__(self, rt, nrt, _arrival_slot=None):
        self.rt = _integers(rt, 'RT queue')
        self.nrt = _integers(nrt, 'NRT totals')
        if self.rt.ndim != 2 or self.rt.shape[1] < 1:
            raise ValueError('RT queue shape must be M by positive TTL')
        if self.nrt.shape != (len(self.rt),):
            raise ValueError('NRT totals must have shape M')
        self._arrival_slot = _arrival_slot

    @classmethod
    def from_state(cls, state):
        return cls(state.rt, state.nrt, state._arrival_slot)

    def copy(self):
        result = object.__new__(LightState)
        result.rt, result.nrt = self.rt.copy(), self.nrt.copy()
        result._arrival_slot = self._arrival_slot
        return result

    def add(self, arrivals, absolute_slot):
        arrivals = _integers(arrivals, 'arrivals')
        if arrivals.shape != (len(self.rt), 2):
            raise ValueError('arrivals must have shape M by 2 (RT,NRT)')
        if self._arrival_slot is not None:
            raise ValueError('arrivals already added; advance the slot before adding again')
        bound = np.iinfo(np.int64).max
        if np.any(self.rt[:, 0] > bound - arrivals[:, 0]):
            raise OverflowError('RT arrival count exceeds int64')
        if np.any(self.nrt > bound - arrivals[:, 1]):
            raise OverflowError('NRT arrival count exceeds int64')
        self.rt[:, 0] += arrivals[:, 0]
        self.nrt += arrivals[:, 1]
        self._arrival_slot = int(absolute_slot)


def advance(state, plan_h, channel, h, absolute_slot):
    """Advance all cells; the original channel implements power and capacity."""
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
    before_rt, before_nrt = state.rt.copy(), state.nrt.copy()
    active = plan_h & ((before_rt.sum(axis=1) + before_nrt) > 0)[None, :]
    edges = np.argwhere(active)
    capacities = np.zeros_like(plan_h, dtype=np.int64)
    interference = np.zeros_like(plan_h, dtype=float)
    cell_cap = np.zeros(channel.M, dtype=np.int64)
    if len(edges):
        powers = channel.power(h, edges, edges)
        np.fill_diagonal(powers, 0.0)
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
    state.nrt -= service_nrt
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
            'weighted_wait': weighted_wait, 'events': [],
            'absolute_slot': int(absolute_slot)}


@dataclass
class LightTrace:
    F: Fraction
    bits: int
    W: Fraction
    weighted_wait: int
    end_state: LightState
    slots: list
    states: list
    plan: np.ndarray
    arrivals: np.ndarray
    abs_start: int
    channel: object
    propagated_slots: int

    @property
    def F_float(self):
        return float(self.F)

    @classmethod
    def from_full_trace(cls, trace):
        return from_full_trace(trace)


def _copy_slot(item):
    return {key: (value.copy() if isinstance(value, np.ndarray) else value)
            for key, value in item.items() if key != 'events'} | {'events': []}


def from_full_trace(trace):
    """Convert stored full data without propagating or claiming FIFO identity."""
    return LightTrace(trace.F, trace.bits, trace.W, trace.weighted_wait,
                      LightState.from_state(trace.end_state),
                      [_copy_slot(item) for item in trace.slots],
                      [LightState.from_state(state) for state in trace.states],
                      trace.plan.copy(), trace.arrivals.copy(), trace.abs_start,
                      trace.channel, 0)


def _same_state(a, b):
    return (a._arrival_slot == b._arrival_slot
            and np.array_equal(a.rt, b.rt) and np.array_equal(a.nrt, b.nrt))


def light_replay(plan, Q0, A, channel, abs_start=0, start=0, prefix=None, meter=None):
    """Exact frozen objective using totals-only NRT; prefix inputs are checked."""
    if meter is not None:
        meter.check()
    plan = np.asarray(plan)
    A = _integers(A, 'planning arrivals')
    validate(plan, channel)
    plan = plan.astype(bool)
    if A.shape != (channel.H, channel.M, 2):
        raise ValueError('planning arrivals must have shape H by M by 2')
    if Q0.rt.shape != (channel.M, int(channel.config['ttl'])) or Q0._arrival_slot is not None:
        raise ValueError('Q0 must be complete arrival-before state with configured TTL')
    if not 0 <= int(start) <= channel.H or int(start) != start:
        raise ValueError('invalid replay suffix start')
    start = int(start)
    if start:
        if prefix is None or prefix.channel is not channel or prefix.abs_start != int(abs_start):
            raise ValueError('suffix requires same immutable channel and absolute horizon')
        if len(prefix.slots) < start or len(prefix.states) < start + 1:
            raise ValueError('suffix prefix does not contain the required completed slots')
        if (not _same_state(Q0, prefix.states[0]) or not np.array_equal(A, prefix.arrivals)
                or not np.array_equal(plan[:start], prefix.plan[:start])):
            raise ValueError('suffix prefix inputs differ from reference trajectory')
        slots = [_copy_slot(item) for item in prefix.slots[:start]]
        states = [LightState.from_state(state) for state in prefix.states[:start+1]]
        state = states[-1].copy()
    else:
        slots, states = [], [LightState.from_state(Q0)]
        state = states[-1].copy()
    propagated = 0
    for h in range(start, channel.H):
        if meter is not None:
            meter.check()
        state.add(A[h], int(abs_start)+h)
        slots.append(advance(state, plan[h], channel, h, int(abs_start)+h))
        states.append(state.copy())
        propagated += 1
        if meter is not None:
            meter.replay('light_slot', 1)
    packets = sum(int(item['service_rt'].sum()) + int(item['service_nrt'].sum()) for item in slots)
    weighted_wait = sum(item['weighted_wait'] for item in slots)
    bits = packets * int(channel.config['packet_bits'])
    W = decimal_fraction(channel.config['dt']) * Fraction(weighted_wait, int(channel.config['ttl']))
    alpha = decimal_fraction(channel.config['alpha'])
    objective = alpha*bits/decimal_fraction(channel.config['B_ref']) - (1-alpha)*W/decimal_fraction(channel.config['W_ref'])
    return LightTrace(objective, bits, W, weighted_wait, state.copy(), slots, states,
                      plan, A, int(abs_start), channel, propagated)


def light_replay_suffix(plan, Q0, A, channel, reference_trace, first_changed, abs_start=0, meter=None):
    return light_replay(plan, Q0, A, channel, abs_start=abs_start,
                        start=first_changed, prefix=reference_trace, meter=meter)


def compare_light_to_full(light, full):
    """Zero-tolerance observable comparison; deliberately excludes FIFO/hash."""
    for key in ('F', 'W', 'bits', 'weighted_wait', 'abs_start'):
        if getattr(light, key) != getattr(full, key):
            raise AssertionError(f'trace {key} mismatch')
    if len(light.slots) != len(full.slots) or len(light.states) != len(full.states):
        raise AssertionError('trace length mismatch')
    arrays = ('qplus_rt', 'qplus_nrt', 'active', 'capacities', 'interference',
              'service_rt', 'service_nrt', 'served_age', 'expired', 'residual')
    for h, (left, right) in enumerate(zip(light.slots, full.slots)):
        for key in arrays:
            if not np.array_equal(left[key], right[key]):
                raise AssertionError(f'slot {h} {key} mismatch')
        for key in ('weighted_wait', 'absolute_slot'):
            if left[key] != right[key]:
                raise AssertionError(f'slot {h} {key} mismatch')
    for h, (left, right) in enumerate(zip(light.states, full.states)):
        if not _same_state(left, right):
            raise AssertionError(f'state {h} RT/NRT totals mismatch')
    if not _same_state(light.end_state, full.end_state):
        raise AssertionError('terminal RT/NRT totals mismatch')
    if not np.array_equal(light.plan, full.plan) or not np.array_equal(light.arrivals, full.arrivals):
        raise AssertionError('trace input mismatch')
    return {'status': 'PASS', 'integer_and_fraction_tolerance': 0,
            'physical_array_comparison': 'exact', 'full_fifo_identity_claimed': False}
