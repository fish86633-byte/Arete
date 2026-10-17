"""Read-only, all-link local service utility for a fixed arrived queue.

This is an exploration component, not a replacement physical model or an
accepted planner.  It scores a complete feasible single-slot arrangement using
the original channel power/capacity routines and the original exact utility.
It does not add arrivals, advance queues, read FIFO birth times, generate delay
events, or claim a full-horizon objective.  Every caller must supply the same
fixed Q+ when comparing arrangements at a slot.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math

import numpy as np

from arete._frozen.pcqr.search import value


@dataclass
class SlotScore:
    """Owned outputs; the input queue and arrangement remain unchanged."""

    V: Fraction
    delivered: int
    weighted_served_rt: int
    actual: np.ndarray
    capacities: np.ndarray
    interference: np.ndarray
    cell_capacity: np.ndarray
    service_rt: np.ndarray
    service_nrt: np.ndarray
    served_age: np.ndarray
    residual_rt: np.ndarray
    expired: np.ndarray
    weighted_wait: int


def _queue_array(array, shape, label):
    """Check numeric queue input without touching or copying any FIFO."""
    array = np.asarray(array)
    if array.shape != shape or array.dtype.kind not in 'iu':
        raise ValueError(f'{label} must be a nonnegative integer array of shape {shape}')
    if np.any(array < 0):
        raise ValueError(f'{label} contains negative counts')
    if array.dtype.kind == 'u' and np.any(array > np.iinfo(np.int64).max):
        raise OverflowError(f'{label} exceeds int64')
    # Authoritative engine states are int64.  Conversion only owns numeric
    # arrays when a different integer representation is explicitly supplied.
    return np.asarray(array, dtype=np.int64)


def _weighted_age_sum(counts):
    """Use the engine's safe vector reduction with its integer fallback."""
    D = counts.shape[1]
    bound = int(counts.max(initial=0)) * len(counts) * D * (D + 1) // 2
    if bound <= np.iinfo(np.int64).max:
        return int(np.sum(counts * np.arange(1, D + 1, dtype=np.int64), dtype=np.int64))
    return sum((age + 1) * int(count) for row in counts for age, count in enumerate(row))


def score_slot(rt, nrt, plan_h, channel, h, *, meter=None):
    """Return exact local V for an entire arrangement at fixed arrived Q+.

    rt is M-by-TTL, nrt is M-vector, and plan_h is S-by-M.  h is the planning
    slot index.  Queue arrays must already contain this slot's planning arrival
    exactly once.  They are only read.  Optional meter checks bracket work;
    callers own the phase timer and must charge every completed/interrupted
    scoring attempt.  This function has no cache or persistent mutable state.

    With fixed Q+, V equals the single-slot objective contribution plus the
    arrangement-independent initial RT weighted-wait constant.  It therefore
    ranks one-slot changes exactly, while providing no guarantee for the
    remaining dynamic suffix.
    """
    if meter is not None:
        meter.check()
    if isinstance(h, (bool, np.bool_)) or int(h) != h or not 0 <= int(h) < channel.H:
        raise ValueError('invalid planning slot index')
    h = int(h)
    D = int(channel.config['ttl'])
    rt = _queue_array(rt, (channel.M, D), 'RT queue')
    nrt = _queue_array(nrt, (channel.M,), 'NRT totals')
    plan_h = np.asarray(plan_h)
    if plan_h.shape != (channel.S, channel.M) or not np.all((plan_h == 0) | (plan_h == 1)):
        raise ValueError('slot arrangement must be binary with shape S by M')
    plan_h = plan_h.astype(bool)
    if (np.any(plan_h & ~channel.visible[h])
            or np.any(plan_h.sum(axis=1) > int(channel.config['K']))
            or np.any(plan_h.sum(axis=0) > 1)):
        raise ValueError('infeasible single-slot arrangement')

    # The rows and columns follow original engine.slot argwhere order.  Every
    # actually active beam contributes, including same-satellite beams; the
    # coordination graph never truncates physical interference here.
    actual = plan_h & ((rt.sum(axis=1) + nrt) > 0)[None, :]
    edges = np.argwhere(actual)
    capacities = np.zeros_like(plan_h, dtype=np.int64)
    interference = np.zeros_like(plan_h, dtype=float)
    cell_capacity = np.zeros(channel.M, dtype=np.int64)
    if len(edges):
        powers = channel.power(h, edges, edges)
        np.fill_diagonal(powers, 0.0)
        incoming = np.asarray([
            math.fsum(powers[:, j]) + float(channel.config.get('I_ext', 0.0))
            for j in range(len(edges))
        ])
        caps = channel.cap(channel.signal[h, edges[:, 0], edges[:, 1]], incoming)
        capacities[edges[:, 0], edges[:, 1]] = caps
        interference[edges[:, 0], edges[:, 1]] = incoming
        cell_capacity[edges[:, 1]] = caps

    # These service formulas are the original engine.slot formulas.  They do
    # not mutate rt/nrt and do not reconstruct or consume NRT FIFO batches.
    oldest = rt[:, ::-1]
    older = np.cumsum(oldest, axis=1) - oldest
    served_age = np.minimum(oldest, np.maximum(cell_capacity[:, None] - older, 0))[:, ::-1]
    service_rt = served_age.sum(axis=1)
    service_nrt = np.minimum(nrt, np.maximum(cell_capacity - service_rt, 0))
    residual_rt = rt - served_age
    expired = residual_rt[:, -1].copy()
    delivered = sum(int(x) for x in service_rt) + sum(int(x) for x in service_nrt)
    # Python integer reductions precede Fraction construction.  No floating
    # surrogate or finer/coarser capacity policy is introduced.
    weighted_served_rt = _weighted_age_sum(served_age)
    weighted_wait = _weighted_age_sum(residual_rt)
    V = value(delivered, weighted_served_rt, channel.config)
    if meter is not None:
        meter.check()
    return SlotScore(V, delivered, weighted_served_rt, actual, capacities,
                     interference, cell_capacity, service_rt, service_nrt,
                     served_age, residual_rt, expired, weighted_wait)
