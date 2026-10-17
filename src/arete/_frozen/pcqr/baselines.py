"""R2-only TPG/DPG fixed-order greedy and visibility-aware round robin."""
from __future__ import annotations

import numpy as np

from .engine import replay, slot


def _ids(channel):
    return np.asarray(getattr(channel, "sat_ids", np.arange(channel.S)), dtype=np.int64)


def _capacities(channel):
    value = channel.config["K"]
    return np.broadcast_to(np.asarray(value, dtype=np.int64), (channel.S,)).copy()


def _check(plan, channel):
    if plan.shape != (channel.H, channel.S, channel.M):
        raise ValueError("Plan shape mismatch")
    if np.any(plan & ~channel.visible):
        raise ValueError("Invisible service edge")
    if np.any(plan.sum(axis=1) > 1):
        raise ValueError("Multiple service stars in cell-slot")
    if np.any(plan.sum(axis=2) > _capacities(channel)[None, :]):
        raise ValueError("Satellite beam capacity exceeded")


def greedy(Q0, A, channel, mode, meter, abs_start=0):
    """One frozen edge ordering per slot, followed by full physical queue update.

    RT urgency is ranked by its exact integer numerator; the shared age horizon
    denominator cannot change this ordering. The shared packet length likewise
    cancels from throughput rankings.
    """
    if mode not in ("TPG", "DPG"):
        raise ValueError("R2 greedy mode must be TPG or DPG")
    state = Q0.copy()
    plan = np.zeros((channel.H, channel.S, channel.M), dtype=bool)
    ids = _ids(channel)
    beam_caps = _capacities(channel)
    phase = "TPG_initialization" if mode == "TPG" else "residual"
    with meter.phase(phase):
        for h in range(channel.H):
            meter.check()
            state.add(A[h], abs_start + h)
            isolated = channel.cap(channel.signal[h], channel.config.get("I_ext", 0))
            cells_total = state.rt.sum(axis=1) + state.nrt
            candidates = np.argwhere(channel.visible[h])
            rows = []
            for s, m in candidates:
                s, m = int(s), int(m)
                capacity = int(isolated[s, m])
                rho_t_packets = min(int(cells_total[m]), capacity)
                if rho_t_packets <= 0:
                    continue
                if mode == "TPG":
                    # Signal differs from equivalent useful gain only by the
                    # common, frozen 120 W beam power in this experiment.
                    key = (-rho_t_packets, -capacity, -float(channel.signal[h, s, m]),
                           int(ids[s]), m)
                else:
                    remaining = capacity
                    urgency = 0
                    for age in range(state.rt.shape[1] - 1, -1, -1):
                        served = min(remaining, int(state.rt[m, age]))
                        urgency += (age + 1) * served
                        remaining -= served
                        if remaining == 0:
                            break
                    key = (-urgency, -rho_t_packets, -capacity, int(ids[s]), m)
                rows.append((key, s, m))
            rows.sort(key=lambda row: row[0])
            available = beam_caps.copy()
            occupied = np.zeros(channel.M, dtype=bool)
            for _, s, m in rows:
                if available[s] > 0 and not occupied[m]:
                    plan[h, s, m] = True
                    available[s] -= 1
                    occupied[m] = True
            # The state already contains this slot's arrival; slot() must not
            # add it again. The tentative queue never replaces the real queue.
            slot(state, plan[h], channel, h, abs_start + h)
            meter.replay(f"{mode}_construction_slot", 1)
        _check(plan, channel)
        meter.check()
        trace = replay(plan, Q0, A, channel, abs_start=abs_start)
        meter.replay(f"{mode}_initial_full", channel.H)
    return plan, trace


def round_robin(Q0, A, channel, meter, pointers=None, abs_start=0):
    """R2 VA-RR schedules without accessing queues or planning traffic.

    Q0/A are first consumed by the final common replay, after the entire plan
    and all persistent pointer updates have been fixed.
    """
    pointers = (np.zeros(channel.S, dtype=np.int64) if pointers is None
                else np.array(pointers, dtype=np.int64, copy=True))
    if pointers.shape != (channel.S,) or np.any(pointers < 0) or np.any(pointers >= channel.M):
        raise ValueError("Invalid round-robin pointers")
    plan = np.zeros((channel.H, channel.S, channel.M), dtype=bool)
    ids = _ids(channel)
    beam_caps = _capacities(channel)
    with meter.phase("residual"):
        for h in range(channel.H):
            meter.check()
            isolated = channel.cap(channel.signal[h], channel.config.get("I_ext", 0))
            scanned = np.zeros(channel.S, dtype=np.int64)
            remaining = beam_caps.copy()
            occupied = np.zeros(channel.M, dtype=bool)
            while np.any((remaining > 0) & (scanned < channel.M)):
                meter.check()
                proposals = {}
                for s in sorted(range(channel.S), key=lambda s: int(ids[s])):
                    requested = 0
                    while scanned[s] < channel.M and requested < remaining[s]:
                        m = int(pointers[s])
                        pointers[s] = (m + 1) % channel.M
                        scanned[s] += 1
                        if not channel.visible[h, s, m] or occupied[m]:
                            continue
                        proposals.setdefault(m, []).append(s)
                        requested += 1
                if not proposals:
                    break
                for m in sorted(proposals):
                    winner = min(proposals[m], key=lambda s: (-int(isolated[s, m]), int(ids[s])))
                    plan[h, winner, m] = True
                    remaining[winner] -= 1
                    occupied[m] = True
        _check(plan, channel)
    with meter.phase("final_verification"):
        meter.check()
        trace = replay(plan, Q0, A, channel, abs_start=abs_start)
        meter.replay("VA_RR_full", channel.H)
    return plan, trace, pointers
