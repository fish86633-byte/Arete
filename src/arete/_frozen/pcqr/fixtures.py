"""Literal, preregistered toy inputs; no fitted scenario or candidate search.

Power tables are complete tx-s,tx-m,rx-s,rx-m linear-W arrays. They are synthetic
physical contracts for finite checks, and make no claim to be S24 geometry.
"""
from itertools import product
import numpy as np

from .engine import State
from .physics import capacity


class TableChannel:
    def __init__(self, record):
        self.config = dict(record['config'])
        self.H, self.S, self.M = map(int, record['shape'])
        self.sat_ids = np.asarray(record['sat_ids'], dtype=np.int64)
        self.visible = np.asarray(record['visible'], dtype=bool)
        self.noise = float(record['noise'])
        self.power_table = np.asarray(record['power_table'], dtype=float)
        expected = (self.H, self.S, self.M, self.S, self.M)
        if self.power_table.shape != expected:
            raise ValueError('literal power table has the wrong shape')
        self.signal = np.zeros((self.H, self.S, self.M), dtype=float)
        for h, s, m in product(range(self.H), range(self.S), range(self.M)):
            self.signal[h, s, m] = self.power_table[h, s, m, s, m]
        self.graphs = np.zeros((self.H, self.S, self.S), dtype=bool)
        for h in range(self.H):
            for s in range(self.S):
                for t in range(s+1, self.S):
                    shared = bool(np.any(self.visible[h, s] & self.visible[h, t]))
                    coupling = any(
                        m != n and max(self.power_table[h, s, m, t, n],
                                       self.power_table[h, t, n, s, m]) / self.noise >= self.config['zeta_I']
                        for m in np.flatnonzero(self.visible[h, s])
                        for n in np.flatnonzero(self.visible[h, t]))
                    self.graphs[h, s, t] = self.graphs[h, t, s] = shared or coupling

    def power(self, h, tx_edges, rx_edges):
        tx = np.asarray(tx_edges, dtype=np.int64).reshape(-1, 2)
        rx = np.asarray(rx_edges, dtype=np.int64).reshape(-1, 2)
        return self.power_table[h, tx[:, 0, None], tx[:, 1, None],
                                rx[None, :, 0], rx[None, :, 1]].copy()

    def cap(self, signal, interference):
        return capacity(signal, interference, self.noise, self.config)


def materialize(record):
    channel = TableChannel(record)
    state = State(np.asarray(record['rt'], dtype=np.int64), record['nrt_batches'])
    arrivals = np.asarray(record['arrivals'], dtype=np.int64)
    return channel, state, arrivals


def edge_plan(record, name):
    plan = np.zeros(tuple(record['shape']), dtype=bool)
    for edge in record['plans'][name]:
        plan[tuple(edge)] = True
    return plan


def legal_slot_plans(channel, h):
    """Independent exhaustive enumeration for K=1, including all idle choices."""
    if channel.config['K'] != 1:
        raise ValueError('finite exhaustive fixture enumerator requires K=1')
    choices = [[-1] + list(map(int, np.flatnonzero(channel.visible[h, s])))
               for s in range(channel.S)]
    plans = []
    for selected in product(*choices):
        cells = [m for m in selected if m != -1]
        if len(cells) != len(set(cells)):
            continue
        plan = np.zeros((channel.S, channel.M), dtype=bool)
        for s, m in enumerate(selected):
            if m != -1:
                plan[s, m] = True
        plans.append(plan)
    return plans


def legal_plans(channel):
    return (np.stack(slots) for slots in product(
        *(legal_slot_plans(channel, h) for h in range(channel.H))))
