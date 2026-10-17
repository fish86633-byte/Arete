"""V2 exploration: targeted legal changes to the current slot arrangement.

The fixed local scoring and slot ordering are shared with V1.  Unlike V1,
reference allocations are retained outside the proposed deletion/replacement/
exchange.  The outer unchanged owner evaluates the full dynamic suffix and
verifies strict improvements with the authoritative complete FIFO engine.
"""
from __future__ import annotations

from fractions import Fraction
from itertools import combinations
import time

import numpy as np

from arete._frozen.pcqr.support import plan_key

from .candidate_search import COMMIT_RESERVE_S, _rank_slots, _stop_name
from .local_score import score_slot


VARIANT = 'V2'
NEIGHBORHOOD_LIMIT = 128
TOP_COMPLETE_CANDIDATES = 4
TYPE_ORDER = {'DELETE': 0, 'INSERT_REPLACE': 1, 'SWAP_TWO': 2}


def _neighbors(reference_h, row, channel, meter):
    """Three fixed streams, round-robin, yielding complete legal changes."""
    h = row['h']
    ids = np.asarray(channel.sat_ids, dtype=np.int64)
    original = sorted(((int(s), int(m)) for s, m in np.argwhere(reference_h)),
                      key=lambda edge: (int(ids[edge[0]]), edge[1]))

    def delete():
        for edge in original:
            meter.check(reserve=.03)
            Y = reference_h.copy()
            Y[edge] = False
            yield Y, {'type': 'DELETE'}

    def insert_replace():
        for s, m in row['pool']:
            meter.check(reserve=.03)
            if reference_h[s, m]:
                continue
            Y = reference_h.copy()
            owner = np.flatnonzero(Y[:, m])
            for old_s in owner:
                Y[int(old_s), m] = False
            if int(Y[s].sum()) >= int(channel.config['K']):
                # Enumerate each fixed-ID occupation edge instead of choosing
                # an edge by its own isolated utility or service alone.
                for old_m in np.flatnonzero(Y[s]):
                    meter.check(reserve=.03)
                    candidate = Y.copy()
                    candidate[s, int(old_m)] = False
                    candidate[s, m] = True
                    yield candidate, {'type': 'INSERT_REPLACE'}
            else:
                Y[s, m] = True
                yield Y, {'type': 'INSERT_REPLACE'}

    def swap():
        for (s, m), (t, n) in combinations(original, 2):
            meter.check(reserve=.03)
            if s == t or not channel.visible[h, s, n] or not channel.visible[h, t, m]:
                continue
            Y = reference_h.copy()
            Y[s, m] = False
            Y[t, n] = False
            Y[s, n] = True
            Y[t, m] = True
            yield Y, {'type': 'SWAP_TWO'}

    streams = [iter(delete()), iter(insert_replace()), iter(swap())]
    live = [True, True, True]
    while any(live):
        for index, stream in enumerate(streams):
            if not live[index]:
                continue
            meter.check(reserve=.03)
            try:
                yield next(stream)
            except StopIteration:
                live[index] = False


def _edge_keys(mask, channel, h):
    return tuple(sorted((h, int(channel.sat_ids[s]), int(m))
                        for s, m in np.argwhere(mask)))


def _score_neighborhood(X, row, tr, channel, meter, scan, event, exploration):
    """Keep only completed, distinct, legal arrangements in the local pool."""
    h = row['h']
    original = X[h]
    rt, nrt = tr.slots[h]['qplus_rt'], tr.slots[h]['qplus_nrt']
    seen = {original.tobytes()}
    candidates = []
    iterator = iter(_neighbors(original, row, channel, meter))
    while len(candidates) < NEIGHBORHOOD_LIMIT:
        meter.check(reserve=.03)
        with meter.phase('direct_generation'):
            try:
                arrangement, source = next(iterator)
            except StopIteration:
                event['neighborhood_exhausted'] = True
                break
            event['neighbor_generated'] += 1
            key = arrangement.tobytes()
            if key in seen:
                event['neighbor_duplicates'] += 1
                continue
            # Streams intentionally generate legal moves; retain an explicit
            # gate before admission.  An illegal move is a correctness failure,
            # not a performance stop or silently discarded experiment.
            if (np.any(arrangement & ~channel.visible[h])
                    or np.any(arrangement.sum(axis=1) > int(channel.config['K']))
                    or np.any(arrangement.sum(axis=0) > 1)):
                raise AssertionError('V2 neighborhood emitted an illegal arrangement')
            seen.add(key)
            removed = _edge_keys(original & ~arrangement, channel, h)
            added = _edge_keys(arrangement & ~original, channel, h)
            scan.take('V2_targeted_score', {'h': h, 'type': source['type'],
                                           'removed': removed, 'added': added})
            event['score_attempts'] += 1
            exploration['score_attempts'] += 1
            score = score_slot(rt, nrt, arrangement, channel, h, meter=meter)
            event['score_completed'] += 1
            exploration['score_completed'] += 1
            delta = score.V - row['reference_actual_V']
            # Nonpositive local deltas remain eligible for true suffix scoring.
            source.update(V_slot_delta=delta, removed=removed, added=added)
            rank = (-delta, TYPE_ORDER[source['type']], removed, added)
            candidates.append((rank, arrangement, source))
    with meter.phase('direct_generation'):
        if len(candidates) >= NEIGHBORHOOD_LIMIT:
            event['neighborhood_exhausted'] = False
            event['neighborhood_cap_reached'] = True
        candidates.sort(key=lambda candidate: candidate[0])
        event['distinct_scored_count'] = len(candidates)
        event['selected_top_count'] = min(TOP_COMPLETE_CANDIDATES, len(candidates))
        event['ranked_top'] = [source for _, _, source in candidates[:TOP_COMPLETE_CANDIDATES]]
    return candidates[:TOP_COMPLETE_CANDIDATES]


def search_reference(X, tr, Q0, A, channel, meter, ledger, cycle, result,
                     roundinfo, scan, counts, whole_plan_memo, evaluate,
                     best_F, *, variant):
    """V2 candidate preparation; acceptance remains with the reliable owner."""
    if variant != VARIANT or int(channel.config['L_eval']) != 16:
        raise ValueError('this frozen V2 mechanism requires variant V2 and L_eval=16')
    exploration = result.setdefault('exploration', {
        'variant': VARIANT, 'policy': 'TARGETED_REPLACEMENT_EXCHANGE',
        'organization': 'FIRST_STRICT_IMPROVEMENT_THEN_COMMIT',
        'commit_reserve_s': COMMIT_RESERVE_S, 'pool_limit': 32,
        'neighborhood_limit': NEIGHBORHOOD_LIMIT,
        'top_complete_candidates': TOP_COMPLETE_CANDIDATES,
        'score_attempts': 0, 'score_completed': 0,
        'construct_started': 0, 'construct_completed': 0,
        'construct_interrupted': 0, 'references': [], 'events': []})
    if exploration['variant'] != VARIANT:
        raise ValueError('V2 exploration log has another variant')
    version = int(result['accepted'])
    ref_record = {'ref_version': version, 'F_reference': tr.F,
                  'status': 'RANKING', 'slot_order': [], 'slots': []}
    exploration['references'].append(ref_record)
    roundinfo['search_policy'] = 'V2_TARGETED_REPLACEMENT_EXCHANGE'
    roundinfo['exploration'] = ref_record
    baseline = sum(counts.values())
    own_keys = {plan_key(X, channel)}
    limited = False
    search_stop = 'SLOT_POOL_EXHAUSTED'
    try:
        rows = _rank_slots(X, tr, channel, meter)
        ref_record['slot_order'] = [row['h'] for row in rows]
        ref_record['slots'] = [{key: value for key, value in row.items() if key != 'pool'}
                               | {'pool_count': len(row['pool'])} for row in rows]
        ref_record['status'] = 'PREPARING_NEIGHBORHOODS'
        for priority_index, row in enumerate(rows):
            if sum(counts.values()) >= int(channel.config['L_eval']):
                limited = True
                search_stop = 'COMPLETED_EVALUATION_CAP'
                break
            if meter.remaining <= COMMIT_RESERVE_S:
                limited = True
                search_stop = 'COMMIT_RESERVE_STOP'
                exploration['events'].append({
                    'ref_version': version, 'status': search_stop,
                    'next_slot': row['h'], 'remaining_s': meter.remaining,
                    'commit_reserve_s': COMMIT_RESERVE_S,
                    'candidate_submitted': False})
                break
            meter.check(reserve=.03)
            h = row['h']
            event = {'ref_version': version, 'slot': h, 'priority_index': priority_index,
                     'status': 'PREPARING_NEIGHBORHOOD', 'gap': row['gap'],
                     'reference_actual_V': row['reference_actual_V'],
                     'reference_isolated_V': row['reference_isolated_V'],
                     'edge_pool_count': len(row['pool']),
                     'edge_pool': [(int(channel.sat_ids[s]), m) for s, m in row['pool']],
                     'neighborhood_limit': NEIGHBORHOOD_LIMIT,
                     'top_complete_candidates': TOP_COMPLETE_CANDIDATES,
                     'neighbor_generated': 0, 'neighbor_duplicates': 0,
                     'score_attempts': 0, 'score_completed': 0,
                     'neighborhood_exhausted': False,
                     'neighborhood_cap_reached': False}
            exploration['events'].append(event)
            exploration['construct_started'] += 1
            started, cpu_started = time.perf_counter(), time.process_time()
            try:
                candidates = _score_neighborhood(X, row, tr, channel, meter, scan,
                                                 event, exploration)
                event['status'] = 'NEIGHBORHOOD_COMPLETED'
                exploration['construct_completed'] += 1
            except Exception as error:
                event.update(status=_stop_name(error), error=repr(error),
                             candidate_submitted=False)
                exploration['construct_interrupted'] += 1
                raise
            finally:
                event['construction_wall_s'] = time.perf_counter() - started
                event['construction_cpu_s'] = time.process_time() - cpu_started
            for local_rank, (_, arrangement, source) in enumerate(candidates):
                if sum(counts.values()) >= int(channel.config['L_eval']):
                    limited = True
                    search_stop = 'COMPLETED_EVALUATION_CAP'
                    break
                with meter.phase('direct_generation'):
                    Y = X.copy()
                    Y[h] = arrangement
                    key = plan_key(Y, channel)
                    difference = Y != X
                    changed_satellites = np.flatnonzero(np.any(difference, axis=(0, 2)))
                    changed_slots = np.flatnonzero(np.any(difference, axis=(1, 2)))
                    metadata = {**source, 'variant': VARIANT, 'slot': h,
                                'priority_index': priority_index, 'local_rank': local_rank,
                                'candidate_pool_count': len(row['pool']),
                                'candidate_pool': event['edge_pool'],
                                'neighborhood_limit': NEIGHBORHOOD_LIMIT,
                                'top_complete_candidates': TOP_COMPLETE_CANDIDATES,
                                'score_attempts': event['score_attempts'],
                                'score_completed': event['score_completed'],
                                'changed_satellite_count': len(changed_satellites),
                                'changed_slot_count': len(changed_slots),
                                'first_changed': int(changed_slots[0]),
                                'true_two_slot_change': len(changed_slots) >= 2,
                                'removed_edge_count': len(source['removed']),
                                'added_edge_count': len(source['added'])}
                attempt = {'ref_version': version, 'kind': 'TARGETED_CANDIDATE',
                           'status': 'CONSTRUCTED', 'source': metadata}
                exploration['events'].append(attempt)
                if key in own_keys:
                    attempt.update(status='DUPLICATE_SKIPPED', candidate_submitted=False)
                    continue
                own_keys.add(key)
                before = sum(counts.values())
                try:
                    meter.check(reserve=.03)
                    attempt.update(status='EVALUATING', candidate_submitted=True)
                    evaluate(Y, metadata, 'direct')
                    attempt.update(status=('EVALUATED' if sum(counts.values()) > before
                                           else 'EVALUATOR_DUPLICATE_OR_INVALID'),
                                   complete_evaluation_count=sum(counts.values()) - before,
                                   streaming_best_F=best_F())
                except Exception as error:
                    attempt.update(status=_stop_name(error), error=repr(error))
                    raise
                if best_F() > tr.F + Fraction(str(channel.config['epsilon_F'])):
                    limited = True
                    search_stop = 'FIRST_STRICT_IMPROVEMENT_FOR_COMMIT'
                    attempt['search_stop'] = search_stop
                    break
            if search_stop in ('FIRST_STRICT_IMPROVEMENT_FOR_COMMIT',
                               'COMPLETED_EVALUATION_CAP'):
                break
        ref_record.update(status='COMPLETED', policy_limited=limited,
                          search_stop=search_stop,
                          completed_evaluations=sum(counts.values()) - baseline,
                          streaming_best_F=best_F())
    except Exception as error:
        ref_record.update(status=_stop_name(error), error=repr(error),
                          completed_evaluations=sum(counts.values()) - baseline)
        raise
    finally:
        roundinfo['policy_limited'] = limited
        assert sum(counts.values()) <= int(channel.config['L_eval'])
    return limited
