"""Frozen V3/V4 finite neighborhoods using one paid union-power slot scorer.

V3 retains exploration_007 V2 decisions and limits, replacing only repeated
local physical/service scoring.  V4 uses the same neighborhood but changes
top-four consumption to R1,V1,R2,V2 and delegates constrained acceptance to its
own wrapper.  Neither variant executes the old backgrounds or price mechanism.
"""
from __future__ import annotations

from fractions import Fraction
import time

import numpy as np

from arete._frozen.pcqr.support import plan_key
from arete._frozen.exploration_007.candidate_search import COMMIT_RESERVE_S, _rank_slots, _stop_name
from arete._frozen.exploration_007.candidate_targeted import _neighbors, _edge_keys, TYPE_ORDER

from .fast_score import SlotScorer


NEIGHBORHOOD_LIMIT = 128
TOP_COMPLETE_CANDIDATES = 4
CRITICAL_RT_BUCKETS = 12


def _neighbor_pool(X, row, channel, meter, event):
    """Generate the identical fixed three-flow V2 neighborhood before scoring."""
    h = row['h']
    original = X[h]
    seen = {original.tobytes()}
    pool = []
    iterator = iter(_neighbors(original, row, channel, meter))
    while len(pool) < NEIGHBORHOOD_LIMIT:
        meter.check(reserve=.03)
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
        if (np.any(arrangement & ~channel.visible[h])
                or np.any(arrangement.sum(axis=1) > int(channel.config['K']))
                or np.any(arrangement.sum(axis=0) > 1)):
            raise AssertionError('008 neighborhood emitted an illegal arrangement')
        seen.add(key)
        source.update(removed=_edge_keys(original & ~arrangement, channel, h),
                      added=_edge_keys(arrangement & ~original, channel, h))
        pool.append({'arrangement': arrangement, 'source': source, 'key': key})
    event['distinct_generated_count'] = len(pool)
    event['neighborhood_cap_reached'] = len(pool) >= NEIGHBORHOOD_LIMIT
    return pool


def _ordinary_rank(candidate):
    source = candidate['source']
    return (-source['V_slot_delta'], TYPE_ORDER[source['type']],
            source['removed'], source['added'])


def _critical_rank(candidate):
    source = candidate['source']
    return (-source['critical_RT_service_delta'],
            -source['normalized_service_delta'],
            -source['V_slot_delta'], TYPE_ORDER[source['type']],
            source['removed'], source['added'])


def _select_candidates(pool, variant):
    """At most four: V3 original V order; V4 fixed R1,V1,R2,V2 order."""
    ordered_v = sorted(pool, key=_ordinary_rank)
    if variant == 'V3':
        for rank, candidate in enumerate(ordered_v[:TOP_COMPLETE_CANDIDATES]):
            candidate['source'].update(consumer_stream='V', consumer_rank=rank)
        return ordered_v[:TOP_COMPLETE_CANDIDATES]
    ordered_r = sorted(pool, key=_critical_rank)
    streams = [iter(ordered_r), iter(ordered_v)]
    names, ranks, live = ['R', 'V'], [0, 0], [True, True]
    seen, out = set(), []
    while any(live) and len(out) < TOP_COMPLETE_CANDIDATES:
        for index, stream in enumerate(streams):
            if not live[index] or len(out) >= TOP_COMPLETE_CANDIDATES:
                continue
            while True:
                try:
                    candidate = next(stream)
                except StopIteration:
                    live[index] = False
                    break
                rank = ranks[index]
                ranks[index] += 1
                if candidate['key'] in seen:
                    continue
                seen.add(candidate['key'])
                candidate['source'].update(consumer_stream=names[index],
                                            consumer_rank=rank)
                out.append(candidate)
                break
    return out


def _reference_proxy(item, denominator):
    critical = sum(int(count) for row in item['served_age'][:, -CRITICAL_RT_BUCKETS:]
                   for count in row)
    normalized = sum((Fraction(int(rt) + int(nrt), denominator[m])
                      for m, (rt, nrt) in enumerate(zip(item['service_rt'], item['service_nrt']))
                      if denominator[m] > 0 and (int(rt) + int(nrt)) > 0), Fraction(0))
    return critical, normalized


def _score_pool(pool, X, row, tr, channel, meter, scan, event, exploration,
                variant, denominator):
    """One paid union build, then compact exact scoring of the fixed pool."""
    h = row['h']
    item = tr.slots[h]
    event['union_prebuild_status'] = 'RUNNING'
    build_started, build_cpu_started = time.perf_counter(), time.process_time()
    scorer = None
    try:
        # Construction is invocation-local and includes the entire reference
        # plus already generated neighbors; there is no inherited free table.
        scorer = SlotScorer(item['qplus_rt'], item['qplus_nrt'], channel, h,
                            [X[h]] + [candidate['arrangement'] for candidate in pool],
                            meter=meter)
        event['union_prebuild_status'] = 'COMPLETED'
    except Exception as error:
        event['union_prebuild_status'] = _stop_name(error)
        raise
    finally:
        event['union_prebuild_wall_s'] = time.perf_counter() - build_started
        event['union_prebuild_cpu_s'] = time.process_time() - build_cpu_started
    reference_critical, reference_normalized = ((None, None) if variant == 'V3'
                                                else _reference_proxy(item, denominator))
    try:
        for candidate in pool:
            meter.check(reserve=.03)
            source = candidate['source']
            scan.take('008_compact_local_score', {
                'variant': variant, 'h': h, 'type': source['type'],
                'removed': source['removed'], 'added': source['added']})
            event['score_attempts'] += 1
            exploration['score_attempts'] += 1
            score = scorer.score(candidate['arrangement'])
            event['score_completed'] += 1
            exploration['score_completed'] += 1
            source['V_slot_delta'] = score.V - row['reference_actual_V']
            source['local_evaluator'] = 'PAID_UNION_POWER_COMPACT_EXACT_SERVICE'
            if variant == 'V4':
                normalized = sum((Fraction(int(rt) + int(nrt), denominator[m])
                                  for m, (rt, nrt) in enumerate(zip(score.service_rt,
                                                                   score.service_nrt))
                                  if denominator[m] > 0 and (int(rt) + int(nrt)) > 0),
                                 Fraction(0))
                source['critical_RT_service_delta'] = (int(score.critical_RT_served)
                                                        - reference_critical)
                source['normalized_service_delta'] = normalized - reference_normalized
        event['distinct_scored_count'] = len(pool)
    finally:
        # This table's lifetime is one paid slot preparation.  Retention and
        # cleanup of the scorer are inside the caller's measured phase; its
        # diagnostics are children, never separately added to Meter costs.
        event['fast_score_stats'] = scorer.stats()
        event['fast_score_stats_accounting'] = 'NESTED_IN_DIRECT_GENERATION'


def search_reference(X, tr, Q0, A, channel, meter, ledger, cycle, result,
                     roundinfo, scan, counts, whole_plan_memo, evaluate,
                     best_F, *, variant):
    if variant not in ('V3', 'V4') or int(channel.config['L_eval']) != 16:
        raise ValueError('008 requires V3/V4 and L_eval=16')
    exploration = result.setdefault('exploration', {
        'variant': variant, 'policy': 'TARGETED_NEIGHBORHOOD_UNION_POWER',
        'organization': 'FIRST_ELIGIBLE_STRICT_IMPROVEMENT_THEN_FULL_COMMIT',
        'commit_reserve_s': COMMIT_RESERVE_S, 'pool_limit': 32,
        'neighborhood_limit': NEIGHBORHOOD_LIMIT,
        'top_complete_candidates': TOP_COMPLETE_CANDIDATES,
        'score_attempts': 0, 'score_completed': 0,
        'construct_started': 0, 'construct_completed': 0,
        'construct_interrupted': 0, 'references': [], 'events': []})
    if exploration['variant'] != variant:
        raise ValueError('008 exploration log variant differs')
    exploration['consumer_policy'] = ('LOCAL_V_FIRST_FOUR' if variant == 'V3' else 'R1,V1,R2,V2_DEDUP_REFILL_IN_SAME_STREAM')
    if variant == 'V4':
        exploration['R_sort'] = ['-critical_RT_service_delta_last12',
                                 '-normalized_service_delta', '-V_slot_delta',
                                 'type_order', 'removed_global_edges', 'added_global_edges']
        exploration['V_sort'] = ['-V_slot_delta', 'type_order',
                                 'removed_global_edges', 'added_global_edges']
        exploration['internal_proxy_denominator'] = 'Q0_alive_RT+Q0_NRT+SUM_FIXED_PLANNING_A'
        exploration['internal_proxy_is_not_F'] = True
    denominator = result.get('fixed_planning_demand') if variant == 'V4' else None
    if variant == 'V4' and denominator is None:
        raise ValueError('V4 fixed planning demand must be prepared by its wrapper')
    version = int(result['accepted'])
    ref_record = {'ref_version': version, 'F_reference': tr.F,
                  'status': 'RANKING', 'slot_order': [], 'slots': []}
    exploration['references'].append(ref_record)
    roundinfo['search_policy'] = f'EXPLORATION_008_{variant}_TARGETED_UNION_POWER'
    roundinfo['exploration'] = ref_record
    baseline = sum(counts.values())
    own_keys = {plan_key(X, channel)}
    limited, search_stop = False, 'SLOT_POOL_EXHAUSTED'
    try:
        rows = _rank_slots(X, tr, channel, meter)
        ref_record['slot_order'] = [row['h'] for row in rows]
        ref_record['slots'] = [{key: value for key, value in row.items() if key != 'pool'}
                               | {'pool_count': len(row['pool'])} for row in rows]
        ref_record['status'] = 'PREPARING_NEIGHBORHOODS'
        for priority_index, row in enumerate(rows):
            if sum(counts.values()) >= int(channel.config['L_eval']):
                limited, search_stop = True, 'COMPLETED_EVALUATION_CAP'
                break
            if meter.remaining <= COMMIT_RESERVE_S:
                limited, search_stop = True, 'COMMIT_RESERVE_STOP'
                exploration['events'].append({
                    'ref_version': version, 'status': search_stop,
                    'next_slot': row['h'], 'remaining_s': meter.remaining,
                    'commit_reserve_s': COMMIT_RESERVE_S, 'candidate_submitted': False})
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
                with meter.phase('direct_generation'):
                    pool = _neighbor_pool(X, row, channel, meter, event)
                    _score_pool(pool, X, row, tr, channel, meter, scan, event,
                                exploration, variant, denominator)
                    candidates = _select_candidates(pool, variant)
                    event['selected_top_count'] = len(candidates)
                    event['ranked_top'] = [candidate['source'] for candidate in candidates]
                event['status'] = 'NEIGHBORHOOD_COMPLETED'
                exploration['construct_completed'] += 1
            except Exception as error:
                event.update(status=_stop_name(error), error=repr(error), candidate_submitted=False)
                exploration['construct_interrupted'] += 1
                raise
            finally:
                event['construction_wall_s'] = time.perf_counter() - started
                event['construction_cpu_s'] = time.process_time() - cpu_started
            for local_rank, candidate in enumerate(candidates):
                if sum(counts.values()) >= int(channel.config['L_eval']):
                    limited, search_stop = True, 'COMPLETED_EVALUATION_CAP'
                    break
                with meter.phase('direct_generation'):
                    Y = X.copy()
                    Y[h] = candidate['arrangement']
                    key = plan_key(Y, channel)
                    difference = Y != X
                    changed_satellites = np.flatnonzero(np.any(difference, axis=(0, 2)))
                    changed_slots = np.flatnonzero(np.any(difference, axis=(1, 2)))
                    source = candidate['source']
                    metadata = {**source, 'variant': variant, 'slot': h,
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
                    limited, search_stop = True, 'FIRST_ELIGIBLE_STRICT_IMPROVEMENT_FOR_COMMIT'
                    attempt['search_stop'] = search_stop
                    break
            if search_stop in ('FIRST_ELIGIBLE_STRICT_IMPROVEMENT_FOR_COMMIT',
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
