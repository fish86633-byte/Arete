"""V1 exploration: paid all-link marginal reconstruction of one slot.

No background, price coordination, matching, or new queue/physical primitive is
introduced here.  The outer planner owns initialization, completed-candidate
evaluation, strict full-horizon acceptance and reliable full-Trace commits.
This module constructs independent candidates against one fixed reference.
"""
from __future__ import annotations

from fractions import Fraction
import time

import numpy as np

from arete._frozen.pcqr.physics import NumericUnresolved
from arete._frozen.pcqr.search import ScanLimit, utility, value
from arete._frozen.pcqr.support import BudgetExceeded, plan_key

from .local_score import score_slot


POOL_LIMIT = 32
VARIANT = 'V1'
COMMIT_RESERVE_S = .25


def _stop_name(error):
    if isinstance(error, BudgetExceeded):
        return 'TIME_LIMIT'
    if isinstance(error, ScanLimit):
        return 'SCAN_LIMIT'
    if isinstance(error, NumericUnresolved):
        return 'NUMERIC_UNRESOLVED'
    return 'ERROR'


def _reference_value(slot_item, config):
    """Exact actual slot service utility from an already completed Trace."""
    delivered = (sum(int(x) for x in slot_item['service_rt'])
                 + sum(int(x) for x in slot_item['service_nrt']))
    weighted = sum((age + 1) * int(count)
                   for row in slot_item['served_age']
                   for age, count in enumerate(row))
    return value(delivered, weighted, config)


def _rank_slots(X, reference, channel, meter):
    """Pay once for fixed-reference slot priority and explicit candidate pools."""
    records = []
    ids = np.asarray(channel.sat_ids, dtype=np.int64)
    for h in range(channel.H):
        meter.check(reserve=.03)
        with meter.phase('direct_generation'):
            item = reference.slots[h]
            rt, nrt = item['qplus_rt'], item['qplus_nrt']
            edges = np.argwhere(channel.visible[h])
            if len(edges):
                caps = channel.cap(channel.signal[h, edges[:, 0], edges[:, 1]],
                                   channel.config.get('I_ext', 0.0))
                vals = utility(rt[edges[:, 1]], nrt[edges[:, 1]], caps, channel.config)
                isolated = {(int(s), int(m)): (v, int(cap))
                            for (s, m), v, cap in zip(edges, vals, caps)}
            else:
                isolated = {}
            def edge_order(edge):
                s, m = edge
                v, cap = isolated[edge]
                return (-v, -cap, -float(channel.signal[h, s, m]), int(ids[s]), m)
            positive = sorted((edge for edge, (v, _) in isolated.items() if v > 0),
                              key=edge_order)
            original = tuple((int(s), int(m)) for s, m in np.argwhere(X[h]))
            # Original arrangements are legal.  Empty-queue old edges remain
            # explicit pool members but cannot be selected by a positive gain.
            pool = set(positive[:POOL_LIMIT]) | set(original)
            pool = tuple(sorted(pool, key=edge_order))
            potential = sum((isolated[edge][0] for edge in original), Fraction(0))
            actual = _reference_value(item, channel.config)
            records.append({'h': h, 'gap': potential - actual,
                            'reference_actual_V': actual,
                            'reference_isolated_V': potential,
                            'positive_visible_count': len(positive),
                            'pool': pool, 'original': original})
        meter.check(reserve=.03)
    return sorted(records, key=lambda row: (-row['gap'], row['h']))


def _construct_slot(row, reference, channel, meter, scan, event, exploration):
    """Greedily choose maximum strictly positive complete-arrangement gain."""
    h = row['h']
    rt = reference.slots[h]['qplus_rt']
    nrt = reference.slots[h]['qplus_nrt']
    selected = np.zeros((channel.S, channel.M), dtype=bool)
    current_V = Fraction(0)
    ids = np.asarray(channel.sat_ids, dtype=np.int64)
    choices = []
    step = 0
    while True:
        meter.check(reserve=.03)
        remaining = int(channel.config['K']) - selected.sum(axis=1)
        occupied = selected.any(axis=0)
        best = None
        for s, m in row['pool']:
            if selected[s, m] or remaining[s] <= 0 or occupied[m]:
                continue
            meter.check(reserve=.03)
            with meter.phase('direct_generation'):
                scan.take('V1_marginal_score', {'h': h, 's': int(ids[s]), 'm': m,
                                               'construction_step': step})
                tentative = selected.copy()
                tentative[s, m] = True
                event['score_attempts'] += 1
                exploration['score_attempts'] += 1
                score = score_slot(rt, nrt, tentative, channel, h, meter=meter)
                event['score_completed'] += 1
                exploration['score_completed'] += 1
                gain = score.V - current_V
                key = (-gain, int(ids[s]), m)
                if gain > 0 and (best is None or key < best[0]):
                    best = (key, s, m, score.V, gain)
        # A partial pass never chooses its current best when interrupted.  Only
        # completion of the entire fixed feasible pool admits another beam.
        if best is None:
            break
        _, s, m, next_V, gain = best
        selected[s, m] = True
        current_V = next_V
        choices.append({'step': step, 'edge': (int(ids[s]), m),
                        'V_after': current_V, 'gain': gain})
        step += 1
    event['construction_choices'] = choices
    event['slot_V'] = current_V
    return selected, current_V


def search_reference(X, tr, Q0, A, channel, meter, ledger, cycle, result,
                     roundinfo, scan, counts, whole_plan_memo, evaluate,
                     best_F, *, variant):
    """Construct V1 candidates; update streaming best only through evaluate.

    Return the same policy-limited boolean expected by the outer B wrapper.
    Q0/A are never decoded or advanced here: every local score uses the fixed
    reference's arrived planning queue.  No actual future business input is
    accessed.  Invocation-local candidates do not inherit another method's
    cache; whole_plan_memo remains exclusively owned by the outer evaluator.
    """
    if variant == 'V2':
        from .candidate_targeted import search_reference as targeted_search
        return targeted_search(X, tr, Q0, A, channel, meter, ledger, cycle,
                               result, roundinfo, scan, counts,
                               whole_plan_memo, evaluate, best_F, variant=variant)
    if variant != VARIANT:
        raise ValueError('only the frozen V1 exploration mechanism is implemented')
    c = channel.config
    if int(c['L_eval']) != 16:
        raise ValueError('V1 exploration is registered for L_eval=16 only')
    exploration = result.setdefault('exploration', {
        'variant': variant, 'policy': 'ALL_LINK_MARGINAL_SINGLE_SLOT',
        'organization': 'FIRST_STRICT_IMPROVEMENT_THEN_COMMIT',
        'commit_reserve_s': COMMIT_RESERVE_S,
        'pool_limit': POOL_LIMIT, 'score_attempts': 0, 'score_completed': 0,
        'construct_started': 0, 'construct_completed': 0,
        'construct_interrupted': 0, 'references': [], 'events': []})
    if exploration['variant'] != variant:
        raise ValueError('exploration log variant disagrees with this invocation')
    version = int(result['accepted'])
    ref_record = {'ref_version': version, 'status': 'RANKING',
                  'F_reference': tr.F, 'slot_order': [], 'slots': []}
    exploration['references'].append(ref_record)
    roundinfo['search_policy'] = 'V1_ALL_LINK_MARGINAL_SINGLE_SLOT'
    roundinfo['exploration'] = ref_record
    completed_before = sum(counts.values())
    own_keys = {plan_key(X, channel)}
    limited = False
    search_stop = 'SLOT_POOL_EXHAUSTED'
    try:
        rows = _rank_slots(X, tr, channel, meter)
        ref_record['slot_order'] = [row['h'] for row in rows]
        ref_record['slots'] = [{key: value for key, value in row.items() if key != 'pool'}
                               | {'pool_count': len(row['pool'])}
                              for row in rows]
        ref_record['status'] = 'CONSTRUCTING'
        for order_index, row in enumerate(rows):
            if sum(counts.values()) >= int(c['L_eval']):
                limited = True
                search_stop = 'COMPLETED_EVALUATION_CAP'
                break
            # This is a declared V1 search-organization rule, not additional
            # wall budget and not a change to the outer reliable-output path.
            # A cooperative reserve permits an already completed improvement
            # to enter full verification before starting another construction.
            if meter.remaining <= COMMIT_RESERVE_S:
                limited = True
                search_stop = 'COMMIT_RESERVE_STOP'
                exploration['events'].append({
                    'ref_version': version, 'status': 'COMMIT_RESERVE_STOP',
                    'next_slot': row['h'], 'priority_index': order_index,
                    'remaining_s': meter.remaining,
                    'commit_reserve_s': COMMIT_RESERVE_S,
                    'candidate_submitted': False})
                break
            meter.check(reserve=.03)
            h = row['h']
            event = {'ref_version': version, 'slot': h, 'priority_index': order_index,
                     'status': 'CONSTRUCTING', 'gap': row['gap'],
                     'reference_actual_V': row['reference_actual_V'],
                     'reference_isolated_V': row['reference_isolated_V'],
                     'pool_count': len(row['pool']),
                     'pool': [(int(channel.sat_ids[s]), m) for s, m in row['pool']],
                     'positive_visible_count': row['positive_visible_count'],
                     'score_attempts': 0, 'score_completed': 0,
                     'construction_choices': []}
            exploration['events'].append(event)
            exploration['construct_started'] += 1
            started = time.perf_counter()
            cpu_started = time.process_time()
            try:
                arrangement, V = _construct_slot(row, tr, channel, meter, scan,
                                                 event, exploration)
                with meter.phase('direct_generation'):
                    Y = X.copy()
                    Y[h] = arrangement
                    difference = Y != X
                    changed_slots = np.flatnonzero(np.any(difference, axis=(1, 2)))
                    changed_satellites = np.flatnonzero(np.any(difference, axis=(0, 2)))
                    key = plan_key(Y, channel)
                    event.update(status='CONSTRUCTED', V_slot_delta=V - row['reference_actual_V'],
                                 changed_satellites=[int(channel.sat_ids[s]) for s in changed_satellites],
                                 changed_satellite_count=len(changed_satellites),
                                 changed_slots=changed_slots.tolist(),
                                 changed_slot_count=len(changed_slots),
                                 first_changed=int(changed_slots[0]) if len(changed_slots) else None,
                                 true_two_slot_change=len(changed_slots) >= 2,
                                 added_edge_count=int(np.count_nonzero(Y & ~X)),
                                 removed_edge_count=int(np.count_nonzero(X & ~Y)))
                exploration['construct_completed'] += 1
            except Exception as error:
                event.update(status=_stop_name(error), error=repr(error),
                             candidate_submitted=False)
                exploration['construct_interrupted'] += 1
                raise
            finally:
                event['construction_wall_s'] = time.perf_counter() - started
                event['construction_cpu_s'] = time.process_time() - cpu_started
            if key in own_keys:
                event.update(status='DUPLICATE_SKIPPED', candidate_submitted=False)
                continue
            own_keys.add(key)
            source = {'type': 'ALL_LINK_MARGINAL_SLOT', 'variant': variant,
                      'slot': h, 'priority_index': order_index,
                      'reference_gap': row['gap'],
                      'reference_actual_V': row['reference_actual_V'],
                      'reference_isolated_V': row['reference_isolated_V'],
                      'V_slot_delta': event['V_slot_delta'],
                      'candidate_pool_count': len(row['pool']),
                      'candidate_pool': event['pool'],
                      'score_attempts': event['score_attempts'],
                      'score_completed': event['score_completed'],
                      'changed_satellite_count': event['changed_satellite_count'],
                      'changed_slot_count': event['changed_slot_count'],
                      'first_changed': event['first_changed'],
                      'true_two_slot_change': event['true_two_slot_change'],
                      'added_edge_count': event['added_edge_count'],
                      'removed_edge_count': event['removed_edge_count']}
            before = sum(counts.values())
            try:
                # Negative local slot delta is still a candidate: a changed
                # suffix can recover value that this local ranking cannot see.
                meter.check(reserve=.03)
                event.update(status='EVALUATING', candidate_submitted=True)
                evaluate(Y, source, 'direct')
                event.update(status=('EVALUATED' if sum(counts.values()) > before
                                     else 'EVALUATOR_DUPLICATE_OR_INVALID'),
                             complete_evaluation_count=sum(counts.values()) - before,
                             streaming_best_F=best_F())
            except Exception as error:
                event.update(status=_stop_name(error), error=repr(error))
                raise
            if best_F() > tr.F + Fraction(str(c['epsilon_F'])):
                # The candidate is not reliable yet.  Return to the unchanged
                # outer full-Trace verification immediately; that owner alone
                # decides whether it can be committed or must be discarded.
                limited = True
                search_stop = 'FIRST_STRICT_IMPROVEMENT_FOR_COMMIT'
                event['search_stop'] = search_stop
                break
        ref_record.update(status='COMPLETED', policy_limited=limited,
                          search_stop=search_stop,
                          completed_evaluations=sum(counts.values()) - completed_before,
                          streaming_best_F=best_F())
    except Exception as error:
        ref_record.update(status=_stop_name(error), error=repr(error),
                          completed_evaluations=sum(counts.values()) - completed_before)
        raise
    finally:
        roundinfo['policy_limited'] = limited
        assert sum(counts.values()) <= int(c['L_eval'])
    return limited
