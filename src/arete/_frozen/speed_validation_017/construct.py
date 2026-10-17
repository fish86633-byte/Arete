"""Private constructor copy of 010; original source and default remain untouched.

Every original meter check, pool rebuild, slot confirmation, queue transition,
reserve stop and proposal suffix rule is retained. Only the score-selection
interface selects the U/I/UI computation implementation. R uses the old source.
"""
from fractions import Fraction
import hashlib
import numpy as np

from arete._frozen.confirmation_010.marginal_fast import FastM1Scorer
from arete._frozen.exploration_009.marginal import (_ReserveStop, _check, _confirm, fixed_demand,
    fixed_pool, RESERVE_S, CRITICAL_RT_BUCKETS)
from arete._frozen.pcqr.engine import slot
from arete._frozen.pcqr.search import value
from arete._frozen.pcqr.support import BudgetExceeded, state_hash
from arete._frozen.speed_validation_017.upper import UpperScorer, UpperBoundMixin

def construct_proposal(Q0, A, channel, tpg_plan, meter, *, variant, abs_start=0,
                       implementation="U", oracle_callback=None):
    """Private same-decision construction; only the score winner interface differs."""
    if variant != 'M1-E':
        raise ValueError('private speed validation supports only frozen M1-E')
    if implementation == 'U':
        scorer_class = UpperScorer
    elif implementation == 'I':
        from arete._frozen.speed_validation_017.interference import ReuseScorer
        scorer_class = ReuseScorer
    elif implementation == 'UI':
        from arete._frozen.speed_validation_017.interference import ReuseScorer
        class CombinedScorer(UpperBoundMixin, ReuseScorer):
            pass
        scorer_class = CombinedScorer
    else:
        raise ValueError('U, I or UI required; R must use the original constructor')
    proposal = np.asarray(tpg_plan, dtype=bool).copy()
    state = Q0.copy()
    denominator = fixed_demand(Q0, A)
    prefix_served = [0] * channel.M
    positive = [demand for demand in denominator if demand]
    mean_demand = Fraction(sum(positive), len(positive)) if positive else Fraction(0)
    a, b = value(1, 0, channel.config), value(0, 1, channel.config)
    events = []
    record = {'variant': variant, 'status': 'RUNNING', 'stop': None,
              'prefix_complete_slots': 0, 'slots_started': 0, 'slots': events,
              'fixed_planning_demand': denominator, 'mean_positive_demand': mean_demand,
              'M2_extra_coefficients': [1, 1], 'critical_RT_buckets': CRITICAL_RT_BUCKETS,
              'a': a, 'b': b, 'commit_reserve_s': RESERVE_S,
              'prefix_served': prefix_served, 'candidate_rows_attempted': 0,
              'candidate_rows_completed': 0, 'batch_calls': 0,
              'tie_rule': 'largest_exact_delta_then_global_satellite_id_then_cell',
              'proxy_is_not_full_F': True, 'intermediate_H_replays': 0,
              'optimization': 'SPARSE_ACTIVE_ROWS_INTEGER_EXACT_UTILITY',
              'externality_loss_in_marginal': variant == 'M1-E',
              'private_implementation': implementation, 'score_selection_rule_unchanged': True}
    try:
        for h in range(channel.H):
            _check(meter)
            event = {'h': h, 'status': 'RUNNING', 'additions': [], 'slot_confirmed': False}
            events.append(event)
            record['slots_started'] += 1
            event['absolute_slot'] = abs_start + h
            event['input_state_hash'] = state_hash(state)
            state.add(A[h], abs_start + h)
            nrt = state.nrt
            event.update(arrival_add_calls=1, arrival_applied_before_service=True,
                         qplus_rt_sha256=hashlib.sha256(state.rt.tobytes()).hexdigest(),
                         qplus_nrt_sha256=hashlib.sha256(nrt.tobytes()).hexdigest())
            edges, pool_record = fixed_pool(state.rt, nrt, channel, h, tpg_plan[h],
                                            denominator, prefix_served)
            event['pool'] = pool_record
            scorer = None
            reserve_stop = False
            try:
                scorer = scorer_class(state.rt, nrt, channel, h, edges,
                                      meter=meter, reserve=True)
                if hasattr(scorer, 'oracle_callback'):
                    scorer.oracle_callback = oracle_callback
                selected = []
                arrangement = np.zeros((channel.S, channel.M), dtype=bool)
                current_sparse = scorer.empty_sparse()
                current_score = scorer.materialize_sparse(current_sparse)
                current_proxy_numerator = 0
                current_proxy = scorer.exact_value(current_proxy_numerator)
                while True:
                    _check(meter)
                    satellite_counts = arrangement.sum(axis=1)
                    occupied = arrangement.any(axis=0)
                    candidates = [i for i, (s, m) in enumerate(scorer.edges)
                                  if not occupied[m] and satellite_counts[s] < int(channel.config['K'])]
                    if not candidates:
                        event['fill_stop'] = 'NO_FEASIBLE_POOL_EDGE'
                        break
                    if hasattr(scorer, 'select_addition'):
                        winner = scorer.select_addition(selected, candidates, current_sparse)
                        event.setdefault('selection_certificates', []).append(
                            scorer.last_selection_certificate())
                        if winner is None:
                            event['fill_stop'] = 'NO_POSITIVE_EXACT_MARGINAL'
                            break
                        index, score = winner
                        delta = score.numerator - current_sparse.numerator
                        s, m = scorer.edges[index]
                        best = (-delta, int(channel.sat_ids[s]), int(m), index, score)
                    else:
                        scores = scorer.score_additions_sparse(selected, candidates)
                        ranked = []
                        for index, score in zip(candidates, scores):
                            delta = score.numerator - current_sparse.numerator
                            s, m = scorer.edges[index]
                            ranked.append((-delta, int(channel.sat_ids[s]), int(m), index, score))
                        best = min(ranked, key=lambda row: row[:3])
                        if oracle_callback is not None:
                            winner = (best[3], best[4]) if -best[0] > 0 else None
                            oracle_callback(scorer, np.asarray(selected, dtype=np.int64),
                                np.asarray(candidates, dtype=np.int64), current_sparse,
                                winner, {'policy': 'NO_SCREENING_I', 'bounds': []})
                    if -best[0] <= 0:
                        event['fill_stop'] = 'NO_POSITIVE_EXACT_MARGINAL'
                        break
                    _, gid, m, index, current_sparse = best
                    current_score = scorer.materialize_sparse(current_sparse)
                    current_proxy_numerator = (current_sparse.numerator if variant == 'M1-E'
                                               else current_proxy_numerator - best[0])
                    current_proxy = scorer.exact_value(current_proxy_numerator)
                    s = int(scorer.edges[index, 0])
                    selected.append(index)
                    arrangement[s, m] = True
                    event['additions'].append({'global_satellite_id': gid, 'cell': m,
                        'delta_proxy': scorer.exact_value(-best[0]), 'proxy': current_proxy,
                        'actual_V': current_score.V})
            except _ReserveStop:
                reserve_stop = True
                event['fill_stop'] = 'COMMIT_RESERVE_STOP'
                if scorer is None:
                    event['status'] = 'RESERVE_BEFORE_SCORER_COMPLETE'
                    raise
            finally:
                if scorer is not None:
                    event['fast_score_stats'] = scorer.stats()
                    for key in ('candidate_rows_attempted', 'candidate_rows_completed', 'batch_calls'):
                        record[key] += event['fast_score_stats'][key]
            meter.check()
            item = slot(state, arrangement, channel, h, abs_start + h)
            meter.replay('009_construction_slot', 1)
            _confirm(current_score, scorer, item)
            proposal[h] = arrangement
            for m, (rt, nonrt) in enumerate(zip(item['service_rt'], item['service_nrt'])):
                prefix_served[m] += int(rt) + int(nonrt)
            event.update(status='COMPLETED', slot_confirmed=True,
                         delivered=current_score.delivered, actual_V=current_score.V,
                         weighted_served_rt=current_score.weighted_served_rt,
                         critical_RT_served=current_score.critical_RT_served,
                         output_state_hash=state_hash(state),
                         service_rt=item['service_rt'].tolist(),
                         service_nrt=item['service_nrt'].tolist(),
                         expired=item['expired'].tolist(),
                         service_rt_sha256=hashlib.sha256(item['service_rt'].tobytes()).hexdigest(),
                         service_nrt_sha256=hashlib.sha256(item['service_nrt'].tobytes()).hexdigest(),
                         expired_sha256=hashlib.sha256(item['expired'].tobytes()).hexdigest())
            record['prefix_complete_slots'] += 1
            if reserve_stop:
                record.update(status='PARTIAL_PREFIX', stop='COMMIT_RESERVE_STOP')
                break
        else:
            record.update(status='COMPLETED', stop='HORIZON_COMPLETED')
    except _ReserveStop:
        record.update(status='PARTIAL_PREFIX', stop='COMMIT_RESERVE_STOP')
    except BudgetExceeded:
        record.update(status='PARTIAL_PREFIX', stop='TIME_LIMIT')
        if events and not events[-1]['slot_confirmed']:
            events[-1]['status'] = 'TIME_LIMIT_SLOT_DISCARDED'
    except Exception as error:
        record.update(status='ERROR', stop='CORRECTNESS_FAILURE', error=repr(error),
                      error_details=getattr(error, 'details', None))
        error.construction_record = record
        error.proposal_plan = proposal
        raise
    return proposal, record
