"""Paid own-TPG fallback and one fully verified M1-E/M1-NX proposal."""
from __future__ import annotations

from fractions import Fraction
import hashlib
import time

import numpy as np

from arete._frozen.pcqr.support import Meter, Ledger, BudgetExceeded, plan_key, state_hash, sha, resolve_stop
from arete._frozen.pcqr.engine import replay, validate
from arete._frozen.pcqr.physics import NumericUnresolved
from arete._frozen.pcqr.baselines import greedy
from arete._frozen.pcqr.preparation import register_silent, prepare_inputs
from arete._frozen.exploration_008.candidate_wrapper import trace_summary, satisfaction_summary
from .marginal_fast import construct_proposal, fixed_demand, RESERVE_S


def _full_summary(trace, denominator):
    summary = trace_summary(trace)
    if trace is not None:
        summary.update(weighted_wait=int(trace.weighted_wait),
                       service_rt_packets=sum(int(x) for item in trace.slots for x in item['service_rt']),
                       service_nrt_packets=sum(int(x) for item in trace.slots for x in item['service_nrt']),
                       per_cell_service=[sum(int(item['service_rt'][m]) + int(item['service_nrt'][m])
                                             for item in trace.slots) for m in range(len(denominator))],
                       planning_proxy=satisfaction_summary(trace, denominator),
                       full_trace_prefix=[{'h': h, 'absolute_slot': item['absolute_slot'],
                           'input_state_hash': state_hash(trace.states[h]),
                           'output_state_hash': state_hash(trace.states[h + 1]),
                           'qplus_rt_sha256': hashlib.sha256(item['qplus_rt'].tobytes()).hexdigest(),
                           'qplus_nrt_sha256': hashlib.sha256(item['qplus_nrt'].tobytes()).hexdigest(),
                           'service_rt_sha256': hashlib.sha256(item['service_rt'].tobytes()).hexdigest(),
                           'service_nrt_sha256': hashlib.sha256(item['service_nrt'].tobytes()).hexdigest(),
                           'expired_sha256': hashlib.sha256(item['expired'].tobytes()).hexdigest()}
                           for h, item in enumerate(trace.slots)])
    return summary


def plan(Q0, A, channel, cycle=0, meter=None, safety_fallback=None, variant='M1-E'):
    if variant not in ('M1-E', 'M1-NX'):
        raise ValueError('CONFIRMATION_010 requires M1-E or M1-NX')
    c = channel.config
    if c.get('r36_batch_arrivals', False):
        raise ValueError('R3.6 must remain disabled')
    if int(c['Q_max']) != 1 or int(c['L_eval']) != 1:
        raise ValueError('M1-E/M1-NX require Q_max=1 and L_eval=1')
    meter = meter or Meter(c['planning_budget_s'])
    X, registration = register_silent(channel, safety_fallback)
    ledger = Ledger(meter)
    result = {'method': 'CONFIRMATION_010:' + variant, 'variant': variant, 'cycle': cycle,
              'run_status': 'OK', 'stop': None, 'accepted': 0, 'candidates': [],
              'rounds': [], 'solver_calls': [], 'messages': ledger.records,
              'requested_initial_source': 'TPG', 'initial_source': None,
              'initial_completed': False, 'initial': trace_summary(None), 'initial_key': None,
              'final': trace_summary(None), 'returned_initial': None,
              'fallback_used': False, 'fallback_source': None, 'fallback_reason': None,
              'silent_registration': registration, 'output_published': False,
              'proposal_plan': None, 'proposal_full': trace_summary(None),
              'proposal': {'construction': None, 'evaluation_status': 'NOT_STARTED',
                           'adopted': False, 'rejection_reason': None,
                           'trace_summary': trace_summary(None), 'full_replay_attempts': 0,
                           'full_replay_completed': 0, 'full_replay_interrupted': 0},
              'prototype_evaluation': {'light_completed': 0, 'light_interrupted': 0,
                  'light_cache_hits': 0, 'full_commit_completed': 0,
                  'full_commit_interrupted': 0, 'unverified_best_discarded': 0, 'events': []},
              'prototype': {'variant': variant, 'experiment_family': 'CONFIRMATION_010',
                  'candidate_evaluator': 'ORIGINAL_FULL_REPLAY_ONCE',
                  'reliable_commit': 'ORIGINAL_FULL_REPLAY_EXACT_CHECK',
                  'candidate_order': ('FORWARD_ALL_LINK_EXACT_MARGINAL' if variant == 'M1-E' else 'FORWARD_NEW_LINK_ONLY_TRUE_RECEIVER_CAPACITY'),
                  'nrt_fifo_hash_available_for_light': False},
              'acceptance_policy': {'name': 'ORIGINAL_STRICT_F',
                  'comparison_reference': 'OWN_PAID_COMPLETED_TPG_FULL_TRACE',
                  'objective_definition_unchanged': True,
                  'maximum_proposal_full_replays': 1, 'maximum_accepted_updates': 1,
                  'construction_commit_reserve_s': RESERVE_S}}
    init = init_tr = tr = None
    denominator = None
    stops = []
    reliable_source = 'ALL_SILENT_REGISTERED'
    operation = 'PREPARATION'
    try:
        if c.get('r3_engineering', False) and hasattr(channel, 'begin_engineering_cache'):
            channel.begin_engineering_cache()
        meter.preparation.start()
        try:
            meter.check()
            Q0, A = prepare_inputs(Q0, A, channel, cycle, meter, ledger)
            meter.check()
        finally:
            meter.preparation.stop()
        operation = 'INITIALIZATION'
        X, tr = greedy(Q0, A, channel, 'TPG', meter, abs_start=cycle * channel.H)
        init, init_tr = X.copy(), tr
        reliable_source = 'TPG_REPLAYED'
        result.update(initial_completed=True, initial_source='TPG',
                      initial=trace_summary(tr), initial_key=plan_key(X, channel))
        with meter.phase('direct_generation'):
            denominator = fixed_demand(Q0, A)
            result['fixed_planning_demand'] = denominator
            result['initial_planning_proxy'] = satisfaction_summary(tr, denominator)
            result['original_own_TPG'] = {**_full_summary(tr, denominator),
                'planning_proxy': result['initial_planning_proxy'],
                'plan_key': result['initial_key'], 'paid_in_this_invocation': True}
        operation = 'RECONSTRUCTION'
        started, cpu_started = time.perf_counter(), time.process_time()
        try:
            with meter.phase('direct_generation'):
                Y, construction = construct_proposal(Q0, A, channel, init, meter,
                    variant=variant, abs_start=cycle * channel.H)
            result['proposal_plan'] = Y
            result['proposal']['construction'] = construction
        except Exception as error:
            result['proposal']['construction'] = getattr(error, 'construction_record', None)
            result['proposal_plan'] = getattr(error, 'proposal_plan', None)
            raise
        finally:
            result['proposal']['construction_wall_s'] = time.perf_counter() - started
            result['proposal']['construction_cpu_s'] = time.process_time() - cpu_started
        proposal = result['proposal']
        proposal['plan_key'] = plan_key(Y, channel)
        proposal['plan_hash'] = sha(repr(proposal['plan_key']).encode())
        if construction['stop'] == 'TIME_LIMIT' or meter.remaining <= 0:
            proposal.update(evaluation_status='TIME_LIMIT_BEFORE_FULL_REPLAY',
                            rejection_reason='UNVERIFIED_PROPOSAL_TIME_LIMIT')
            stops.append('TIME_LIMIT')
            result['fallback_reason'] = 'UNVERIFIED_PROPOSAL_TIME_LIMIT'
        elif np.array_equal(Y, init):
            proposal.update(evaluation_status='DUPLICATE_TPG', rejection_reason='UNCHANGED_PLAN',
                            trace_summary=trace_summary(init_tr),
                            planning_proxy=result['initial_planning_proxy'])
            result['proposal_full'] = result['original_own_TPG'].copy()
            stops.append('POOL_NO_IMPROVEMENT')
        else:
            operation = 'PROPOSAL_FULL_VERIFICATION'
            event = {'kind': 'PROPOSAL_FULL_VERIFICATION', 'status': 'RUNNING',
                     'requested_slots': channel.H, 'propagated_slots': 0,
                     'ref_version': 0, 'plan_hash': proposal['plan_hash']}
            result['prototype_evaluation']['events'].append(event)
            proposal['full_replay_attempts'] = 1
            started, cpu_started = time.perf_counter(), time.process_time()
            try:
                meter.check()
                with meter.phase('final_verification'):
                    meter.replay('009_proposal_full', channel.H)
                    verified = replay(Y, Q0, A, channel, abs_start=cycle * channel.H)
                event.update(status='COMPLETED', propagated_slots=channel.H)
                proposal['full_replay_completed'] = 1
                proposal['evaluation_status'] = 'COMPLETED_FULL_REPLAY'
                result['prototype_evaluation']['full_commit_completed'] = 1
                with meter.phase('final_verification'):
                    proposal['trace_summary'] = trace_summary(verified)
                    proposal['planning_proxy'] = satisfaction_summary(verified, denominator)
                    result['proposal_full'] = _full_summary(verified, denominator)
                    improved = verified.F > init_tr.F + Fraction(str(c['epsilon_F']))
                    proposal['delta_F'] = verified.F - init_tr.F
                    proposal['adopted'] = bool(improved)
                    proposal['rejection_reason'] = None if improved else 'F_NOT_STRICTLY_IMPROVED'
                    source = {'type': 'ONE_FORWARD_RECONSTRUCTION', 'variant': variant,
                              'prefix_complete_slots': construction['prefix_complete_slots'],
                              'construction_stop': construction['stop'],
                              'net_R': plan_key(init & ~Y, channel),
                              'net_A': plan_key(Y & ~init, channel)}
                    result['candidates'].append({'ref_version': 0, 'source': source, 'lane': 'direct',
                        'status': 'EVALUATED', 'plan_hash': proposal['plan_hash'],
                        'plan': proposal['plan_key'], 'delta_F': proposal['delta_F'],
                        'eligible': bool(improved), 'eligibility_required': True,
                        'rejection_reasons': [] if improved else ['F_NOT_STRICTLY_IMPROVED'],
                        'planning_proxy': proposal['planning_proxy'], **trace_summary(verified)})
                if improved:
                    X, tr = Y, verified
                    reliable_source = 'PROPOSAL_FULL_REPLAYED'
                    result['accepted'] = 1
                    result['rounds'].append({'ref_version': 0, 'F_before': init_tr.F,
                        'F_best': verified.F, 'improved': True, 'counts': {'direct': 1, 'match': 0},
                        'scan_count': 0, 'scan_records': [],
                        'truncated': construction['prefix_complete_slots'] < channel.H,
                        'full_replay_verified': True, 'light_best_verified': True})
                    stops.append('Q_MAX')
                else:
                    stops.append('POOL_NO_IMPROVEMENT')
            except BudgetExceeded:
                event['status'] = 'TIME_LIMIT'
                proposal.update(evaluation_status='TIME_LIMIT_BEFORE_FULL_REPLAY',
                                rejection_reason='UNVERIFIED_PROPOSAL_TIME_LIMIT',
                                full_replay_interrupted=1)
                result['prototype_evaluation']['full_commit_interrupted'] = 1
                stops.append('TIME_LIMIT')
                result['fallback_reason'] = 'UNVERIFIED_PROPOSAL_TIME_LIMIT'
            except Exception as error:
                event.update(status='ERROR', error=repr(error))
                proposal['full_replay_interrupted'] = 1
                result['prototype_evaluation']['full_commit_interrupted'] = 1
                raise
            finally:
                event['wall_s'] = time.perf_counter() - started
                event['cpu_s'] = time.process_time() - cpu_started
        if meter.remaining <= 0:
            stops.append('TIME_LIMIT')
            result['fallback_reason'] = result['fallback_reason'] or operation + '_TIME_LIMIT'
    except BudgetExceeded:
        stops.append('TIME_LIMIT')
        result['fallback_reason'] = operation + '_TIME_LIMIT'
    except NumericUnresolved as error:
        stops.append('NUMERIC_UNRESOLVED')
        result['run_status'] = 'ERROR'
        result['numeric_evidence'] = {'reason': str(error), 'inputs': error.inputs}
    except Exception as error:
        result['run_status'] = 'ERROR'
        result['error'] = repr(error)
        result['error_details'] = getattr(error, 'details', None)
    # Full replay snapshots are the sole numerical authority.  Publication is
    # structural and remains paid after a cooperative time-limit completion.
    if X is not None:
        try:
            operation = 'FINAL_VERIFICATION'
            with meter.phase('final_verification'):
                validate(X, channel)
                if tr is None:
                    if np.any(X):
                        raise AssertionError('unreplayed output is not registered silent plan')
                    result['final_verification_source'] = 'ALL_SILENT_STRUCTURAL'
                else:
                    if (len(tr.slots) != channel.H or len(tr.states) != channel.H + 1
                            or tr.channel is not channel or tr.abs_start != cycle * channel.H
                            or not np.array_equal(X, tr.plan)):
                        raise AssertionError('completed full trace does not match output')
                    result['final_verification_source'] = 'COMPLETED_REPLAY_SNAPSHOT'
            operation = 'PLAN_DISTRIBUTION'
            with meter.phase('plan_distribution'):
                published = np.zeros_like(X)
                for s, gid in enumerate(channel.sat_ids):
                    decoded = ledger.emit('PLAN_DISTRIBUTION', 'gateway', [f'sat:{gid}'],
                        {'cycle': cycle, 'version': result['accepted'], 'global_satellite_id': int(gid),
                         'beam_slots': np.argwhere(X[:, s]).tolist()})
                    if (decoded['cycle'] != cycle or decoded['version'] != result['accepted']
                            or decoded['global_satellite_id'] != int(gid)):
                        raise ValueError('downlink plan version or destination mismatch')
                    for h, m in decoded['beam_slots']:
                        published[h, s, m] = True
                if not np.array_equal(published, X):
                    raise AssertionError('downlink changed plan')
                X = published
                result['output_published'] = True
            with meter.phase('final_verification'):
                result['final'] = trace_summary(tr)
                result['final_key'] = plan_key(X, channel)
                result['returned_initial'] = np.array_equal(X, init) if init is not None else None
                if tr is not None and denominator is not None:
                    result['final_planning_proxy'] = satisfaction_summary(tr, denominator)
        except NumericUnresolved as error:
            stops.append('NUMERIC_UNRESOLVED')
            result['run_status'] = 'ERROR'
            result['numeric_evidence'] = {'reason': str(error), 'inputs': error.inputs}
        except Exception as error:
            result['run_status'] = 'ERROR'
            result['error'] = repr(error)
    if c.get('r3_engineering', False) and hasattr(channel, 'end_engineering_cache'):
        with meter.phase('residual'):
            result['engineering_cache'] = channel.end_engineering_cache()
    result['revision'] = {'name': 'CONFIRMATION_010:' + variant,
        'engineering': bool(c.get('r3_engineering', False)),
        'legacy_r3_search_rules_executed': False, 'legacy_price_executed': False,
        'reliable_output_rule': 'LAST_COMMITTED_COMPLETE_FULL_TRACE',
        'construction': 'ONE_FORWARD_RECONSTRUCTION_WITH_TPG_SUFFIX',
        'intermediate_H_replays': 0}
    result['prototype_evaluation']['event_time_accounting'] = 'NESTED_IN_ORIGINAL_METER_PHASES'
    result['prototype_evaluation']['original_full_engine_recorded_calls'] = {
        key: int(count) for key, count in meter.replays.items()
        if key not in ('TPG_construction_slot', '009_construction_slot')}
    result['prototype_evaluation']['original_full_engine_recorded_slots'] = {
        key: int(count) for key, count in meter.replayed_slots.items()
        if key not in ('TPG_construction_slot', '009_construction_slot')}
    result['costs'] = meter.finish()
    result['overrun'] = result['costs']['total_s'] > c['planning_budget_s']
    if result['overrun'] or meter.remaining <= 0:
        stops.append('TIME_LIMIT')
    result['stop_labels'] = list(dict.fromkeys(stops))
    result['time_limit_occurred'] = 'TIME_LIMIT' in stops
    result['stop'] = (('NUMERIC_UNRESOLVED' if 'NUMERIC_UNRESOLVED' in stops else 'CORRECTNESS_FAILURE')
                      if result['run_status'] != 'OK' else resolve_stop(*stops))
    if result['time_limit_occurred']:
        result['fallback_used'] = True
        result['fallback_source'] = reliable_source
        result['fallback_reason'] = result['fallback_reason'] or operation + '_TIME_LIMIT'
    result['reliable_plan_source'] = reliable_source
    result['preparation_metrics'] = result['costs']['preparation_metrics']
    result['plan'], result['trace'], result['initial_plan'] = X, tr, init
    return result
