"""Isolated exploration wrapper retaining the frozen B reliability contract.

Only candidate search is delegated to exploration_008.candidate_search. This
wrapper does not execute frozen PCQR backgrounds, direct candidate ordering,
matching or price coordination. It retains paid input preparation and TPG,
aggregate candidate suffix evaluation, exact full-FIFO commit verification,
last-committed-reference fallback, complete timing and plan publication.

The compatibility method selector is PCQR-BH, but result identities explicitly
name EXPLORATION_008:<variant>; no old result or algorithm label is reused.
"""
from fractions import Fraction
import time
import numpy as np
from arete._frozen.pcqr.support import Meter,Ledger,BudgetExceeded,plan_key,state_hash,sha,resolve_stop
from arete._frozen.pcqr.engine import replay,validate
from arete._frozen.pcqr.physics import NumericUnresolved
from arete._frozen.pcqr.baselines import greedy
from arete._frozen.pcqr.search import Scan,ScanLimit
from arete._frozen.pcqr.preparation import register_silent,prepare_inputs
from arete._frozen.prototype.light_engine import LightTrace,light_replay_suffix,compare_light_to_full

def demand_denominator(Q0,A):
    """Fixed available planning demand; includes every positive-demand cell."""
    nrt=Q0.nrt
    return tuple(sum(int(x) for x in Q0.rt[m])+int(nrt[m])
                 +sum(int(x) for x in A[:,m].flat)
                 for m in range(len(Q0.rt)))

def satisfaction_summary(tr,denominator):
    """Exact Jain index of per-cell served/available-demand fractions.

    Zero-demand cells are excluded, regardless of visibility.  No positive
    demand is silently excluded.  All-zero service on nonempty demand has Jain
    zero; an empty workload has Jain one as a neutral constraint convention.
    This planning proxy is neither a continuous satisfaction result nor a
    fairness optimality certificate.
    """
    served=[0]*len(denominator)
    for item in tr.slots:
        for m,(rt,nrt) in enumerate(zip(item['service_rt'],item['service_nrt'])):
            served[m]+=int(rt)+int(nrt)
    ratios=[]
    for m,demand in enumerate(denominator):
        if served[m]<0 or served[m]>demand:
            raise AssertionError('planning service exceeds fixed available demand')
        if demand>0: ratios.append(Fraction(served[m],demand))
        elif served[m]: raise AssertionError('zero-demand cell has positive service')
    total=sum(ratios,Fraction(0))
    squared=sum((ratio*ratio for ratio in ratios),Fraction(0))
    jain=(total*total/(len(ratios)*squared) if squared
          else Fraction(0) if ratios else Fraction(1))
    return {'Jain_satisfaction':jain,'positive_demand_cells':len(ratios),
            'sum_satisfaction':total,'sum_squared_satisfaction':squared,
            'zero_satisfied_cells':sum(ratio==0 for ratio in ratios),
            'minimum_satisfaction':min(ratios) if ratios else None,
            'empty_workload_neutral_convention':not ratios,
            'planning_proxy_only':True}

def _expired(tr):
    return sum(int(count) for item in tr.slots for count in item['expired'])

def _eligible(candidate,reference,candidate_proxy,reference_proxy,config):
    reasons=[]
    if candidate.F<=reference.F+Fraction(str(config['epsilon_F'])):
        reasons.append('F_NOT_STRICTLY_IMPROVED')
    if candidate.W>reference.W: reasons.append('W_RT_INCREASED')
    if _expired(candidate)>_expired(reference): reasons.append('RT_EXPIRED_INCREASED')
    if candidate_proxy['Jain_satisfaction']<reference_proxy['Jain_satisfaction']:
        reasons.append('JAIN_SATISFACTION_DECREASED')
    return not reasons,reasons

def trace_summary(tr):
    if tr is None:
        return {key:None for key in ('F','F_float','bits','W','rt_expired','end_rt','end_nrt','end_hash')}
    return {'F':tr.F,'F_float':float(tr.F),'bits':tr.bits,'W':tr.W,
            'rt_expired':sum(int(s['expired'].sum()) for s in tr.slots),
            'end_rt':int(tr.end_state.rt.sum()),'end_nrt':int(tr.end_state.nrt.sum()),
            'end_hash':None if isinstance(tr,LightTrace) else state_hash(tr.end_state)}

def is_better(candidate_F,candidate_key,best_F,best_key,reference_key):
    """The shared streaming best rule never terminates on an ordinary rejection."""
    return candidate_F>best_F or (candidate_F==best_F and best_key!=reference_key and candidate_key<best_key)

def plan(Q0,A,channel,method='PCQR-BH',cycle=0,pointers=None,meter=None,safety_fallback=None,*,variant):
    if method != 'PCQR-BH':
        raise ValueError('The common candidate wrapper uses the PCQR-BH compatibility selector')
    if variant not in ('V3','V4'):
        raise ValueError('A named exploration_008 V3/V4 variant is required')
    if channel.config.get('r36_batch_arrivals',False):
        raise ValueError('R3.6 must remain disabled')
    meter=meter or Meter(channel.config['planning_budget_s']); c=channel.config
    if int(c['L_eval'])!=16 or int(c['Q_max'])!=(5 if variant=='V3' else 20):
        raise ValueError('V3 requires Q_max=5; V4 requires Q_max=20; both require L_eval=16')
    # Registration precedes JSON encoding, decoded state construction, and the
    # first budget check. A timeout therefore always has a structural output.
    X,silent_registration=register_silent(channel,safety_fallback)
    ledger=Ledger(meter)
    requested_initial='TPG'
    result={'method':'EXPLORATION_008:'+variant,'variant':variant,'cycle':cycle,'run_status':'OK','stop':None,'accepted':0,
            'candidates':[],'rounds':[],'solver_calls':[],'messages':ledger.records,
            'requested_initial_source':requested_initial,'initial_source':None,
            'initial_completed':False,'initial':trace_summary(None),'initial_key':None,
            'final':trace_summary(None),'returned_initial':None,
            'fallback_used':False,'fallback_source':None,'fallback_reason':None,
            'silent_registration':silent_registration,'output_published':False}
    result['prototype']={'variant':variant,
        'experiment_family':'EXPLORATION_008',
        'reliable_wrapper_source':'prototype.pcqr_light.plan',
        'reliable_wrapper_implementation_commit':'2b9f896603a52816695a4c2928adb8c3bf797406',
        'candidate_evaluator':'LIGHT_AGGREGATE',
        'candidate_order':'EXPLORATION_008_VARIANT_POLICY',
        'background_policy':'NONE_FROM_FROZEN_PCQR',
        'price_policy':'NONE_FROM_FROZEN_PCQR',
        'reliable_commit':'ORIGINAL_FULL_REPLAY_EXACT_CHECK',
        'nrt_fifo_hash_available_for_light':False}
    result['prototype_evaluation']={'light_completed':0,'light_interrupted':0,
        'light_cache_hits':0,'full_commit_completed':0,'full_commit_interrupted':0,
        'unverified_best_discarded':0,'events':[]}
    result['acceptance_policy']={
        'name':('ORIGINAL_STRICT_F' if variant=='V3' else 'STRICT_F_RT_JAIN_NONDEGRADATION'),
        'comparison_reference':'LAST_COMMITTED_COMPLETE_FULL_TRACE',
        'fairness_denominator':'Q0_RT+Q0_NRT+SUM_FIXED_PLANNING_ARRIVALS',
        'positive_demand_cells':'ALL_WITHOUT_VISIBILITY_FILTER',
        'zero_service_Jain':0,'empty_workload_Jain':1,
        'evaluated_eligible':0,'evaluated_rejected':0,'proxy_metric_calls':0,
        'objective_definition_unchanged':True}
    denominator=None
    tr=None; init=None; init_tr=None; stops=[]
    reliable_source='ALL_SILENT_REGISTERED'; operation='PREPARATION'
    try:
        if c.get('r3_engineering',False) and hasattr(channel,'begin_engineering_cache'):
            # Cold on every call; allocation and release stay inside Meter.
            channel.begin_engineering_cache()
        meter.preparation.start()
        try:
            meter.check()
            Q0,A=prepare_inputs(Q0,A,channel,cycle,meter,ledger)
            meter.check()
        finally:
            meter.preparation.stop()
        operation='INITIALIZATION'
        X,tr=greedy(Q0,A,channel,'TPG',meter,abs_start=cycle*channel.H)
        # Tuple assignment above occurs only after the full initialization
        # replay returned. Partial TPG construction never becomes X/tr.
        result['initial_completed']=True; result['initial_source']=requested_initial
        reliable_source='TPG_REPLAYED'
        init=X.copy(); init_tr=tr; result['initial']=trace_summary(tr); result['initial_key']=plan_key(X,channel)
        if variant=='V4':
            with meter.phase('candidate_replay'):
                denominator=demand_denominator(Q0,A)
                result['fixed_planning_demand']=denominator
                result['initial_planning_proxy']=satisfaction_summary(tr,denominator)
                result['acceptance_policy']['proxy_metric_calls']+=1
        whole_plan_memo={plan_key(X,channel):tr}
        # A light trace intentionally has no FIFO and must never escape into
        # a reliable full-FIFO reference memo or become a reference trace.
        light_memo={}
        result['whole_plan_reuse_count']=0
        operation='SEARCH'
        while result['accepted']<c['Q_max']:
            meter.check(reserve=.03)
            ref_version=result['accepted']; bestX=X; besttr=tr; bestkey=plan_key(X,channel)
            refkey=bestkey; seen={refkey:tr}; provenance={}; counts={'direct':0,'match':0}; truncated=False
            scan=Scan(c['L_scan'],meter); roundinfo={'ref_version':ref_version,'F_before':tr.F}
            pending_stop=None
            reference_proxy=None
            if variant=='V4':
                with meter.phase('candidate_replay'):
                    reference_proxy=satisfaction_summary(tr,denominator)
                    result['acceptance_policy']['proxy_metric_calls']+=1
                roundinfo['reference_planning_proxy']=reference_proxy
            def evaluate(Y,source,lane,known=None):
                nonlocal bestX,besttr,bestkey
                source={**source,'net_R':plan_key(X & ~Y,channel),'net_A':plan_key(Y & ~X,channel)}
                key=plan_key(Y,channel)
                if key in seen:
                    provenance.setdefault(key,[]).append(source)
                    result['candidates'].append({'ref_version':ref_version,'source':source,'lane':lane,'status':'DUPLICATE','plan_hash':sha(repr(key).encode())})
                    return
                try: validate(Y,channel)
                except ValueError as e:
                    result['candidates'].append({'ref_version':ref_version,'source':source,'lane':lane,'status':'INVALID','reason':str(e)})
                    return
                if known is None and key in whole_plan_memo:
                    known=whole_plan_memo[key]; result['whole_plan_reuse_count']+=1
                if known is None and key in light_memo:
                    known=light_memo[key]
                    result['prototype_evaluation']['light_cache_hits']+=1
                if known is None:
                    meter.check(reserve=.03)
                    changed=np.flatnonzero(np.any(Y!=X,axis=(1,2))); first=int(changed[0])
                    event={'kind':'LIGHT_CANDIDATE','ref_version':ref_version,
                           'first_changed':first,'requested_slots':channel.H-first,
                           'plan_hash':sha(repr(key).encode()),'status':'RUNNING'}
                    result['prototype_evaluation']['events'].append(event)
                    started=time.perf_counter(); cpu_started=time.process_time()
                    slots_before=int(meter.replayed_slots.get('light_slot',0))
                    try:
                        with meter.phase('candidate_replay'):
                            ty=light_replay_suffix(Y,Q0,A,channel,tr,first,
                                abs_start=cycle*channel.H,meter=meter)
                        event['status']='COMPLETED'
                        result['prototype_evaluation']['light_completed']+=1
                    except BudgetExceeded:
                        event['status']='TIME_LIMIT'
                        result['prototype_evaluation']['light_interrupted']+=1
                        raise
                    except Exception as error:
                        event['status']='ERROR';event['error']=repr(error)
                        result['prototype_evaluation']['light_interrupted']+=1
                        raise
                    finally:
                        event['wall_s']=time.perf_counter()-started
                        event['cpu_s']=time.process_time()-cpu_started
                        event['propagated_slots']=int(meter.replayed_slots.get('light_slot',0))-slots_before
                else: ty=known
                if isinstance(ty,LightTrace): light_memo[key]=ty
                else: whole_plan_memo[key]=ty
                seen[key]=ty; counts[lane]+=1; provenance[key]=[source]
                eligible=True;rejection_reasons=[];candidate_proxy=None
                if variant=='V4':
                    with meter.phase('candidate_replay'):
                        candidate_proxy=satisfaction_summary(ty,denominator)
                        result['acceptance_policy']['proxy_metric_calls']+=1
                        eligible,rejection_reasons=_eligible(ty,tr,candidate_proxy,reference_proxy,c)
                    result['acceptance_policy']['evaluated_eligible' if eligible else 'evaluated_rejected']+=1
                record={'ref_version':ref_version,'source':source,'lane':lane,'status':'EVALUATED',
                                            'plan_hash':sha(repr(key).encode()),'plan':key,
                                            'delta_F':ty.F-tr.F,**trace_summary(ty)}
                record.update(eligible=eligible,eligibility_required=variant=='V4',
                              rejection_reasons=rejection_reasons)
                if candidate_proxy is not None:
                    record['planning_proxy']=candidate_proxy
                    record['reference_planning_proxy']=reference_proxy
                    record['delta_Jain_satisfaction']=(candidate_proxy['Jain_satisfaction']
                                                      -reference_proxy['Jain_satisfaction'])
                result['candidates'].append(record)
                if eligible and is_better(ty.F,key,besttr.F,bestkey,refkey):
                    bestX=Y.copy(); besttr=ty; bestkey=key
            try:
                from arete._frozen.exploration_008.candidate_search import search_reference
                truncated=search_reference(X,tr,Q0,A,channel,meter,ledger,cycle,
                    result,roundinfo,scan,counts,whole_plan_memo,evaluate,
                    lambda:besttr.F,variant=variant)
            except ScanLimit: truncated=True
            except BudgetExceeded:
                pending_stop='TIME_LIMIT'; result['fallback_reason']='SEARCH_TIME_LIMIT'
            except NumericUnresolved as e:
                pending_stop='NUMERIC_UNRESOLVED'; result['run_status']='ERROR'
                result['numeric_evidence']={'reason':str(e),'inputs':e.inputs}
            truncated=truncated or roundinfo.get('policy_limited',False)
            improved=besttr.F>tr.F+Fraction(str(c['epsilon_F']))
            roundinfo.update(counts=counts,scan_count=scan.count,scan_records=scan.records,improved=improved,
                             F_best=besttr.F,truncated=truncated)
            result['rounds'].append(roundinfo)
            if improved and isinstance(besttr,LightTrace):
                # The old full engine remains the authority. Starting this
                # verification requires budget, and only a completed exact
                # agreement can replace the last reliable FIFO trace. Like
                # the frozen engine, an already-started full replay is a
                # cooperative atomic operation: a completed overrun is
                # retained and receives TIME_LIMIT below.
                event={'kind':'FULL_COMMIT_VERIFICATION','ref_version':ref_version,
                       'requested_slots':channel.H,'propagated_slots':0,'status':'RUNNING'}
                result['prototype_evaluation']['events'].append(event)
                started=time.perf_counter();cpu_started=time.process_time()
                try:
                    meter.check(reserve=.03)
                    with meter.phase('final_verification'):
                        meter.replay('light_commit_full',channel.H)
                        verified=replay(bestX,Q0,A,channel,abs_start=cycle*channel.H)
                        event['propagated_slots']=channel.H
                        comparison=compare_light_to_full(besttr,verified)
                        event['comparison']=comparison
                    # compare_light_to_full must either return a passing
                    # comparison or raise; a false summary is never accepted.
                    if comparison is False or (isinstance(comparison,dict)
                        and (comparison.get('equal') is False or
                             comparison.get('passed') is False or
                             comparison.get('status') in ('FAIL','ERROR'))):
                        raise AssertionError('light/full commit verification differs')
                    besttr=verified;whole_plan_memo[bestkey]=verified
                    event['status']='COMPLETED'
                    result['prototype_evaluation']['full_commit_completed']+=1
                except BudgetExceeded:
                    event['status']='TIME_LIMIT'
                    result['prototype_evaluation']['full_commit_interrupted']+=1
                    result['prototype_evaluation']['unverified_best_discarded']+=1
                    improved=False;pending_stop='TIME_LIMIT'
                    result['fallback_reason']='UNVERIFIED_LIGHT_BEST_TIME_LIMIT'
                except Exception as error:
                    event['status']='ERROR';event['error']=repr(error)
                    result['prototype_evaluation']['full_commit_interrupted']+=1
                    result['prototype_failure']={'kind':'LIGHT_FULL_COMMIT_MISMATCH',
                        'plan':bestkey,'error':repr(error)}
                    raise
                finally:
                    event['wall_s']=time.perf_counter()-started
                    event['cpu_s']=time.process_time()-cpu_started
                roundinfo['light_best_verified']=bool(improved)
                roundinfo['improved']=improved
            if improved:
                X=bestX; tr=besttr; result['accepted']+=1
                reliable_source='BEST_REPLAYED'
            if pending_stop: stops.append(pending_stop); break
            if meter.remaining<=0:
                stops.append('TIME_LIMIT'); result['fallback_reason']='SEARCH_TIME_LIMIT'; break
            if result['accepted']>=c['Q_max']: stops.append('Q_MAX'); break
            if improved: continue
            stops.append('CANDIDATE_LIMIT' if truncated else 'POOL_NO_IMPROVEMENT'); break
    except BudgetExceeded:
        stops.append('TIME_LIMIT')
        result['fallback_reason']=operation+'_TIME_LIMIT'
    except NumericUnresolved as e:
        stops.append('NUMERIC_UNRESOLVED'); result['run_status']='ERROR'
        result['numeric_evidence']={'reason':str(e),'inputs':e.inputs}
    except Exception as e:
        result['run_status']='ERROR'; result['error']=repr(e)
    # A budget exit may publish the pre-registered silent plan without inventing
    # an objective. Replayed snapshots retain their original value and plan.
    if X is not None:
        try:
            if meter.remaining<=0 and 'TIME_LIMIT' not in stops:
                stops.append('TIME_LIMIT'); result['fallback_reason']=operation+'_TIME_LIMIT'
            operation='FINAL_VERIFICATION'
            if meter.remaining<=0: stops.append('TIME_LIMIT')
            with meter.phase('final_verification'):
                validate(X,channel)
                if tr is None:
                    if np.any(X): raise AssertionError('un-replayed output is not registered silent plan')
                    result['final_verification_source']='ALL_SILENT_STRUCTURAL'
                else:
                    if (len(tr.slots)!=channel.H or len(tr.states)!=channel.H+1
                            or tr.channel is not channel or tr.abs_start!=cycle*channel.H
                            or not np.array_equal(X,tr.plan)):
                        raise AssertionError('reliable completed Replay snapshot does not match output plan')
                    if 'TIME_LIMIT' in stops or meter.remaining<=0:
                        stops.append('TIME_LIMIT')
                        result['final_verification_source']='COMPLETED_REPLAY_SNAPSHOT'
                    else:
                        meter.replay('final_verification',channel.H)
                        final=replay(X,Q0,A,channel,abs_start=cycle*channel.H)
                        if final.F!=tr.F or state_hash(final.end_state)!=state_hash(tr.end_state):
                            raise AssertionError('final replay differs from accepted snapshot')
                        tr=final
                        result['final_verification_source']='FULL_REPLAY'
            if meter.remaining<=0 and 'TIME_LIMIT' not in stops:
                stops.append('TIME_LIMIT'); result['fallback_reason']='FINAL_VERIFICATION_TIME_LIMIT'
            operation='PLAN_DISTRIBUTION'
            with meter.phase('plan_distribution'):
                published=np.zeros_like(X)
                for s,gid in enumerate(channel.sat_ids):
                    decoded=ledger.emit('PLAN_DISTRIBUTION','gateway',[f'sat:{gid}'],
                         {'cycle':cycle,'version':result['accepted'],'global_satellite_id':int(gid),
                          'beam_slots':np.argwhere(X[:,s]).tolist()})
                    if (decoded['cycle']!=cycle or decoded['version']!=result['accepted']
                            or decoded['global_satellite_id']!=int(gid)):
                        raise ValueError('downlink plan version or destination mismatch')
                    for h,m in decoded['beam_slots']: published[h,s,m]=True
                if not np.array_equal(published,X): raise AssertionError('downlink changed plan')
                X=published
                result['output_published']=True
            result['final']=trace_summary(tr); result['final_key']=plan_key(X,channel)
            if variant=='V4' and tr is not None and denominator is not None:
                with meter.phase('final_verification'):
                    result['final_planning_proxy']=satisfaction_summary(tr,denominator)
                    result['acceptance_policy']['proxy_metric_calls']+=1
            result['returned_initial']=np.array_equal(X,init) if init is not None else None
        except NumericUnresolved as e:
            stops.append('NUMERIC_UNRESOLVED'); result['run_status']='ERROR'
            result['numeric_evidence']={'reason':str(e),'inputs':e.inputs}
        except Exception as e:
            result['run_status']='ERROR'; result['error']=repr(e)
    if meter.remaining<=0: stops.append('TIME_LIMIT')
    if c.get('r3_engineering',False) and hasattr(channel,'end_engineering_cache'):
        with meter.phase('residual'):
            result['engineering_cache']=channel.end_engineering_cache()
    result['revision']={'name':'EXPLORATION_008:'+variant,
        'engineering':bool(c.get('r3_engineering',False)),
        'candidate_order':'EXPLORATION_008_VARIANT_POLICY',
        'legacy_r3_search_rules_executed':False,
        'legacy_price_executed':False,
        'reliable_output_rule':'LAST_COMMITTED_COMPLETE_FULL_TRACE'}
    # Event times are diagnostic children of the original exclusive Meter
    # phases and must not be added to those phases in any total ledger.
    result['prototype_evaluation']['event_time_accounting']='NESTED_IN_ORIGINAL_METER_PHASES'
    result['prototype_evaluation']['light_propagated_slots']=int(meter.replayed_slots.get('light_slot',0))
    result['prototype_evaluation']['full_commit_propagated_slots']=sum(
        event.get('propagated_slots',0) for event in result['prototype_evaluation']['events']
        if event['kind']=='FULL_COMMIT_VERIFICATION')
    result['prototype_evaluation']['original_full_engine_recorded_calls']={
        key:int(value) for key,value in meter.replays.items()
        if key not in ('light_slot','TPG_construction_slot')}
    result['prototype_evaluation']['original_full_engine_recorded_slots']={
        key:int(value) for key,value in meter.replayed_slots.items()
        if key not in ('light_slot','TPG_construction_slot')}
    result['prototype_evaluation']['counter_notes']=(
        'light_slot counts completed slot propagations, not complete evaluations; '
        'full-engine counters retain their original call sites; TPG construction '
        'slots are separately present in costs.replayed_slots; event timers are nested.')
    result['costs']=meter.finish(); result['overrun']=result['costs']['total_s']>c['planning_budget_s']
    if result['overrun']: stops.append('TIME_LIMIT')
    result['stop_labels']=list(dict.fromkeys(stops))
    result['time_limit_occurred']='TIME_LIMIT' in stops
    if result['run_status']!='OK':
        result['stop']=('NUMERIC_UNRESOLVED' if 'NUMERIC_UNRESOLVED' in stops
                        else 'CORRECTNESS_FAILURE')
    else:
        result['stop']=resolve_stop(*stops)
    if result['stop']=='TIME_LIMIT':
        result['fallback_used']=True; result['fallback_source']=reliable_source
        result['fallback_reason']=result['fallback_reason'] or (operation+'_TIME_LIMIT' if operation!='SEARCH' else 'SEARCH_TIME_LIMIT')
    result['reliable_plan_source']=reliable_source
    result['preparation_metrics']=result['costs']['preparation_metrics']
    result['plan']=X; result['trace']=tr; result['initial_plan']=init
    return result
