"""Exclusive wall accounting, explicit protocol, canonical plans and provenance."""
from collections import Counter
from contextlib import contextmanager
from fractions import Fraction
import hashlib
import json
import time
import numpy as np
from .prep_metrics import PreparationMetrics
from .state_digest import state_sha256,begin_cache,end_cache

PHASES = ('geometry_visibility', 'traffic_preparation', 'TPG_initialization', 'w', 'mu',
          'matching', 'direct_generation', 'background_construction', 'candidate_replay',
          'final_verification', 'codec_protocol', 'plan_distribution', 'residual')
STOPS = ('NUMERIC_UNRESOLVED','TIME_LIMIT','Q_MAX','CANDIDATE_LIMIT','POOL_NO_IMPROVEMENT')

class BudgetExceeded(RuntimeError):
    pass

def jsonable(value):
    if hasattr(value, 'to_json'): return value.to_json()
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, Fraction): return {'numerator':value.numerator,'denominator':value.denominator}
    if isinstance(value, set): return sorted(value)
    raise TypeError(type(value).__name__)

def dumps(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',',':'), default=jsonable)

def sha(data):
    return hashlib.sha256(data).hexdigest()

def plan_key(plan, channel=None):
    e=np.argwhere(plan)
    if channel is not None and len(e): e[:,1]=channel.sat_ids[e[:,1]]
    return tuple(sorted(map(tuple, e.tolist())))

def state_hash(state):
    return state_sha256(state,dumps)

class Meter:
    def __init__(self, limit=5.0):
        self.started=time.perf_counter(); self.cpu_started=time.process_time()
        self.limit=limit; self.costs=Counter(); self.replays=Counter(); self.replayed_slots=Counter()
        self._phase='residual'; self._tick=self.started; self.ended=None
        self.cpu_costs=Counter();self._cpu_tick=self.cpu_started
        self.preparation=PreparationMetrics()
        self._digest_cache,self._digest_token=begin_cache()
    @property
    def elapsed(self): return (self.ended or time.perf_counter())-self.started
    @property
    def remaining(self): return self.limit-self.elapsed
    def check(self, reserve=0.0):
        if self.remaining<=reserve: raise BudgetExceeded('whole-plan budget exhausted')
    @contextmanager
    def phase(self, name):
        now=time.perf_counter(); self.costs[self._phase]+=now-self._tick
        cpu_now=time.process_time();self.cpu_costs[self._phase]+=cpu_now-self._cpu_tick
        old=self._phase; self._phase=name; self._tick=now
        self._cpu_tick=cpu_now
        try: yield
        finally:
            now=time.perf_counter(); self.costs[self._phase]+=now-self._tick
            cpu_now=time.process_time();self.cpu_costs[self._phase]+=cpu_now-self._cpu_tick
            self._phase=old; self._tick=now
            self._cpu_tick=cpu_now
    def replay(self, kind, slots):
        self.replays[kind]+=1; self.replayed_slots[kind]+=int(slots)
    def finish(self):
        self.preparation.stop()
        # Releasing all retained encodings/storage references is paid inside
        # this invocation, before its planning clock stops.
        digest_cost=end_cache(self._digest_cache,self._digest_token)
        self.ended=time.perf_counter(); self.costs[self._phase]+=self.ended-self._tick
        cpu_now=time.process_time();self.cpu_costs[self._phase]+=cpu_now-self._cpu_tick
        self.cpu=cpu_now-self.cpu_started
        self.preparation.reconcile(self.costs['traffic_preparation'],self.cpu_costs['traffic_preparation'])
        return {**{k:float(self.costs[k]) for k in PHASES},
                'total_s':self.elapsed,'cpu_s':self.cpu,
                'closure_error_s':sum(self.costs.values())-self.elapsed,
                'replay_calls':dict(self.replays),'replayed_slots':dict(self.replayed_slots),
                'cpu_components_s':{k:float(self.cpu_costs[k]) for k in PHASES},
                'cpu_closure_error_s':sum(self.cpu_costs.values())-self.cpu,
                'preparation_metrics':self.preparation.snapshot(),
                'state_digest_cache':digest_cost}

class Ledger:
    """UTF-8 JSON v1 envelopes, measured per recipient, no invented wire cost.

    The returned decoded payload is the actual interface consumed by the caller.
    Co-located gateway operations remain local even when serialized. Satellite
    workers are logical nodes; there is no claim of real network execution.
    """
    def __init__(self, meter): self.meter=meter; self.records=[]
    def emit(self, kind, sender, recipients, payload):
        recipients=list(recipients)
        with self.meter.phase('codec_protocol'):
            decoded=None
            for recipient in recipients:
                envelope={'schema':1,'type':kind,'sender':str(sender),
                          'recipient':str(recipient),'payload':payload}
                encoded=dumps(envelope).encode('utf-8')
                obj=json.loads(encoded)
                assert obj['type']==kind and obj['schema']==1
                decoded=obj['payload']
                self.records.append({'seq':len(self.records),'type':kind,'sender':str(sender),
                                     'recipient':str(recipient),'local':str(sender)==str(recipient),
                                     'application_bytes':len(encoded),'payload':obj['payload'],
                                     'sha256':sha(encoded),'used':True})
            return decoded

def resolve_stop(*labels):
    return next((s for s in STOPS if s in labels), None)
