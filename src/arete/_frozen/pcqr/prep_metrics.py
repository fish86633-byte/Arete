"""Preparation diagnostics, with exclusive wall/CPU phases and overlapping GC.

No timing result changes a decision. Payload copy bytes count only explicitly
measured copies; Python object allocation/copy bytes remain unmeasured.
"""
from collections import Counter
from contextlib import contextmanager
import gc
import resource
import sys
import time

PARTS=('scene_copy','message_postprocessing','rt_state_construction',
       'nrt_fifo_construction','item_validation','array_conversion','other')

def peak_rss_bytes():
    value=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform=='darwin' else value*1024)

class PreparationMetrics:
    def __init__(self):
        self.wall=Counter();self.cpu=Counter();self.calls=Counter();self.copies=Counter()
        self.gc_records=[];self._gc_pending={};self._active=None;self._tick=None;self._cpu_tick=None
        self.started=None;self.ended=None;self.cpu_started=None;self.cpu_ended=None
        self.rss_before=None;self.rss_after=None;self.rss_by_part={};self.fifo_batches=None
        self.traffic_wall=None;self.traffic_cpu=None;self._installed=False
    def start(self):
        if self.started is not None:return
        self.started=time.perf_counter();self.cpu_started=time.process_time();self.rss_before=peak_rss_bytes()
        gc.callbacks.append(self._gc_callback);self._installed=True
    def _gc_callback(self,phase,info):
        generation=info.get('generation',-1)
        if phase=='start':self._gc_pending[generation]=(time.perf_counter(),time.process_time(),self._active)
        elif generation in self._gc_pending:
            w,c,part=self._gc_pending.pop(generation)
            self.gc_records.append({'generation':generation,'part':part,'wall_s':time.perf_counter()-w,
                'cpu_s':time.process_time()-c,'collected':info.get('collected'),'uncollectable':info.get('uncollectable')})
    def _charge(self,wall,cpu):
        if self._active is not None:
            self.wall[self._active]+=wall-self._tick;self.cpu[self._active]+=cpu-self._cpu_tick
    @contextmanager
    def part(self,name):
        if name not in PARTS:raise ValueError(name)
        self.start();w=time.perf_counter();c=time.process_time();self._charge(w,c)
        old=self._active;self._active=name;self._tick=w;self._cpu_tick=c;self.calls[name]+=1
        try:yield
        finally:
            w=time.perf_counter();c=time.process_time();self._charge(w,c)
            self.rss_by_part[name]=peak_rss_bytes()
            self._active=old;self._tick=time.perf_counter();self._cpu_tick=time.process_time()
    def copied(self,nbytes,kind):
        self.copies[str(kind)]+=int(nbytes)
    def set_fifo_batches(self,count):self.fifo_batches=int(count)
    def stop(self):
        if self.ended is not None:return
        if self.started is None:self.start()
        self.ended=time.perf_counter();self.cpu_ended=time.process_time();self.rss_after=peak_rss_bytes()
        if self._installed:
            gc.callbacks.remove(self._gc_callback);self._installed=False
    def reconcile(self,wall_s,cpu_s):
        self.traffic_wall=float(wall_s);self.traffic_cpu=float(cpu_s)
    def snapshot(self):
        now=self.ended if self.ended is not None else time.perf_counter()
        cpu_now=self.cpu_ended if self.cpu_ended is not None else time.process_time()
        parts={k:{'wall_s':float(self.wall[k]),'cpu_s':float(self.cpu[k]),'calls':self.calls[k],
            'process_peak_rss_bytes_at_part_end':self.rss_by_part.get(k)} for k in PARTS}
        # The remainder covers instrumentation and unsectioned preparation work;
        # this is an accounting identity, not an invented measured code section.
        residual_wall=None if self.traffic_wall is None else self.traffic_wall-sum(self.wall.values())
        residual_cpu=None if self.traffic_cpu is None else self.traffic_cpu-sum(self.cpu.values())
        if residual_wall is not None:
            parts['other']['explicit_wall_s']=parts['other']['wall_s']
            parts['other']['unsectioned_and_instrumentation_wall_s']=residual_wall
            parts['other']['wall_s']+=residual_wall
            parts['other']['explicit_cpu_s']=parts['other']['cpu_s']
            parts['other']['unsectioned_and_instrumentation_cpu_s']=residual_cpu
            parts['other']['cpu_s']+=residual_cpu
        return {'parts':parts,'traffic_preparation_wall_s':self.traffic_wall,'traffic_preparation_cpu_s':self.traffic_cpu,
            'span_wall_s':None if self.started is None else now-self.started,
            'span_cpu_s':None if self.cpu_started is None else cpu_now-self.cpu_started,
            'fifo_batch_count':self.fifo_batches,'known_payload_copy_bytes':sum(self.copies.values()),
            'copy_bytes_by_kind':dict(self.copies),'python_object_copy_or_allocation_bytes':None,
            'gc_count':len(self.gc_records),'gc_counts_by_generation':dict(Counter(r['generation'] for r in self.gc_records)),
            'gc_wall_s':sum(r['wall_s'] for r in self.gc_records),'gc_cpu_s':sum(r['cpu_s'] for r in self.gc_records),
            'gc_records':self.gc_records,'process_peak_rss_bytes_before':self.rss_before,
            'process_peak_rss_bytes_after':self.rss_after,'exclusive_preparation_peak_rss_bytes':None,
            'scope':'Seven preparation parts are exclusive. Codec is separate. GC overlaps measured phases and must not be added again. RSS is process high-water, not exclusive preparation allocation; missing measurements are N/A.'}
