"""R2 §§5/7/8 only: common deterministic backgrounds and direct candidates."""
from fractions import Fraction
from itertools import combinations
import math
import numpy as np
from .support import plan_key

class ScanLimit(RuntimeError): pass

def power_sum(matrix):
    return np.asarray([math.fsum(matrix[:,j]) for j in range(matrix.shape[1])])

class Scan:
    def __init__(self, maximum, meter): self.maximum=maximum; self.count=0; self.meter=meter; self.records=[]
    def take(self, source, detail):
        self.meter.check()
        if self.count>=self.maximum: raise ScanLimit('construction scan cap')
        self.count+=1; self.records.append({'source':source,'detail':detail})

def components(rt,nrt,cap):
    rt=np.asarray(rt); cap=np.asarray(cap)
    oldest=rt[...,::-1]; older=np.cumsum(oldest,axis=-1)-oldest
    served=np.minimum(oldest,np.maximum(cap[...,None]-older,0))
    delivered=served.sum(axis=-1)+np.minimum(nrt,np.maximum(cap-served.sum(axis=-1),0))
    weighted=(served*np.arange(rt.shape[-1],0,-1)).sum(axis=-1)
    return delivered,weighted

def value(delivered,weighted,c):
    if c.get('r32_exact_utility',False):
        coefficients=_utility_coefficients(c)
        if coefficients is not None:
            return _value_with_coefficients(delivered,weighted,coefficients)
    return (Fraction(str(c['alpha']))*c['packet_bits']*int(delivered)/Fraction(str(c['B_ref']))
            +(1-Fraction(str(c['alpha'])))*Fraction(str(c['dt']))*int(weighted)
             /int(c['ttl'])/Fraction(str(c['W_ref'])))

def _utility_coefficients(c):
    """One paid, invocation-local reduction; no state or configuration cache.

Only the rational value expression is regrouped. Queue/capacity arithmetic in
components remains untouched, including NumPy integer overflow semantics.
The fallback preserves configurations whose original packet_bits arithmetic
produces floats rather than Fraction values.
    """
    alpha=Fraction(str(c['alpha']))
    bits=alpha*c['packet_bits']/Fraction(str(c['B_ref']))
    wait=(1-alpha)*Fraction(str(c['dt']))/int(c['ttl'])/Fraction(str(c['W_ref']))
    if not isinstance(bits,Fraction) or not isinstance(wait,Fraction):
        return None
    denominator=math.lcm(bits.denominator,wait.denominator)
    return (bits.numerator*(denominator//bits.denominator),
            wait.numerator*(denominator//wait.denominator),denominator)

def _value_with_coefficients(delivered,weighted,coefficients):
    bits,wait,denominator=coefficients
    return Fraction(bits*int(delivered)+wait*int(weighted),denominator)

def utility(rt,nrt,cap,c):
    d,a=components(rt,nrt,cap)
    if c.get('r32_exact_utility',False):
        # Empty arrays previously read no objective coefficients at all.
        if np.ndim(d)>0 and not d.size: return np.empty(d.shape,dtype=object)
        coefficients=_utility_coefficients(c)
        if coefficients is not None:
            if np.ndim(d)==0: return _value_with_coefficients(d,a,coefficients)
            return np.array([_value_with_coefficients(x,y,coefficients)
                             for x,y in zip(d.flat,a.flat)],dtype=object).reshape(d.shape)
    if np.ndim(d)==0: return value(d,a,c)
    return np.array([value(x,y,c) for x,y in zip(d.flat,a.flat)],dtype=object).reshape(d.shape)

def edge_id(e,ch):
    h,s,m=e; return (int(h),int(ch.sat_ids[s]),int(m))

def edges_of(X,ch): return sorted(map(tuple,np.argwhere(X)),key=lambda e:edge_id(e,ch))

def background_metrics(X,tr,ch,meter):
    c=ch.config; phi={}; xi={}; zeta={}; pairs=[]
    exact_utility=c.get('r32_exact_utility',False); phi_coefficients=None; phi_prepared=False
    for h in range(ch.H):
        meter.check()
        t=tr.slots[h]; rt=t['qplus_rt']; nr=t['qplus_nrt']
        planned=np.argwhere(X[h]); active=np.argwhere(t['active'])
        for s,m in planned:
            weighted=int(t['served_age'][m]@np.arange(1,c['ttl']+1))
            if exact_utility and not phi_prepared:
                phi_coefficients=_utility_coefficients(c); phi_prepared=True
            delivered=t['service_rt'][m]+t['service_nrt'][m]
            phi[(h,s,m)]=(_value_with_coefficients(delivered,weighted,phi_coefficients)
                          if phi_coefficients is not None else value(delivered,weighted,c))
            xi[(h,s,m)]=Fraction(0)
        if len(active):
            p=ch.power(h,active,active); np.fill_diagonal(p,0)
            I=power_sum(p)+c.get('I_ext',0.)
            base=utility(rt[active[:,1]],nr[active[:,1]],ch.cap(ch.signal[h,active[:,0],active[:,1]],I),c)
            for i,(s,m) in enumerate(active):
                remain=power_sum(p[np.arange(len(active))!=i])+c.get('I_ext',0.)
                cap=ch.cap(ch.signal[h,active[:,0],active[:,1]],remain)
                after=utility(rt[active[:,1]],nr[active[:,1]],cap,c)
                xi[(h,s,m)]=sum((max(Fraction(0),after[j]-base[j]) for j in range(len(active)) if j!=i),Fraction(0))
        legal=np.argwhere(ch.visible[h]); cap=ch.cap(ch.signal[h,legal[:,0],legal[:,1]],c.get('I_ext',0.))
        vals=utility(rt[legal[:,1]],nr[legal[:,1]],cap,c)
        for (s,m),v,cc in zip(legal,vals,cap): zeta[(h,s,m)]=(v,int(cc),float(ch.signal[h,s,m]))
        if len(planned)>1:
            p=ch.power(h,planned,planned)
            for i,j in combinations(range(len(planned)),2):
                e=(h,*planned[i]); f=(h,*planned[j])
                if edge_id(e,ch)>edge_id(f,ch): e,f=f,e
                pairs.append((e,f,float(p[i,j]+p[j,i])))
    return phi,xi,zeta,pairs

def backgrounds(X,tr,ch,meter,scan):
    """Shared declared (R,D) keys; distinct D survives even for identical Z."""
    phi,xi,zeta,pairs=background_metrics(X,tr,ch,meter)
    all_edges=edges_of(X,ch); seen=set(); out=[]; stats={'generated':0,'absent':0,'duplicates':0,'truncated':False}
    def one():
        for e in sorted(all_edges,key=lambda e:(phi[e],edge_id(e,ch))): yield (e,),((e[0],e[1]),)
    def two():
        for e in sorted(all_edges,key=lambda e:(-xi[e],phi[e],edge_id(e,ch))):
            h,s,m=e; nodes=[s]+np.flatnonzero(ch.graphs[h,s]).tolist()
            yield (e,),tuple(sorted((h,int(j)) for j in nodes))
    def three():
        cells=[]
        for h in range(ch.H):
            for m in np.flatnonzero(ch.visible[h].sum(axis=0)>=2):
                stars=np.flatnonzero(ch.visible[h,:,m]); best=max(zeta[(h,s,m)][0] for s in stars)
                cells.append((-best,h,int(m),stars))
        for _,h,m,stars in sorted(cells,key=lambda x:x[:3]):
            old=np.flatnonzero(X[h,:,m]); s0=int(old[0]) if len(old) else None
            choices=sorted((int(s) for s in stars if s!=s0),key=lambda s:(-zeta[(h,s,m)][0],-zeta[(h,s,m)][1],-zeta[(h,s,m)][2],int(ch.sat_ids[s])))
            for s1 in choices:
                if ch.config['K']<=0: continue
                release=[] if s0 is None else [(h,s0,m)]
                if X[h,s1].sum()>=ch.config['K']:
                    old_edges=[(h,s1,int(n)) for n in np.flatnonzero(X[h,s1])]
                    release.append(min(old_edges,key=lambda e:(phi[e],edge_id(e,ch))))
                D={(h,s1)}
                if s0 is not None: D.add((h,s0))
                yield tuple(release),tuple(sorted(D))
    def four():
        for cross in (True,False):
            group=[p for p in pairs if (p[0][1]!=p[1][1])==cross]
            positive=any(p[2]>0 for p in group)
            key=lambda p:((-p[2],) if positive else ())+ (phi[p[0]]+phi[p[1]],edge_id(p[0],ch),edge_id(p[1],ch))
            for e,f,p in sorted(group,key=key): yield (e,f),tuple(sorted({(e[0],e[1]),(f[0],f[1])}))
    for b,g in enumerate((one,two,three,four),1):
        found=False
        try:
            for R,D in g():
                scan.take('background',{'template':b,'R':R,'D':D})
                R=tuple(sorted(R,key=lambda e:edge_id(e,ch))); D=tuple(sorted(D))
                key=(R,D)
                if key in seen: stats['duplicates']+=1; continue
                if any(not X[e] for e in R): continue
                Z=X.copy()
                for e in R: Z[e]=False
                seen.add(key); out.append({'id':b,'R':R,'D':D,'Z':Z}); found=True; stats['generated']+=1; break
        except ScanLimit:
            stats['truncated']=True; break
        if not found: stats['absent']+=1
    return out,stats,{'phi':[(edge_id(e,ch),v) for e,v in phi.items()],
                     'xi':[(edge_id(e,ch),v) for e,v in xi.items()]}

class LazyBackgrounds:
    """PCQR-only, one request per template with shared declared (R,D) keys.

    Metric arithmetic and template orderings are copied from the frozen eager
    functions above. Call order chooses which duplicate key survives. No plan,
    Replay, phase timer, or cross-reference cache is created here; the caller
    charges construction and each request to background_construction.
    """
    def __init__(self,X,tr,ch,meter,scan):
        self.X=X; self.tr=tr; self.ch=ch; self.meter=meter; self.scan=scan
        self._phi=None; self._xi=None; self._zeta=None; self._pairs=None
        self._edges=None; self._seen=set(); self._requested=set()
        self.stats={'generated':0,'absent':0,'duplicates':0,'truncated':False,
                    'requested_templates':[]}

    @property
    def metrics(self):
        # Missing metrics remain explicitly marked uncomputed. Fraction values
        # use the same canonical support.dumps encoding as eager metrics.
        return {'phi':[(edge_id(e,self.ch),v) for e,v in (self._phi or {}).items()],
                'xi':[(edge_id(e,self.ch),v) for e,v in (self._xi or {}).items()],
                'computed':{'phi':self._phi is not None,'xi':self._xi is not None,
                            'zeta':self._zeta is not None,'pairs':self._pairs is not None},
                'requested_templates':list(self.stats['requested_templates'])}

    @property
    def metric_values(self):
        """Exact completed metric values for finite equivalence gates."""
        return self._phi,self._xi,self._zeta,self._pairs

    def _prepare_phi(self):
        if self._phi is not None: return
        X,tr,ch,meter=self.X,self.tr,self.ch,self.meter
        c=ch.config; phi={}; coefficients=None; prepared=False
        for h in range(ch.H):
            meter.check(); t=tr.slots[h]
            for s,m in np.argwhere(X[h]):
                weighted=int(t['served_age'][m]@np.arange(1,c['ttl']+1))
                if c.get('r32_exact_utility',False) and not prepared:
                    coefficients=_utility_coefficients(c); prepared=True
                delivered=t['service_rt'][m]+t['service_nrt'][m]
                phi[(h,s,m)]=(_value_with_coefficients(delivered,weighted,coefficients)
                              if coefficients is not None else value(delivered,weighted,c))
        self._phi=phi

    def _prepare_xi(self):
        if self._xi is not None: return
        X,tr,ch,meter=self.X,self.tr,self.ch,self.meter
        c=ch.config; xi={}
        for h in range(ch.H):
            meter.check(); t=tr.slots[h]; rt=t['qplus_rt']; nr=t['qplus_nrt']
            for s,m in np.argwhere(X[h]): xi[(h,s,m)]=Fraction(0)
            active=np.argwhere(t['active'])
            if len(active):
                p=ch.power(h,active,active); np.fill_diagonal(p,0)
                I=power_sum(p)+c.get('I_ext',0.)
                base=utility(rt[active[:,1]],nr[active[:,1]],ch.cap(ch.signal[h,active[:,0],active[:,1]],I),c)
                for i,(s,m) in enumerate(active):
                    remain=power_sum(p[np.arange(len(active))!=i])+c.get('I_ext',0.)
                    cap=ch.cap(ch.signal[h,active[:,0],active[:,1]],remain)
                    after=utility(rt[active[:,1]],nr[active[:,1]],cap,c)
                    xi[(h,s,m)]=sum((max(Fraction(0),after[j]-base[j]) for j in range(len(active)) if j!=i),Fraction(0))
        self._xi=xi

    def _prepare_zeta(self):
        if self._zeta is not None: return
        tr,ch,meter=self.tr,self.ch,self.meter
        c=ch.config; zeta={}
        for h in range(ch.H):
            meter.check(); t=tr.slots[h]; rt=t['qplus_rt']; nr=t['qplus_nrt']
            legal=np.argwhere(ch.visible[h]); cap=ch.cap(ch.signal[h,legal[:,0],legal[:,1]],c.get('I_ext',0.))
            vals=utility(rt[legal[:,1]],nr[legal[:,1]],cap,c)
            for (s,m),v,cc in zip(legal,vals,cap): zeta[(h,s,m)]=(v,int(cc),float(ch.signal[h,s,m]))
        self._zeta=zeta

    def _prepare_pairs(self):
        if self._pairs is not None: return
        X,ch,meter=self.X,self.ch,self.meter
        pairs=[]
        for h in range(ch.H):
            meter.check(); planned=np.argwhere(X[h])
            if len(planned)>1:
                p=ch.power(h,planned,planned)
                for i,j in combinations(range(len(planned)),2):
                    e=(h,*planned[i]); f=(h,*planned[j])
                    if edge_id(e,ch)>edge_id(f,ch): e,f=f,e
                    pairs.append((e,f,float(p[i,j]+p[j,i])))
        self._pairs=pairs

    def _candidates(self,template_id):
        X,ch=self.X,self.ch; phi=self._phi
        if self._edges is None: self._edges=edges_of(X,ch)
        if template_id==1:
            for e in sorted(self._edges,key=lambda e:(phi[e],edge_id(e,ch))):
                yield (e,),((e[0],e[1]),)
        elif template_id==2:
            for e in sorted(self._edges,key=lambda e:(-self._xi[e],phi[e],edge_id(e,ch))):
                h,s,m=e; nodes=[s]+np.flatnonzero(ch.graphs[h,s]).tolist()
                yield (e,),tuple(sorted((h,int(j)) for j in nodes))
        elif template_id==3:
            zeta=self._zeta; cells=[]
            for h in range(ch.H):
                for m in np.flatnonzero(ch.visible[h].sum(axis=0)>=2):
                    stars=np.flatnonzero(ch.visible[h,:,m]); best=max(zeta[(h,s,m)][0] for s in stars)
                    cells.append((-best,h,int(m),stars))
            for _,h,m,stars in sorted(cells,key=lambda x:x[:3]):
                old=np.flatnonzero(X[h,:,m]); s0=int(old[0]) if len(old) else None
                choices=sorted((int(s) for s in stars if s!=s0),key=lambda s:(-zeta[(h,s,m)][0],-zeta[(h,s,m)][1],-zeta[(h,s,m)][2],int(ch.sat_ids[s])))
                for s1 in choices:
                    if ch.config['K']<=0: continue
                    release=[] if s0 is None else [(h,s0,m)]
                    if X[h,s1].sum()>=ch.config['K']:
                        old_edges=[(h,s1,int(n)) for n in np.flatnonzero(X[h,s1])]
                        release.append(min(old_edges,key=lambda e:(phi[e],edge_id(e,ch))))
                    D={(h,s1)}
                    if s0 is not None: D.add((h,s0))
                    yield tuple(release),tuple(sorted(D))
        else:
            for cross in (True,False):
                group=[p for p in self._pairs if (p[0][1]!=p[1][1])==cross]
                positive=any(p[2]>0 for p in group)
                key=lambda p:((-p[2],) if positive else ())+ (phi[p[0]]+phi[p[1]],edge_id(p[0],ch),edge_id(p[1],ch))
                for e,f,p in sorted(group,key=key):
                    yield (e,f),tuple(sorted({(e[0],e[1]),(f[0],f[1])}))

    def next(self,template_id):
        if template_id not in (1,2,3,4): raise ValueError('unknown background template')
        if template_id in self._requested: raise ValueError('background template requested twice')
        if self.stats['truncated']: raise ScanLimit('construction scan cap already reached')
        self._requested.add(template_id); self.stats['requested_templates'].append(template_id)
        self._prepare_phi()
        if template_id==2: self._prepare_xi()
        elif template_id==3: self._prepare_zeta()
        elif template_id==4: self._prepare_pairs()
        try:
            for R,D in self._candidates(template_id):
                self.scan.take('background',{'template':template_id,'R':R,'D':D})
                R=tuple(sorted(R,key=lambda e:edge_id(e,self.ch))); D=tuple(sorted(D))
                key=(R,D)
                if key in self._seen: self.stats['duplicates']+=1; continue
                if any(not self.X[e] for e in R): continue
                Z=self.X.copy()
                for e in R: Z[e]=False
                self._seen.add(key); self.stats['generated']+=1
                return {'id':template_id,'R':R,'D':D,'Z':Z}
        except ScanLimit:
            self.stats['truncated']=True
            raise
        self.stats['absent']+=1
        return None

def direct_candidates(X,ch,meter,scan):
    old=edges_of(X,ch)
    def delete():
        for e in old: yield (e,),()
    def insert():
        for e in edges_of(ch.visible & ~X,ch):
            h,s,m=e; occupied=np.flatnonzero(X[h,:,m]); R=[(h,int(t),m) for t in occupied]
            remaining=int(X[h,s].sum())-sum(t==s for _,t,_ in R)
            if remaining>=ch.config['K']:
                for n in np.flatnonzero(X[h,s]):
                    f=(h,s,int(n))
                    if f not in R: yield tuple(R+[f]),(e,)
            else: yield tuple(R),(e,)
    def swap():
        for e,f in combinations(old,2):
            h,s,m=e; t,j,n=f
            yield (e,f),((h,s,n),(t,j,m))
    def shift():
        for e in old:
            h,s,m=e
            for t in range(ch.H):
                if t!=h: yield (e,),((t,s,m),)
    streams=[iter(g()) for g in (delete,insert,swap,shift)]; names=['DELETE','INSERT_REPLACE','SWAP_TWO','SHIFT_ONE']
    live=[True]*4
    while any(live):
        for i,stream in enumerate(streams):
            if not live[i]: continue
            with meter.phase('direct_generation'):
                try: R,A=next(stream)
                except StopIteration: live[i]=False; continue
                scan.take('direct',{'type':names[i],'R':R,'A':A})
                Y=X.copy()
                for e in R: Y[e]=False
                for e in A: Y[e]=True
                # Invalid/unchanged attempts are deliberately yielded and paid.
            yield Y,{'type':names[i],'R':R,'A':A}

def score_background(bg,tr,ch,meter):
    Z=bg['Z']; c=ch.config; kappa=c['K']-Z.sum(axis=2); chi=1-Z.sum(axis=1)
    edges=[]; ws=[]; mus=[]
    for h,s in bg['D']:
        meter.check()
        if kappa[h,s]<=0: continue
        cells=np.flatnonzero(ch.visible[h,s] & (chi[h]>0) & ~Z[h,s])
        if not len(cells): continue
        t=tr.slots[h]; rt=t['qplus_rt']; nr=t['qplus_nrt']; active=np.argwhere(t['active'])
        candidates=np.column_stack((np.full(len(cells),s),cells)); alive=rt[cells].sum(axis=1)+nr[cells]>0
        with meter.phase('w'):
            p=ch.power(h,active,candidates)
            caps=ch.cap(ch.signal[h,s,cells],c.get('I_ext',0.)+power_sum(p))
            w=utility(rt[cells],nr[cells],caps,c)
        with meter.phase('mu'):
            mu=np.full(len(cells),Fraction(0),dtype=object)
            if len(active):
                matrix=ch.power(h,active,active); np.fill_diagonal(matrix,0)
                I=c.get('I_ext',0.)+power_sum(matrix)
                vmask=(active[:,0]==s)|ch.graphs[h,s,active[:,0]]
                victims=active[vmask]; baseI=I[vmask]
                if len(victims):
                    base=utility(rt[victims[:,1]],nr[victims[:,1]],ch.cap(ch.signal[h,victims[:,0],victims[:,1]],baseI),c)
                    added=ch.power(h,candidates,victims)
                    cap_after=ch.cap(ch.signal[h,victims[:,0],victims[:,1]][None,:],baseI[None,:]+added)
                    after=utility(rt[victims[:,1]][None,:,:],nr[victims[:,1]][None,:],cap_after,c)
                    for i in range(len(cells)):
                        if alive[i]: mu[i]=sum((max(Fraction(0),base[j]-after[i,j]) for j in range(len(victims)) if victims[j,1]!=cells[i]),Fraction(0))
        edges.extend((h,s,int(m)) for m in cells); ws.extend(w.tolist()); mus.extend(mu.tolist())
    return np.asarray(edges,dtype=int).reshape(-1,3),ws,mus,kappa
