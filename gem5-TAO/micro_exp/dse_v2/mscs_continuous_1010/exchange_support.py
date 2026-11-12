"""Signed resource variables with no simultaneous up/down occupancy."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax,expit,logsumexp
from secant_support import config_key
from core import legal,get_param_values


def weights(v,basis,names):
    """One-sided derivative at zero; each parameter occupies at most one edge."""
    ids={k:i for i,k in enumerate(names)};positive={a['parameter'] for a in basis if a['sign']==1}
    w=np.zeros(len(basis));jac=np.zeros((len(basis),len(names)))
    for j,a in enumerate(basis):
        i=ids[a['parameter']];s=a['sign']
        if s*v[i]>0 or (v[i]==0 and (s==1 or a['parameter'] not in positive)):
            w[j]=s*v[i];jac[j,i]=s
    return w,jac


def response(v,kind,basis,names,ell,dirs,width):
    w,jac=weights(v,basis,names);de=np.einsum('r,rnsk->nsk',w,dirs)
    if kind=='scalar':
        q=softmax(ell,-1);u=logsumexp(ell,-1)+np.sum(q*de,-1);b=expit(-u)
        raw=1/(np.maximum(b,.01)*width+1e-8);cs=np.minimum(100.,raw)
        factor=width*b*(1-b)*cs**2*(b>.01)*(raw<100)
        edge=np.einsum('nsk,rnsk->rns',q,dirs)*factor[None]
    else:
        assert kind=='mscs';e=ell+de
        m=softmax(np.concatenate([np.zeros_like(e[...,:1]),e],-1),-1);b=m[...,0]
        raw=1/(np.maximum(b,.01)*width+1e-8);cs=np.minimum(100.,raw)
        factor=width*b[...,None]*m[...,1:]*cs[...,None]**2*((b>.01)&(raw<100))[...,None]
        edge=np.einsum('nsk,rnsk->rns',factor,dirs)
    return float(cs.mean()),edge.mean((1,2))@jac


def bounds_for(basis,names):
    return [(-1. if any(a['parameter']==k and a['sign']==-1 for a in basis) else 0.,
              1. if any(a['parameter']==k and a['sign']==1 for a in basis) else 0.) for k in names]


def solve(kind,root,basis,names,areas,root_area,ell,dirs,width,root_cpi):
    bounds=bounds_for(basis,names);da=np.asarray(areas)-root_area
    capacity=np.array([(1 if a['parameter']=='numROBEntries' else -1 if a['parameter'] in ('numIQEntries','LQEntries','SQEntries') else 0)*
        (a['config'][a['parameter']]-root[a['parameter']]) for a in basis],dtype=float)
    slack=root['numROBEntries']-2*root['dispatchWidth']-root['numIQEntries']-root['LQEntries']-root['SQEntries']
    def area(v):
        w,j=weights(v,basis,names);return float(-da@w/root_area),-da@j/root_area
    def cap(v):
        w,j=weights(v,basis,names);return float(slack+capacity@w),capacity@j
    def fn(v):
        f,g=response(v,kind,basis,names,ell,dirs,width);return f/root_cpi,g/root_cpi
    starts=[np.zeros(len(names)),np.array([.25*hi for lo,hi in bounds]),np.array([.25*lo for lo,hi in bounds])]
    results=[]
    for start in starts:
        r=minimize(fn,start,jac=True,method='SLSQP',bounds=bounds,
            constraints=[dict(type='ineq',fun=lambda v:area(v)[0],jac=lambda v:area(v)[1]),dict(type='ineq',fun=lambda v:cap(v)[0],jac=lambda v:cap(v)[1])],
            options=dict(ftol=1e-10,maxiter=100))
        v=np.clip(r.x,[p[0] for p in bounds],[p[1] for p in bounds]);finite=bool(np.isfinite(v).all())
        results.append(dict(start=start.tolist(),signed_fractions=v.tolist(),success=bool(r.success),message=str(r.message),iterations=int(r.nit),
            normalized_predicted_cpi=fn(v)[0] if finite else None,area_slack=area(v)[0] if finite else None,
            capacity_slack=cap(v)[0] if finite else None,admissible=finite and area(v)[0]>=-1e-7 and cap(v)[0]>=-1e-7))
    return results


def rounding(root,basis,names,v):
    w,_=weights(np.asarray(v),basis,names);targets={k:float(root[k]) for k in names}
    low=dict(root);high=dict(root)
    for a,f in zip(basis,w):
        k=a['parameter'];targets[k]+=f*(a['config'][k]-root[k]);low[k]=min(low[k],a['config'][k]);high[k]=max(high[k],a['config'][k])
    seen=set();out=[]
    for mode in ('floor','nearest'):
        c=dict(root)
        for k in names:
            values=[t for t in get_param_values(k) if low[k]<=t<=high[k]]
            c[k]=max(t for t in values if t<=targets[k]+1e-9) if mode=='floor' else min(values,key=lambda t:(abs(t-targets[k]),t))
        needed=2*c['dispatchWidth']+c['numIQEntries']+c['LQEntries']+c['SQEntries'];repair=None
        if c['numROBEntries']<needed:
            if needed>high['numROBEntries']:continue
            repair=dict(before=c['numROBEntries'],after=needed);c['numROBEntries']=needed
        key=config_key(c)
        if c==root or not legal(c) or key in seen:continue
        seen.add(key);out.append(dict(config=c,rounding=mode,repair=repair,
            edits={k:dict(before=root[k],after=c[k]) for k in root if c[k]!=root[k]}))
    return out
