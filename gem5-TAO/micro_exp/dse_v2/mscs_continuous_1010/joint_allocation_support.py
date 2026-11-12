"""Constrained resource-allocation proposals from two finite local models."""
import numpy as np
from scipy.optimize import minimize
from secant_support import config_key
from core import legal,get_param_values


def objective(w,kind,root_cs,root_ell,stage_dirs,logodds_dirs,width):
    """Return normalized mean CPI and its analytic gradient on a feature chord."""
    lo=1/width;hi=min(100.,1/(.01*width+1e-8));denom=float(root_cs.mean())
    if kind=='scalar':
        ex=np.maximum(root_cs-lo,1e-10);ds=stage_dirs/ex[None]
        arg=np.einsum('r,rns->ns',w,ds);ec=np.exp(np.clip(arg,-80,80))
        raw=lo+ex*ec;c=np.clip(raw,lo,hi)
        active=(raw>lo)&(raw<hi)&(arg>-80)&(arg<80)
        dc=stage_dirs*ec[None]*active[None]
    else:
        assert kind=='mscs'
        ell=root_ell+np.einsum('r,rnsk->nsk',w,logodds_dirs)
        ep=np.exp(np.clip(ell,-80,80));total=1+ep.sum(-1);base=1/total
        raw=1/(np.maximum(base,.01)*width+1e-8);c=np.minimum(100.,raw)
        active=(base>.01)&(raw<100.)
        weight=width*ep/(width+1e-8*total[...,None])**2
        weight*=active[...,None]*(ell>-80)*(ell<80)
        dc=np.einsum('nsk,rnsk->rns',weight,logodds_dirs)
    return float(c.mean()/denom),dc.mean((1,2))/denom


def allocation(kind,root,basis,area_delta,area_cap,root_area,root_cs,root_ell,stage_dirs,logodds_dirs,width):
    n=len(basis);slack=root['numROBEntries']-2*root['dispatchWidth']-root['numIQEntries']-root['LQEntries']-root['SQEntries']
    capacity=np.array([(1 if a['parameter']=='numROBEntries' else -1 if a['parameter'] in ('numIQEntries','LQEntries','SQEntries') else 0)*
                       (a['config'][a['parameter']]-root[a['parameter']]) for a in basis],dtype=float)
    costs=np.array(area_delta)/root_area;budget=(area_cap-root_area)/root_area
    fn=lambda w:objective(w,kind,root_cs,root_ell,stage_dirs,logodds_dirs,width)
    constraints=[dict(type='ineq',fun=lambda w:budget-costs@w,jac=lambda w:-costs),
                 dict(type='ineq',fun=lambda w:slack+capacity@w,jac=lambda w:capacity)]
    result=minimize(fn,np.zeros(n),jac=True,method='SLSQP',bounds=[(0.,1.)]*n,
                    constraints=constraints,options=dict(ftol=1e-10,maxiter=100))
    w=np.clip(result.x,0,1)
    return dict(fractions=w.tolist(),success=bool(result.success),message=str(result.message),iterations=int(result.nit),
                normalized_predicted_cpi=float(result.fun),area_constraint_slack=float(budget-costs@w),
                capacity_constraint_slack=float(slack+capacity@w),area_delta=list(map(float,area_delta)))


def round_configs(root,basis,fractions,area_delta,root_area,cap):
    out=[];seen=set();bounds={a['parameter']:a['config'][a['parameter']] for a in basis}
    for mode in ('floor','nearest'):
        c=dict(root)
        for a,f in zip(basis,fractions):
            k=a['parameter'];target=root[k]+f*(a['config'][k]-root[k])
            values=[v for v in get_param_values(k) if root[k]<=v<=a['config'][k]]
            c[k]=max(v for v in values if v<=target+1e-10) if mode=='floor' else min(values,key=lambda v:(abs(v-target),v))
        minrob=2*c['dispatchWidth']+c['numIQEntries']+c['LQEntries']+c['SQEntries'];repair=None
        if c['numROBEntries']<minrob:
            values=[v for v in get_param_values('numROBEntries') if minrob<=v<=bounds.get('numROBEntries',root['numROBEntries'])]
            if not values:continue
            repair=dict(parameter='numROBEntries',before=c['numROBEntries'],after=min(values));c['numROBEntries']=min(values)
        effective=np.array([(c[a['parameter']]-root[a['parameter']])/(a['config'][a['parameter']]-root[a['parameter']]) for a in basis])
        area=float(root_area+np.dot(effective,area_delta));key=config_key(c)
        if key in seen or c==root or not legal(c) or area>cap+1e-9:continue
        seen.add(key);out.append(dict(config=c,rounding=mode,repair=repair,fractions=effective.tolist(),predicted_area=area))
    return out
