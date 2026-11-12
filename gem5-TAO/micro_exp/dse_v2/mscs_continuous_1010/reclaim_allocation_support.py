"""Area-minimizing resource shrinkage with a finite-response CPI constraint."""
import numpy as np
from scipy.optimize import minimize
from joint_allocation_support import objective
from core import legal,get_param_values
from secant_support import config_key

TOLERANCE=.003


def allocation(kind,root,basis,area_delta,root_area,root_cs,root_ell,stage_dirs,logodds_dirs,width):
    slack=root['numROBEntries']-2*root['dispatchWidth']-root['numIQEntries']-root['LQEntries']-root['SQEntries']
    capacity=np.array([(1 if a['parameter']=='numROBEntries' else -1 if a['parameter'] in ('numIQEntries','LQEntries','SQEntries') else 0)*
        (a['config'][a['parameter']]-root[a['parameter']]) for a in basis],dtype=float)
    costs=np.array(area_delta)/root_area
    response=lambda w:objective(w,kind,root_cs,root_ell,stage_dirs,logodds_dirs,width)
    constraints=[dict(type='ineq',fun=lambda w:1+TOLERANCE-response(w)[0],jac=lambda w:-response(w)[1]),
                 dict(type='ineq',fun=lambda w:slack+capacity@w,jac=lambda w:capacity)]
    result=minimize(lambda w:(float(costs@w),costs),np.zeros(len(basis)),jac=True,method='SLSQP',bounds=[(0.,1.)]*len(basis),
        constraints=constraints,options=dict(ftol=1e-10,maxiter=100))
    w=np.clip(result.x,0,1)
    return dict(fractions=w.tolist(),success=bool(result.success),message=str(result.message),iterations=int(result.nit),
        normalized_predicted_cpi=response(w)[0],predicted_area=float(root_area+np.dot(area_delta,w)),
        cpi_constraint_slack=float(1+TOLERANCE-response(w)[0]),capacity_constraint_slack=float(slack+capacity@w))


def area_estimate(root,basis,area_delta,root_area,c):
    fractions=np.array([(c[a['parameter']]-root[a['parameter']])/(a['config'][a['parameter']]-root[a['parameter']]) for a in basis])
    return float(root_area+np.dot(fractions,area_delta)),fractions.tolist()


def round_configs(root,basis,fractions,area_delta,root_area):
    out=[];seen=set()
    for mode in ('floor','nearest'):
        c=dict(root)
        for a,f in zip(basis,fractions):
            k=a['parameter'];target=root[k]+f*(a['config'][k]-root[k])
            values=[v for v in get_param_values(k) if a['config'][k]<=v<=root[k]]
            c[k]=max(v for v in values if v<=target+1e-10) if mode=='floor' else min(values,key=lambda v:(abs(v-target),v))
        minrob=2*c['dispatchWidth']+c['numIQEntries']+c['LQEntries']+c['SQEntries'];repair=None
        if c['numROBEntries']<minrob:
            values=[v for v in get_param_values('numROBEntries') if minrob<=v<=root['numROBEntries']]
            if not values:continue
            repair=dict(parameter='numROBEntries',before=c['numROBEntries'],after=min(values));c['numROBEntries']=min(values)
        key=config_key(c);area,effective=area_estimate(root,basis,area_delta,root_area,c)
        if key in seen or c==root or not legal(c) or area>=root_area:continue
        seen.add(key);out.append(dict(config=c,rounding=mode,repair=repair,fractions=effective,predicted_area=area))
    return out


def outcome(root_area,cpi_cap,trials):
    known=[r for r in trials if r['cpi']<=cpi_cap and r['area'] is not None and r['area']<root_area]
    best=min(known,key=lambda r:r['area'],default=None)
    unknown=[r for r in trials if r['cpi']<=cpi_cap and r['area'] is None]
    high=root_area if best is None else best['area'];low=0. if unknown else high
    return dict(resolved=not unknown,area_lower=low,area_upper=high,best_area=high if not unknown else None,
        saving_pct=100*(1-high/root_area) if not unknown else None,
        accepted_config=None if best is None or unknown else best['config'],
        accepted_cpi=None if best is None or unknown else best['cpi'])
