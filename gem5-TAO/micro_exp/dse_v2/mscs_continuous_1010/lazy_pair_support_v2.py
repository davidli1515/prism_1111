"""Exact short-circuit feasibility for the same individually-feasible pair domain."""
from secant_support import config_key


def choose_pair(pairs,scores,singles,area_get,cap):
    """No performance labels. The callback returns (exact area or None, error)."""
    known={};trace=[];choice=None;resolved=True
    for index in sorted(range(len(pairs)),key=lambda i:(float(scores[i]),i)):
        pair=pairs[index];record=dict(pair_index=index,endpoint_checks=[],pair_area=None,status=None)
        skip=False
        # Reuse known rejections BEFORE requesting an unknown partner's area.
        def check_order(sid):
            key=config_key(singles[sid]['config'])
            if key not in known:return 2
            aa,_=known[key]
            return 0 if aa is None or aa>cap+1e-9 else 1
        ordered=sorted(pair['indices'],key=check_order)
        for sid in ordered:
            c=singles[sid]['config'];key=config_key(c);fresh=key not in known
            if fresh:known[key]=area_get(c)
            aa,error=known[key];record['endpoint_checks'].append(dict(index=sid,area=aa,error=error,new_request=fresh))
            if aa is None:
                record['status']='unknown_endpoint_area';resolved=False;skip=True;break
            if aa>cap+1e-9:record['status']='endpoint_outside_declared_domain';skip=True;break
        if skip:
            trace.append(record)
            if not resolved:break
            continue
        aa,error=area_get(pair['config']);record['pair_area']=aa;record['pair_error']=error
        if aa is None:record['status']='unknown_joint_area';resolved=False;trace.append(record);break
        if aa<=cap+1e-9:record['status']='selected';choice=index;trace.append(record);break
        record['status']='joint_over_cap';trace.append(record)
    return dict(choice=choice,resolved=resolved,trace=trace,
        endpoint_requests=sum(r['new_request'] for t in trace for r in t['endpoint_checks']),
        joint_requests=sum(t['status'] in ('selected','joint_over_cap','unknown_joint_area') for t in trace))
