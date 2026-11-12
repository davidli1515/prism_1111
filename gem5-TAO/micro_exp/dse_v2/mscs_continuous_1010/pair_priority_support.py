"""Dense single-resource grid and a common automatic pair universe."""
from secant_support import config_key,compute_w_min
from core import PARAMS,STRIDE,get_param_values,legal
from width_priority_support import width_moves


def dense_singles(root,seen):
    used=set(seen)|{config_key(root)};out=[]
    for name in PARAMS:
        vals=get_param_values(name)
        for sign in (1,-1):
            for stride in (1,2,3,4):
                idx=max(0,min(len(vals)-1,vals.index(root[name])+sign*stride*STRIDE.get(name,1)))
                c=dict(root,**{name:vals[idx]});key=config_key(c)
                if key in used or not legal(c) or c['fetchBufferSize']<16:continue
                used.add(key);out.append(dict(parameter=name,sign=sign,stride=stride,kind='single_resource',config=c,
                    edits={name:dict(before=root[name],after=c[name])}))
    for a in width_moves(root,4,seen):
        key=config_key(a['config'])
        if a['kind']!='width_with_explicit_closure' or key in used or a['config']['fetchBufferSize']<16:continue
        used.add(key);out.append(a)
    return out


def pair_universe(root,singles,eligible,seen):
    out=[];used=set(seen)|{config_key(root)}
    for ii,i in enumerate(eligible):
        a=singles[i]
        assert a['kind']=='single_resource' and a['sign']==1
        for j in eligible[ii+1:]:
            b=singles[j]
            assert b['kind']=='single_resource' and b['sign']==1
            if a['parameter']==b['parameter']:continue
            c=dict(root);c[a['parameter']]=a['config'][a['parameter']];c[b['parameter']]=b['config'][b['parameter']]
            key=config_key(c)
            if key in used or not legal(c):continue
            assert compute_w_min(c)==compute_w_min(root)
            used.add(key);out.append(dict(indices=[i,j],config=c,kind='two_resource_increase',
                edits={k:dict(before=root[k],after=c[k]) for k in root if c[k]!=root[k]}))
    return out
