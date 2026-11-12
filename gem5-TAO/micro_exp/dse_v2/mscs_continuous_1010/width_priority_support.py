"""Variable-width local models and explicit common constraint-closed actions."""
import torch
from secant_support import config_key,compute_w_min
from core import moves,legal,get_param_values,probabilities,cpis

WIDTHS=('fetchWidth','decodeWidth','dispatchWidth','commitWidth')


def close_width(root,name,target):
    c=dict(root);c[name]=target
    if name=='fetchWidth':
        c['decodeWidth']=min(c['decodeWidth'],target)
        c['dispatchWidth']=min(c['dispatchWidth'],c['decodeWidth'])
    elif name=='decodeWidth':
        c['fetchWidth']=max(c['fetchWidth'],target)
        c['dispatchWidth']=min(c['dispatchWidth'],target)
    elif name=='dispatchWidth':
        c['decodeWidth']=max(c['decodeWidth'],target)
        c['fetchWidth']=max(c['fetchWidth'],c['decodeWidth'])
        c['commitWidth']=max(c['commitWidth'],target)
        c['wbWidth']=max(c['wbWidth'],target)
    else:
        assert name=='commitWidth'
        c['dispatchWidth']=min(c['dispatchWidth'],target)
    needed=2*c['dispatchWidth']+c['numIQEntries']+c['LQEntries']+c['SQEntries']
    if c['numROBEntries']<needed:
        values=[v for v in get_param_values('numROBEntries') if v>=needed]
        if not values:return None
        c['numROBEntries']=min(values)
    return c if legal(c) else None


def width_moves(root,radius=4,seen=()):
    pool=moves(root,radius,seen);used=set(seen)|{config_key(a['config']) for a in pool}|{config_key(root)}
    for a in pool:
        a['edits']={k:dict(before=root[k],after=a['config'][k]) for k in root if root[k]!=a['config'][k]}
        a['kind']='single_resource'
    for name in WIDTHS:
        vals=get_param_values(name)
        for sign in (1,-1):
            for stride in (1,2,4):
                if stride>radius:continue
                target=max(vals[0],min(vals[-1],root[name]+sign*stride))
                c=close_width(root,name,target)
                if c is None or config_key(c) in used:continue
                used.add(config_key(c))
                pool.append(dict(parameter=name,sign=sign,stride=stride,config=c,kind='width_with_explicit_closure',
                    edits={k:dict(before=root[k],after=c[k]) for k in root if root[k]!=c[k]}))
    assert all(legal(a['config']) and a['config']['l1_size']==root['l1_size'] and a['config']['bp_type']==root['bp_type'] for a in pool)
    return pool


def predict(root_ell,logodds_slope,widths):
    """No candidate labels. Scalar only uses the aggregate u and its directional derivative."""
    u=torch.logsumexp(root_ell,-1)
    du=(torch.softmax(root_ell,-1)[None]*logodds_slope).sum(-1)
    base=torch.sigmoid(-(u[None]+du))
    w=torch.as_tensor(widths,dtype=base.dtype)[:,None,None]
    scalar=(1/(base.clamp(min=.01)*w+1e-8)).clamp(max=100.)
    ell=root_ell[None]+logodds_slope
    mm=torch.softmax(torch.cat([torch.zeros_like(ell[...,:1]),ell],-1),-1)
    mc=cpis(mm,w)
    return dict(scalar_cpi=scalar.mean((1,2)),mscs_cpi=mc.mean((1,2)),scalar_stage=scalar,
        mscs_stage=mc,predicted_m=mm,root_u=u,scalar_u_slope=du)


def check_scalar_gradient(model,local):
    xx=local.x.detach().requires_grad_(True);z,_=probabilities(model,xx)
    u=torch.logsumexp(z[...,1:]-z[...,:1],-1)
    direct=torch.stack([torch.autograd.grad(u[:,s].sum(),xx,retain_graph=s<3)[0] for s in range(4)],1)
    analytic=(torch.softmax(local.ell,-1)[...,None]*local.gell).sum(2)
    torch.testing.assert_close(direct,analytic,rtol=3e-4,atol=2e-6)
    return dict(passed=True,max_abs_error=float((direct-analytic).abs().max()),diagnostic_forwards=1,diagnostic_backward_outputs=4)
