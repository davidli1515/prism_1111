"""Base-only quadratic scalar control for changing known pipeline width."""
import torch
from quadratic_support import quadratic_fit
from width_priority_support import predict


def scalar_response(m):
    b=m[...,0].clamp(min=1e-30,max=1-1e-7)
    return torch.log1p(-b)-torch.log(b)


def scalar_quadratic(local,xs,widths,hx,hy):
    delta=xs-local.x[None];hd=hx-local.x[None]
    root_u=torch.logsumexp(local.ell,-1)
    gu=(torch.softmax(local.ell,-1)[...,None]*local.gell).sum(2)
    past=torch.einsum('nsd,hnd->hns',gu,hd)
    slope=torch.einsum('nsd,cnd->cns',gu,delta)
    residual=hy-root_u[None]-past
    correction,info=quadratic_fit(delta,hd,residual)
    u=root_u[None]+slope+correction;b=torch.sigmoid(-u)
    w=torch.tensor(widths,dtype=b.dtype)[:,None,None]
    stages=(1/(b.clamp(min=.01)*w+1e-8)).clamp(max=100.)
    return stages.mean((1,2)),dict(info,root_u=root_u,history_response=hy,history_slope=past,
        scalar_u_slope=slope,correction=correction,corrected_stages=stages)
