"""The same fixed quadratic correction applied to component relative logits."""
import torch
from quadratic_support import quadratic_fit
from core import cpis


def mscs_response(m):
    return torch.log(m[...,1:].clamp(min=1e-30))-torch.log(m[...,:1].clamp(min=1e-30))


def mscs_quadratic(local,xs,widths,hx,hy):
    hd=hx-local.x[None];delta=xs-local.x[None]
    past=torch.einsum('nskd,hnd->hnsk',local.gell,hd)
    slope=torch.einsum('nskd,cnd->cnsk',local.gell,delta)
    residual=hy-local.ell[None]-past;correction,info=quadratic_fit(delta,hd,residual)
    ell=local.ell[None]+slope+correction
    mm=torch.softmax(torch.cat([torch.zeros_like(ell[...,:1]),ell],-1),-1)
    w=torch.tensor(widths,dtype=mm.dtype)[:,None,None];stages=cpis(mm,w)
    return stages.mean((1,2)),dict(info,history_response=hy,history_slope=past,logodds_slope=slope,
        correction=correction,corrected_m=mm,corrected_stages=stages)
