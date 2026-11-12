"""Online local response correction; the checkpoint is read-only."""
from pathlib import Path
import sys
import torch

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'mscs_fresh_disagreement_1010'))
from fresh_disagreement_support import (read,sha,save,array_save,LocalModel,Inputs,
    FeatureCache,load_model,exact,config_key,compute_w_min,moves,contributions,pressure)
from core import cpis

ARMS=('scalar','mscs','scalar_secant','mscs_secant')
HISTORY=6


def response(m,width,space):
    if space=='scalar':
        return torch.log((cpis(m,width)-1/width).clamp(min=1e-10))
    assert space=='mscs'
    return torch.log(m[...,1:].clamp(min=1e-30))-torch.log(m[...,:1].clamp(min=1e-30))


def fit_correction(candidate_delta,history_delta,residual):
    """API knows only response residuals, not labels or candidate outcomes."""
    c,n,_=candidate_delta.shape;h=history_delta.shape[0]
    shape=residual.shape[2:]
    if h==0:
        return torch.zeros((c,n,*shape),dtype=candidate_delta.dtype),dict()
    hd=history_delta.double();cd=candidate_delta.double()
    gram=torch.einsum('hnd,knd->nhk',hd,hd)
    cross=torch.einsum('cnd,hnd->nch',cd,hd)
    inverse=torch.linalg.pinv(gram,rtol=1e-6,hermitian=True)
    weights=torch.einsum('nch,nhk->nck',cross,inverse)
    flat=residual.permute(1,0,*range(2,residual.ndim)).reshape(n,h,-1).double()
    correction=torch.einsum('nch,nhv->ncv',weights,flat).permute(1,0,2).reshape(c,n,*shape)
    return correction.to(candidate_delta.dtype),dict(gram=gram,cross=cross,weights=weights,residual=residual)


def predict(local,xs,base,space,history_xs,history_response):
    """Scalar is passed transformed CPI only. MSCS has its own richer response."""
    delta=xs-local.x[None];hd=history_xs-local.x[None]
    lo=1/local.width;hi=min(100.,1/(.01*local.width+1e-8))
    excess=(local.cs-lo).clamp(min=1e-10)
    if space=='scalar':
        anchor=torch.log(excess)
        past_slope=torch.einsum('nsd,hnd->hns',local.gc,hd)/excess[None]
        slope=base['stage_slope']/excess[None]
    else:
        anchor=local.ell
        past_slope=torch.einsum('nskd,hnd->hnsk',local.gell,hd)
        slope=base['logodds_slope']
    residual=history_response-anchor[None]-past_slope
    correction,info=fit_correction(delta,hd,residual)
    if space=='scalar':
        stages=(lo+excess[None]*torch.exp((slope+correction).clamp(-80,80))).clamp(lo,hi)
        m=None
    else:
        ell=anchor[None]+slope+correction
        m=torch.softmax(torch.cat([torch.zeros_like(ell[...,:1]),ell],-1),-1)
        stages=cpis(m,local.width)
    assert torch.isfinite(stages).all()
    return stages.mean((1,2)),m,dict(info,history_response=history_response,
        history_slope=past_slope,correction=correction,corrected_stage_cpi=stages)
