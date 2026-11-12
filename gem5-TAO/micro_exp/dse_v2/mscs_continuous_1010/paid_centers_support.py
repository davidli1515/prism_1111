"""All-paid response interpolation with exact candidate pipeline widths."""
import torch
from shepard_support import blend_responses


def predict(centers,xs,widths):
    us=[torch.logsumexp(c.ell,-1) for c in centers]
    gus=[(torch.softmax(c.ell,-1)[...,None]*c.gell).sum(2) for c in centers]
    u,si=blend_responses(xs,[c.x for c in centers],us,gus)
    ell,mi=blend_responses(xs,[c.x for c in centers],[c.ell for c in centers],[c.gell for c in centers])
    assert torch.equal(si['weights'],mi['weights'])
    base_s=torch.sigmoid(-u)
    mm=torch.softmax(torch.cat([torch.zeros_like(ell[...,:1]),ell],-1),-1)
    w=torch.tensor(widths,dtype=xs.dtype)[:,None,None]
    ss=(1/(base_s.clamp(min=.01)*w+1e-8)).clamp(max=100.)
    ms=(1/(mm[...,0].clamp(min=.01)*w+1e-8)).clamp(max=100.)
    arrays={}
    for kind,info,stages in (('scalar',si,ss),('mscs',mi,ms)):
        arrays.update({kind+'__'+k:v for k,v in info.items()})
        arrays[kind+'__stage']=stages
    arrays['predicted_m']=mm
    return dict(scalar_blend=ss.mean((1,2)),mscs_blend=ms.mean((1,2))),arrays
