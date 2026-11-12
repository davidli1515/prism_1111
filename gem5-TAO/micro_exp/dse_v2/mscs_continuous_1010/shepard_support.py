"""Blend local response models from an arm's own visited roots."""
import torch
from quadratic_support import *

ARMS=('scalar_shepard','mscs_shepard','hybrid_shepard')


def normalized_weights(dist2):
    # [center, candidate, slice]; exact duplicates get equal mass, all others zero.
    zero=dist2==0
    inverse=1/dist2.clamp(min=torch.finfo(dist2.dtype).tiny)
    # Normalize by closest positive distance first to avoid overflow or underflow.
    nearest=dist2.clamp(min=torch.finfo(dist2.dtype).tiny).amin(0,keepdim=True)
    relative=nearest/dist2.clamp(min=torch.finfo(dist2.dtype).tiny)
    normal=relative/relative.sum(0,keepdim=True)
    exact=zero.to(dist2.dtype)/zero.sum(0,keepdim=True).clamp(min=1)
    return torch.where(zero.any(0,keepdim=True),exact,normal)


def blend_responses(candidate_x,center_x,center_y,center_grad):
    # Generic dimension trailing outputs, gradients [center,slice,...,feature].
    predictions=[];dist=[];slopes=[]
    for x,y,g in zip(center_x,center_y,center_grad):
        delta=candidate_x-x[None]
        dist.append(delta.double().square().sum(-1))
        shape=g.shape[1:-1]
        slope=torch.einsum('nvd,cnd->cnv',g.reshape(g.shape[0],-1,g.shape[-1]),delta).reshape(*delta.shape[:2],*shape)
        slopes.append(slope);predictions.append(y[None]+slope)
    dist=torch.stack(dist);weights=normalized_weights(dist)
    predicted=torch.stack(predictions).double()
    expanded=weights.reshape(*weights.shape,*([1]*(predicted.ndim-weights.ndim)))
    value=(expanded*predicted).sum(0).to(candidate_x.dtype)
    return value,dict(center_response=torch.stack(center_y),center_slope=torch.stack(slopes),
                      distance_squared=dist,weights=weights,blended_response=value)


def predict_centers(centers,xs,space,base):
    current=centers[-1];lo=1/current.width;hi=min(100.,1/(.01*current.width+1e-8))
    yy=[];gg=[]
    for center in centers:
        if space=='scalar':
            excess=(center.cs-lo).clamp(min=1e-10)
            yy.append(torch.log(excess));gg.append(center.gc/excess[...,None])
        else:
            yy.append(center.ell);gg.append(center.gell)
    value,info=blend_responses(xs,[c.x for c in centers],yy,gg)
    if space=='scalar':
        excess=(current.cs-lo).clamp(min=1e-10)
        ratio=value-torch.log(excess)[None]
        stages=(lo+excess[None]*torch.exp(ratio.clamp(-80,80))).clamp(lo,hi);m=None
        if len(centers)==1:
            stages=(lo+excess[None]*torch.exp((base['stage_slope']/excess[None]).clamp(-80,80))).clamp(lo,hi)
    else:
        m=torch.softmax(torch.cat([torch.zeros_like(value[...,:1]),value],-1),-1)
        stages=cpis(m,current.width)
        if len(centers)==1:m=base['mscs'];stages=cpis(m,current.width)
    return stages.mean((1,2)),m,dict(info,blended_stage_cpi=stages)
