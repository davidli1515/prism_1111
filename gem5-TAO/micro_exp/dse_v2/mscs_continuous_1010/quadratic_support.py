"""Squared inner-product kernel leaves the current exact Jacobian unchanged."""
from collections import OrderedDict
import time
import torch
import secant_support as linear
from secant_support import *

ARMS=('scalar','mscs','scalar_secant','mscs_secant','hybrid_secant')


def quadratic_fit(candidate_delta,history_delta,residual):
    c,n,_=candidate_delta.shape;h=history_delta.shape[0];shape=residual.shape[2:]
    if h==0:return torch.zeros((c,n,*shape),dtype=candidate_delta.dtype),{}
    hd=history_delta.double();cd=candidate_delta.double()
    gram_linear=torch.einsum('hnd,knd->nhk',hd,hd)
    cross_linear=torch.einsum('cnd,hnd->nch',cd,hd)
    gram=gram_linear.square();cross=cross_linear.square()
    weights=torch.einsum('nch,nhk->nck',cross,torch.linalg.pinv(gram,rtol=1e-6,hermitian=True))
    flat=residual.permute(1,0,*range(2,residual.ndim)).reshape(n,h,-1).double()
    correction=torch.einsum('nch,nhv->ncv',weights,flat).permute(1,0,2).reshape(c,n,*shape)
    return correction.to(candidate_delta.dtype),dict(gram=gram,cross=cross,weights=weights,residual=residual,
        gram_linear=gram_linear,cross_linear=cross_linear)


def predict(local,xs,base,space,history_xs,history_response):
    # Reuse the exact response reconstruction; only the explicitly declared fit is rebound in this process.
    previous=linear.fit_correction
    linear.fit_correction=quadratic_fit
    try:return linear.predict(local,xs,base,space,history_xs,history_response)
    finally:linear.fit_correction=previous


class FeatureCache(linear.FeatureCache):
    def get(self,c):
        key=config_key(c)
        if key in self.data:
            self.data.move_to_end(key);return self.data[key]
        t=time.perf_counter();x=self.inputs.build(c)
        self.seconds+=time.perf_counter()-t;self.builds+=1;self.data[key]=x
        while len(self.data)>512:self.data.popitem(last=False)
        return x
