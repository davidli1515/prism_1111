"""Reuse frozen numerical primitives; change only the proposal allocation."""
from pathlib import Path
import sys
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PREVIOUS = ROOT/'mscs_jacobian_greedy_1010'
sys.path.insert(0, str(PREVIOUS))
from core import (LocalModel as OriginalLocalModel, PARAMS, STEPS, TOPK,
                  DEFAULT_CKPT, NORM_CACHE, config_key, compute_w_min, legal, moves,
                  Inputs, FeatureCache, load_model, exact, cpis, contributions, probabilities)

ARMS = ('scalar_excess', 'mscs_logit', 'hybrid')


class LocalModel(OriginalLocalModel):
    def __init__(self, model, x, width, arm):
        assert arm in ARMS
        super().__init__(model, x, width, 'mscs_logit' if arm == 'hybrid' else arm)

    @torch.no_grad()
    def predict(self, xs):
        pred = super().predict(xs)
        lo, hi = 1/self.width, min(100., 1/(.01*self.width+1e-8))
        excess = (self.cs-lo).clamp(min=1e-10)
        scalar = (lo+excess[None]*torch.exp((pred['stage_slope']/excess[None]).clamp(-80,80))).clamp(lo,hi).mean((1,2))
        pred['scalar_cpi'] = scalar
        pred['mscs_cpi'] = pred['cpi'] if self.arm == 'mscs_logit' else None
        return pred


def choose(arm, scalar_scores, mscs_scores, topk=2):
    """No truth, history labels, or area outcomes enter this rule."""
    assert arm in ARMS and topk == 2
    scalar_scores = np.asarray(scalar_scores)
    assert scalar_scores.ndim == 1 and np.isfinite(scalar_scores).all()
    if len(scalar_scores) == 0:
        return [], []
    if arm == 'scalar_excess':
        ids = np.argsort(scalar_scores, kind='stable')[:topk].tolist()
        return ids, ['scalar']*len(ids)
    mscs_scores = np.asarray(mscs_scores)
    assert mscs_scores.shape == scalar_scores.shape and np.isfinite(mscs_scores).all()
    order = np.argsort(mscs_scores, kind='stable').tolist()
    if arm == 'mscs_logit':
        return order[:topk], ['mscs']*min(topk,len(order))
    anchor = int(np.argmin(scalar_scores))
    alternative = [i for i in order if i != anchor][:1]
    return [anchor]+alternative, ['scalar_anchor']+['mscs_alternative']*len(alternative)
