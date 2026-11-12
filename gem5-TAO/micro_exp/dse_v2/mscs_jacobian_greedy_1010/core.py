"""Finite-action input-space tangents for a frozen PRISM checkpoint.

The encoder is discrete. Rebuild its COMPLETE input at both legal endpoints and
apply the network Jacobian to that feature displacement. This is a local model
in feature space, not a claim of an exact derivative through a discrete encoder.
"""
from collections import OrderedDict
from pathlib import Path
import sys
import time
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from common import (GEMModel, DEFAULT_CKPT, PRECOMPUTE, NORM_CACHE, get_normalizer,
                    config_key, compute_w_min, get_param_values, ALL_PARAMS)
from config_dse import is_valid_config
from dse_infer import load_precomputed, dse_infer

ARMS = ('scalar_linear', 'scalar_excess', 'mscs_logit')
STEPS = 6
TOPK = 2
PARAMS = ('numROBEntries', 'numIQEntries', 'LQEntries', 'SQEntries', 'IntALU',
          'IntMultDiv', 'FP_ALU', 'FP_MultDiv', 'RdWrPort', 'fetchBufferSize',
          'l1i_assoc', 'l1d_assoc', 'l2_size', 'l2_assoc', 'wbWidth')
STRIDE = {'numROBEntries': 32, 'numIQEntries': 8, 'LQEntries': 8, 'SQEntries': 8}


def legal(c):
    return (all(c[k] in get_param_values(k) for k in ALL_PARAMS) and is_valid_config(c)
            and max(c['l1i_assoc'], c['l1d_assoc']) <= c['l1_size'])


def moves(c, radius=4, seen=()):
    """No projection, collateral edits, width changes, trace switches or BP flips."""
    out, used = [], set(seen)
    for name in PARAMS:
        vals = get_param_values(name)
        for sign in (1, -1):
            for stride in (1, 2, 4):
                if stride > radius:
                    continue
                idx = vals.index(c[name])+sign*stride*STRIDE.get(name, 1)
                idx = max(0, min(len(vals)-1, idx))
                cn = dict(c, **{name: vals[idx]})
                k = config_key(cn)
                if cn == c or k in used or not legal(cn):
                    continue
                assert sum(cn[p] != c[p] for p in c) == 1
                assert compute_w_min(cn) == compute_w_min(c)
                assert cn['l1_size'] == c['l1_size'] and cn['bp_type'] == c['bp_type']
                used.add(k)
                out.append({'parameter': name, 'sign': sign, 'stride': stride, 'config': cn})
    return out


def probabilities(model, x):
    z = model.head(model.backbone(x)).reshape(-1, 4, 4)
    return z, torch.softmax(z, -1)


def cpis(m, width):
    return (1/(m[..., 0].clamp(min=.01)*width+1e-8)).clamp(max=100.)


def contributions(m, width):
    """CPI units, separately for each stage; stages are alternative accounts."""
    cs = cpis(m, width)
    return (cs[..., None]*m[..., 1:]).mean(dim=-3)


class LocalModel:
    def __init__(self, model, x, width, arm):
        assert arm in ARMS and not model.training
        self.width, self.arm, self.x = width, arm, x.detach()
        xx = x.detach().requires_grad_(True)
        t = time.perf_counter()
        z, m = probabilities(model, xx); cs = cpis(m, width)
        self.forward_seconds = time.perf_counter()-t
        self.m, self.cs = m.detach(), cs.detach()
        self.ell = (z[..., 1:]-z[..., :1]).detach()
        self.cpi = float(cs.mean())
        self.backward_outputs = 12 if arm == 'mscs_logit' else 4
        t = time.perf_counter()
        if arm == 'mscs_logit':
            ell = z[..., 1:]-z[..., :1]
            g = [torch.autograd.grad(ell[:, s, k].sum(), xx, retain_graph=(3*s+k < 11))[0]
                 for s in range(4) for k in range(3)]
            self.gell = torch.stack(g, dim=1).reshape(len(x), 4, 3, x.shape[1]).detach()
            weight = width*self.m[..., :1]*self.m[..., 1:]*self.cs[..., None]**2
            active = (self.m[..., 0] > .01) & (self.cs < 100.)
            self.gc = (self.gell*weight[..., None]*active[..., None, None]).sum(2)
        else:
            self.gc = torch.stack([torch.autograd.grad(cs[:, s].sum(), xx, retain_graph=(s < 3))[0]
                                   for s in range(4)], dim=1).detach()
            self.gell = None
        self.backward_seconds = time.perf_counter()-t

    @torch.no_grad()
    def predict(self, xs):
        """xs: candidate, slice, ALL input dimensions. Never receives true labels."""
        delta = xs-self.x[None]
        dc = torch.einsum('nsd,cnd->cns', self.gc, delta)
        lo, hi = 1/self.width, min(100., 1/(.01*self.width+1e-8))
        mhat = None
        de = None
        if self.arm == 'scalar_linear':
            chat = (self.cs[None]+dc).clamp(lo, hi)
        elif self.arm == 'scalar_excess':
            excess = (self.cs-lo).clamp(min=1e-10)
            chat = (lo+excess[None]*torch.exp((dc/excess[None]).clamp(-80, 80))).clamp(lo, hi)
        else:
            de = torch.einsum('nskd,cnd->cnsk', self.gell, delta)
            ellhat = self.ell[None]+de
            mhat = torch.softmax(torch.cat([torch.zeros_like(ellhat[..., :1]), ellhat], dim=-1), -1)
            chat = cpis(mhat, self.width)
        return {'cpi': chat.mean((1, 2)), 'cpi_slope': dc.mean((1, 2)), 'mscs': mhat,
                'stage_slope': dc, 'logodds_slope': de}

    @torch.no_grad()
    def partial_arch_slope(self, xs):
        return torch.einsum('nsd,cnd->cns', self.gc[:, :, 3850:3873],
                            xs[:, :, 3850:3873]-self.x[None, :, 3850:3873]).mean((1, 2))


class Inputs:
    """Precomputed workload is shared; normalized-config caches are per trajectory."""
    def __init__(self, bench, l1):
        t = time.perf_counter()
        paths = sorted((Path(PRECOMPUTE)/f'l1_{l1}'/bench).glob('slice_*/dse_precomputed.npy'))
        paths = [p for p in paths if int(p.parent.name.split('_')[1]) % 2 == 0][:50]
        assert len(paths) == 50
        self.ids = [int(p.parent.name.split('_')[1]) for p in paths]
        self.pre = [load_precomputed(p) for p in paths]
        for kind in ('mean', 'std'):
            assert (Path(NORM_CACHE)/f'18k_s42_{kind}.npy').exists()
        self.norm = get_normalizer()
        self.l1 = l1
        self.load_seconds = time.perf_counter()-t

    def build(self, c):
        assert c['l1_size'] == self.l1
        with np.errstate(divide='ignore', invalid='ignore'):
            raw = np.stack([dse_infer(p, c) for p in self.pre])
        x = torch.from_numpy(self.norm.transform(raw)).float()
        assert x.shape == (50, 4681) and torch.isfinite(x).all()
        return x


class FeatureCache:
    def __init__(self, inputs):
        self.inputs, self.data = inputs, OrderedDict()
        self.builds, self.seconds = 0, 0.

    def get(self, c):
        k = config_key(c)
        if k in self.data:
            self.data.move_to_end(k)
            return self.data[k]
        t = time.perf_counter(); x = self.inputs.build(c)
        self.seconds += time.perf_counter()-t; self.builds += 1
        self.data[k] = x
        while len(self.data) > 128:
            self.data.popitem(last=False)
        return x


def load_model():
    torch.set_num_threads(1)
    model = GEMModel(4681, [512, 256], 0.).eval()
    model.load_state_dict(torch.load(DEFAULT_CKPT, map_location='cpu', weights_only=True))
    return model


@torch.no_grad()
def exact(model, x, width):
    z, m = probabilities(model, x)
    return float(cpis(m, width).mean()), m
