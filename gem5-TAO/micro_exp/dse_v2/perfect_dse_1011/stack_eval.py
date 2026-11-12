"""
Perfect-component DSE (2026-10-11), shared pieces.  Nothing outside perfect_dse_1011/ is modified.

Model / data = the surrogate whose stacks were validated against gem5 in gem5_bounds/
(dse_v2.common.DEFAULT_CKPT, 50 slices per workload, per-sample CPI = 4-stage mean), with the
bit-exact FastEngine features from coord_ablation/fast_infer.py.

Per (config, workload) we keep the stage-resolved CPI stack:
  cs[s]        = 1 / (Base_s * W_min)                per slice, stage s (Dec, Disp, Iss, Com)
  comp[c, s]   = mean_slices( cs[s] * share_{c,s} )  c in FP, BP, SP     (CPI units)
  base[s]      = mean_slices( cs[s] * Base_s )
  cpi          = mean_slices( mean_s cs[s] )         (identical to SurrogateWL)

Perfect-component objective (Eyerman et al., CAL'18 / ISPASS'18): eliminating stall class c
reduces CPI by an amount bounded by the class component at the dispatch / issue / commit stacks:
  gain_lo = min_{s in Disp,Iss,Com} comp[c,s]     gain_hi = max_{...}
  gain_pt = comp[c, Disp] for FE and SPEC, comp[c, Com] for BE   (ISPASS: dispatch is more accurate
            for frontend / bpred, commit for backend)
  CPI_X   = max(cpi - gain, mean_s base[s])
Joint objective = geometric mean over workloads of CPI_X.
"""
import os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DSE_V2 = os.path.dirname(HERE)
sys.path.insert(0, DSE_V2)
sys.path.insert(0, os.path.join(DSE_V2, "coord_ablation"))
import torch
from common import Surrogate, DEFAULT_CKPT, GEM_INPUT_DIM, BASE_IDX, compute_w_min, config_key

MECH = {"FE": 1, "BE": 2, "SPEC": 3}            # MSCS category index removed by the perfect component
PT_STAGE = {"FE": 1, "BE": 3, "SPEC": 1}        # ISPASS point estimate: Disp for FE/bpred, Com for BE
BOUND_STAGES = [1, 2, 3]                        # dispatch, issue, commit (as in Eyerman)


class StackEval:
    def __init__(self, workloads, max_slices=50, fast=True, check=True):
        self.S = Surrogate(DEFAULT_CKPT, workloads, "18k", device="cpu", max_slices=max_slices)
        self.workloads = list(workloads)
        self.check_cfgs, self.check_vals = [], []
        self.fast = fast
        self.n = 0

    @torch.no_grad()
    def eval(self, c, wl):
        S = self.S
        w = float(compute_w_min(c))
        X = np.stack([f[:GEM_INPUT_DIM] for f in S.engine.get_features(c, workload=wl)])
        m = S.model(torch.from_numpy(S.norm.transform(X)).float())
        base = torch.clamp(m[:, BASE_IDX], min=0.01)
        cs = torch.clamp(1.0 / (base * w + 1e-8), max=100.0)            # (n, 4) as SurrogateWL
        cpi = cs.mean(1).mean().item()
        m = m.numpy(); cs = cs.numpy()
        comp = np.array([[np.mean(cs[:, s] * m[:, 4 * s + k]) for s in range(4)] for k in (1, 2, 3)])
        bse = np.array([np.mean(cs[:, s] * m[:, 4 * s]) for s in range(4)])
        self.n += 1
        return cpi, comp, bse

    def enable_fast(self, probe_cfgs):
        """Evaluate probes with the original features, switch to FastEngine, require identical CPI."""
        from fast_infer import FastEngine
        ref = [[self.eval(c, wl)[0] for wl in self.workloads] for c in probe_cfgs]
        self.S.engine = FastEngine(self.S.engine)
        new = [[self.eval(c, wl)[0] for wl in self.workloads] for c in probe_cfgs]
        assert ref == new, f"fast features differ: {ref} vs {new}"
        return len(probe_cfgs) * len(self.workloads)


def objective(cpi, comp, bse, mech, variant):
    """cpi (...), comp (..., 3, 4), bse (..., 4) -> CPI with the perfect component (...)."""
    c = MECH[mech] - 1
    stk = comp[..., c, :]
    if variant == "lo":
        g = stk[..., BOUND_STAGES].min(-1)
    elif variant == "hi":
        g = stk[..., BOUND_STAGES].max(-1)
    else:
        g = stk[..., PT_STAGE[mech]]
    return np.maximum(cpi - g, bse.mean(-1))
