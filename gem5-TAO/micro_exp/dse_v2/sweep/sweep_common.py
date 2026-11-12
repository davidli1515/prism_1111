"""
Shared pieces for the sweep-scale re-analyses (dse_v2/sweep/).  Nothing outside sweep/ is modified;
originals are imported read-only.

Models: the ORIGINAL DSE pipeline, 100% labels: PRISM = exp1_v2 gem_full.pt, Concorde = exp1_v2
concorde.pt, dse_exp normalizer cache (gem & con), GEMPredictor / ConcordePredictor slice sampling,
bit-exact fast features (coord_ablation/fast_infer.py).  PRISM CPI = commit-stage CPI as in
GEMPredictor.predict_cpi.  Design space: dse17/space17 (17 free parameters, L2 / BP / fetch buffer fixed).

Sweep endpoints of parameter p at config c: c with p := min and c with p := max (others at c),
both passed through space17.project17 (validity rules may move coupled parameters, e.g. widths or
ROB >= 2*dispatch + IQ + LQ + SQ; the achieved values are recorded).
"""
import os, sys
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "dse17"))
import space17 as S                                    # sets dse_exp / exp1 / coord_ablation paths
C = S.C
from fast_infer import FastEngine
from config_dse import get_param_values, compute_w_min

BENCHES = ["602.gcc_s", "605.mcf_s", "623.xalancbmk_s", "638.imagick_s", "641.leela_s", "648.exchange2_s"]
PARAMS = list(S.SEARCH)                                 # 17 free parameters
BASE_IDX = [0, 4, 8, 12]
PRESSURE = [1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15]   # FP, BP, SP x 4 stages (shapley_pruning MSCS_12)


class Models:
    """PRISM + Concorde predictors on one benchmark, cached per config."""
    def __init__(self, bench):
        import io, contextlib
        import run_gem_dse as R
        from dse_engine import GEMPredictor, ConcordePredictor
        from model import ConcordeModel
        with contextlib.redirect_stdout(io.StringIO()):
            gm, gnorm, _ = C.load_original_model()
            _, cnorm = R.get_normalizers()
            cm = ConcordeModel(input_dim=R.CONCORDE_INPUT_DIM, hidden_dims=R.CONCORDE_HIDDEN_DIMS, dropout=0.0)
            cm.load_state_dict(torch.load(R.CONCORDE_WEIGHT_PATH, map_location="cpu")); cm.eval()
            eng = FastEngine(C.single_workload_engine(R.PRECOMPUTE_BASE, bench))
            self.gp = GEMPredictor(gm, gnorm, eng, workload=bench, device="cpu")
            self.cp = ConcordePredictor(cm, cnorm, eng, workload=bench, device="cpu")
        self.cache = {}

    @torch.no_grad()
    def __call__(self, c):
        """-> dict(cpi_p, cpi_c, share(16), cyc(16)) ; cyc = CPI-weighted per-cell cycles."""
        k = C.ckey(c)
        if k not in self.cache:
            x = self.gp._get_features(c, self.gp.cpi_indices)
            m = self.gp.model(x)
            w = float(compute_w_min(c))
            cpi_p = (1.0 / (m[:, 12] * w + 1e-8)).mean().item()
            cs = 1.0 / (torch.clamp(m[:, BASE_IDX], min=0.01) * w + 1e-8)        # per-stage CPI per slice
            cyc = np.zeros(16)
            for s in range(4):
                idx = [4 * s + j for j in range(4)]
                cyc[idx] = ((cs[:, s:s + 1] * m[:, idx]).mean(0)).numpy()
            cpi_c = self.cp.model(self.cp._get_features(c, self.cp.cpi_indices)).mean().item()
            self.cache[k] = {"cpi_p": cpi_p, "cpi_c": cpi_c, "share": m.mean(0).numpy(), "cyc": cyc}
        return self.cache[k]


def endpoints(c, p):
    vals = get_param_values(p)
    lo = S.project17(dict(c, **{p: min(vals)}))
    hi = S.project17(dict(c, **{p: max(vals)}))
    if C.ckey(lo) == C.ckey(hi):
        return None
    moved = sorted(k for k in PARAMS if k != p and (lo[k] != c[k] or hi[k] != c[k]))
    return lo, hi, {"lo": lo[p], "hi": hi[p], "coupled": moved}


def bases(n, seed):
    rng = np.random.RandomState(seed)
    return [S.sample17(rng) for _ in range(n)]
