"""
Greedy-trajectory collection: shared pieces.

Model / CPI / MSCS: exactly dse_v2/refine_v3.SurrogateWL (ckpt common.DEFAULT_CKPT = exp1
lambda_sweep_1004/t10_c1_m0p01_s42.pt, 18k normaliser, first 50 even slices, CPI = mean over slices of
the 4-stage-mean CPI, MSCS = slice mean) -- the model whose stacks were validated against gem5 in
gem5_bounds/. Features via coord_ablation/fast_infer.FastEngine (bit-exact to dse_infer).
Space: dse17 (17 free parameters, L2 1 MB/8-way, TAGE, fetch buffer 32 fixed).
Area for decisions: dse17 fast area model (McPAT is checked on a sample afterwards).

Moves ("resolvable" steps, decided before running): one parameter per step, up or down by
  list parameters (l1_size, l1d_assoc, l1i_assoc): one list index (x2 / /2)
  integer parameters: k strides with k = max(1, round(range / 8 / stride))
     -> widths 1, FUs 1, IQ 16, LQ/SQ 32, ROB 112
followed by the dse17 projection (validity rules may change coupled parameters; recorded).
Greedy (scalar-driven; MSCS is only recorded):
  1. evaluate all up/down moves of the current design
  2. RECLAIM if some down-move loses <= L_MAX CPI (relative) -> take the one saving most area
  3. else ADVANCE with the up-move of largest gain / area-increase among gains >= G_MIN
  4. else stop (plateau). Never revisit a design; at most MAX_STEPS steps.
"""
import os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
DSE_V2 = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(DSE_V2, "dse17"))
import space17 as S                                    # also puts coord_ablation, dse_exp, exp1 on path
C = S.C
sys.path.insert(0, DSE_V2)
import common as CM                                    # dse_v2/common (read-only)
from fast_infer import FastEngine
from config_dse import ALL_PARAMS, compute_w_min
from dse_engine import PARAM_STRIDE
from model import GEMModel

SEARCH = list(S.SEARCH)
MOVES = [(p, d) for p in SEARCH for d in (+1, -1)]     # 34 candidate slots, fixed order
G_MIN, L_MAX, MAX_STEPS, MAX_SLICES = 0.005, 0.002, 60, 50


def step_size(p):
    spec = ALL_PARAMS[p]
    if isinstance(spec, list):
        return None
    lo, hi, _ = spec
    s = PARAM_STRIDE.get(p) or 1
    return max(1, int(round((hi - lo) / 8 / s))) * s


STEP = {p: step_size(p) for p in SEARCH}


def ckey(c):
    return tuple(c[k] for k in sorted(c))


def move(c, p, d):
    """Returns (new_config, coupled_params) or (None, None)."""
    spec = ALL_PARAMS[p]
    if isinstance(spec, list):
        i = spec.index(c[p]) if c[p] in spec else 0
        j = i + d
        if not 0 <= j < len(spec):
            return None, None
        nv = spec[j]
    else:
        lo, hi, _ = spec
        nv = min(max(c[p] + d * STEP[p], lo), hi)
    if nv == c[p]:
        return None, None
    cn = S.project17(dict(c, **{p: nv}))
    if ckey(cn) == ckey(c):
        return None, None
    coupled = [q for q in SEARCH if q != p and cn[q] != c[q]]
    return cn, coupled


class WLPredictor:
    """One workload; same numbers as refine_v3.SurrogateWL(..., [wl], max_slices=50)."""
    def __init__(self, wl, max_slices=MAX_SLICES):
        torch.set_num_threads(1)
        self.wl = wl
        self.eng = FastEngine(C.single_workload_engine(CM.PRECOMPUTE, wl, max_slices))
        self.model = GEMModel(CM.GEM_INPUT_DIM, [512, 256], 0.0)
        self.model.load_state_dict(torch.load(CM.DEFAULT_CKPT, map_location="cpu"))
        self.model.eval()
        self.norm = CM.get_normalizer("18k")
        self.cache = {}

    @torch.no_grad()
    def __call__(self, c):
        k = ckey(c)
        if k not in self.cache:
            if len(self.cache) > 200000:          # bound memory; clear BEFORE inserting
                self.cache.clear()
            X = np.stack([f[:CM.GEM_INPUT_DIM] for f in self.eng.get_features(c, self.wl)])
            m = self.model(torch.from_numpy(self.norm.transform(X)).float())
            base = torch.clamp(m[:, CM.BASE_IDX], min=0.01)
            cs = torch.clamp(1.0 / (base * float(compute_w_min(c)) + 1e-8), max=100.0)
            self.cache[k] = (cs.mean(1).mean().item(), m.mean(0).numpy().astype(np.float32))
        return self.cache[k]


def walk(c0, evaluate, area_many):
    """evaluate(list of configs) -> list of dicts {cpi, mscs (16,) or (W,16), extra...};
    area_many(list) -> np.array. Returns list of state records."""
    c, visited, recs = c0, {ckey(c0)}, []
    for step in range(MAX_STEPS + 1):
        cands = [move(c, p, d) for p, d in MOVES]
        cfgs = [c] + [cn for cn, _ in cands if cn is not None]
        ev = evaluate(cfgs)
        ar = area_many(cfgs)
        s0, a0 = ev[0], float(ar[0])
        slot, j = [], 1
        for (cn, cp) in cands:
            if cn is None:
                slot.append(None)
            else:
                slot.append((cn, cp, ev[j], float(ar[j]))); j += 1
        rec = {"step": step, "cfg": [c[p] for p in SEARCH], "area": a0, "ev": s0, "cand": slot,
               "kind": 0, "act": -1, "stop": ""}
        if step == MAX_STEPS:
            rec["stop"] = "max_steps"; recs.append(rec); break
        cpi0 = s0["cpi"]
        best, kind = None, 0
        rc = [(i, a0 - x[3]) for i, x in enumerate(slot) if x is not None and MOVES[i][1] < 0
              and (x[2]["cpi"] - cpi0) / cpi0 <= L_MAX and x[3] < a0 and ckey(x[0]) not in visited]
        if rc:
            best, kind = max(rc, key=lambda t: t[1])[0], 2
        else:
            up = [(i, ((cpi0 - x[2]["cpi"]) / cpi0) / max((x[3] - a0) / a0, 1e-3))
                  for i, x in enumerate(slot) if x is not None and MOVES[i][1] > 0
                  and (cpi0 - x[2]["cpi"]) / cpi0 >= G_MIN and ckey(x[0]) not in visited]
            if up:
                best, kind = max(up, key=lambda t: t[1])[0], 1
        if best is None:
            rec["stop"] = "plateau"; recs.append(rec); break
        rec["kind"], rec["act"] = kind, best
        recs.append(rec)
        c = slot[best][0]; visited.add(ckey(c))
    return recs


def pack(walks, start_ids, joint_W=None):
    """walks: list of record lists. Returns dict of numpy arrays (one row per state)."""
    rows = [(sid, r) for sid, recs in zip(start_ids, walks) for r in recs]
    n, K = len(rows), len(MOVES)
    out = {"start": np.array([s for s, _ in rows], np.int32),
           "step": np.array([r["step"] for _, r in rows], np.int16),
           "cfg": np.array([r["cfg"] for _, r in rows], np.int32),
           "area": np.array([r["area"] for _, r in rows], np.float32),
           "cpi": np.array([r["ev"]["cpi"] for _, r in rows], np.float32),
           "kind": np.array([r["kind"] for _, r in rows], np.int8),
           "act": np.array([r["act"] for _, r in rows], np.int8),
           "stop": np.array([r["stop"] for _, r in rows]),
           "k_valid": np.zeros((n, K), bool), "k_coupled": np.zeros((n, K), bool),
           "k_cpi": np.full((n, K), np.nan, np.float32), "k_area": np.full((n, K), np.nan, np.float32)}
    if joint_W is None:
        out["mscs"] = np.array([r["ev"]["mscs"] for _, r in rows], np.float32)
        out["k_mscs"] = np.full((n, K, 16), np.nan, np.float32)
    else:
        out["cpi_wl"] = np.array([r["ev"]["cpi_wl"] for _, r in rows], np.float32)
        out["mscs_wl"] = np.array([r["ev"]["mscs_wl"] for _, r in rows], np.float32)
        out["k_cpi_wl"] = np.full((n, K, joint_W), np.nan, np.float32)
        out["k_mscs"] = np.full((n, K, 16), np.nan, np.float32)        # mean over workloads
    for i, (_, r) in enumerate(rows):
        for k, x in enumerate(r["cand"]):
            if x is None:
                continue
            out["k_valid"][i, k] = True; out["k_coupled"][i, k] = bool(x[1])
            out["k_cpi"][i, k] = x[2]["cpi"]; out["k_area"][i, k] = x[3]
            if joint_W is None:
                out["k_mscs"][i, k] = x[2]["mscs"]
            else:
                out["k_cpi_wl"][i, k] = x[2]["cpi_wl"]; out["k_mscs"][i, k] = x[2]["mscs_wl"].mean(0)
    return out
