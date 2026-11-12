"""
Shared pieces for the MSCS-guidance feasibility checks (mscs_guide/).

Question: can MSCS tell the search WHAT TO EVALUATE (never override an evaluated result)?
  check_prune.py  A1  rank the 21 "+1" moves from the current MSCS; is the scalar-best move
                      (max gain/area) in the top-k?   A2  order the reclaim candidates so the
                      accepted one is reached with fewer queries
  check_pairs.py  B   on low-gain states, propose k two-parameter moves from the MSCS and
                      compare with the exhaustive pair oracle and scalar-only proposals

Relief map R (built by build_map.py, offline, on random configs):
  R_w[p, i] = sum_s (m_wi(x_s) - m_wi(x_s + p)) / sum_s m_wi(x_s)
  = fraction of MSCS component i that a +1 stride of parameter p removes on workload w
  (signed: negative = p raises that pressure).  For a target benchmark the LOO map
  (mean over the other 16 workloads) is used, so the map costs the target no queries.
MSCS layout: index i = 4*stage + category, stages Dec/Disp/Iss/Com, categories Base/FP/BP/SP.
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import numpy as np
from common import ALL_PARAMS, pareto_mask
from refine_v3 import _shift

PARAMS = list(ALL_PARAMS)
STAGES, CATS = ["Dec", "Disp", "Iss", "Com"], ["Base", "FP", "BP", "SP"]
PRESSURE = [1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15]
TEST_BENCHES = ["638.imagick_s", "623.xalancbmk_s", "641.leela_s", "648.exchange2_s",
                "602.gcc_s", "605.mcf_s"]


def comp_name(i):
    return f"{CATS[i % 4]}@{STAGES[i // 4]}"


def load_map(path, bench, mode="loo"):
    """Returns R (P,16) and the static mean gain per param (P,) for `bench`."""
    z = np.load(path, allow_pickle=True)
    wls = list(z["workloads"])
    assert list(z["params"]) == PARAMS
    if mode == "own":
        sel = [wls.index(bench)]
    elif mode == "loo":
        sel = [i for i, w in enumerate(wls) if w != bench]
    else:
        sel = list(range(len(wls)))
    return z["R"][sel].mean(0), z["mean_gain"][sel].mean(0)


# ---------------- MSCS scores (no query needed beyond the current point) ----------------

def relief_scores(m, R):
    """Predicted total pressure relief of each +1 move: sum_i m_i * R[p, i]."""
    return (R[:, PRESSURE] * m[PRESSURE]).sum(1)


def top_pressures(m):
    """c1 = largest pressure component; c2 = largest one of a different category."""
    order = [PRESSURE[j] for j in np.argsort(-m[PRESSURE])]
    c1 = order[0]
    c2 = next((c for c in order[1:] if c % 4 != c1 % 4), order[1])
    return c1, c2


def dom_scores(m, R):
    return R[:, top_pressures(m)[0]]


def pair_cover(m, R, i, j):
    """Pressure covered by doing both moves (complementary relief counts, overlap doesn't)."""
    Rc = np.clip(R[:, PRESSURE], 0, 1)
    return float((m[PRESSURE] * (1 - (1 - Rc[i]) * (1 - Rc[j]))).sum())


# ---------------- exact evaluation helpers ----------------

def eval_inc(c, S, A, pre):
    """All +1 neighbours of c, exactly evaluated: list of dicts (name, cfg, gain, darea, eff)."""
    cpi0, a0 = S.cpi(c), A(c)
    nb = [(n, cn) for n in PARAMS for cn in [_shift(c, n, +1)] if cn is not None]
    pre([cn for _, cn in nb])
    out = []
    for n, cn in nb:
        a = A(cn)
        if a is None:
            continue
        g = (cpi0 - S.cpi(cn)) / cpi0
        dr = max(a - a0, 1e-6) / a0
        out.append({"name": n, "cfg": cn, "gain": g, "area": a, "darea": dr, "eff": g / dr})
    return out


def bo_front_starts(src_json, bo_queries, n_starts):
    """Evenly spaced points of the pure-BO front (first bo_queries) of an existing run."""
    idx = [i for i, (t, q) in enumerate(zip(src_json["tags"], src_json["queries"]))
           if t.startswith("bo") and q <= bo_queries]
    P = np.c_[np.array(src_json["cpis"])[idx], np.array(src_json["areas"])[idx]]
    front = sorted((idx[j] for j in np.where(pareto_mask(P))[0]), key=lambda i: src_json["cpis"][i])
    pick = np.unique(np.linspace(0, len(front) - 1, min(n_starts, len(front))).round().astype(int))
    return [src_json["configs"][front[j]] for j in pick], len(front)
