#!/usr/bin/env python3
"""
Rebuttal B4/D1: direct gem5 validation of (1) individual MSCS components and (2) MSCS finite
differences between configurations, incl. whether Dispatch errors cancel under differencing.

Data: the 671 gem5 runs in runs/ (12 SPEC benchmarks x ~10 BO-front points; per point the base
config + configs with one resource group maxed: FE / L1 / BE / SP / ALL). gem5 MSCS = the 16-dim
cumulative stack read from the O3 trace; PRISM MSCS = cycle-weighted over 50 slices (check_bounds.stack),
so both are the share of each stage's cycles.
Checkpoints: exp1 lambda_sweep_1004 (lam_task 0.1 = paper setting, and lam_task 10).
Writes only validate_mscs_diff_<tag>.json / .txt (refuses to overwrite).
"""
import os, sys, json, glob, argparse
import numpy as np, torch
from scipy.stats import spearmanr, pearsonr

HERE = os.path.dirname(os.path.abspath(__file__))
DSE_V2 = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(DSE_V2, "mscs_guide")); sys.path.insert(0, DSE_V2)
EXP1 = os.path.join(os.path.dirname(DSE_V2), "exp1", "results", "lambda_sweep_1004")
CKPTS = {"lt0p1": os.path.join(EXP1, "t0p1_c1_m0p01_s42.pt"), "lt10": os.path.join(EXP1, "t10_c1_m0p01_s42.pt")}
CATS = ["Base", "FP", "BP", "SP"]; STAGES = ["Dec", "Disp", "Iss", "Com"]


def gem5_sh(m16):
    m = np.asarray(m16, float)
    return np.array([[m[4 * s + k] for s in range(4)] for k in range(4)])          # (cat, stage)


def predict(runs, ckpt):
    from refine_v3 import SurrogateWL
    from check_bounds import stack
    out = {}
    for b in sorted({r["bench"] for r in runs}):
        S = SurrogateWL(ckpt, [b], "18k", device="cpu", max_slices=50)
        with torch.no_grad():
            for r in runs:
                if r["bench"] == b:
                    cpi, sh = stack(S, r["config"])
                    out[(b, r["idx"], r["tag"])] = (cpi, np.array([sh[c] for c in CATS]))
    return out


def dom_acc(G, P):                     # (n, cat, stage): dominant pressure type per stage
    return [(G[:, 1:, s].argmax(1) == P[:, 1:, s].argmax(1)).mean() for s in range(4)]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", choices=list(CKPTS), default="lt0p1")
    args = ap.parse_args()
    jpath = os.path.join(HERE, f"validate_mscs_diff_{args.ckpt}.json"); tpath = jpath.replace(".json", ".txt")
    if os.path.exists(jpath):
        sys.exit(f"exists, refusing to overwrite: {jpath}")
    runs = [json.load(open(f)) for f in glob.glob(os.path.join(HERE, "runs", "*", "*", "result.json"))]
    R = {(r["bench"], r["idx"], r["tag"]): r for r in runs}
    P = predict(runs, CKPTS[args.ckpt])
    lines = []; pr = lambda s="": (print(s), lines.append(s))
    pr(f"checkpoint {args.ckpt}: {CKPTS[args.ckpt]}   gem5 runs {len(runs)}")

    keys = sorted(R)
    G = np.stack([gem5_sh(R[k]["mscs16"]) for k in keys]); Pm = np.stack([P[k][1] for k in keys])
    pr("\n=== 1. component level (all configs, n=%d): MAE of the share, |pred - gem5| ===" % len(keys))
    pr("        " + "  ".join(f"{s:>6s}" for s in STAGES))
    for k, c in enumerate(CATS):
        pr(f"  {c:5s} " + "  ".join(f"{np.abs(Pm[:, k, s] - G[:, k, s]).mean():6.3f}" for s in range(4)))
    pr("  Pearson r per component (pooled over configs):")
    for k, c in enumerate(CATS):
        pr(f"  {c:5s} " + "  ".join(f"{pearsonr(Pm[:, k, s], G[:, k, s])[0]:6.2f}" for s in range(4)))
    pr("  dominant pressure type (argmax FP/BP/SP) agreement per stage: " +
       "  ".join(f"{s} {a:.3f}" for s, a in zip(STAGES, dom_acc(G, Pm))))

    pairs = [(k, (k[0], k[1], "base")) for k in keys if k[2] != "base" and (k[0], k[1], "base") in R]
    dG = np.stack([gem5_sh(R[a]["mscs16"]) - gem5_sh(R[b]["mscs16"]) for a, b in pairs])
    dP = np.stack([P[a][1] - P[b][1] for a, b in pairs])
    eA = np.stack([P[a][1] - gem5_sh(R[a]["mscs16"]) for a, b in pairs])        # level error, modified
    eB = np.stack([P[b][1] - gem5_sh(R[b]["mscs16"]) for a, b in pairs])        # level error, base
    pr(f"\n=== 2. finite differences dMSCS = MSCS(modified) - MSCS(base), n={len(pairs)} config pairs ===")
    pr("  (a) Pearson r between predicted and gem5 dMSCS, per component")
    pr("        " + "  ".join(f"{s:>6s}" for s in STAGES))
    for k, c in enumerate(CATS):
        pr(f"  {c:5s} " + "  ".join(f"{pearsonr(dP[:, k, s], dG[:, k, s])[0]:6.2f}" for s in range(4)))
    pr("  (b) sign agreement of dMSCS where |gem5 d| > 0.02")
    for k, c in enumerate(CATS):
        cells = []
        for s in range(4):
            m = np.abs(dG[:, k, s]) > 0.02
            cells.append(f"{(np.sign(dP[m, k, s]) == np.sign(dG[m, k, s])).mean():5.2f}({m.sum():3d})" if m.sum() else "   -     ")
        pr(f"  {c:5s} " + "  ".join(cells))
    pr("  (c) error cancellation: MAE of level vs MAE of difference, and corr(err_base, err_modified)")
    pr("      cancellation <=> errors of the two configs are positively correlated, so diff MAE << sqrt(2) x level MAE")
    for k, c in enumerate(CATS):
        cells = []
        for s in range(4):
            lvl = np.abs(np.r_[eA[:, k, s], eB[:, k, s]]).mean(); dif = np.abs(dP[:, k, s] - dG[:, k, s]).mean()
            cells.append(f"{lvl:.3f}/{dif:.3f} r={pearsonr(eA[:, k, s], eB[:, k, s])[0]:+.2f}")
        pr(f"  {c:5s} " + "   ".join(cells))
    pr("  (d) direction-level: per change, does the predicted largest |dMSCS| pressure component (12) match gem5?")
    big = np.abs(dG[:, 1:, :]).reshape(len(pairs), -1).max(1) > 0.02
    hit = (np.abs(dP[:, 1:, :]).reshape(len(pairs), -1).argmax(1) == np.abs(dG[:, 1:, :]).reshape(len(pairs), -1).argmax(1))
    pr(f"      top changed component match {hit[big].mean():.2f} (n={big.sum()})")
    pr("  (e) Dispatch stage only, dominant-type agreement: level (base configs) vs which type shrinks most under the change")
    Gb = np.stack([gem5_sh(R[b]["mscs16"]) for a, b in pairs]); Pb = np.stack([P[b][1] for a, b in pairs])
    lvl_acc = (Gb[:, 1:, 1].argmax(1) == Pb[:, 1:, 1].argmax(1)).mean()
    m = np.abs(dG[:, 1:, 1]).max(1) > 0.02
    d_acc = (dG[m, 1:, 1].argmin(1) == dP[m, 1:, 1].argmin(1)).mean()
    pr(f"      Dispatch dominant type (level) {lvl_acc:.2f}   Dispatch most-reduced type (difference) {d_acc:.2f} (n={m.sum()})")
    pr("\nNote: changes are resource-group moves (one group of parameters maxed, or BP type switched), not single +1 steps.")
    open(tpath, "w").write("\n".join(lines) + "\n")
    json.dump({"ckpt": CKPTS[args.ckpt], "keys": [list(k) for k in keys], "gem5": G.tolist(), "pred": Pm.tolist(),
               "pairs": [[list(a), list(b)] for a, b in pairs]}, open(jpath, "w"))


if __name__ == "__main__":
    main()
