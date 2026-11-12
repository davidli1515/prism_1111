#!/usr/bin/env python3
"""
Is Dispatch balance a signal of "removable stalls exhausted"? (random-start greedy walks, results/).
A naive test (does the entropy peak / plateau line up with the knee of the CPI-area curve) is confounded:
the walk stops when the gain drops below G_MIN, so ANY saturating quantity (area, step count) lines up with
the knee. Instead, a prediction test with grouped cross-validation (folds = walks, 5-fold), per benchmark:
  target  = log(CPI_t / CPI_end)  (CPI reduction still to come on this walk)
  base    = log CPI_t, log area_t, step t
  +bal    = base + Dispatch Jain and normalised entropy at t
  +mscs   = base + the full 16-dim MSCS at t (upper reference: everything the stack knows)
  +null   = base + a shuffled copy of Jain (same marginal, no information; guards against model capacity)
Model = HistGradientBoostingRegressor (fixed hyper-parameters). Reports out-of-fold R^2 and the share of the
MSCS gain that balance alone recovers. Also the classification view: will the walk stop within 2 steps.
Writes results/balance_signal_v1.txt (never overwrites).
"""
import os, glob
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, "results")
FP_SUITE = {"508.namd_r", "526.blender_r", "607.cactuBSSN_s", "619.lbm_s", "621.wrf_s", "638.imagick_s",
            "644.nab_s", "654.roms_s"}
OUT = []


def say(s=""):
    print(s, flush=True); OUT.append(s)


def short(b):
    return b.split(".")[1].replace("_s", "").replace("_r", "") + ("*" if b in FP_SUITE else "")


def oof(X, y, groups, clf=False):
    p = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, groups):
        m = (HistGradientBoostingClassifier if clf else HistGradientBoostingRegressor)(max_iter=200, learning_rate=0.1,
                                                                                       random_state=0)
        m.fit(X[tr], y[tr]); p[te] = m.predict_proba(X[te])[:, 1] if clf else m.predict(X[te])
    return p


def main():
    out = os.path.join(R, "balance_signal_v1.txt")
    if os.path.exists(out):
        out = out.replace(".txt", f".{os.getpid()}.txt")
    B = sorted(d for d in os.listdir(R) if d[0].isdigit())
    rng = np.random.RandomState(0)
    say("out-of-fold R^2 of log(CPI_t/CPI_end) | AUC of 'walk stops within 2 steps'   (folds = walks)")
    say(f"{'bench':11s} {'n':>6s} | R2 base  +null  +bal  +mscs  bal/mscs gain | AUC base  +null  +bal  +mscs")
    S = []
    for b in B:
        fs = sorted(glob.glob(os.path.join(R, b, "part_*.npz")))
        d = {k: np.concatenate([np.load(f)[k] for f in fs]) for k in ("start", "step", "cpi", "area", "mscs")}
        st = d["start"]; last = np.r_[np.where(np.diff(st) != 0)[0], len(st) - 1]
        first = np.r_[0, last[:-1] + 1]; n_of = np.repeat(last - first + 1, last - first + 1)
        cend = np.repeat(d["cpi"][last], last - first + 1)
        y = np.log(d["cpi"] / cend); ystop = ((n_of - 1 - d["step"]) <= 2).astype(int)
        x = np.stack([d["cpi"] * d["mscs"][:, 4 + k] for k in (1, 2, 3)], 1)
        J = x.sum(1) ** 2 / (3 * (x ** 2).sum(1)); p = x / x.sum(1, keepdims=True); H = -(p * np.log(p + 1e-12)).sum(1) / np.log(3)
        base = np.stack([np.log(d["cpi"]), np.log(d["area"]), d["step"]], 1)
        sets = {"base": base, "+null": np.c_[base, rng.permutation(J)], "+bal": np.c_[base, J, H], "+mscs": np.c_[base, d["mscs"]]}
        r2 = {k: r2_score(y, oof(X, y, st)) for k, X in sets.items()}
        auc = {k: roc_auc_score(ystop, oof(X, ystop, st, clf=True)) for k, X in sets.items()}
        share = (r2["+bal"] - r2["base"]) / max(r2["+mscs"] - r2["base"], 1e-9)
        S.append((r2, auc, share))
        say(f"{short(b):11s} {len(y):6d} | {r2['base']:.3f} {r2['+null']:.3f} {r2['+bal']:.3f} {r2['+mscs']:.3f}  {share:5.2f}"
            f"          | {auc['base']:.3f} {auc['+null']:.3f} {auc['+bal']:.3f} {auc['+mscs']:.3f}")
    g = lambda f: np.median([f(s) for s in S])
    say(f"\nmedian over benchmarks: R2 gain +bal {g(lambda s: s[0]['+bal']-s[0]['base']):+.3f}, +null {g(lambda s: s[0]['+null']-s[0]['base']):+.3f}, "
        f"+mscs {g(lambda s: s[0]['+mscs']-s[0]['base']):+.3f}; balance recovers {g(lambda s: s[2]):.2f} of the MSCS gain")
    say(f"AUC gain +bal {g(lambda s: s[1]['+bal']-s[1]['base']):+.3f}, +null {g(lambda s: s[1]['+null']-s[1]['base']):+.3f}, "
        f"+mscs {g(lambda s: s[1]['+mscs']-s[1]['base']):+.3f}; +bal > +null in {sum(s[0]['+bal'] > s[0]['+null'] for s in S)}/{len(S)} (R2)")
    open(out, "w").write("\n".join(OUT) + "\n"); print("->", out)


if __name__ == "__main__":
    main()
