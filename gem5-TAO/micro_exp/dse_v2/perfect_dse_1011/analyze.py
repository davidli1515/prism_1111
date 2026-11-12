#!/usr/bin/env python3
"""
Step 2 of the first look: perfect-component fronts from the rescored archives (stacks/*.npz).

For every benchmark (and joint_17), every perfect component X in {FE, BE, SPEC} and every
estimate v in {lo, pt, hi} (stack_eval.objective), at area budgets B:
  real      min CPI over configs with area <= B                       (today's best design)
  as-is     CPI_X of that same design                                 (add X, keep the core)
  re-opt    min CPI_X over configs with area <= B                     (add X and re-optimize)
  re-opt gain = as-is / re-opt - 1  -> what re-optimizing the core around X adds
  changed   is the re-optimized design a different configuration?
  param shifts of re-opt vs real design (sign per parameter)
  iso-CPI area saving: smallest area whose CPI_X <= real min CPI at B, vs B
  mechanism ranking at B by pt, robust if lo-gain(best) >= hi-gain(2nd)
NOTE: configs come from searches that optimized the real CPI, so re-opt is a lower bound on what
a dedicated perfect-X search would find.
Usage: python analyze.py [tags...]   -> results/summary.txt, results/summary.json
"""
import os, sys, json, glob, collections
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from stack_eval import objective, MECH

BUDGETS = [15, 20, 25, 30, 40, 50, 60, 80, 100, 130]
VARIANTS = ["lo", "pt", "hi"]


def load(tag):
    fs = sorted(glob.glob(os.path.join(HERE, "stacks", f"{tag}.npz")) +
                glob.glob(os.path.join(HERE, "stacks", f"{tag}_c*-of-*.npz")))
    if not fs:
        return None
    if "-of-" in fs[-1]:
        n = int(fs[-1].split("-of-")[1].split(".")[0])
        if len(fs) != n:
            return None
    zs = [np.load(f, allow_pickle=True) for f in fs]
    cat = lambda k: np.concatenate([z[k] for z in zs])
    return {"configs": [json.loads(s) for s in cat("configs")], "area": cat("area"),
            "cpi": cat("cpi"), "comp": cat("comp"), "base": cat("base"), "wl": list(zs[0]["workloads"])}


def gmean(x):
    return np.exp(np.log(x).mean(-1))


def analyze(tag):
    D = load(tag)
    if D is None:
        return None
    A = D["area"]; real = gmean(D["cpi"])
    obj = {(X, v): gmean(objective(D["cpi"], D["comp"], D["base"], X, v)) for X in MECH for v in VARIANTS}
    rows, shifts = [], collections.defaultdict(lambda: collections.Counter())
    for B in BUDGETS:
        m = A <= B
        if not m.any():
            continue
        ir = np.where(m)[0][np.argmin(real[m])]
        r = {"budget": B, "real": float(real[ir]), "real_area": float(A[ir]), "mech": {}}
        for X in MECH:
            e = {}
            for v in VARIANTS:
                o = obj[(X, v)]; io = np.where(m)[0][np.argmin(o[m])]
                asis, reopt = float(o[ir]), float(o[io])
                iso = A[o <= real[ir]]
                e[v] = {"asis": asis, "reopt": reopt, "gain_asis": 1 - asis / real[ir],
                        "gain_reopt": 1 - reopt / real[ir], "reopt_extra": asis / reopt - 1,
                        "changed": bool(io != ir), "area": float(A[io]),
                        "iso_area_saving": float(1 - iso.min() / A[ir]) if len(iso) else None}
                if v == "pt" and io != ir:
                    for p, val in D["configs"][io].items():
                        d = np.sign(val - D["configs"][ir][p])
                        if d:
                            shifts[X][(p, "+" if d > 0 else "-")] += 1
            r["mech"][X] = e
        g = sorted(((r["mech"][X]["pt"]["gain_reopt"], X) for X in MECH), reverse=True)
        r["rank"] = [X for _, X in g]
        r["robust_top"] = bool(r["mech"][g[0][1]]["lo"]["gain_reopt"] >= r["mech"][g[1][1]]["hi"]["gain_reopt"])
        rows.append(r)
    return {"tag": tag, "n_configs": len(A), "rows": rows,
            "shifts": {X: {f"{p}{s}": n for (p, s), n in c.most_common(8)} for X, c in shifts.items()}}


def report(res, out):
    P = lambda *a: print(*a, file=out)
    P(f"\n######## {res['tag']}  ({res['n_configs']} archived configs)")
    P("  budget  real-CPI |" + "|".join(f"  {X:4s} gain as-is / re-opt  [lo..hi]  extra  chg  iso-area " for X in MECH) + "| rank (robust)")
    for r in res["rows"]:
        s = f"  {r['budget']:5d}  {r['real']:.4f}  |"
        for X in MECH:
            e = r["mech"][X]
            iso = e["pt"]["iso_area_saving"]
            iso_s = "" if iso is None else f"{100*iso:5.1f}%"
            s += (f"  {100*e['pt']['gain_asis']:5.1f}% / {100*e['pt']['gain_reopt']:5.1f}%"
                  f" [{100*e['lo']['gain_reopt']:4.1f}..{100*e['hi']['gain_reopt']:4.1f}]"
                  f" {100*e['pt']['reopt_extra']:+5.1f}%  {'Y' if e['pt']['changed'] else '.'}"
                  f"  {iso_s:>6s}  |")
        P(s + f" {'>'.join(r['rank'])} ({'Y' if r['robust_top'] else 'n'})")
    for X, c in res["shifts"].items():
        P(f"  param shifts when re-optimizing for perfect {X}: {c}")


def main():
    tags = sys.argv[1:] or sorted({os.path.basename(f).split("_c")[0].replace(".npz", "")
                                   for f in glob.glob(os.path.join(HERE, "stacks", "*.npz"))})
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    allres = [r for r in (analyze(t) for t in tags) if r]
    with open(os.path.join(HERE, "results", "summary.txt"), "w") as f:
        for r in allres:
            report(r, f); report(r, sys.stdout)
        # cross-benchmark aggregate (single benchmarks only)
        single = [r for r in allres if r["tag"] != "joint_17"]
        if single:
            msg = "\n######## aggregate over single benchmarks (pt estimate)\n"
            for X in MECH:
                g = [x["mech"][X]["pt"] for r in single for x in r["rows"]]
                msg += (f"  {X:4s} gain as-is {100*np.mean([e['gain_asis'] for e in g]):5.1f}%  re-opt "
                        f"{100*np.mean([e['gain_reopt'] for e in g]):5.1f}%  re-opt extra "
                        f"{100*np.mean([e['reopt_extra'] for e in g]):+5.2f}%  design changed "
                        f"{np.mean([e['changed'] for e in g]):.2f}  (n={len(g)} bench x budget)\n")
            tops = collections.Counter(x["rank"][0] for r in single for x in r["rows"])
            rob = np.mean([x["robust_top"] for r in single for x in r["rows"]])
            msg += f"  best component per (bench, budget): {dict(tops)}   robust under [lo,hi]: {rob:.2f}\n"
            print(msg); f.write(msg)
    json.dump(allres, open(os.path.join(HERE, "results", "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
