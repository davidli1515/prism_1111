#!/usr/bin/env python3
"""Headline-only run of val6_trainfeat: training-identical features for the 13 seen-checkpoint benchmarks only,
and only for designs whose 10 windowed gem5 runs are all finished (re-polls until every seen design is covered).
Reuses val6_trainfeat.run (same work dirs, resumable, never overwrites).
  python calib/val6_trainfeat_seen.py --workers 8
  python calib/val6_trainfeat_seen.py --summary     # design-level error only, 2 decimals"""
import os, sys, json, time, argparse
from concurrent.futures import ThreadPoolExecutor
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import val6_trainfeat as TF
GV, T = TF.GV, TF.T
UNSEEN = {"526.blender_r", "623.xalancbmk_s", "631.deepsjeng_s", "638.imagick_s"}


def seen_designs():
    return [d for d in GV.designs() if d["bench"] not in UNSEEN]


def gem5_done(d):
    return all(os.path.exists(os.path.join(GV.OUT, "runs", d["bench"], d["name"], f"w{k}", "result.json"))
               for k in GV.slices(d["bench"], d["config"]["l1_size"]))


def feat_path(d, k):
    return os.path.join(TF.OUT, d["bench"], d["name"], f"w{k}", "gem_data.npy")


def loop(workers):
    D = seen_designs()
    with ThreadPoolExecutor(workers) as ex:
        while True:
            todo = [(d, k) for d in D if gem5_done(d) for k in GV.slices(d["bench"], d["config"]["l1_size"]) if not os.path.exists(feat_path(d, k))]
            covered = sum(gem5_done(d) and all(os.path.exists(feat_path(d, k)) for k in GV.slices(d["bench"], d["config"]["l1_size"])) for d in D)
            print(f"seen designs fully covered {covered}/{len(D)}; feature jobs queued {len(todo)}", flush=True)
            if covered == len(D):
                return
            if not todo:
                time.sleep(120); continue
            list(ex.map(TF.run, todo))


def summary():
    import torch, diag_slice_coverage as DS
    model = T.GEMModel(T.CM.GEM_INPUT_DIM, [512, 256], 0.0); model.load_state_dict(torch.load(T.CM.DEFAULT_CKPT, map_location="cpu")); model.eval()
    norm = T.CM.get_normalizer("18k"); rows = []; D = seen_designs()
    for d in D:
        b = d["bench"]; ks = GV.slices(b, d["config"]["l1_size"])
        if not gem5_done(d) or not all(os.path.exists(feat_path(d, k)) for k in ks):
            continue
        g = np.array([json.load(open(os.path.join(GV.OUT, "runs", b, d["name"], f"w{k}", "result.json")))["cpi_stack"] for k in ks])
        Xt = np.stack([np.load(feat_path(d, k))[:T.CM.GEM_INPUT_DIM] for k in ks])
        c = T.S.project17({kk: d["config"][kk] for kk in T.SEARCH})
        Xe = np.stack([f[:T.CM.GEM_INPUT_DIM] for f in DS.engine_multi(b, {c["l1_size"]: ks}).get_features(c, b)])
        wm = float(T.compute_w_min(c))
        with torch.no_grad():
            cp = [torch.clamp(1.0 / (torch.clamp(model(torch.from_numpy(norm.transform(X)).float())[:, T.CM.BASE_IDX], min=0.01) * wm + 1e-8), max=100.0).mean(1).numpy() for X in (Xt, Xe)]
        rows.append((b, abs(cp[0].mean() - g.mean()) / g.mean(), abs(cp[1].mean() - g.mean()) / g.mean()))
    p = os.path.join(TF.OUT, f"designlevel_seen_{len(rows)}.txt")
    if os.path.exists(p):
        sys.exit(f"exists, refusing to overwrite: {p}")
    L = [f"design-level |error| vs windowed gem5 (cycles per uop, 10 windows/design), 13 seen-checkpoint benchmarks: {len(rows)} / {len(D)} designs",
         "bench            designs | training-identical features | DSE-engine features"]
    for b in sorted({r[0] for r in rows}):
        R = [r for r in rows if r[0] == b]
        L.append(f"{b:16s} {len(R):4d} | {np.mean([r[1] for r in R]):6.2%} | {np.mean([r[2] for r in R]):6.2%}")
    if rows:
        L.append(f"{'SEEN (13)':16s} {len(rows):4d} | {np.mean([r[1] for r in rows]):6.2%} | {np.mean([r[2] for r in rows]):6.2%}")
    print("\n".join(L)); open(p, "w").write("\n".join(L) + "\n"); print("wrote", p)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=8); ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    summary() if a.summary else loop(a.workers)
