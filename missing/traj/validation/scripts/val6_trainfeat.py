#!/usr/bin/env python3
"""Training-identical input features for the gem5_val6 designs.
For every design x the 10 windows of gem5_val6_windowed: copy the DSE o3trace of that slice into a work dir, write the
design's arch_params.npy (training encoding), run concord_analytical exactly as gen_spec_traindata.py does
(ext-sim warmup from simpletrace + ext-sim on the trace + features), keep gem_data.npy only.
Reproduction check: two 18k training rows (gcc slices 0, 37) give byte-identical 4681 features this way.
Work dirs:  results_calibrated/val6_trainfeat/<bench>/<design>/w<k>/gem_data.npy  (resumable, never overwritten)
  python calib/val6_trainfeat.py --workers 4
  python calib/val6_trainfeat.py --summary      # model on training-identical vs DSE-engine features vs windowed gem5"""
import os, sys, json, glob, shutil, argparse, subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); TRAJ = os.path.dirname(HERE); sys.path.insert(0, TRAJ); sys.path.insert(0, HERE)
import gem5_val6_windowed as GV
T = GV.T
OUT = os.path.join(TRAJ, "results_calibrated", "val6_trainfeat")
CONC = "/scratch/gem5-TAO/concord_analytical/concord_analytical"
TRACES = "/scratch/gem5-TAO/micro_exp/dse_exp/traces"
CKPT = "/scratch2/wl730/micro_ckpt/m5out_cpts"
ORDER = ["fetchWidth", "decodeWidth", "renameWidth", "dispatchWidth", "issueWidth", "wbWidth", "commitWidth", "numROBEntries",
         "numIQEntries", "LQEntries", "SQEntries", "fetchBufferSize", "IntALU", "IntMultDiv", "FP_ALU", "FP_MultDiv", "RdWrPort",
         "l1_size", "l2_size", "l1d_assoc", "l1i_assoc", "l2_assoc", "bp_type"]


def arch(c):
    p = dict(c); p.setdefault("issueWidth", p["dispatchWidth"]); p.setdefault("renameWidth", p["dispatchWidth"])
    return np.array([float(p[k]) for k in ORDER], np.float32)


def job_list():
    J, per = [], {}
    for d in GV.designs():
        per.setdefault(d["bench"], []).append(d)
    done = lambda d: len(glob.glob(os.path.join(GV.OUT, "runs", d["bench"], d["name"], "w*", "result.json")))
    for b in per:                                   # designs with finished gem5 windows first
        per[b].sort(key=lambda d: -done(d))
    while any(per.values()):
        for b in sorted(per):
            if per[b]:
                d = per[b].pop(0)
                J += [(d, k) for k in GV.slices(b, d["config"]["l1_size"])]
    return J


def run(item):
    d, k = item; b = d["bench"]; w = os.path.join(OUT, b, d["name"], f"w{k}"); res = os.path.join(w, "gem_data.npy")
    if os.path.exists(res):
        return True
    os.makedirs(w, exist_ok=True)
    np.save(os.path.join(w, "arch_params.npy"), arch(d["config"]))
    shutil.copyfile(os.path.join(TRACES, b, f"slice_{k}", "o3trace.txt"), os.path.join(w, "o3trace.txt"))
    tick = GV.GP.BENCH[b]["tick"]
    rc = subprocess.run([CONC, w, "--ckpt-dir", os.path.join(CKPT, b, tick), "--concord_warmup", str(GV.WARM + k * GV.SLICE)],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode
    for f in os.listdir(w):                         # keep only the features and the config
        if f not in ("gem_data.npy", "arch_params.npy"):
            os.remove(os.path.join(w, f))
    return rc == 0 and os.path.exists(res)


def summary():
    import torch, diag_slice_coverage as DS
    model = T.GEMModel(T.CM.GEM_INPUT_DIM, [512, 256], 0.0); model.load_state_dict(torch.load(T.CM.DEFAULT_CKPT, map_location="cpu")); model.eval()
    norm = T.CM.get_normalizer("18k"); rows = []
    for d in GV.designs():
        b = d["bench"]; ks = GV.slices(b, d["config"]["l1_size"])
        G = {k: os.path.join(GV.OUT, "runs", b, d["name"], f"w{k}", "result.json") for k in ks}
        F = {k: os.path.join(OUT, b, d["name"], f"w{k}", "gem_data.npy") for k in ks}
        if not all(os.path.exists(G[k]) and os.path.exists(F[k]) for k in ks):
            continue
        g = np.array([json.load(open(G[k]))["cpi_stack"] for k in ks])
        Xt = np.stack([np.load(F[k])[:T.CM.GEM_INPUT_DIM] for k in ks])
        c = T.S.project17({kk: d["config"][kk] for kk in T.SEARCH})
        eng = DS.engine_multi(b, {c["l1_size"]: ks}); Xe = np.stack([f[:T.CM.GEM_INPUT_DIM] for f in eng.get_features(c, b)])
        wm = float(T.compute_w_min(c))
        with torch.no_grad():
            cp = [torch.clamp(1.0 / (torch.clamp(model(torch.from_numpy(norm.transform(X)).float())[:, T.CM.BASE_IDX], min=0.01) * wm + 1e-8), max=100.0).mean(1).numpy() for X in (Xt, Xe)]
        rows.append((b, np.abs(cp[0] - g) / g, np.abs(cp[1] - g) / g, (cp[0].mean() - g.mean()) / g.mean(), (cp[1].mean() - g.mean()) / g.mean()))
    p = os.path.join(OUT, f"summary_{len(rows)}.txt")
    if os.path.exists(p):
        sys.exit(f"exists, refusing to overwrite: {p}")
    unseen = {"526.blender_r", "623.xalancbmk_s", "631.deepsjeng_s", "638.imagick_s"}
    L = [f"gem5_val6 designs with windowed gem5 AND training-identical features: {len(rows)}; cycles per uop, same 10 windows",
         "bench            designs | per-window MAPE: training-identical features / DSE-engine features | design-level |err|: training / engine"]
    for b in sorted({r[0] for r in rows}):
        R = [r for r in rows if r[0] == b]
        L.append(f"{b:16s} {len(R):4d} | {np.concatenate([r[1] for r in R]).mean():5.1%} / {np.concatenate([r[2] for r in R]).mean():5.1%} | "
                 f"{np.mean([abs(r[3]) for r in R]):5.1%} / {np.mean([abs(r[4]) for r in R]):5.1%}")
    for name, sel in (("ALL", lambda r: True), ("seen checkpoint", lambda r: r[0] not in unseen), ("unseen checkpoint", lambda r: r[0] in unseen)):
        R = [r for r in rows if sel(r)]
        if R:
            t, e = np.concatenate([r[1] for r in R]), np.concatenate([r[2] for r in R])
            L.append(f"{name:18s} {len(R):4d} | per-window MAPE {t.mean():5.1%} (p90 {np.percentile(t, 90):5.1%}) / {e.mean():5.1%} (p90 {np.percentile(e, 90):5.1%}) | "
                     f"design-level {np.mean([abs(r[3]) for r in R]):5.1%} / {np.mean([abs(r[4]) for r in R]):5.1%}")
    print("\n".join(L)); open(p, "w").write("\n".join(L) + "\n"); print("wrote", p)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=4); ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    if a.summary:
        return summary()
    J = job_list(); print(f"{len(J)} feature jobs, {a.workers} workers", flush=True)
    with ThreadPoolExecutor(a.workers) as ex:
        for i, f in enumerate(as_completed([ex.submit(run, j) for j in J])):
            if not f.result():
                print("failed", i, flush=True)
            if (i + 1) % 50 == 0:
                print(f"{i + 1}/{len(J)}", flush=True)


if __name__ == "__main__":
    main()
