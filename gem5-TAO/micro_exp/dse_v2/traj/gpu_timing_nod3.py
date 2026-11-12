#!/usr/bin/env python3
"""Timing of PRISM without d3 (Concorde feature set, first N_FEATURES = 3873 dims), Concorde-style:
  precompute (one-time, per benchmark): every config-dependent block of every slice for EVERY value of the
      parameter it reads (full DSE index ranges, all L1 sizes) -- as Concorde sweeps each parameter.
  per design: gather the blocks (table lookup) + raw config + normaliser + MLP forward.
Block arithmetic is fast_infer.FastSliceInfer's (imported, not modified); the gathered vector is checked
bit-exact against FastEngine features[:3873] on --check designs.
Model: GEMModel(3873, [512, 256]) with random weights -- timing only (weights do not change run time).
Designs = gpu_timing.designs (first N distinct small-start designs of the benchmark).
  python gpu_timing_nod3.py --bench 644.nab_s --n 2000
Writes gpu_timing/<bench>_nod3_n<N>.txt (never overwrites)."""
import os, sys, time, argparse
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import traj_common as T
import gpu_timing as G
from config_dse import ALL_PARAMS, compute_w_min
from fast_infer import encode_distribution
from dse_infer import (WINDOW_SIZE, N_FEATURES, DSE_ROB_START, DSE_N_ROB, DSE_LQ_START, DSE_N_LQ, DSE_SQ_START,
                       DSE_N_SQ, DSE_FW_START, DSE_N_FW, DSE_NB_START, DSE_N_NB, DSE_DW_START, DSE_N_DW)

NF = N_FEATURES
FU = list(range(ALL_PARAMS["IntALU"][0], ALL_PARAMS["IntALU"][1] + 1))      # 1..8 (IntALU / FP_ALU / RdWrPort)


def pipes(s, nls, nl=0):                       # = FastSliceInfer.pipes
    K, W, nld, nst = WINDOW_SIZE, s.pre["W"], s.nld, s.nst
    tot = nls + nl
    if tot == 0:
        lo = hi = np.zeros(W)
    else:
        tm = nld / tot + np.where(nls > 0, nst / nls, 0.0); lo = np.where(tm > 0, K / tm, float(K))
        ts = np.where(nls > 0, nst / nls, 0.0); rem = np.maximum(0.0, nld - (ts * nl if nl > 0 else 0.0))
        tn = ts + rem / tot; hi = np.where(tn > 0, K / tn, float(K))
    return np.concatenate([encode_distribution(lo), encode_distribution(hi)])


def build(slices):
    """slices: list of _Slice (one L1 size) -> dict of (n_slices, n_values, dim) tables + template."""
    K = WINDOW_SIZE; enc = encode_distribution
    tb = {"tmpl": np.stack([s.template[:NF] for s in slices])}
    tb["rob"] = np.stack([[enc(s.pre["rob_thr"][i]) for i in range(DSE_N_ROB)] for s in slices])
    tb["lq"] = np.stack([[enc(s.pre["lq_thr"][i]) for i in range(DSE_N_LQ)] for s in slices])
    tb["sq"] = np.stack([[enc(s.pre["sq_thr"][i]) for i in range(DSE_N_SQ)] for s in slices])
    for nm, key in (("alu", "n_alu"), ("fp", "n_fp"), ("ls", "n_ls")):
        tb[nm] = np.stack([[enc(np.where(s.pre[key] > 0, K * w / s.pre[key], float(K))) for w in FU] for s in slices])
    tb["fw"] = np.stack([[enc(s.pre["fw_thr"][i]) for i in range(DSE_N_FW)] for s in slices])
    tb["pipes"] = np.stack([[pipes(s, w) for w in FU] for s in slices])
    tb["fb"] = np.stack([[enc(s.pre["fb_thr"][i]) for i in range(DSE_N_NB * DSE_N_DW)] for s in slices])
    return tb


def gather(tb, c):
    cl = lambda v, a, n: max(0, min(v - a, n - 1))
    X = tb["tmpl"].copy()
    X[:, 0:101] = tb["rob"][:, cl(c["numROBEntries"], DSE_ROB_START, DSE_N_ROB)]
    X[:, 101:202] = tb["lq"][:, cl(c["LQEntries"], DSE_LQ_START, DSE_N_LQ)]
    X[:, 202:303] = tb["sq"][:, cl(c["SQEntries"], DSE_SQ_START, DSE_N_SQ)]
    X[:, 303:404] = tb["alu"][:, c["IntALU"] - FU[0]]
    X[:, 404:505] = tb["fp"][:, c["FP_ALU"] - FU[0]]
    X[:, 505:606] = tb["ls"][:, c["RdWrPort"] - FU[0]]
    X[:, 606:707] = tb["fw"][:, cl(c["fetchWidth"], DSE_FW_START, DSE_N_FW)]
    X[:, 707:909] = tb["pipes"][:, c["RdWrPort"] - FU[0]]
    X[:, 1010:1111] = tb["fb"][:, cl(c["fetchBufferSize"], DSE_NB_START, DSE_N_NB) * DSE_N_DW
                                  + cl(c["decodeWidth"], DSE_DW_START, DSE_N_DW)]
    d = c["dispatchWidth"]
    X[:, 3850:3873] = [c["fetchWidth"], c["decodeWidth"], d, d, d, c["wbWidth"], c["commitWidth"],
                       c["numROBEntries"], c["numIQEntries"], c["LQEntries"], c["SQEntries"], c["fetchBufferSize"],
                       c["IntALU"], c["IntMultDiv"], c["FP_ALU"], c["FP_MultDiv"], c["RdWrPort"], c["l1_size"],
                       c["l2_size"], c["l1d_assoc"], c["l1i_assoc"], c["l2_assoc"], c["bp_type"]]
    return X


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--bench", default="644.nab_s"); ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--check", type=int, default=200); ap.add_argument("--chunk", type=int, default=1000)
    a = ap.parse_args()
    out = os.path.join(T.HERE, "gpu_timing", f"{a.bench}_nod3_n{a.n}.txt")
    if os.path.exists(out):
        sys.exit(f"exists, refusing to overwrite: {out}")
    L = []
    say = lambda s: (print(s, flush=True), L.append(s))
    P = T.WLPredictor(a.bench); cfgs = G.designs(a.bench, a.n); N = len(cfgs)
    wmin = [float(compute_w_min(c)) for c in cfgs]
    mu, sd = P.norm.mean[:NF].astype(np.float64), P.norm.std[:NF].astype(np.float64)
    say(f"bench {a.bench}; {N} distinct small-start designs; no d3 (dims 0:{NF}); GPU {torch.cuda.get_device_name(0)}")

    t = time.perf_counter()
    tabs = {l1: build(d[a.bench]) for l1, d in P.eng.slices.items() if a.bench in d}
    t_pre = time.perf_counter() - t
    S = len(next(iter(tabs.values()))["tmpl"])
    say(f"precompute (one-time): {t_pre:.1f} s for {len(tabs)} L1 sizes x {S} slices "
        f"({sum(v.nbytes for tb in tabs.values() for v in tb.values())/2**20:.0f} MB)")

    ok = all(np.array_equal(gather(tabs[c["l1_size"]], c), np.stack([f[:NF] for f in P.eng.get_features(c, a.bench)]))
             for c in cfgs[:a.check])
    say(f"gathered features bit-exact vs FastEngine[:{NF}] on {a.check} designs: {ok}")

    t = time.perf_counter()
    Xs = [torch.from_numpy((gather(tabs[c["l1_size"]], c) - mu) / sd).float() for c in cfgs]
    t_feat = time.perf_counter() - t

    torch.manual_seed(0)
    m = T.GEMModel(NF, [512, 256], 0.0).eval()
    with torch.no_grad():
        torch.set_num_threads(1)
        t = time.perf_counter(); [G.head(m(x), w) for x, w in zip(Xs, wmin)]; t_cs = time.perf_counter() - t
        mg = T.GEMModel(NF, [512, 256], 0.0); mg.load_state_dict(m.state_dict()); mg = mg.cuda().eval()
        for _ in range(20):
            mg(Xs[0].cuda())
        torch.cuda.synchronize()
        t = time.perf_counter(); Xg = torch.cat(Xs).cuda()
        Mg = torch.cat([mg(Xg[i:i + a.chunk * S]) for i in range(0, len(Xg), a.chunk * S)]).cpu()
        torch.cuda.synchronize(); t_gb = time.perf_counter() - t
        Mc = m(torch.cat(Xs))
    say(f"max |GPU - CPU| model output: {float((Mg - Mc).abs().max()):.1e}")
    say(f"{'path':26s}{'total s':>10s}{'ms/design':>12s}")
    for nm, tt in (("feat gather+norm (CPU)", t_feat), ("model cpu_single", t_cs), ("model gpu_batched", t_gb)):
        say(f"{nm:26s}{tt:10.3f}{1000*tt/N:12.4f}")
    say("end-to-end per design: " + ", ".join(f"{n} {1000*(t_feat + tt)/N:.3f} ms"
                                               for n, tt in (("CPU single", t_cs), ("GPU batched", t_gb))))
    open(out, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
