#!/usr/bin/env python3
"""CPU vs GPU timing of the trajectory predictor (traj_common.WLPredictor: FastEngine features + GEMModel,
50 slices per design). Designs = the first N distinct designs visited by the small-start greedy walks of one
benchmark (results_small/<bench>/part_*.npz), i.e. real designs of Section V. Timed separately:
  feat         FastEngine.get_features + normaliser (CPU numpy), per design
  cpu_single   model forward per design, 1 thread (exactly as the trajectory runs)
  cpu_batched  one forward over all designs (chunks), all CPU threads
  gpu_single   model forward per design on the GPU
  gpu_batched  one forward over all designs (chunks) on the GPU
Each path reproduces WLPredictor's CPI / MSCS; the max difference vs cpu_single is reported.
  python gpu_timing.py --bench 644.nab_s --n 2000
Writes gpu_timing/<bench>_n<N>.txt (never overwrites)."""
import os, sys, glob, time, argparse
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import traj_common as T
from config_dse import ALL_PARAMS, compute_w_min

HERE = T.HERE; CM = T.CM


def designs(bench, n):
    fs = sorted(glob.glob(os.path.join(HERE, "results_small", bench, "part_*.npz")))
    out, seen = [], set()
    for f in fs:
        for row in np.load(f)["cfg"]:
            c = dict(T.S.FIXED)
            for p, v in zip(T.SEARCH, row):
                c[p] = type(ALL_PARAMS[p][0])(v) if isinstance(ALL_PARAMS[p], list) else int(round(float(v)))
            k = T.ckey(c)
            if k not in seen:
                seen.add(k); out.append(c)
            if len(out) == n:
                return out
    return out


def head(m, wmin):
    """m: (slices, 16) tensor -> (CPI, MSCS mean) as WLPredictor."""
    base = torch.clamp(m[:, CM.BASE_IDX], min=0.01)
    cs = torch.clamp(1.0 / (base * wmin + 1e-8), max=100.0)
    return cs.mean(1).mean().item(), m.mean(0)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--bench", default="644.nab_s"); ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--chunk", type=int, default=1000)
    a = ap.parse_args()
    os.makedirs(os.path.join(HERE, "gpu_timing"), exist_ok=True)
    out = os.path.join(HERE, "gpu_timing", f"{a.bench}_n{a.n}.txt")
    if os.path.exists(out):
        sys.exit(f"exists, refusing to overwrite: {out}")
    L = []
    say = lambda s: (print(s, flush=True), L.append(s))
    P = T.WLPredictor(a.bench); cfgs = designs(a.bench, a.n); N = len(cfgs)
    wmin = [float(compute_w_min(c)) for c in cfgs]
    say(f"bench {a.bench}; {N} distinct small-start designs; GPU {torch.cuda.get_device_name(0)}; CPU threads {os.cpu_count()}")

    t = time.perf_counter()
    Xs = [torch.from_numpy(P.norm.transform(np.stack([f[:CM.GEM_INPUT_DIM] for f in P.eng.get_features(c, a.bench)]))).float() for c in cfgs]
    t_feat = time.perf_counter() - t
    S = Xs[0].shape[0]

    torch.set_num_threads(1); m_cpu = P.model
    with torch.no_grad():
        t = time.perf_counter(); ref = [head(m_cpu(x), w) for x, w in zip(Xs, wmin)]; t_cs = time.perf_counter() - t
        torch.set_num_threads(os.cpu_count())
        Xall = torch.cat(Xs); t = time.perf_counter()
        Mc = torch.cat([m_cpu(Xall[i:i + a.chunk * S]) for i in range(0, len(Xall), a.chunk * S)]); t_cb = time.perf_counter() - t
        torch.set_num_threads(1)
        m_gpu = T.GEMModel(CM.GEM_INPUT_DIM, [512, 256], 0.0); m_gpu.load_state_dict(m_cpu.state_dict()); m_gpu = m_gpu.cuda().eval()
        for _ in range(20):                                       # warm-up
            m_gpu(Xs[0].cuda())
        torch.cuda.synchronize()
        t = time.perf_counter(); gs = []
        for x, w in zip(Xs, wmin):
            cpi, ms = head(m_gpu(x.cuda()), w); gs.append((cpi, ms.cpu()))
        torch.cuda.synchronize(); t_gs = time.perf_counter() - t
        t = time.perf_counter()
        Xg = Xall.cuda()
        Mg = torch.cat([m_gpu(Xg[i:i + a.chunk * S]) for i in range(0, len(Xg), a.chunk * S)]).cpu()
        torch.cuda.synchronize(); t_gb = time.perf_counter() - t

    def err(M):
        M = M.view(N, S, -1); d_cpi = d_m = 0.0
        for j in range(N):
            cpi, ms = head(M[j], wmin[j]); d_cpi = max(d_cpi, abs(cpi - ref[j][0]) / ref[j][0]); d_m = max(d_m, float((ms - ref[j][1]).abs().max()))
        return d_cpi, d_m
    e_cb, e_gb = err(Mc), err(Mg)
    e_gs = (max(abs(g[0] - r[0]) / r[0] for g, r in zip(gs, ref)), max(float((g[1] - r[1]).abs().max()) for g, r in zip(gs, ref)))
    say(f"slices per design {S}; feature dim {Xs[0].shape[1]}")
    say(f"{'path':14s}{'total s':>10s}{'ms/design':>12s}{'max rel dCPI':>14s}{'max dMSCS':>11s}")
    for name, tt, e in (("feat (CPU)", t_feat, None), ("cpu_single", t_cs, (0, 0)), ("cpu_batched", t_cb, e_cb),
                        ("gpu_single", t_gs, e_gs), ("gpu_batched", t_gb, e_gb)):
        say(f"{name:14s}{tt:10.2f}{1000*tt/N:12.3f}" + (f"{e[0]:14.1e}{e[1]:11.1e}" if e else ""))
    say("end-to-end per design (features + forward): " + ", ".join(
        f"{n} {1000*(t_feat + tt)/N:.2f} ms" for n, tt in (("CPU single", t_cs), ("GPU batched", t_gb))))
    open(out, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
