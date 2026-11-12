#!/usr/bin/env python3
"""CPI definition mismatch.  PRISM's label is 1/(Base_commit * W_min) = cycles per committed micro-op (ARM splits
instructions into uops), while the traj validation compares it with gem5 stats `cpi` = cycles per instruction.
For the same (bench, config, insts) rows as diag_slice_coverage.npz this script adds
  g_uop = gem5 cycles per uop (stack-derived, from result.json mscs16; = numCycles / committed ops)
  r     = committed ops / insts of that run (stats.txt, switch_cpus_1)
and reports the per-benchmark bias of the surrogate (50 slices; gem5 range) against both definitions.
Read-only. Writes traj/results_calibrated/diag_cpi_definition.{txt,npz}; never overwrites."""
import os, sys, re, json, glob
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); TRAJ = os.path.dirname(HERE); sys.path.insert(0, TRAJ); sys.path.insert(0, HERE)
import diag_slice_coverage as DS
OUT = os.path.join(TRAJ, "results_calibrated", "diag_cpi_definition")


def stat(txt, k):
    m = re.search(r"^system\.switch_cpus_1\." + re.escape(k) + r"\s+([\d.e+-]+)", txt, re.M)
    return float(m.group(1)) if m else None


def main():
    for e in (".txt", ".npz"):
        if os.path.exists(OUT + e):
            sys.exit(f"exists, refusing to overwrite: {OUT + e}")
    Z = np.load(DS.OUT + ".npz"); R = DS.runs()
    # rebuild the per-(bench, config, insts) gem5 values in the same order as diag_slice_coverage.run()
    guop, ratio, gchk, bch = [], [], [], []
    for b, D in sorted(R.items()):
        for (k, ins), (c, g) in D.items():
            us, rs = [], []
            for v in DS.VALS:
                for f in glob.glob(os.path.join(TRAJ, v, "runs", b, "*", "result.json")):
                    d = json.load(open(f))
                    if d.get("error") or not d.get("cpi") or d["insts"] != ins or T_key(d) != k:
                        continue
                    m = d["mscs16"]; cc = d["config"]
                    w = min(int(cc.get(x, cc["dispatchWidth"])) for x in ("fetchWidth", "decodeWidth", "renameWidth", "dispatchWidth", "issueWidth", "commitWidth"))
                    us.append(sum(m[12:16]) / m[12] / w)
                    st = open(f.replace("result.json", "stats.txt")).read() if os.path.exists(f.replace("result.json", "stats.txt")) else ""
                    ops, ins_ = stat(st, "thread_0.numOps"), stat(st, "thread_0.numInsts")
                    rs.append(ops / ins_ if ops and ins_ else np.nan)
            guop.append(np.mean(us)); ratio.append(np.nanmean(rs) if rs else np.nan); gchk.append(np.mean(g)); bch.append(b)
    guop, ratio, gchk, bch = map(np.array, (guop, ratio, gchk, bch))
    assert (bch == Z["bench"]).all() and np.allclose(gchk, Z["gem5"]), "row order mismatch"
    g, m50, mall, b = Z["gem5"], Z["m50"], Z["mall"], Z["bench"]
    L = [f"rows {len(b)}; stack-derived cycles/uop vs numCycles/numOps checked on the pilot designs",
         "bench            n | uops/inst (median, min-max) | bias vs gem5 CPI: 50 sl / gem5 range | bias vs gem5 cycles/uop: 50 sl / gem5 range"]
    def e(m, ref, k): return (m[k] - ref[k]) / ref[k]
    bias = {n: [] for n in ("cpi50", "cpiall", "uop50", "uopall")}
    for bb in sorted(set(b)):
        k = b == bb; rr = ratio[k]
        vals = [e(m50, g, k), e(mall, g, k), e(m50, guop, k), e(mall, guop, k)]
        for n, v in zip(bias, vals):
            bias[n].append(v.mean())
        L.append(f"{bb:16s} {k.sum():3d} | {np.nanmedian(rr):.3f} ({np.nanmin(rr):.3f}-{np.nanmax(rr):.3f}) | "
                 + " / ".join(f"{v.mean():+6.1%} ({np.abs(v).mean():4.1%})" for v in vals[:2]) + " | "
                 + " / ".join(f"{v.mean():+6.1%} ({np.abs(v).mean():4.1%})" for v in vals[2:]))
    k = np.ones(len(b), bool)
    L.append("ALL                  |                              | " + " / ".join(f"{e(m, ref, k).mean():+6.1%} ({np.abs(e(m, ref, k)).mean():4.1%})"
             for m, ref in ((m50, g), (mall, g))) + " | " + " / ".join(f"{e(m, ref, k).mean():+6.1%} ({np.abs(e(m, ref, k)).mean():4.1%})" for m, ref in ((m50, guop), (mall, guop))))
    for n, v in bias.items():
        v = np.array(v); L.append(f"{n:7s}: benchmarks with |mean bias| > 8%: {(np.abs(v) > 0.08).sum()}/17, > 5%: {(np.abs(v) > 0.05).sum()}/17; mean |bias| {np.abs(v).mean():.1%}")
    print("\n".join(L)); open(OUT + ".txt", "w").write("\n".join(L) + "\n")
    np.savez_compressed(OUT + ".npz", bench=b, gem5_cpi=g, gem5_cpuop=guop, uops_per_inst=ratio, m50=m50, mall=mall)


def T_key(d):
    c = DS.T.S.project17({kk: d["config"][kk] for kk in DS.T.SEARCH})
    return DS.T.ckey(c)


if __name__ == "__main__":
    main()
