#!/usr/bin/env python3
"""
SimPoint-style Basic Block Vectors (BBV) per 100K-instruction window.

Sources
  o3 : micro_exp/dse_exp/traces/<bm>/slice_k/o3trace.txt  (the 400 slices of the
       checkpoint used by the 18k dataset; one file per window, commit order)
  st : micro_ckpt/m5out_cpts/<bm>/<ckpt>/simpletrace.txt   (other checkpoints =
       unseen simpoints; instructions [20M, 60M) cut into 400 windows of 100K,
       the same region the slices cover; boundaries approximate)

Trace line format (both): idx,"asm",PC,addr,mnemonic,f1..f5,...; f5 = control-flow flag.
Micro-ops of one instruction appear as consecutive lines with the same PC and are
merged. A basic block ends at a control-flow instruction or a PC discontinuity.
BBV[block start PC] += block length (= executed instructions); normalized to sum 1
by the consumer.

Output: bbv/raw/<bm>__<source>.pkl = {"windows": [dict(pc -> count)], "stats": [...]}

Usage: python compute_bbv.py --workers 24
"""
import os, sys, pickle, argparse, itertools
from multiprocessing import Pool
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
TRACE_DIR = "/path/to/gem5/micro_exp/dse_exp/traces"
CKPT_BASE = "/data/micro_ckpt/m5out_cpts"
OUT_DIR = os.path.join(HERE, "raw")

SLICES = 400
WIN = 100_000
ST_START = 20_000_000

# checkpoint whose 400 slices are in 18k (= dse_exp traces); see make_unseen notes.
# gcc/lbm: 18k also contains a 2nd ckpt (windows 400-799), not covered by o3 traces.
UNSEEN_CKPTS = {     # simpoints absent from 18k (trace-signature verified earlier)
    "508.namd_r":      ["420400000000", "696800000000"],
    "526.blender_r":   ["580900000000", "958700000000"],
    "607.cactuBSSN_s": ["30600000000"],
    "623.xalancbmk_s": ["377600000000", "885500000000"],
    "631.deepsjeng_s": ["804200000000", "1384000000000"],
    "638.imagick_s":   ["561400000000", "653700000000"],
    "641.leela_s":     ["1120100000000", "1429200000000"],
}


def parse(line):
    """-> (pc, is_control, is_mem) or None."""
    parts = line.split('"')
    if len(parts) < 3:
        return None
    f = parts[2].split(",")
    try:
        return int(f[1]), f[8] == "1", f[4] == "1"
    except (IndexError, ValueError):
        return None


class BBVBuilder:
    def __init__(self):
        self.reset()

    def reset(self):
        self.bbv = defaultdict(int)
        self.start, self.length, self.prev_pc = None, 0, None
        self.n = self.n_ctrl = self.n_mem = 0

    def add(self, pc, ctrl, mem):
        """Feed one micro-op line; returns True when a new instruction was counted."""
        if pc == self.prev_pc:                 # micro-op of the same instruction
            return False
        if self.start is not None and pc != self.prev_pc + 4:   # taken jump without flag
            self._close()
        if self.start is None:
            self.start = pc
        self.length += 1
        self.prev_pc = pc
        self.n += 1; self.n_ctrl += ctrl; self.n_mem += mem
        if ctrl:
            self._close()
        return True

    def _close(self):
        if self.start is not None and self.length:
            self.bbv[self.start] += self.length
        self.start, self.length = None, 0

    def finish(self):
        self._close()
        out = (dict(self.bbv), {"n_instr": self.n,
                                "frac_ctrl": self.n_ctrl / max(self.n, 1),
                                "frac_mem": self.n_mem / max(self.n, 1)})
        self.reset()
        return out


def job_o3(bm):
    windows, stats = [], []
    for k in range(SLICES):
        b = BBVBuilder()
        path = os.path.join(TRACE_DIR, bm, f"slice_{k}", "o3trace.txt")
        with open(path) as fh:
            for line in fh:
                r = parse(line)
                if r: b.add(*r)
        w, s = b.finish(); s["slice"] = k
        windows.append(w); stats.append(s)
    return bm, "o3", windows, stats


def job_st(bm, ckpt):
    path = os.path.join(CKPT_BASE, bm, ckpt, "simpletrace.txt")
    windows, stats = [], []
    b = BBVBuilder()
    with open(path) as fh:
        for line in itertools.islice(fh, ST_START, None):
            r = parse(line)
            if r and b.add(*r) and b.n >= WIN:
                w, s = b.finish(); s["slice"] = len(windows)
                windows.append(w); stats.append(s)
                if len(windows) == SLICES:
                    break
    return bm, f"st{ckpt}", windows, stats


def run(job):
    kind, args = job
    bm, src, windows, stats = job_o3(*args) if kind == "o3" else job_st(*args)
    with open(os.path.join(OUT_DIR, f"{bm}__{src}.pkl"), "wb") as fh:
        pickle.dump({"windows": windows, "stats": stats}, fh)
    return f"{bm}__{src}: {len(windows)} windows"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    bms = sorted(d for d in os.listdir(TRACE_DIR) if os.path.isdir(os.path.join(TRACE_DIR, d)))
    if args.only:
        bms = [b for b in bms if b in args.only]
    jobs = [("o3", (bm,)) for bm in bms]
    jobs += [("st", (bm, c)) for bm in bms for c in UNSEEN_CKPTS.get(bm, [])]
    jobs = [j for j in jobs if not os.path.exists(os.path.join(
        OUT_DIR, f"{j[1][0]}__{'o3' if j[0] == 'o3' else 'st' + j[1][1]}.pkl"))]
    print(f"{len(jobs)} jobs, {args.workers} workers", flush=True)
    with Pool(args.workers) as pool:
        for msg in pool.imap_unordered(run, jobs):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
