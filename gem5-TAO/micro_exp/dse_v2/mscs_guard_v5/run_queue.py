#!/usr/bin/env python3
"""Run the predeclared six benchmarks x three seeds, preserving every result."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
BENCHES = ("602.gcc_s", "605.mcf_s", "623.xalancbmk_s", "638.imagick_s", "641.leela_s", "648.exchange2_s")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--mcpat-workers", type=int, default=2)
    ap.add_argument("--total-queries", type=int, default=5000)
    ap.add_argument("--bo-queries", type=int, default=500)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = args.out / "queue.json"
    if manifest.exists():
        raise FileExistsError(manifest)
    jobs = [(b, s) for s in (42, 43, 44) for b in BENCHES]
    spec = {"started_unix": time.time(), "python": sys.executable,
            "jobs": [{"bench": b, "seed": s} for b, s in jobs],
            "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
    manifest.write_text(json.dumps(spec, indent=2))
    def work(bench, seed):
        path = args.out / f"{bench}_s{seed}.log"
        cmd = [sys.executable, str(HERE / "run.py"), "--bench", bench, "--seed", str(seed),
               "--total-queries", str(args.total_queries), "--bo-queries", str(args.bo_queries),
               "--mcpat-workers", str(args.mcpat_workers), "--out", str(args.out)]
        start = time.time()
        print(f"START {bench} s{seed}", flush=True)
        with path.open("x") as log:
            p = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                               env=dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
                                        PYTHONDONTWRITEBYTECODE="1"))
        r = {"bench": bench, "seed": seed, "returncode": p.returncode, "seconds": time.time()-start}
        print(json.dumps(r), flush=True)
        return r
    results = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(work, b, s) for b, s in jobs]
        for f in as_completed(futures):
            results.append(f.result())
            tmp = args.out / "queue_status.tmp"
            tmp.write_text(json.dumps({"completed": results, "total": len(jobs)}, indent=2))
            tmp.replace(args.out / "queue_status.json")
    if any(r["returncode"] for r in results):
        raise SystemExit("One or more runs failed; see per-run logs. Existing results were preserved.")
    subprocess.run([sys.executable, str(HERE / "summarize.py"), "--data", str(args.out)], check=True)


if __name__ == "__main__":
    main()
