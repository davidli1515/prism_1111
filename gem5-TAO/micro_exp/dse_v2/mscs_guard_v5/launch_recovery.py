#!/usr/bin/env python3
"""Supervise only missing seed-42 arms; independent of an interactive session."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--jobs", type=int, default=6)
    args = ap.parse_args()
    lock = (args.out / "recovery_queue.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    scope = json.loads((args.out / "queue_effective.json").read_text())
    jobs = scope["jobs"]
    if any(j["seed"] != 42 for j in jobs):
        raise ValueError("Recovery was authorized for seed 42 only")
    started = time.time()
    status = {"started_unix": started, "pid": os.getpid(), "jobs": jobs, "completed": []}
    def persist():
        status["updated_unix"] = time.time()
        temp = args.out / "recovery_status.tmp"
        temp.write_text(json.dumps(status, indent=2))
        temp.replace(args.out / "recovery_status.json")
    persist()
    def work(job):
        bench, seed = job["bench"], job["seed"]
        log_path = args.out / f"{bench}_s{seed}_recovery_{int(started)}.log"
        command = [sys.executable, str(HERE / "resume_single_seed.py"), "--bench", bench,
                   "--seed", str(seed), "--out", str(args.out), "--mcpat-workers", "2"]
        print(f"RECOVERY START {bench} s{seed}", flush=True)
        with log_path.open("x") as output:
            p = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT,
                               env=dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
                                        PYTHONDONTWRITEBYTECODE="1"))
        result = dict(job, returncode=p.returncode, log=str(log_path), finished_unix=time.time())
        print(json.dumps(result), flush=True)
        return result
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for future in as_completed([pool.submit(work, j) for j in jobs]):
            status["completed"].append(future.result())
            persist()
    if any(r["returncode"] for r in status["completed"]):
        raise SystemExit("Recovery failed; see saved logs; no completed results overwritten")
    subprocess.run([sys.executable, str(HERE / "summarize.py"), "--data", str(args.out)], check=True)
    status["finished_unix"] = time.time()
    persist()


if __name__ == "__main__":
    main()
