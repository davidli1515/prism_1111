"""Check live queue scope before importing the model or starting an experiment."""
import argparse
import json
from pathlib import Path
import time


def skip_cancelled_run(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--bench")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path)
    args, _ = parser.parse_known_args(argv)
    if args.out is None:
        return False
    control_path = args.out / "run_control.json"
    if not control_path.exists():
        return False
    control = json.loads(control_path.read_text())
    if args.seed in control["allowed_seeds"]:
        return False
    record = {"bench": args.bench, "seed": args.seed, "status": "cancelled_before_start",
              "surrogate_queries": 0, "time_unix": time.time(), "reason": control["reason"]}
    path = args.out / f"cancelled_{args.bench}_s{args.seed}.json"
    if not path.exists():
        with path.open("x") as f:
            json.dump(record, f, indent=2)
    print(json.dumps(record), flush=True)
    return True
