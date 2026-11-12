#!/usr/bin/env python3
"""Recheck a small, fixed sample of old McPAT results in isolated directories."""
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import config_key
from refine_v3 import _shift
from safe_area import SafeArea


def main():
    out = HERE / "results/area_audit.json"
    if out.exists():
        raise SystemExit(f"Refusing to overwrite {out}")
    old = {}
    for p in (HERE.parent / "area_cache").glob("*.json"):
        for k, v in json.loads(p.read_text()).items():
            old.setdefault(k, set()).add(v)
    rows = []
    for p in sorted((HERE.parent / "mscs_guide/results").glob("masking_*.json")):
        d = json.loads(p.read_text())
        s = d["rows"][0]
        configs = [("base", s["cfg"], s["area"])]
        configs += [(n, _shift(s["cfg"], n, 1), None) for n in list(s["singles"])[:3]]
        for name, c, saved in configs:
            k = hashlib.md5(json.dumps(config_key(c)).encode()).hexdigest()
            rows.append({"bench": d["bench"], "action": name, "config": c,
                         "saved_base_area": saved,
                         "old_cached_values": sorted(old.get(k, set()), key=lambda v: -1 if v is None else v)})
    area = SafeArea()
    area.prefetch([r["config"] for r in rows], workers=4)
    for r in rows:
        r["isolated_area"] = area(r["config"])
        r["error"] = area.errors.get(area.key(r["config"]))
    # Fresh serial repetitions, bypassing the cache, verify the new concurrency path.
    repeats = []
    for r in rows[:3]:
        _, value, error = area._compute(r["config"])
        repeats.append({"bench": r["bench"], "action": r["action"],
                        "parallel": r["isolated_area"], "serial": value, "error": error})
    area.flush()
    result = {"note": "Fixed sample: first saved state of each benchmark, base + first 3 recorded singles. "
                       "Old cache values can originate from different historical runs; mismatches alone "
                       "do not identify their origin. Source-level shared XML race is independently verified.",
              "rows": rows, "fresh_serial_repeats": repeats,
              "old_keys_with_multiple_values": sum(len(v) > 1 for v in old.values())}
    out.parent.mkdir(exist_ok=True)
    with out.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
