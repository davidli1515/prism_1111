#!/usr/bin/env python3
"""Supplement to gem5_val4 (small-start representative walks): uniform 12-state sampling can skip a one-step
peak. Rule applied to all 17 walks: if the state with the highest model Dispatch entropy was not sampled, add
it. Plan -> gem5_val4/plan_peak.json; runs go to gem5_val4/runs/ (same layout, existing results skipped)."""
import os, json, glob
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import gem5_val4  # noqa: F401  (sets gem5_val3 globals to the small-start setting)
import gem5_val3 as G

if __name__ == "__main__":
    jp = os.path.join(G.OUT, "plan_peak.json")
    base = json.load(open(os.path.join(G.OUT, "plan.json")))
    if os.path.exists(jp):
        extra = json.load(open(jp))
    else:
        extra = []
        for b in sorted({j["bench"] for j in base}):
            s = [j for j in base if j["bench"] == b][0]["idx"]
            fs = sorted(glob.glob(os.path.join(G.R, b, "part_*.npz")))
            d = {k: np.concatenate([np.load(f)[k] for f in fs]) for k in ("start", "step", "cfg", "cpi", "mscs")}
            r = np.where(d["start"] == s)[0]; i = r[np.argmax(G.ent(d["cpi"][r], d["mscs"][r]))]
            if any(j["bench"] == b and j["step"] == int(d["step"][i]) for j in base):
                continue
            c = G.T.S.project17(dict(zip(G.T.SEARCH, map(int, d["cfg"][i]))))
            extra.append({"bench": b, "idx": int(s), "tag": f"s{int(d['step'][i]):02d}", "config": c, "step": int(d["step"][i]),
                          "walk_len": len(r), "added": "peak", "model_cpi": float(d["cpi"][i]), "model_mscs": d["mscs"][i].tolist(),
                          "insts": G.surrogate_insts(b, c["l1_size"])})
        json.dump(extra, open(jp, "w"), indent=1)
    print(len(extra), "peak states added:", [(j["bench"].split(".")[1], j["tag"]) for j in extra], flush=True)
    with ThreadPoolExecutor(max(1, len(extra))) as ex:
        for r in ex.map(lambda j: G.run_job({k: j[k] for k in ("bench", "idx", "tag", "config", "insts")}, G.OUT), extra):
            print(r["bench"], r["tag"], "skip" if r.get("skipped") else r.get("error") or "ok", flush=True)
