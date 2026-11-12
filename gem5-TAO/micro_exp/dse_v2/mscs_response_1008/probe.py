#!/usr/bin/env python3
"""Audit legacy masking data and test response composition at equal pair budgets.

Default: saved mean MSCS only (an explicitly approximate diagnostic).
--exact: re-evaluate base + ALL valid singles, independently of McPAT success,
using the frozen checkpoint and the original 50-slice selection. Compose before
averaging. Pair CPI labels come from the existing masking files; no area/HV claim.
No legacy code, checkpoint, data, area cache or result is written.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from common import DEFAULT_CKPT, GEM_INPUT_DIM, PRECOMPUTE, get_normalizer, config_key
from config_dse import ALL_PARAMS, compute_w_min
from refine_v3 import _shift
from response import cpi_samples, compose, config_additive, predict_pair


class SliceOracle:
    """Read-only, workload-lazy equivalent of Surrogate, retaining slice outputs."""
    def __init__(self, bench):
        import torch
        from model import GEMModel
        self.torch, self.bench = torch, bench
        torch.set_num_threads(1)
        self.model = GEMModel(GEM_INPUT_DIM, [512, 256], 0.0).eval()
        self.model.load_state_dict(torch.load(DEFAULT_CKPT, map_location="cpu"))
        self.norm = get_normalizer()
        self.data, self.cache = {}, {}
        self.n_queries = 0

    def query(self, c):
        from dse_infer import load_precomputed, dse_infer
        key = config_key(c)
        if key in self.cache:
            return self.cache[key]
        l1 = c["l1_size"]
        if l1 not in self.data:
            paths = sorted((Path(PRECOMPUTE) / f"l1_{l1}" / self.bench).glob(
                "slice_*/dse_precomputed.npy"))
            paths = [p for p in paths if int(p.parent.name.split("_")[1]) % 2 == 0][:50]
            if not paths:
                raise ValueError(f"Missing precompute: {self.bench} L1={l1}")
            self.data[l1] = [(int(p.parent.name.split("_")[1]), load_precomputed(p)) for p in paths]
        ids, pre = zip(*self.data[l1])
        with np.errstate(divide="ignore", invalid="ignore"):
            x = np.stack([dse_infer(p, c) for p in pre])
        with self.torch.no_grad():
            t = self.torch.from_numpy(self.norm.transform(x)).float()
            m = self.model(t).numpy()
        cpi = float(cpi_samples(m, compute_w_min(c)).mean())
        self.n_queries += 1
        self.cache[key] = (m, cpi, ids)
        return self.cache[key]


def pair_config(c, p, q):
    # Reproduce the old collector's order/fallback, not a new action space.
    for a, b in ((p, q), (q, p)):
        ca = _shift(c, a, 1)
        cab = _shift(ca, b, 1) if ca is not None else None
        if cab is not None and config_key(cab) != config_key(ca):
            return cab
    raise ValueError(f"Cannot reproduce saved pair {p}, {q}")


def ranking_metrics(gains, predictions, baseline, rng):
    """Same candidate set/budget for every method. Baseline single is retained."""
    gains = np.asarray(gains)
    optimum = max(float(gains.max()), baseline, 0.0)
    out = {}
    if optimum <= 0:
        return out
    for name, score in predictions.items():
        orders = [np.argsort(-np.asarray(score), kind="stable")]
        if name == "random":
            orders = [rng.permutation(len(gains)) for _ in range(200)]
        out[name] = {}
        for k in (1, 3, 5):
            best = [max(float(gains[o[:k]].max()), baseline, 0.0) for o in orders]
            out[name][str(k)] = {
                "retained_gain": float(np.mean(best) / optimum),
                "regret": float(optimum - np.mean(best)),
                "pair_queries": min(k, len(gains)),
            }
    return out


def run_file(path, exact, rng):
    data = json.loads(path.read_text())
    oracle = SliceOracle(data["bench"]) if exact else None
    counts, rows = Counter(), []
    for s in data["rows"]:
        c, sing = s["cfg"], s["singles"]
        ci = {n: _shift(c, n, 1) for n in ALL_PARAMS}
        counts["states"] += 1
        counts["valid_singles_missing_in_legacy"] += sum(
            cn is not None and n not in sing for n, cn in ci.items())
        samples, cpis = {}, {}
        if exact:
            m0, cpi0, ids0 = oracle.query(c)
            if not np.isclose(cpi0, s["cpi"], rtol=3e-6):
                raise ValueError(f"Baseline CPI drift: {cpi0} vs {s['cpi']}")
            for n, cn in ci.items():
                if cn is None:
                    continue
                m, v, ids = oracle.query(cn)
                if ids != ids0:
                    raise ValueError("Slice alignment mismatch")
                samples[n], cpis[n] = m, v
                if n in sing and not np.isclose(v, s["cpi"] * (1-sing[n]["gain"]), rtol=3e-6):
                    raise ValueError(f"Single CPI drift: {n}")
        else:
            m0, cpi0 = np.array(s["mscs"])[None, :], s["cpi"]
            for n, v in sing.items():
                samples[n] = np.array(v["mscs"])[None, :]
                cpis[n] = cpi0 * (1-v["gain"])
        best_single = max([0.0] + [1-v/cpi0 for v in cpis.values()])
        pools = {"composable": [], "same_l1": []}
        seen = set()
        for p in s["pairs"]:
            a, b = p["p"], p["q"]
            counts["legacy_pairs"] += 1
            missing = any(n not in sing and n not in s["noop"] for n in (a, b))
            counts["legacy_pairs_with_unknown_single_gain"] += missing
            if a not in samples or b not in samples:
                continue
            cab = pair_config(c, a, b)
            if not config_additive(c, ci[a], ci[b], cab):
                counts["nonadditive_projection_pairs"] += 1
                continue
            key = config_key(cab)
            if key in seen:
                counts["duplicate_configs"] += 1
                continue
            seen.add(key)
            w0, wa, wb, wab = [compute_w_min(x) for x in (c, ci[a], ci[b], cab)]
            va, vb = cpis[a], cpis[b]
            pred = {
                "scalar_add": va + vb - cpi0,
                "scalar_mult": va * vb / cpi0,
                "scalar_width_add": (va*wa + vb*wb - cpi0*w0) / wab,
                "scalar_width_mult": (va*wa * vb*wb / (cpi0*w0)) / wab,
                "mscs_compose": predict_pair(m0, samples[a], samples[b], wab),
                # Strong control: retain Base per sample/stage but no pressure labels.
                "base_detail": float(np.minimum(
                    1.0 / (np.maximum(m0[..., ::4], .01)*w0+1e-8) *
                    (np.maximum(m0[..., ::4], .01) /
                     np.maximum(samples[a][..., ::4], .01)) *
                    (np.maximum(m0[..., ::4], .01) /
                     np.maximum(samples[b][..., ::4], .01)) * w0 / wab,
                    100.0).mean()),
            }
            record = {"p": a, "q": b, "gain": p["gain"],
                      "actual_synergy": p["gain"] - (1-va/cpi0) - (1-vb/cpi0),
                      "pred_gain": {n: 1-v/cpi0 for n, v in pred.items()}}
            pools["composable"].append(record)
            if c["l1_size"] == cab["l1_size"]:
                pools["same_l1"].append(record)
        for pool, pairs in pools.items():
            if not pairs:
                continue
            preds = {n: [p["pred_gain"][n] for p in pairs] for n in pairs[0]["pred_gain"]}
            preds["random"] = np.zeros(len(pairs))
            metrics = ranking_metrics([p["gain"] for p in pairs], preds, best_single, rng)
            rows.append({"state": s["state"], "pool": pool, "n_pairs": len(pairs),
                         "best_single_gain": best_single, "metrics": metrics,
                         "pairs": pairs})
        print(f"{data['bench']} state={s['state']} pairs={len(pools['composable'])} "
              f"queries={oracle.n_queries if oracle else 0}", flush=True)
    return {"bench": data["bench"], "audit": dict(counts), "rows": rows,
            "new_queries": oracle.n_queries if oracle else 0}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exact", action="store_true")
    ap.add_argument("--bench", default="*")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dest = Path(args.out)
    if dest.exists():
        ap.error(f"Refusing to overwrite {dest}")
    files = sorted((ROOT / "mscs_guide/results").glob(f"masking_{args.bench}.json"))
    if not files:
        ap.error("No masking data matched")
    start = time.time()
    results = [run_file(p, args.exact, np.random.RandomState(0)) for p in files]
    summary = {}
    for pool in ("composable", "same_l1"):
        rows = [r for d in results for r in d["rows"] if r["pool"] == pool and r["metrics"]]
        summary[pool] = {"states": len(rows), "pairs": sum(r["n_pairs"] for r in rows)}
        for name in rows[0]["metrics"] if rows else []:
            summary[pool][name] = {
                k: float(np.mean([r["metrics"][name][k]["retained_gain"] for r in rows]))
                for k in ("1", "3", "5")}
    output = {"args": vars(args), "checkpoint": DEFAULT_CKPT,
              "checkpoint_sha256": hashlib.sha256(Path(DEFAULT_CKPT).read_bytes()).hexdigest(),
              "note": "Fixed, selected plateau states; surrogate-only pair labels. No HV/area claim. "
                      "All policies pay the same base/singles and k unique pair queries. "
                      "Exact mode repairs missing singles; saved-mean mode cannot.",
              "wall_seconds": time.time()-start, "summary": summary, "results": results}
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("x") as f:
        json.dump(output, f, indent=2, allow_nan=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
