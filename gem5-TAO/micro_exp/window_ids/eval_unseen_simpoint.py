"""
Evaluate existing exp1 / exp4 checkpoints (trained on 18k) on simpoints that
never appear in 18k ("unseen simpoint" test set, from make_simpoint_ids.py).

For every checkpoint it reports CPI MAPE on
  * in-dist test      : the original 1/16 row-split test set (all 23 bms)
  * in-dist (subset)  : same, restricted to bms that have an unseen simpoint
  * unseen simpoint   : rows of the 40k simpoints that share nothing with 18k
plus per-benchmark numbers. The in-dist number is recomputed and checked
against the checkpoint's saved JSON (sanity check of normalizer + loading).

Normalizer: exp1 and exp4 both split 18k with RandomState(seed) sequentially over
sorted benchmarks (15/16 per bm) and fit the normalizers on the full train split,
so both are reproduced exactly by exp1/dataset.create_exp1_dataloaders(seed).

Model type is inferred from the state_dict: head 16-dim -> GEM (CPI from MSCS),
1-dim -> Concorde; input 4681 -> Concorde+D3 features, 3873 -> Concorde features.

Not supported: exp4 PRISM with 40k extra unlabeled (exp4_18k_extra_unlabeled):
its normalizer includes 40k rows and its unlabeled set contains the unseen
simpoints themselves, so it is not a clean unseen-simpoint test.

Usage:
    python eval_unseen_simpoint.py --gpu 1
    python eval_unseen_simpoint.py --ckpts "../exp4/results_77_ablation/exp4_18k/checkpoints/*.pt"
"""
import os, sys, glob, json, argparse
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
MICRO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(MICRO, "exp1"))

from config import (
    SEED, CONCORDE_FEAT_END, D3_FEAT_END, MSCS_START, MSCS_END,
    WMIN_INDICES, BASE_INDICES, BENCHMARK_NAMES,
)
from model import GEMModel, ConcordeModel, compute_cpi_mean
from dataset import load_all_benchmarks, create_exp1_dataloaders

PACKED_18K = "/scratch2/wl730/micro_traindata_packed/gem"
CONCORD_18K = "/scratch2/wl730/micro_traindata_packed/concord"
PACKED_40K = "/scratch2/wl730/micro_traindata_packed_all/gem"
UNSEEN_DIR = os.path.join(HERE, "unseen_simpoint")

DEFAULT_CKPTS = [
    os.path.join(MICRO, "exp1/results/exp1_v3_76/exp1_512_256_183k/*.pt"),
    os.path.join(MICRO, "exp4/results_77/exp4_18k/checkpoints/*.pt"),
    os.path.join(MICRO, "exp4/results_77/exp4_18k_supervised_only/checkpoints/*.pt"),
    os.path.join(MICRO, "exp4/results_77/exp4_18k_concorde/checkpoints/*.pt"),
]


def mape(p, t):
    return float((torch.abs(p - t) / (torch.abs(t) + 1e-8) * 100).mean())


def build_model(state):
    """Rebuild GEMModel / ConcordeModel from state_dict shapes."""
    layers = sorted({int(k.split(".")[1]) for k in state if k.startswith("backbone.")})
    hidden = [state[f"backbone.{i}.block.0.weight"].shape[0] for i in layers]
    in_dim = state["backbone.0.block.0.weight"].shape[1]
    out_dim = state["head.weight"].shape[0]
    if out_dim == 16:
        model, kind = GEMModel(in_dim, hidden, 0.0), "gem"
    elif out_dim == 1:
        model, kind = ConcordeModel(in_dim, hidden, 0.0), "concorde"
    else:
        raise ValueError(f"unexpected head dim {out_dim}")
    model.load_state_dict(state)
    feat = "gem" if in_dim == D3_FEAT_END else "con"
    if in_dim not in (D3_FEAT_END, CONCORDE_FEAT_END):
        raise ValueError(f"unexpected input dim {in_dim}")
    return model.eval(), kind, feat, hidden


@torch.no_grad()
def predict(model, kind, x, w_min, device, bs=50000):
    out = []
    for s in range(0, len(x), bs):
        xb = x[s:s + bs].to(device)
        if kind == "gem":
            out.append(compute_cpi_mean(model(xb), w_min[s:s + bs].to(device)).cpu())
        else:
            out.append(model(xb).cpu())
    return torch.cat(out)


def saved_mape(ckpt):
    """Locate the checkpoint's result JSON (exp1: <stem>.json; exp4: ../exp4_<ratio>.json)."""
    d, stem = os.path.split(ckpt)
    stem = stem[:-3]
    cands = [os.path.join(d, f"{stem}.json")]
    if os.path.basename(d) == "checkpoints":
        cands.append(os.path.join(os.path.dirname(d), f"exp4_{stem.rsplit('_', 1)[-1]}.json"))
    for c in cands:
        if os.path.exists(c):
            j = json.load(open(c))
            if "overall_mape" in j:
                return float(j["overall_mape"]), c
    return None, None


def load_unseen(gem_norm, con_norm):
    """Unseen-simpoint rows, same field definitions as exp1/dataset.load_benchmark."""
    data = {}
    for f in sorted(glob.glob(os.path.join(UNSEEN_DIR, "*.npy"))):
        name = os.path.basename(f)[:-4]
        idx = np.load(f)
        raw = np.load(os.path.join(PACKED_40K, f"{name}.npy"), mmap_mode="r")
        rows = np.asarray(raw[np.sort(idx)], dtype=np.float32)
        mscs = rows[:, MSCS_START:MSCS_END]
        w_min = rows[:, WMIN_INDICES].min(axis=1)
        cpi = 1.0 / (mscs[:, BASE_INDICES[3]] * w_min + 1e-8)
        data[name] = {
            "gem": torch.from_numpy(gem_norm.transform(rows[:, :D3_FEAT_END]).astype(np.float32)),
            "con": torch.from_numpy(con_norm.transform(rows[:, :CONCORDE_FEAT_END]).astype(np.float32)),
            "cpi": torch.from_numpy(cpi.astype(np.float32)),
            "wmin": torch.from_numpy(w_min.astype(np.float32)),
        }
        print(f"  unseen simpoint {name:20s}: {len(rows):,} rows")
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", default=DEFAULT_CKPTS, help="checkpoint paths or globs")
    ap.add_argument("--seed", type=int, default=SEED, help="split seed used at training time")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(UNSEEN_DIR, "eval_unseen_simpoint.json"))
    args = ap.parse_args()

    ckpts = sorted({p for g in args.ckpts for p in glob.glob(g)})
    bad = [c for c in ckpts if "extra_unlabeled" in c]
    if bad:
        sys.exit(f"Refusing 40k-extra checkpoints (see docstring): {bad[:3]}")
    if not ckpts:
        sys.exit("No checkpoints matched")
    if not os.path.isdir(UNSEEN_DIR) or not glob.glob(os.path.join(UNSEEN_DIR, "*.npy")):
        sys.exit("Run make_simpoint_ids.py first")
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    print("Rebuilding 18k split + normalizers (seed=%d)..." % args.seed)
    all_data = load_all_benchmarks(PACKED_18K, CONCORD_18K, BENCHMARK_NAMES)
    _, test_loader, gem_norm, con_norm, bm_idx = create_exp1_dataloaders(all_data, seed=args.seed)
    del all_data
    ds = test_loader.dataset
    test = {"gem": ds.gem_features, "con": ds.concord_features,
            "cpi": ds.cpi_golden, "wmin": ds.w_min}

    print("Loading unseen simpoints...")
    unseen = load_unseen(gem_norm, con_norm)
    subset = sorted(unseen)
    sub_idx = torch.cat([torch.arange(*bm_idx[n]) for n in subset])

    results = {"seed": args.seed, "unseen_benchmarks": subset, "models": {}}
    hdr = f"{'checkpoint':58s} {'in-dist':>8s} {'saved':>7s} {'in(sub)':>8s} {'unseen':>8s}"
    print("\n" + hdr + "\n" + "-" * len(hdr))
    for ck in ckpts:
        model, kind, feat, hidden = build_model(torch.load(ck, map_location="cpu"))
        model.to(device)
        p = predict(model, kind, test[feat], test["wmin"], device)
        in_all = mape(p, test["cpi"])
        in_sub = mape(p[sub_idx], test["cpi"][sub_idx])
        per_in = {n: mape(p[slice(*bm_idx[n])], test["cpi"][slice(*bm_idx[n])]) for n in subset}
        per_un, P, T = {}, [], []
        for n in subset:
            u = unseen[n]
            pu = predict(model, kind, u[feat], u["wmin"], device)
            per_un[n] = mape(pu, u["cpi"]); P.append(pu); T.append(u["cpi"])
        un_all = mape(torch.cat(P), torch.cat(T))
        un_bm_avg = float(np.mean(list(per_un.values())))
        ref, ref_path = saved_mape(ck)
        rel = os.path.relpath(ck, MICRO)
        results["models"][rel] = {
            "kind": kind, "features": feat, "hidden": hidden,
            "in_dist_mape": in_all, "saved_in_dist_mape": ref, "saved_json": ref_path,
            "in_dist_subset_mape": in_sub, "unseen_simpoint_mape": un_all,
            "unseen_simpoint_mape_bm_avg": un_bm_avg,
            "per_benchmark_in_dist": per_in, "per_benchmark_unseen": per_un,
        }
        flag = "" if ref is None or abs(ref - in_all) < 0.05 else "  <-- in-dist != saved (normalizer/seed?)"
        print(f"{rel[-58:]:58s} {in_all:8.2f} {('%.2f' % ref) if ref is not None else '-':>7s} "
              f"{in_sub:8.2f} {un_all:8.2f}{flag}")
        model.cpu()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=2)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
