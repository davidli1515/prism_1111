"""
One trainer for Concorde, V1, V2, V3 and PRISM with identical settings.

Shared by every variant:
  * data split   : 18k row split (seed 42, 15:1) + labeled subset (stable_hash rule), exactly as
                   train_exp4_all.create_exp4_dataloaders / create_dataloaders_with_extra
  * normalizer   : per-feature z-score fit on the same rows for all variants
                   (--norm all: 18k train + 40k extra [default, = original PRISM]; --norm 18k: 18k train)
  * optimizer    : AdamW lr 1e-3, wd 0.3, grad clip 1.0, batch min(50000, n_train_rows)
  * schedule     : --steps optimizer steps for every variant (default 6084 = 1521 epochs of the full
                   18k train set); LR x0.5 at [42, 58, 75, 92]% of --steps
  * shuffling    : GPU randperm, reshuffled after each full pass (drop_last semantics)
  * model        : [512, 256] MLP, dropout from config
  * selection    : test CPI MAPE every --eval-every steps, best checkpoint kept (as in the paper)

Variants (only the training rows, inputs and loss differ):
  concorde  labeled rows      Concorde features (3873)  relative CPI error
  v1        labeled rows      PRISM features (4681)     lt*L_task
  v2        labeled rows      PRISM features            lt*L_task + lc*L_const + lm*L_mono
  v3        18k train rows    PRISM features            v2 on labeled + lu*(lc*L_const_self + lm*L_mono) on unlabeled
  prism     18k train + 40k   PRISM features            same loss as v3
Use --input concorde to give the MSCS variants the Concorde features only.

Usage:
  python train_aligned.py --variant concorde --ratio 0.05 --seed 42 --gpu 0 --out results_aligned
  python train_aligned.py --variant prism    --ratio 0.05 --seed 42 --gpu 0 --out results_aligned
"""
import os, sys, json, time, argparse
import numpy as np
import torch
import torch.optim as optim

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from config import (GEM_INPUT_DIM, CONCORDE_INPUT_DIM, DROPOUT, BENCHMARK_NAMES, SEED, TRAIN_RATIO)
from model import GEMModel, ConcordeModel, loss_task, loss_const, loss_mono
from dataset import load_all_benchmarks, FeatureNormalizer
from train_exp4_extra_unlabeled import (find_extra_indices, stable_hash, loss_const_self, DATASET_PATHS,
                                        LEARNING_RATE, WEIGHT_DECAY, BATCH_SIZE, LR_GAMMA, MILESTONE_FRACS)
from train_exp1_lambda_sens import evaluate as evaluate_mscs

HIDDEN = [512, 256]
VARIANTS = ["concorde", "v1", "v2", "v3", "prism"]


def fmt(x):
    return ("%g" % x).replace(".", "p")


def set_seed(s):
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    np.random.seed(s); torch.backends.cudnn.deterministic = True


def build_split(d18, d40, ratio, seed=SEED):
    """Same split/labeled subset as create_exp4_dataloaders and create_dataloaders_with_extra.
    Returns train/test arrays and a per-row source flag (0 = 18k labeled, 1 = 18k unlabeled, 2 = 40k extra)."""
    rng_split = np.random.RandomState(seed)
    tr, te, src = {k: [] for k in ("X", "M", "Y", "W")}, {k: [] for k in ("X", "M", "Y", "W")}, []
    keys = [("X", "gem_features"), ("M", "mscs_target"), ("Y", "cpi_golden"), ("W", "w_min")]
    for name in sorted(d18):
        d = d18[name]
        idx = rng_split.permutation(d["n_samples"])
        cut = int(d["n_samples"] * TRAIN_RATIO)
        tr_idx, te_idx = idx[:cut], idx[cut:]
        n_tr = len(tr_idx)
        n_lab = max(1, int(n_tr * ratio))
        perm = np.random.RandomState(seed + stable_hash(name) + int(round(ratio * 1000))).permutation(n_tr)
        s = np.ones(n_tr, dtype=np.int8); s[perm[:n_lab]] = 0
        for k, f in keys:
            tr[k].append(d[f][tr_idx]); te[k].append(d[f][te_idx])
        src.append(s)
        if d40 is not None:
            ex = find_extra_indices(d, d40[name])
            for k, f in keys:
                tr[k].append(d40[name][f][ex])
            src.append(np.full(len(ex), 2, dtype=np.int8))
    tr = {k: np.concatenate(v) for k, v in tr.items()}
    te = {k: np.concatenate(v) for k, v in te.items()}
    return tr, te, np.concatenate(src)


@torch.no_grad()
def evaluate_concorde(model, X, Y):
    model.eval()
    P = torch.cat([model(X[i:i + BATCH_SIZE]) for i in range(0, len(X), BATCH_SIZE)])
    return {"mape_mean4": ((P - Y).abs() / (Y.abs() + 1e-8)).mean().item() * 100}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=VARIANTS)
    ap.add_argument("--ratio", type=float, required=True)
    ap.add_argument("--lam-task", type=float, default=10.0)
    ap.add_argument("--lam-const", type=float, default=1.0)
    ap.add_argument("--lam-mono", type=float, default=0.0)
    ap.add_argument("--lam-unlab", type=float, default=1.0)
    ap.add_argument("--input", choices=["gem", "concorde"], default=None,
                    help="feature set (default: concorde for Concorde, gem for the MSCS variants)")
    ap.add_argument("--norm", choices=["all", "18k"], default="all")
    ap.add_argument("--steps", type=int, default=6084)
    ap.add_argument("--eval-every", type=int, default=4)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    v = args.variant
    inp = args.input or ("concorde" if v == "concorde" else "gem")
    in_dim = CONCORDE_INPUT_DIM if inp == "concorde" else GEM_INPUT_DIM

    os.makedirs(args.out, exist_ok=True)
    tag = (f"{v}_r{int(round(args.ratio * 100))}" +
           ("" if v == "concorde" else f"_t{fmt(args.lam_task)}_c{fmt(args.lam_const)}_m{fmt(args.lam_mono)}"
                                       f"_u{fmt(args.lam_unlab)}") +
           ("_nod3" if v != "concorde" and inp == "concorde" else "") +
           f"_n{args.norm}_T{args.steps}_s{args.seed}")
    out_json = os.path.join(args.out, f"{tag}.json")
    if os.path.exists(out_json):
        sys.exit(f"{out_json} exists, refusing to overwrite")
    dev = torch.device(f"cuda:{args.gpu}")

    # ---- data (identical split and normalizer for every variant) ----
    d18 = load_all_benchmarks(DATASET_PATHS["18k"]["gem"], DATASET_PATHS["18k"]["concord"], BENCHMARK_NAMES)
    d40 = (load_all_benchmarks(DATASET_PATHS["40k"]["gem"], DATASET_PATHS["40k"]["concord"], BENCHMARK_NAMES)
           if (args.norm == "all" or v == "prism") else None)
    tr, te, src = build_split(d18, d40, args.ratio)
    del d18, d40
    norm = FeatureNormalizer()
    norm.fit(tr["X"] if args.norm == "all" else tr["X"][src < 2])
    tr["X"], te["X"] = norm.transform(tr["X"]), norm.transform(te["X"])
    keep = {"concorde": src == 0, "v1": src == 0, "v2": src == 0, "v3": src < 2, "prism": src <= 2}[v]
    n_lab, n_unl18, n_extra = int((src == 0).sum()), int((src == 1).sum()), int((src == 2).sum())
    T = lambda a, sel=None: torch.from_numpy(np.ascontiguousarray(a if sel is None else a[sel])).float().to(dev)
    Xtr = T(tr["X"][keep][:, :in_dim]); Mtr, Ytr, Wtr = T(tr["M"], keep), T(tr["Y"], keep), T(tr["W"], keep)
    Ltr = torch.from_numpy(src[keep] == 0).to(dev)
    Xte = T(te["X"][:, :in_dim]); Mte, Yte, Wte = T(te["M"]), T(te["Y"]), T(te["W"])
    del tr, te

    # ---- model / optimizer / schedule (identical for every variant) ----
    set_seed(args.seed)
    model = (ConcordeModel(in_dim, HIDDEN, DROPOUT) if v == "concorde" else GEMModel(in_dim, HIDDEN, DROPOUT)).to(dev)
    opt = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    n = len(Xtr); bs = min(BATCH_SIZE, n); per_pass = n // bs if n > bs else 1
    milestones = [int(f * args.steps) for f in MILESTONE_FRACS]
    print(f"[{tag}] labeled={n_lab} unlab18k={n_unl18} extra40k={n_extra} train_rows={n} batch={bs} "
          f"steps={args.steps} (= {args.steps / per_pass:.0f} passes) milestones={milestones} in_dim={in_dim}",
          flush=True)

    def ev(full=False):
        return evaluate_concorde(model, Xte, Yte) if v == "concorde" else \
               evaluate_mscs(model, Xte, Wte, Yte, Mte, full=full)[0]

    best, best_state, best_step, hist = float("inf"), None, 0, []
    perm, pos, t0 = None, per_pass, time.time()
    for step in range(1, args.steps + 1):
        model.train()
        if pos == per_pass:
            perm, pos = torch.randperm(n, device=dev), 0
        b = perm[pos * bs:(pos + 1) * bs]; pos += 1
        if v == "concorde":
            y = Ytr[b]
            loss = ((model(Xtr[b]) - y).abs() / (y.abs() + 1e-8)).mean()
        else:
            mp = model(Xtr[b]); lab = Ltr[b]
            loss = torch.zeros((), device=dev)
            if lab.any():
                p = mp[lab]
                loss = loss + args.lam_task * loss_task(p, Mtr[b][lab])
                if v != "v1":
                    loss = loss + args.lam_const * loss_const(p, Wtr[b][lab], Ytr[b][lab])
                    if args.lam_mono > 0:
                        loss = loss + args.lam_mono * loss_mono(p)
            if v in ("v3", "prism") and (~lab).any():
                u = mp[~lab]
                ul = args.lam_const * loss_const_self(u, Wtr[b][~lab])
                if args.lam_mono > 0:
                    ul = ul + args.lam_mono * loss_mono(u)
                loss = loss + args.lam_unlab * ul
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        for pg in opt.param_groups:
            pg["lr"] = LEARNING_RATE * (LR_GAMMA ** sum(step >= m for m in milestones))
        if step % args.eval_every == 0 or step == args.steps:
            e = ev()
            hist.append({"step": step, "mape_mean4": e["mape_mean4"]})
            if e["mape_mean4"] < best:
                best, best_step = e["mape_mean4"], step
                best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
            if step % (100 * args.eval_every) == 0:
                print(f"[{tag}] step {step:6d} mape={e['mape_mean4']:.2f} best={best:.2f}@{best_step} "
                      f"| {time.time() - t0:.0f}s", flush=True)

    last = ev(full=True)
    model.load_state_dict(best_state)
    sel = ev(full=True)
    res = {"variant": v, "ratio": args.ratio, "input": inp, "norm": args.norm, "steps": args.steps,
           "eval_every": args.eval_every, "lam_task": args.lam_task, "lam_const": args.lam_const,
           "lam_mono": args.lam_mono, "lam_unlab": args.lam_unlab, "seed": args.seed,
           "n_labeled": n_lab, "n_unlabeled_18k": n_unl18, "n_extra_40k": n_extra, "n_train_rows": n,
           "batch": bs, "milestones": milestones, "best_step": best_step, "selected": sel,
           "last_step": last, "train_time_s": time.time() - t0, "history": hist}
    json.dump(res, open(out_json, "w"), indent=2)
    torch.save(best_state, os.path.join(args.out, f"{tag}.pt"))
    print(f"[{tag}] DONE mape={sel['mape_mean4']:.2f} @ step {best_step}", flush=True)


if __name__ == "__main__":
    main()
