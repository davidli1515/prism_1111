"""
Exp2 (v8, fixed per-sample unlabeled weight): identical to train_ood_lowlabel_extra.py
except, with --unlab-weight-ratio r, lambda_unlab is set per run so that one unlabeled
sample weighs r times one labeled sample under --norm subset:
    lambda_unlab = r * n_unlabeled / n_labeled      (counts over the whole training set)
r = 0.1 reproduces the source-20% regime (lambda_unlab ~= 1) at every source ratio.
The effective lambda_unlab is stored in the result JSON (lambdas.unlab).

Exp2 (v7, low-label OOD + 40k extra unlabeled = exp4 PRISM): with --extra40k, the
prism variant also gets the 40k rows not present in 18k as unlabeled data, for every
benchmark (sources and held-out target), as in exp4/train_exp4_extra_unlabeled.py,
EXCEPT rows of held-out 18k windows (target test windows, source val windows) and
rows whose window cannot be identified (window_ids/w40k label -2). Window labels come
from window_ids/make_window_ids_40k.py. Other variants are unchanged.

Exp2 (v6, low-label OOD): exp4 label-efficiency setting under leave-one-benchmark-out.
Sources keep only --src-ratio of their training rows labeled (exp4 sampling rule:
RandomState(seed + stable_hash(bm) + round(ratio*1000))); the rest are unlabeled.
The normalizer is fit on ALL training inputs (labeled + unlabeled, sources + target)
for every variant, as in exp4; non-prism variants then drop unlabeled rows.
BN fix and per-batch loss normalization as in train_ood_bnfix.py.

Exp2 (v5, BN fix): unlabeled rows do not touch BatchNorm statistics. Each batch is
split; labeled rows are forwarded in train mode (BN batch stats + running-stat
update), unlabeled rows in eval mode (BN uses running stats, no update; dropout off).
Loss weights per sample as in --norm batch. With --lam-unlab 0 unlabeled rows are
not forwarded at all.

Exp2 (v4, batch-normalized losses): labeled and unlabeled terms are each
sum-over-samples / batch_size instead of per-subset means, so a sample's weight
depends only on whether it is labeled, not on the labeled:unlabeled ratio in the
batch (--norm batch; --norm subset reproduces train_ood_lambda.py).

Exp2 (v3, lambda): same as train_ood.py, with loss weights as CLI args
(--lam-task/--lam-const/--lam-mono/--lam-unlab; defaults = new working point 10/1/0.01/1).

Exp2 (v2): OOD generalization — leave-one-benchmark-out, aligned with exp1/exp4 (recipe v77).

Setting (joint training, no foundation/finetune stages):
  * 23 benchmarks, 18k dataset only (~232k rows, same data as exp1/exp4).
    One target is held out; the other 22 are sources.
  * Sources:  18k rows split 15/16 train (fully labeled) / 1/16 val.
              The source val set is used for model selection — target labels are
              never used for selection (fixes test-set selection in the old exp2).
  * Split:    every benchmark is split 15/16 / 1/16 by TRACE WINDOW (default,
              --split window): a window's rows (same trace slice, different
              configs) are all on one side. Window ids: micro_exp/window_ids/.
              --split row reproduces the row-level split of exp1/exp4.
  * Target:   15/16 train pool / 1/16 test. The test part is fixed for every N / variant.
              --n-shot N labeled samples are the first N of the train pool (nested
              across N, identical across variants for a seed). Nothing from the
              test part is ever used for training, labeled or unlabeled.
  * Unlabeled (prism only): target train pool minus the shots, labels dropped.
              Sources are fully labeled, so they contribute no unlabeled rows.

Variants (same names as exp4 figures):
  concorde    Concorde features, scalar CPI, relative loss, labeled only
  vanilla     V1: GEM, L_task only, labeled only
  supervised  V2: GEM, task+const+mono, labeled only
  prism       V3/PRISM: supervised + unlabeled target traces (L_const_self +
              L_mono). The gap to `supervised` = value of unlabeled traces
              from the new workload.

Loss / optimizer identical to exp4 (train_exp4_all.py --ablation full, recipe v77):
  labeled:   0.1*L_task + 1.0*L_const + 0.01*L_mono
  unlabeled: 1.0*L_const_self + 0.01*L_mono
  AdamW lr 1e-3 wd 0.3, batch 50k, 1521 epochs, step LR x0.5 at [42,58,75,92]% of steps.

Data feeding: the whole (normalized) dataset is kept on the GPU and batches are
gathered with a GPU permutation each epoch (same batch size / drop_last semantics
as the torch DataLoader in exp4). --cpu-loader switches back to the exp4-style
DataLoader path for equivalence checks.

Usage:
    python train_ood.py --target 605.mcf_s --n-shot 0 --variant prism --gpu 0
    python train_ood.py --target snappy --n-shot 256 --variant concorde --seed 123
"""

import os, sys, time, json, hashlib, pickle, argparse
import numpy as np
import torch
import torch.optim as optim
from collections import defaultdict

sys.path.insert(0, "/scratch/gem5-TAO/micro_exp/exp1")
sys.path.insert(0, "/scratch/gem5-TAO/micro_exp/window_ids")

from config import (
    GEM_INPUT_DIM, CONCORDE_INPUT_DIM, DROPOUT, BENCHMARK_NAMES,
    SEED, TRAIN_RATIO, BASE_INDICES,
    CONCORDE_FEAT_END, D3_FEAT_END, MSCS_START, MSCS_END, WMIN_INDICES,
)
from model import (
    GEMModel, ConcordeModel, compute_cpi_mean,
    loss_task, loss_const, loss_mono,
)
from dataset import discover_benchmarks, FeatureNormalizer
from window_split import load_window_ids, split_by_window

# Recipe v77 (same as exp4)
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.3
BATCH_SIZE = 50000
NUM_EPOCHS = 1521
LR_GAMMA = 0.5
MILESTONE_FRACS = [0.42, 0.58, 0.75, 0.92]
LAMBDA_TASK = 10.0   # overridden by --lam-task
LAMBDA_CONST = 1.0
LAMBDA_MONO = 0.01
LAMBDA_UNLAB = 1.0

VARIANTS = ["concorde", "vanilla", "supervised", "prism"]

DATA_18K = "/scratch2/wl730/micro_traindata_packed/gem"
DATA_40K = "/scratch2/wl730/micro_traindata_packed_all/gem"
W40K_DIR = "/scratch/gem5-TAO/micro_exp/window_ids/w40k"


def extra40k_rows(name, raw18, heldout_windows):
    """Indices of 40k rows usable as extra unlabeled data for benchmark `name`:
    not in 18k (exact match on features), window label known (!= -2) and not a
    held-out 18k window. Returns (idx, n_extra_total, n_dropped_heldout, n_dropped_unknown)."""
    raw40 = open_bench(DATA_40K, name)
    lab40 = np.load(os.path.join(W40K_DIR, f"{name}.npy"))
    assert len(lab40) == raw40.shape[0], name
    in18 = {np.asarray(r[:D3_FEAT_END], dtype=np.float32).tobytes() for r in raw18}
    extra = np.array([i for i, r in enumerate(raw40)
                      if np.asarray(r[:D3_FEAT_END], dtype=np.float32).tobytes() not in in18],
                     dtype=np.int64)
    if len(extra) == 0:
        return extra, 0, 0, 0
    lab = lab40[extra]
    unknown = lab == -2
    held = np.isin(lab, np.fromiter(heldout_windows, dtype=np.int64)) & (lab >= 0)
    keep = ~unknown & ~held
    return extra[keep], len(extra), int(held.sum()), int(unknown.sum())


def set_seed(seed):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    np.random.seed(seed); torch.backends.cudnn.deterministic = True

def stable_hash(name):
    return int(hashlib.md5(name.encode()).hexdigest(), 16) % (2**31)

def step_lr(optimizer, global_step, milestones):
    n_passed = sum(1 for m in milestones if global_step >= m)
    for pg in optimizer.param_groups:
        pg["lr"] = LEARNING_RATE * (LR_GAMMA ** n_passed)

def compute_mape(pred, target, eps=1e-8):
    return (torch.abs(pred - target) / (torch.abs(target) + eps) * 100).mean().item()

def loss_const_self(mscs_pred, w_min, eps=1e-8):
    """Self-consistency: variance of CPI across 4 stages (no golden needed)"""
    base_vals = mscs_pred[:, BASE_INDICES]
    cpi_per_stage = 1.0 / (base_vals * w_min.unsqueeze(1) + eps)
    cpi_mean = cpi_per_stage.mean(dim=1, keepdim=True)
    return ((cpi_per_stage - cpi_mean) ** 2).mean()


# ============================================================
# Data loading (mmap, row subsets; same field definitions as exp1/dataset.py)
# ============================================================

def open_bench(data_dir, name):
    return np.load(os.path.join(data_dir, f"{name}.npy"), mmap_mode="r")

def take_rows(raw, idx, feat_end):
    """Extract fields for rows idx (sorted for sequential mmap reads)."""
    rows = np.asarray(raw[np.sort(idx)], dtype=np.float32)
    mscs = rows[:, MSCS_START:MSCS_END].copy()
    w_min = rows[:, WMIN_INDICES].min(axis=1)
    cpi = 1.0 / (mscs[:, BASE_INDICES[3]] * w_min + 1e-8)
    return {"x": rows[:, :feat_end].copy(), "mscs": mscs, "cpi": cpi, "wmin": w_min}

# ============================================================
# Split
# ============================================================

def build_ood_split(names, target, n_shot, variant, seed=SEED, split_mode="window", src_ratio=1.0,
                    extra40k=False):
    """Returns numpy dicts (train / src-val / tgt-test), fitted normalizer, counts."""
    feat_end = CONCORDE_FEAT_END if variant == "concorde" else D3_FEAT_END
    use_tgt_unlab = True          # always built; dropped for non-prism after normalizer fit

    train, val = defaultdict(list), defaultdict(list)
    test = None
    counts = {"src_labeled": 0, "src_unlabeled": 0, "tgt_shot": 0, "tgt_unlabeled": 0,
              "src_extra40k": 0, "tgt_extra40k": 0, "extra40k_dropped_heldout": 0,
              "extra40k_dropped_unknown": 0}
    use_extra = extra40k and variant == "prism"

    def add(store, d, label):
        for k, v in d.items():
            store[k].append(v)
        store["lab"].append(np.full(len(d["x"]), label, dtype=np.float32))

    for name in sorted(names):
        raw = open_bench(DATA_18K, name)
        n = raw.shape[0]
        rng = np.random.RandomState(seed + stable_hash(name))

        # 15/16 train / 1/16 held-out per benchmark, as exp1/exp4 (per-benchmark
        # seed, so a benchmark's split does not depend on which one is the target).
        # window: whole trace windows on one side (no window in both train and test)
        # row:    row-level split as in exp1/exp4
        if split_mode == "window":
            wid = load_window_ids(name, n)
            tr_idx, te_idx = split_by_window(wid, TRAIN_RATIO, rng)
        else:
            if extra40k:
                raise ValueError("--extra40k requires --split window")
            perm = rng.permutation(n)
            tr_idx, te_idx = perm[:int(n * TRAIN_RATIO)], perm[int(n * TRAIN_RATIO):]
        split = len(tr_idx)
        if use_extra:   # target: test windows; sources: val windows (both held out)
            ex_idx, n_ex, n_held, n_unk = extra40k_rows(name, raw, set(np.unique(wid[te_idx]).tolist()))
            if len(ex_idx) > 0:
                add(train, take_rows(open_bench(DATA_40K, name), ex_idx, feat_end), 0.0)
            counts["tgt_extra40k" if name == target else "src_extra40k"] += len(ex_idx)
            counts["extra40k_dropped_heldout"] += n_held
            counts["extra40k_dropped_unknown"] += n_unk

        if name == target:
            if n_shot > len(tr_idx):
                raise ValueError(f"n_shot={n_shot} > train pool {len(tr_idx)} for {name}")
            if n_shot > 0:
                add(train, take_rows(raw, tr_idx[:n_shot], feat_end), 1.0)  # nested across N
            counts["tgt_shot"] = n_shot
            test = take_rows(raw, te_idx, feat_end)    # never used for training
            if use_tgt_unlab:
                rest = tr_idx[n_shot:]                 # train pool minus shots, labels dropped
                if len(rest) > 0:
                    add(train, take_rows(raw, rest, feat_end), 0.0)
                counts["tgt_unlabeled"] = len(rest)
        else:
            n_lab = max(1, int(split * src_ratio))
            lperm = np.random.RandomState(
                seed + stable_hash(name) + int(round(src_ratio * 1000))).permutation(split)
            lab_idx, unl_idx = tr_idx[lperm[:n_lab]], tr_idx[lperm[n_lab:]]
            add(train, take_rows(raw, lab_idx, feat_end), 1.0)
            if len(unl_idx) > 0:
                add(train, take_rows(raw, unl_idx, feat_end), 0.0)
            add(val, take_rows(raw, te_idx, feat_end), 1.0)
            counts["src_labeled"] += n_lab
            counts["src_unlabeled"] += len(unl_idx)

    tr = {k: np.concatenate(v) for k, v in train.items()}
    va = {k: np.concatenate(v) for k, v in val.items()}
    test["lab"] = np.ones(len(test["x"]), dtype=np.float32)

    # Normalizer: fit on all training inputs (labeled + unlabeled; no labels used).
    # In-place transform == FeatureNormalizer.transform, without a second copy.
    norm = FeatureNormalizer()
    norm.fit(tr["x"])
    for d in (tr, va, test):
        d["x"] -= norm.mean
        d["x"] /= norm.std
    if variant != "prism":            # labeled-only variants: drop unlabeled rows
        keep = tr["lab"] > 0.5
        tr = {k: v[keep] for k, v in tr.items()}

    print(f"  Target: {target} | variant={variant} | n_shot={n_shot}")
    for k, v in counts.items():
        print(f"  {k:12s}: {v:,}")
    print(f"  Train: {len(tr['x']):,} | Src val: {len(va['x']):,} | Tgt test: {len(test['x']):,}")
    return tr, va, test, norm, counts


# ============================================================
# Batch iterators
# ============================================================

class GPUBatches:
    """Whole split resident on GPU; one randperm per epoch. Mirrors DataLoader's
    batch_size / shuffle / drop_last behaviour."""
    def __init__(self, arrays, device, batch_size, shuffle, drop_last):
        self.t = {k: torch.from_numpy(v).to(device) for k, v in arrays.items()}
        self.n = len(arrays["x"])
        self.bs, self.shuffle, self.drop_last = batch_size, shuffle, drop_last
        self.device = device

    def __len__(self):
        return self.n // self.bs if self.drop_last else -(-self.n // self.bs)

    def __iter__(self):
        idx = (torch.randperm(self.n, device=self.device) if self.shuffle
               else torch.arange(self.n, device=self.device))
        for i in range(len(self)):
            b = idx[i * self.bs:(i + 1) * self.bs]
            yield {k: v[b] for k, v in self.t.items()}


class _NumpyDS(torch.utils.data.Dataset):
    def __init__(self, arrays):
        self.t = {k: torch.from_numpy(v) for k, v in arrays.items()}
    def __len__(self): return self.t["x"].shape[0]
    def __getitem__(self, i): return {k: v[i] for k, v in self.t.items()}


def make_loaders(tr, va, test, device, cpu_loader, batch_size=BATCH_SIZE):
    actual_bs = min(batch_size, len(tr["x"]))
    drop = len(tr["x"]) > actual_bs
    if cpu_loader:   # exp4-style path, for equivalence checks
        mk = lambda a, bs, sh, dl: torch.utils.data.DataLoader(
            _NumpyDS(a), batch_size=bs, shuffle=sh, num_workers=4,
            pin_memory=True, drop_last=dl)
    else:
        mk = lambda a, bs, sh, dl: GPUBatches(a, device, bs, sh, dl)
    loaders = (mk(tr, actual_bs, True, drop),
               mk(va, batch_size, False, False),
               mk(test, batch_size, False, False))
    print(f"  Batch: {actual_bs:,} | steps/epoch: {len(loaders[0])} | "
          f"feeding: {'cpu DataLoader' if cpu_loader else 'gpu-resident'}")
    return loaders


# ============================================================
# Loss
# ============================================================

NORM = "batch"   # overridden by --norm


def gem_loss(mscs_pred, mscs_target, w_min, cpi_golden, has_label, variant):
    labeled_mask = has_label > 0.5
    unlabeled_mask = ~labeled_mask
    loss_dict = {}
    total = torch.tensor(0.0, device=mscs_pred.device)
    B = mscs_pred.shape[0]
    # subset-mean -> batch-mean: multiply each term by n_subset / B
    f_l = labeled_mask.sum().item() / B if NORM == "batch" else 1.0
    f_u = unlabeled_mask.sum().item() / B if NORM == "batch" else 1.0

    if labeled_mask.any():
        l_pred = mscs_pred[labeled_mask]
        l_task = loss_task(l_pred, mscs_target[labeled_mask])
        loss_dict["L_task"] = l_task.item()
        if variant == "vanilla":
            total = total + f_l * l_task    # same as exp4/train_exp4_vanilla.py
        else:
            l_const = loss_const(l_pred, w_min[labeled_mask], cpi_golden[labeled_mask])
            l_mono = loss_mono(l_pred)
            total = total + f_l * (LAMBDA_TASK * l_task + LAMBDA_CONST * l_const + LAMBDA_MONO * l_mono)
            loss_dict["L_const"] = l_const.item()

    if unlabeled_mask.any():
        u_pred = mscs_pred[unlabeled_mask]
        u_cself = loss_const_self(u_pred, w_min[unlabeled_mask])
        u_mono = loss_mono(u_pred)
        total = total + f_u * LAMBDA_UNLAB * (LAMBDA_CONST * u_cself + LAMBDA_MONO * u_mono)
        loss_dict["L_const_self"] = u_cself.item()

    loss_dict["L_total"] = total.item()
    return total, loss_dict


def concorde_loss(cpi_pred, cpi_golden, eps=1e-8):
    l = (torch.abs(cpi_pred - cpi_golden) / (torch.abs(cpi_golden) + eps)).mean()
    return l, {"L_rel": l.item()}


# ============================================================
# Eval / Train
# ============================================================

@torch.no_grad()
def predict(model, loader, device, is_concorde):
    model.eval()
    P, G, M = [], [], []
    for b in loader:
        x = b["x"].to(device, non_blocking=True)
        if is_concorde:
            P.append(model(x))
        else:
            m = model(x)
            P.append(compute_cpi_mean(m, b["wmin"].to(device)))
            M.append(m)
        G.append(b["cpi"].to(device))
    return torch.cat(P), torch.cat(G), (torch.cat(M) if M else None)


def bnfix_step(model, x, mscs, wmin, cpi, lab, variant):
    """Labeled rows: train-mode forward. Unlabeled rows: eval-mode forward (BN running
    stats, frozen; no dropout), gradients still flow.
    NORM == "batch":  each term scaled by n_subset / B (per-sample weight 1/B).
    NORM == "subset": each term is a mean over its own subset (exp4 formula)."""
    B = x.shape[0]
    lm = lab > 0.5
    um = ~lm
    total = torch.tensor(0.0, device=x.device)
    ld = {}
    if lm.any():
        mp = model(x[lm])
        f_l = lm.sum().item() / B if NORM == "batch" else 1.0
        l_task = loss_task(mp, mscs[lm])
        ld["L_task"] = l_task.item()
        if variant == "vanilla":
            total = total + f_l * l_task
        else:
            l_const = loss_const(mp, wmin[lm], cpi[lm])
            total = total + f_l * (LAMBDA_TASK * l_task + LAMBDA_CONST * l_const
                                   + LAMBDA_MONO * loss_mono(mp))
            ld["L_const"] = l_const.item()
    if um.any() and LAMBDA_UNLAB > 0:
        model.eval()
        up = model(x[um])
        model.train()
        f_u = um.sum().item() / B if NORM == "batch" else 1.0
        u_cself = loss_const_self(up, wmin[um])
        total = total + f_u * LAMBDA_UNLAB * (LAMBDA_CONST * u_cself + LAMBDA_MONO * loss_mono(up))
        ld["L_const_self"] = u_cself.item()
    ld["L_total"] = total.item()
    return total, ld


def train(model, train_loader, val_loader, test_loader, device, variant,
          num_epochs=NUM_EPOCHS):
    is_concorde = variant == "concorde"
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    total_steps = num_epochs * len(train_loader)
    milestones = [int(f * total_steps) for f in MILESTONE_FRACS]

    best_val, best_state, best_epoch = float("inf"), None, 0
    history = []
    global_step = 0
    t0 = time.time()

    print(f"\n{'='*60}")
    print(f"Training {variant} — {num_epochs} epochs, steps/epoch={len(train_loader)}")
    print(f"  Milestones={milestones}, gamma={LR_GAMMA}")
    print(f"  Model selection: source val MAPE (target test is report-only)")
    print(f"{'='*60}")

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_losses = defaultdict(float); nb = 0
        for b in train_loader:
            x = b["x"].to(device, non_blocking=True)
            cpi_gold = b["cpi"].to(device, non_blocking=True)
            if is_concorde:
                loss, ld = concorde_loss(model(x), cpi_gold)
            else:
                loss, ld = bnfix_step(model, x, b["mscs"].to(device, non_blocking=True),
                                      b["wmin"].to(device, non_blocking=True), cpi_gold,
                                      b["lab"].to(device, non_blocking=True), variant)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            global_step += 1
            step_lr(optimizer, global_step, milestones)
            for k, v in ld.items(): epoch_losses[k] += v
            nb += 1

        vp, vg, _ = predict(model, val_loader, device, is_concorde)
        tp, tg, _ = predict(model, test_loader, device, is_concorde)
        val_mape, test_mape = compute_mape(vp, vg), compute_mape(tp, tg)
        history.append({"epoch": epoch, "val_mape": val_mape, "test_mape": test_mape,
                        **{k: v / nb for k, v in epoch_losses.items()}})

        if val_mape < best_val:
            best_val, best_epoch = val_mape, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            avg = " ".join(f"{k}={v/nb:.5f}" for k, v in epoch_losses.items())
            print(f"  Ep {epoch:4d} | {avg} | SrcVal={val_mape:.2f}% "
                  f"TgtTest={test_mape:.2f}% | lr={optimizer.param_groups[0]['lr']:.2e} "
                  f"| {time.time()-t0:.0f}s", flush=True)

    last_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    print(f"  Best src-val {best_val:.2f}% @ ep {best_epoch} ({time.time()-t0:.0f}s)")
    return model, last_state, best_epoch, history, time.time() - t0


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, default=None)
    parser.add_argument("--n-shot", type=int, default=0,
                        help="Labeled target samples (0 = zero-shot)")
    parser.add_argument("--variant", type=str, default=None, choices=VARIANTS)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--hidden", type=int, nargs="+", default=[512, 256])
    parser.add_argument("--unlab-weight-ratio", type=float, default=None,
                        help="prism: set lambda_unlab = r * n_unlabeled / n_labeled (per-sample weight ratio r)")
    parser.add_argument("--extra40k", action="store_true",
                        help="prism only: add 40k extra rows (minus held-out windows) as unlabeled")
    parser.add_argument("--src-ratio", type=float, default=1.0,
                        help="Fraction of each source benchmark's training rows that are labeled")
    parser.add_argument("--split", type=str, default="window", choices=["window", "row"],
                        help="15:1 split by trace window (default) or by row (exp1/exp4 style)")
    parser.add_argument("--cpu-loader", action="store_true",
                        help="Use exp4-style CPU DataLoader instead of GPU-resident data")
    parser.add_argument("--results-dir", type=str,
                        default="/scratch/gem5-TAO/micro_exp/exp2/results_ood_lowlabel_grid_wfix_1010")
    parser.add_argument("--lam-task", type=float, default=10.0)
    parser.add_argument("--lam-const", type=float, default=1.0)
    parser.add_argument("--lam-mono", type=float, default=0.01)
    parser.add_argument("--lam-unlab", type=float, default=1.0)
    parser.add_argument("--norm", type=str, default="batch", choices=["batch", "subset"])
    args = parser.parse_args()
    global LAMBDA_TASK, LAMBDA_CONST, LAMBDA_MONO, LAMBDA_UNLAB, NORM
    NORM = args.norm
    LAMBDA_TASK, LAMBDA_CONST = args.lam_task, args.lam_const
    LAMBDA_MONO, LAMBDA_UNLAB = args.lam_mono, args.lam_unlab

    names = BENCHMARK_NAMES or discover_benchmarks(DATA_18K)
    if args.target is None or args.variant is None:
        parser.error("--target and --variant are required")
    if args.target not in names:
        raise ValueError(f"Unknown target {args.target}; choices: {names}")

    out_dir = os.path.join(args.results_dir, args.variant, args.target)
    tag = f"n{args.n_shot}_s{args.seed}" + ("" if args.split == "window" else "_row")
    os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(os.path.join(out_dir, f"{tag}.json")):
        sys.exit(f"Refusing to overwrite existing result {out_dir}/{tag}.json")

    set_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Recipe: v77 | Seed: {args.seed} | Out: {out_dir}/{tag}")

    t_load = time.time()
    tr, va, test, norm, counts = build_ood_split(
        names, args.target, args.n_shot, args.variant, seed=args.seed,
        split_mode=args.split, src_ratio=args.src_ratio, extra40k=args.extra40k)
    if args.unlab_weight_ratio is not None and args.variant == "prism":
        if NORM != "subset":
            raise ValueError("--unlab-weight-ratio assumes --norm subset")
        n_lab = int((tr["lab"] > 0.5).sum()); n_unl = int((tr["lab"] <= 0.5).sum())
        LAMBDA_UNLAB = args.unlab_weight_ratio * n_unl / max(n_lab, 1)
        print(f"  lambda_unlab set to {LAMBDA_UNLAB:.4f} (r={args.unlab_weight_ratio}, "
              f"n_unlabeled={n_unl:,}, n_labeled={n_lab:,})")
    train_loader, val_loader, test_loader = make_loaders(
        tr, va, test, device, args.cpu_loader)
    test_mscs = torch.from_numpy(test["mscs"])
    if not args.cpu_loader:
        del tr, va, test        # GPU copies hold the data now
    print(f"  Data ready in {time.time()-t_load:.0f}s")

    set_seed(args.seed)
    is_concorde = args.variant == "concorde"
    if is_concorde:
        model = ConcordeModel(CONCORDE_INPUT_DIM, args.hidden, DROPOUT).to(device)
    else:
        model = GEMModel(GEM_INPUT_DIM, args.hidden, DROPOUT).to(device)
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    model, last_state, best_epoch, history, train_time = train(
        model, train_loader, val_loader, test_loader, device, args.variant,
        num_epochs=args.epochs)

    # Final metrics: selected (best src-val) model; last-epoch and oracle for reference
    tp, tg, tm = predict(model, test_loader, device, is_concorde)
    err = (torch.abs(tp - tg) / (torch.abs(tg) + 1e-8) * 100).cpu().numpy()
    oracle = min(history, key=lambda h: h["test_mape"])
    result = {
        "experiment": "exp2_ood_v2",
        "recipe": "v77",
        "lambdas": {"task": LAMBDA_TASK, "const": LAMBDA_CONST,
                    "mono": LAMBDA_MONO, "unlab": LAMBDA_UNLAB},
        "loss_norm": NORM,
        "src_ratio": args.src_ratio,
        "extra40k": args.extra40k,
        "unlab_weight_ratio": args.unlab_weight_ratio,
        "bn_fix": "unlabeled rows eval-mode BN (no stat update)",
        "target": args.target,
        "variant": args.variant,
        "n_shot": args.n_shot,
        "seed": args.seed,
        "hidden": args.hidden,
        "split": args.split,
        "feeding": "cpu_loader" if args.cpu_loader else "gpu_resident",
        "counts": counts,
        "target_mape": float(err.mean()),          # headline number
        "target_p90": float(np.percentile(err, 90)),
        "target_mape_last_epoch": history[-1]["test_mape"],
        "target_mape_oracle": oracle["test_mape"],  # test-selected; do NOT report as main
        "oracle_epoch": oracle["epoch"],
        "best_epoch": best_epoch,
        "src_val_mape": history[best_epoch - 1]["val_mape"],
        "train_time_seconds": train_time,
        "history": history,
    }
    if tm is not None:
        result["per_dim_mse"] = ((tm.cpu() - test_mscs) ** 2).mean(dim=0).tolist()

    with open(os.path.join(out_dir, f"{tag}.json"), "w") as f:
        json.dump(result, f, indent=2)
    torch.save(model.state_dict(), os.path.join(out_dir, f"{tag}.pt"))
    torch.save(last_state, os.path.join(out_dir, f"{tag}_last.pt"))
    with open(os.path.join(out_dir, f"{tag}_normalizer.pkl"), "wb") as f:
        pickle.dump({"mean": norm.mean, "std": norm.std}, f)

    print(f"\n  [{args.variant}|{args.target}|N={args.n_shot}] "
          f"target MAPE={result['target_mape']:.2f}% "
          f"(last={result['target_mape_last_epoch']:.2f}%, oracle={result['target_mape_oracle']:.2f}%)")
    print(f"  Saved {out_dir}/{tag}.json")


if __name__ == "__main__":
    main()
