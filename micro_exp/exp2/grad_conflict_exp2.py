"""
Gradient Conflict Analysis — Exp2 OOD (5% target data)
Runs all benchmarks, saves per-benchmark JSON, outputs averaged result.

Usage:
    python grad_conflict_exp2.py --gpu 0                    # all benchmarks
    python grad_conflict_exp2.py --gpu 0 --target 619.lbm_s # single (for parallel)
"""
import os, sys, json, glob, argparse, numpy as np, torch, torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, "/path/to/gem5/micro_exp/exp1")
from config import (
    GEM_INPUT_DIM, CONCORDE_HIDDEN_DIMS, DROPOUT,
    LEARNING_RATE, WEIGHT_DECAY, BATCH_SIZE,
    LAMBDA_CONST, LAMBDA_MONO,
    GEM_DATA_DIR, CONCORD_DATA_DIR, BENCHMARK_NAMES, SEED,
)
from model import GEMModel, loss_task, loss_const, loss_mono
from dataset import load_all_benchmarks, FeatureNormalizer, GEMDataset

OOD_RATIO = 0.05
OUT_DIR = "/path/to/gem5/micro_exp/exp2/results_grad_conflict"
EPOCHS = 50


def set_seed(s):
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    np.random.seed(s); torch.backends.cudnn.deterministic = True


def build_split(all_data, target):
    rng = np.random.RandomState(SEED)
    n = all_data[target]["n_samples"]
    perm = rng.permutation(n)
    split = max(1, int(n * OOD_RATIO))
    tr_idx = perm[:split]

    tg, tm, tp, tw = [], [], [], []
    for name, d in sorted(all_data.items()):
        if name == target:
            tg.append(d["gem_features"][tr_idx])
            tm.append(d["mscs_target"][tr_idx])
            tp.append(d["cpi_golden"][tr_idx])
            tw.append(d["w_min"][tr_idx])
        else:
            tg.append(d["gem_features"])
            tm.append(d["mscs_target"])
            tp.append(d["cpi_golden"])
            tw.append(d["w_min"])

    tg = np.concatenate(tg); tm = np.concatenate(tm)
    tp = np.concatenate(tp); tw = np.concatenate(tw)
    norm = FeatureNormalizer()
    tg = norm.fit_transform(tg)
    ds = GEMDataset(tg, tg, tm, tp, tw)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                      num_workers=4, pin_memory=True, drop_last=True), split


def get_grad_vec(loss, model):
    grads = torch.autograd.grad(loss, model.parameters(),
                                 retain_graph=True, allow_unused=True)
    return torch.cat([g.flatten() if g is not None
                      else torch.zeros(p.numel(), device=p.device)
                      for g, p in zip(grads, model.parameters())])


def run_one(all_data, target, dev):
    set_seed(SEED)
    loader, n_tr = build_split(all_data, target)
    model = GEMModel(GEM_INPUT_DIM, CONCORDE_HIDDEN_DIMS, DROPOUT).to(dev)
    opt = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    history = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        cos_tm, cos_tc, cos_cm = [], [], []
        norm_t, norm_c, norm_m = [], [], []

        for i, b in enumerate(loader):
            f = b["gem_features"].to(dev)
            mt, cg, wm = b["mscs_target"].to(dev), b["cpi_golden"].to(dev), b["w_min"].to(dev)
            mp = model(f)
            l_t = loss_task(mp, mt)
            l_c = loss_const(mp, wm, cg)
            l_m = loss_mono(mp)

            if i % 20 == 0:
                gt = get_grad_vec(l_t, model)
                gc = get_grad_vec(l_c, model)
                gm = get_grad_vec(l_m, model)
                cos_tm.append(torch.nn.functional.cosine_similarity(gt, gm, dim=0).item())
                cos_tc.append(torch.nn.functional.cosine_similarity(gt, gc, dim=0).item())
                cos_cm.append(torch.nn.functional.cosine_similarity(gc, gm, dim=0).item())
                norm_t.append(gt.norm().item())
                norm_c.append(gc.norm().item())
                norm_m.append(gm.norm().item())

            loss = l_t + LAMBDA_CONST * l_c + LAMBDA_MONO * l_m
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        sch.step()
        history.append({
            "epoch": epoch,
            "cos_task_mono": np.mean(cos_tm), "cos_task_const": np.mean(cos_tc),
            "cos_const_mono": np.mean(cos_cm),
            "norm_task": np.mean(norm_t), "norm_const": np.mean(norm_c),
            "norm_mono": np.mean(norm_m),
        })

    # Save per-benchmark
    out_path = os.path.join(OUT_DIR, f"grad_conflict_{target}.json")
    with open(out_path, "w") as f:
        json.dump({"target": target, "n_target_train": n_tr, "history": history}, f, indent=2)

    early = history[:10]; late = history[-10:]
    print(f"  {target}: early cos(t,m)={np.mean([r['cos_task_mono'] for r in early]):+.4f}  "
          f"late cos(t,m)={np.mean([r['cos_task_mono'] for r in late]):+.4f}  "
          f"||∇mono|| early={np.mean([r['norm_mono'] for r in early]):.6f} late={np.mean([r['norm_mono'] for r in late]):.6f}")
    return history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--target", type=str, default=None)
    parser.add_argument("--collect", action="store_true", help="Collect & average existing results")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    # Collect mode: just average existing per-benchmark results
    if args.collect:
        files = sorted(glob.glob(os.path.join(OUT_DIR, "grad_conflict_*.json")))
        if not files:
            print("No results found!"); return

        all_histories = []
        for f in files:
            d = json.load(open(f))
            all_histories.append(d["history"])
            print(f"  Loaded {d['target']}")

        # Average across benchmarks per epoch
        n_epochs = min(len(h) for h in all_histories)
        avg_history = []
        for ep in range(n_epochs):
            avg = {}
            for key in ["cos_task_mono", "cos_task_const", "cos_const_mono",
                        "norm_task", "norm_const", "norm_mono"]:
                avg[key] = np.mean([h[ep][key] for h in all_histories])
            avg["epoch"] = ep + 1
            avg_history.append(avg)

        with open(os.path.join(OUT_DIR, "grad_conflict_avg.json"), "w") as f:
            json.dump({"n_benchmarks": len(all_histories), "history": avg_history}, f, indent=2)

        early = avg_history[:10]; late = avg_history[-10:]
        print(f"\n--- Average across {len(all_histories)} benchmarks ---")
        print(f"Early: cos(task,mono)={np.mean([r['cos_task_mono'] for r in early]):+.4f}  "
              f"cos(task,const)={np.mean([r['cos_task_const'] for r in early]):+.4f}")
        print(f"Late:  cos(task,mono)={np.mean([r['cos_task_mono'] for r in late]):+.4f}  "
              f"cos(task,const)={np.mean([r['cos_task_const'] for r in late]):+.4f}")
        print(f"Saved to {OUT_DIR}/grad_conflict_avg.json")
        return

    # Training mode
    dev = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print("Loading...")
    all_data = load_all_benchmarks(GEM_DATA_DIR, CONCORD_DATA_DIR, BENCHMARK_NAMES)
    bms = sorted(all_data.keys())
    targets = [args.target] if args.target else bms

    for i, t in enumerate(targets):
        print(f"\n[{i+1}/{len(targets)}] {t}")
        run_one(all_data, t, dev)

    print("\nDone! Run with --collect to average.")


if __name__ == "__main__":
    main()