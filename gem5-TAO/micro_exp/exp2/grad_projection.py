"""
Gradient Projection Decomposition (PCGrad-style analysis)

Decomposes ∇L_mono into:
  - Component parallel to ∇L_task (redundant with task signal)
  - Component orthogonal to ∇L_task (unique physical signal)

Reports the orthogonal ratio: ||∇L_mono_⊥|| / ||∇L_mono||
  → close to 1: L_mono provides mostly NEW information
  → close to 0: L_mono is redundant with L_task

Runs both ID (exp1) and OOD (exp2, 5% per benchmark) settings.

Usage:
    python grad_projection.py --gpu 0 --mode id
    python grad_projection.py --gpu 0 --mode ood --target 619.lbm_s
    python grad_projection.py --gpu 0 --mode ood   # all benchmarks
"""
import os, sys, json, glob, argparse, numpy as np, torch, torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, "/scratch/gem5-TAO/micro_exp/exp1")
from config import (
    GEM_INPUT_DIM, CONCORDE_HIDDEN_DIMS, DROPOUT,
    LEARNING_RATE, WEIGHT_DECAY, BATCH_SIZE, NUM_EPOCHS,
    LAMBDA_CONST, LAMBDA_MONO,
    GEM_DATA_DIR, CONCORD_DATA_DIR, BENCHMARK_NAMES, SEED,
)
from model import GEMModel, loss_task, loss_const, loss_mono
from dataset import load_all_benchmarks, create_exp1_dataloaders, FeatureNormalizer, GEMDataset

OUT_DIR = "/scratch/gem5-TAO/micro_exp/exp2/results_grad_projection"
OOD_RATIO = 0.05
EPOCHS = 50


def set_seed(s):
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    np.random.seed(s); torch.backends.cudnn.deterministic = True


def get_grad_vec(loss, model):
    grads = torch.autograd.grad(loss, model.parameters(),
                                 retain_graph=True, allow_unused=True)
    return torch.cat([g.flatten() if g is not None
                      else torch.zeros(p.numel(), device=p.device)
                      for g, p in zip(grads, model.parameters())])


def project(a, b, eps=1e-10):
    """Project vector a onto vector b. Returns (parallel, orthogonal)."""
    b_norm_sq = torch.dot(b, b) + eps
    coeff = torch.dot(a, b) / b_norm_sq
    parallel = coeff * b
    orthogonal = a - parallel
    return parallel, orthogonal, coeff.item()


def build_ood_split(all_data, target):
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
                      num_workers=4, pin_memory=True, drop_last=True)


def run_analysis(loader, device, label, epochs=EPOCHS):
    """Train model, measure gradient projection at each epoch."""
    model = GEMModel(GEM_INPUT_DIM, CONCORDE_HIDDEN_DIMS, DROPOUT).to(device)
    opt = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        ortho_ratios_mono = []   # ||∇L_mono_⊥|| / ||∇L_mono||
        ortho_ratios_const = []  # ||∇L_const_⊥|| / ||∇L_const||  (for comparison)
        proj_coeffs = []         # sign of projection coefficient

        for i, b in enumerate(loader):
            f = b["gem_features"].to(device)
            mt, cg, wm = b["mscs_target"].to(device), b["cpi_golden"].to(device), b["w_min"].to(device)
            mp = model(f)
            l_t = loss_task(mp, mt)
            l_c = loss_const(mp, wm, cg)
            l_m = loss_mono(mp)

            if i % 20 == 0:
                g_task = get_grad_vec(l_t, model)
                g_mono = get_grad_vec(l_m, model)
                g_const = get_grad_vec(l_c, model)

                # Project L_mono onto L_task
                _, mono_orth, coeff = project(g_mono, g_task)
                mono_norm = g_mono.norm().item()
                if mono_norm > 1e-10:
                    ortho_ratios_mono.append(mono_orth.norm().item() / mono_norm)
                proj_coeffs.append(coeff)

                # Project L_const onto L_task (for comparison)
                _, const_orth, _ = project(g_const, g_task)
                const_norm = g_const.norm().item()
                if const_norm > 1e-10:
                    ortho_ratios_const.append(const_orth.norm().item() / const_norm)

            # Normal training
            loss = l_t + LAMBDA_CONST * l_c + LAMBDA_MONO * l_m
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        sch.step()

        rec = {
            "epoch": epoch,
            "ortho_ratio_mono": np.mean(ortho_ratios_mono),
            "ortho_ratio_const": np.mean(ortho_ratios_const),
            "proj_coeff_mono": np.mean(proj_coeffs),  # negative = conflicting
        }
        history.append(rec)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  [{label}] Ep {epoch:3d} | "
                  f"ortho_mono={rec['ortho_ratio_mono']:.4f}  "
                  f"ortho_const={rec['ortho_ratio_const']:.4f}  "
                  f"proj_coeff={rec['proj_coeff_mono']:+.4f}")

    return history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--mode", choices=["id", "ood"], default=None)
    parser.add_argument("--target", type=str, default=None)
    parser.add_argument("--collect", action="store_true")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    # Collect mode
    if args.collect:
        files = sorted(glob.glob(os.path.join(OUT_DIR, "proj_ood_*.json")))
        all_h = [json.load(open(f))["history"] for f in files]
        n_ep = min(len(h) for h in all_h)
        avg = []
        for ep in range(n_ep):
            avg.append({
                "epoch": ep + 1,
                "ortho_ratio_mono": np.mean([h[ep]["ortho_ratio_mono"] for h in all_h]),
                "ortho_ratio_const": np.mean([h[ep]["ortho_ratio_const"] for h in all_h]),
                "proj_coeff_mono": np.mean([h[ep]["proj_coeff_mono"] for h in all_h]),
            })
        with open(os.path.join(OUT_DIR, "proj_ood_avg.json"), "w") as f:
            json.dump({"n_benchmarks": len(all_h), "history": avg}, f, indent=2)

        early = avg[:10]; late = avg[-10:]
        print(f"Averaged {len(all_h)} benchmarks:")
        print(f"  Early: ortho_mono={np.mean([r['ortho_ratio_mono'] for r in early]):.4f}  "
              f"ortho_const={np.mean([r['ortho_ratio_const'] for r in early]):.4f}")
        print(f"  Late:  ortho_mono={np.mean([r['ortho_ratio_mono'] for r in late]):.4f}  "
              f"ortho_const={np.mean([r['ortho_ratio_const'] for r in late]):.4f}")
        return

    if not args.mode:
        print("Error: --mode required when not using --collect")
        return

    dev = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    set_seed(SEED)

    print("Loading...")
    all_data = load_all_benchmarks(GEM_DATA_DIR, CONCORD_DATA_DIR, BENCHMARK_NAMES)

    if args.mode == "id":
        (tl, _, _, _, _) = create_exp1_dataloaders(all_data)
        history = run_analysis(tl, dev, "ID")
        with open(os.path.join(OUT_DIR, "proj_id.json"), "w") as f:
            json.dump({"mode": "id", "history": history}, f, indent=2)

        early = history[:10]; late = history[-10:]
        print(f"\n--- ID Summary ---")
        print(f"Early: ortho_mono={np.mean([r['ortho_ratio_mono'] for r in early]):.4f}  "
              f"proj_coeff={np.mean([r['proj_coeff_mono'] for r in early]):+.4f}")
        print(f"Late:  ortho_mono={np.mean([r['ortho_ratio_mono'] for r in late]):.4f}  "
              f"proj_coeff={np.mean([r['proj_coeff_mono'] for r in late]):+.4f}")

    elif args.mode == "ood":
        bms = sorted(all_data.keys())
        targets = [args.target] if args.target else bms

        for i, t in enumerate(targets):
            print(f"\n[{i+1}/{len(targets)}] {t}")
            set_seed(SEED)
            loader = build_ood_split(all_data, t)
            history = run_analysis(loader, dev, t)
            with open(os.path.join(OUT_DIR, f"proj_ood_{t}.json"), "w") as f:
                json.dump({"target": t, "history": history}, f, indent=2)

        if not args.target:
            print("\nRun with --collect to average.")


if __name__ == "__main__":
    main()