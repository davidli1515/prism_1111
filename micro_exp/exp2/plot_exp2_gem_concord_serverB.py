"""
Plot GEM vs Concorde vs GEM-Mix vs GEM-Vanilla across shot sizes.

Usage:
    python plot_exp2_gem_concord.py
    python plot_exp2_gem_concord.py --nomix
    python plot_exp2_gem_concord.py --nomix --novanilla
"""

import os
import json
import glob
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_avg_mape(results_dir, key):
    files = sorted(glob.glob(os.path.join(results_dir, "exp2_ood_*.json")))
    if not files:
        return None, 0
    vals = []
    for f in files:
        with open(f) as fp:
            r = json.load(fp)
        if key in r:
            vals.append(r[key])
    if not vals:
        return None, 0
    return np.mean(vals), len(vals)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str,
                        default="/path/to/gem5/micro_exp/exp2")
    parser.add_argument("--shots", nargs="+", type=int,
                        default=[32, 256, 512, 1024, 2048])
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--save", type=str, default="exp2_comparison.png")
    parser.add_argument("--nomix", action="store_true")
    parser.add_argument("--novanilla", action="store_true")
    args = parser.parse_args()

    shots = args.shots
    gem_avgs, con_avgs, mix_avgs, van_avgs = [], [], [], []
    valid_shots = []
    use_mix = not args.nomix
    use_vanilla = not args.novanilla

    header = f"{'Shots':<8} {'GEM':>10} {'Concorde':>10}"
    if use_mix:     header += f" {'GEM-Mix':>10}"
    if use_vanilla: header += f" {'Vanilla':>10}"
    header += f" {'#BMs':>6}"
    print(header)
    print("-" * len(header))

    for n in shots:
        tag = f"_{args.tag}" if args.tag else ""
        sup_dir = os.path.join(args.base_dir, f"results_{n}shot_ft{tag}")
        mix_dir = os.path.join(args.base_dir, f"results_{n}shot_mix_ft{tag}")
        van_dir = os.path.join(args.base_dir, f"results_{n}shot_vanilla_ft{tag}")

        g, ng = load_avg_mape(sup_dir, "gem_full")
        c, nc = load_avg_mape(sup_dir, "concorde")

        if g is None or c is None:
            print(f"{n:<8} {'MISSING':>10} {'MISSING':>10}")
            continue

        m_val, v_val = None, None
        if use_mix:
            m_val, _ = load_avg_mape(mix_dir, "gem_mix_ft")
        if use_vanilla:
            v_val, _ = load_avg_mape(van_dir, "gem_vanilla")

        valid_shots.append(n)
        gem_avgs.append(g); con_avgs.append(c)
        mix_avgs.append(m_val); van_avgs.append(v_val)

        line = f"{n:<8} {g:>9.2f}% {c:>9.2f}%"
        if use_mix:     line += f" {m_val:>9.2f}%" if m_val is not None else f" {'N/A':>10}"
        if use_vanilla: line += f" {v_val:>9.2f}%" if v_val is not None else f" {'N/A':>10}"
        line += f" {min(ng, nc):>6}"
        print(line)

    if not valid_shots:
        print("No data found!"); return

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(valid_shots))

    ax.plot(x, gem_avgs, 'o-', color="#2196F3", linewidth=2, markersize=8, label="GEM (full loss)")
    ax.plot(x, con_avgs, 's-', color="#4CAF50", linewidth=2, markersize=8, label="Concorde")
    if use_mix and any(m is not None for m in mix_avgs):
        ax.plot(x, [m if m is not None else np.nan for m in mix_avgs],
                'D-', color="#FF9800", linewidth=2, markersize=8, label="GEM (semi-supervised)")
    if use_vanilla and any(v is not None for v in van_avgs):
        ax.plot(x, [v if v is not None else np.nan for v in van_avgs],
                '^-', color="#9C27B0", linewidth=2, markersize=8, label="GEM (L_task only)")

    # Annotate
    for i in range(len(valid_shots)):
        all_v = [gem_avgs[i], con_avgs[i]]
        if use_mix and mix_avgs[i] is not None: all_v.append(mix_avgs[i])
        if use_vanilla and van_avgs[i] is not None: all_v.append(van_avgs[i])
        ax.annotate(f"{gem_avgs[i]:.1f}", (i, gem_avgs[i]), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=7, color="#2196F3")
        ax.annotate(f"{con_avgs[i]:.1f}", (i, con_avgs[i]), textcoords="offset points",
                    xytext=(0, -12), ha="center", fontsize=7, color="#4CAF50")
        if use_mix and mix_avgs[i] is not None:
            ax.annotate(f"{mix_avgs[i]:.1f}", (i, mix_avgs[i]), textcoords="offset points",
                        xytext=(15, 0), ha="left", fontsize=7, color="#FF9800")
        if use_vanilla and van_avgs[i] is not None:
            ax.annotate(f"{van_avgs[i]:.1f}", (i, van_avgs[i]), textcoords="offset points",
                        xytext=(-15, 0), ha="right", fontsize=7, color="#9C27B0")

    ax.set_xlabel("Number of Labeled Samples", fontsize=12)
    ax.set_ylabel("Average MAPE (%)", fontsize=12)
    ax.set_title("OOD Generalization: Few-Shot Fine-Tuning", fontsize=13)
    ax.set_xticks(x); ax.set_xticklabels([str(s) for s in valid_shots])
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(args.save, dpi=200, bbox_inches="tight"); plt.close()
    print(f"\nSaved: {args.save}")

    if use_mix and any(m is not None for m in mix_avgs):
        plot_mix_delta_bars(args, valid_shots)


def load_per_benchmark_deltas(sup_dir, mix_dir):
    sup_files = sorted(glob.glob(os.path.join(sup_dir, "exp2_ood_*.json")))
    mix_files = sorted(glob.glob(os.path.join(mix_dir, "exp2_ood_*.json")))
    sup_map, mix_map = {}, {}
    for f in sup_files:
        with open(f) as fp: r = json.load(fp)
        if "gem_full" in r: sup_map[r["target"]] = r["gem_full"]
    for f in mix_files:
        with open(f) as fp: r = json.load(fp)
        if "gem_mix_ft" in r: mix_map[r["target"]] = r["gem_mix_ft"]
    return [sup_map[n] - mix_map[n] for n in sorted(sup_map) if n in mix_map]


def plot_mix_delta_bars(args, valid_shots):
    improved_avgs, degraded_avgs = [], []
    improved_counts, degraded_counts = [], []
    print(f"\n{'Shots':<8} {'Improved':>10} {'Avg+':>8} {'Degraded':>10} {'Avg-':>8}")
    print("-" * 50)
    for n in valid_shots:
        tag = f"_{args.tag}" if args.tag else ""
        sup_dir = os.path.join(args.base_dir, f"results_{n}shot_ft{tag}")
        mix_dir = os.path.join(args.base_dir, f"results_{n}shot_mix_ft{tag}")
        deltas = load_per_benchmark_deltas(sup_dir, mix_dir)
        if not deltas:
            improved_avgs.append(0); degraded_avgs.append(0)
            improved_counts.append(0); degraded_counts.append(0); continue
        improved = [d for d in deltas if d > 0]
        degraded = [d for d in deltas if d < 0]
        avg_imp = np.mean(improved) if improved else 0
        avg_deg = np.mean(degraded) if degraded else 0
        improved_avgs.append(avg_imp); degraded_avgs.append(avg_deg)
        improved_counts.append(len(improved)); degraded_counts.append(len(degraded))
        print(f"{n:<8} {len(improved):>6}/{len(deltas)}  {avg_imp:>+7.2f}% "
              f"{len(degraded):>6}/{len(deltas)}  {avg_deg:>+7.2f}%")

    x = np.arange(len(valid_shots)); w = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w/2, improved_avgs, w, label="Improved (Mix better)", color="#4CAF50", alpha=0.85)
    ax.bar(x + w/2, degraded_avgs, w, label="Degraded (Mix worse)", color="#F44336", alpha=0.85)
    for i in range(len(valid_shots)):
        if improved_avgs[i] > 0:
            ax.annotate(f"{improved_counts[i]}bm\n+{improved_avgs[i]:.2f}%",
                        (i - w/2, improved_avgs[i]), textcoords="offset points",
                        xytext=(0, 5), ha="center", fontsize=7, color="#2E7D32")
        if degraded_avgs[i] < 0:
            ax.annotate(f"{degraded_counts[i]}bm\n{degraded_avgs[i]:.2f}%",
                        (i + w/2, degraded_avgs[i]), textcoords="offset points",
                        xytext=(0, -12), ha="center", fontsize=7, color="#C62828")
    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.set_xlabel("Number of Labeled Samples", fontsize=12)
    ax.set_ylabel("Average MAPE Change (%)", fontsize=12)
    ax.set_title("GEM-Mix vs GEM Supervised: Per-Benchmark Impact", fontsize=13)
    ax.set_xticks(x); ax.set_xticklabels([str(s) for s in valid_shots])
    ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    delta_save = args.save.replace(".png", "_mix_delta.png")
    fig.savefig(delta_save, dpi=200, bbox_inches="tight"); plt.close()
    print(f"\nSaved: {delta_save}")


if __name__ == "__main__":
    main()