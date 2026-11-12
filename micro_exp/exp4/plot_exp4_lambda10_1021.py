"""
plot_exp4_lambda10_1021.py — Label-efficiency figure (Fig 1 style of plot_exp4_final_717.py)
for lam_task=10, lam_const=1, lam_mono=0. Reads only the JSON written by
collect_exp4_lambda10_1021.py.

Curves: Concorde, V1 (seed 42, results_run2) and V2 / V3 / PRISM at lam_task=10 (3-seed mean,
error bar = std). Reference lines: PRISM and Concorde at 100% labels.

Usage:
    python plot_exp4_lambda10_1021.py [--json results_lambda10_summary_1021.json]
        [--out figures_final/fig1_main_results_lambda10_1021.png] [--skip 10 25]
"""
import os, json, argparse
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 10, 'axes.labelsize': 11, 'axes.titlesize': 12,
    'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 12,
    'figure.dpi': 600, 'axes.linewidth': 1.0,
    'xtick.major.width': 1.0, 'ytick.major.width': 1.0,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.major.size': 4, 'ytick.major.size': 4,
})

KEYS = ["Concorde", "V1", "V2", "V3", "PRISM"]
STYLES = {
    "Concorde": {"color": "#2E7D32", "marker": "s", "lw": 2.0, "ms": 6},
    "V1":       {"color": "#E91E63", "marker": "^", "lw": 2.0, "ms": 6},
    "V2":       {"color": "#455A64", "marker": "D", "lw": 2.0, "ms": 6},
    "V3":       {"color": "#1565C0", "marker": "o", "lw": 2.0, "ms": 6},
    "PRISM":    {"color": "#FF6F00", "marker": "p", "lw": 2.5, "ms": 7},
}
LABELS = {
    "Concorde": "Concorde",
    "V1":       "V1: +MSCS",
    "V2":       "V2: +Physics",
    "V3":       "V3: +Semi-Supervised",
    "PRISM":    "PRISM (+Extra Unlabeled)",
}
OFFSET = {"Concorde": +1, "V1": +1, "V2": -1, "V3": +1, "PRISM": -1}
CUSTOM = {("Concorde", 30): (0, -10, "top"), ("Concorde", 40): (0, -10, "top"),
          ("Concorde", 60): (0, -6, "top"), ("Concorde", 90): (12, 8, "bottom")}
LABEL_KEYS = {"PRISM", "Concorde"}
REF = {"PRISM": {"color": "#FF6F00", "ls": "--"}, "Concorde": {"color": "#2E7D32", "ls": ":"}}


def series(var, skip, max_pct=None):
    pts = sorted((int(p), r["cpi_mape"]["mean"], r["cpi_mape"]["std"], r["n_seeds"])
                 for p, r in var["ratios"].items() if int(p) not in skip)
    return [x for x in pts if max_pct is None or x[0] <= max_pct]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=os.path.join(HERE, "results_lambda10_summary_1021.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "figures_final/fig1_main_results_lambda10_1021.png"))
    ap.add_argument("--skip", type=int, nargs="*", default=[10, 25], help="label %% to omit (Fig 1 omits 10, 25)")
    ap.add_argument("--title", default="Label Efficiency: Incremental Ablation")
    args = ap.parse_args()
    V = json.load(open(args.json))["variants"]
    skip = set(args.skip)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for t in [3.0, 4.0, 5.0]:
        ax.axhline(y=t, color="#BDBDBD", linestyle="--", linewidth=1.2, alpha=0.6, zorder=0)
        ax.text(4, t + 0.1, f"{t:.0f}%", fontsize=8, color="#757575", va="bottom", ha="left")

    all_x = set()
    for k in KEYS:
        pts = series(V[k], skip, max_pct=90)           # curves stop at 90%, 100% is a reference line
        x, y, e, n = zip(*pts)
        all_x |= set(x)
        s = STYLES[k]
        ax.errorbar(x, y, yerr=[v if m > 1 else 0 for v, m in zip(e, n)], color=s["color"],
                    marker=s["marker"], linewidth=s["lw"], markersize=s["ms"], capsize=2.5,
                    elinewidth=1.0, label=LABELS[k], zorder=3)
        if k in LABEL_KEYS:
            for xi, yi in zip(x, y):
                dx, dy, va = CUSTOM.get((k, xi), (0, 8 * OFFSET[k], "bottom" if OFFSET[k] > 0 else "top"))
                ax.annotate(f"{yi:.1f}", (xi, yi), textcoords="offset points", xytext=(dx, dy),
                            ha="center", va=va, fontsize=7, color=s["color"], fontweight="bold")
    for k, r in REF.items():
        full = V[k]["ratios"].get("100")
        if full:
            ax.axhline(y=full["cpi_mape"]["mean"], color=r["color"], linestyle=r["ls"], linewidth=1.5,
                       alpha=0.7, zorder=1, label=f"{k} 100% ({full['cpi_mape']['mean']:.2f})")

    xs = sorted(all_x)
    ax.set_xlabel("Labeled Data (%)", fontweight="bold")
    ax.set_ylabel("Overall MAPE (%)", fontweight="bold")
    ax.set_title(args.title, fontsize=12, fontweight="bold", pad=10)
    ax.set_xticks(xs); ax.set_xticklabels([f"{x}%" for x in xs], rotation=45, ha="right")
    ax.yaxis.grid(True, linestyle=':', color='#A0A0A0', alpha=0.8, zorder=0); ax.xaxis.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_color('black'); sp.set_linewidth(1.0)
    ax.set_ylim(bottom=0)
    leg = ax.legend(loc="upper right", frameon=True, edgecolor='black', facecolor='white',
                    fancybox=False, framealpha=1.0)
    leg.get_frame().set_linewidth(1.0)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.tight_layout()
    fig.savefig(args.out, dpi=600, bbox_inches="tight")
    fig.savefig(args.out.rsplit(".", 1)[0] + ".pdf", format="pdf", bbox_inches="tight")
    print(f"Saved {args.out} (+ .pdf)")


if __name__ == "__main__":
    main()
