#!/usr/bin/env python3
"""
White-background table: labels PRISM saves vs Concorde at matched CPI MAPE (exp4, 18k).

For each PRISM label ratio, the Concorde label ratio reaching the same CPI MAPE is found by
linear interpolation on the Concorde curve (results_run2/exp4_18k_concorde). Saved labels are
converted to gem5 cost with the measured mean of 81 s per labeled sample (15 runs, 7 SPEC
benchmarks), and to wall-clock days on one 512 GB server running PAR gem5 jobs at once
(memory-bound: 32 GB per job).
PRISM MAPE comes from results_run2/exp4_18k_extra_unlabeled.

Usage: python plot_label_cost_table_1012.py [--par 12] [--out results_run2/label_cost_table_1012.png]
"""
import re, glob, json, argparse
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "/path/to/gem5/micro_exp/exp4/results_run2"
N_TRAIN = 217476          # 18k train rows (100% labels)
GEM5_S = 81.0             # measured mean gem5 time per labeled sample (s)
RATIOS = [5, 10, 20, 50, 60]


def curve(d):
    r = {int(re.search(r"_(\d+)pct", p).group(1)): json.load(open(p))["overall_mape"]
         for p in glob.glob(f"{R}/{d}/exp4_*pct.json")}
    return dict(sorted(r.items()))


def match_ratio(conc, target):
    ks = list(conc)
    for a, b in zip(ks, ks[1:]):
        if conc[a] >= target >= conc[b]:
            return a + (b - a) * (conc[a] - target) / (conc[a] - conc[b])
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--par", type=int, default=12, help="concurrent gem5 jobs per server")
    ap.add_argument("--out", default=f"{R}/label_cost_table_1012.png")
    args = ap.parse_args()

    conc, prism = curve("exp4_18k_concorde"), curve("exp4_18k_extra_unlabeled")
    rows = []
    for r in RATIOS:
        m = prism[r]
        cr = match_ratio(conc, m)
        n_p, n_c = r / 100 * N_TRAIN, cr / 100 * N_TRAIN
        saved = n_c - n_p
        core_h = saved * GEM5_S / 3600
        days = saved * GEM5_S / args.par / 86400
        rows.append([f"{r}%", f"{m:.1f}", f"{cr:.1f}%", f"{n_c/n_p:.1f}×",
                     f"{saved/1e3:.1f}k", f"{core_h:,.0f}", f"{days:.1f}"])
        print(f"PRISM {r}% MAPE {m:.2f} -> Concorde {cr:.2f}% | saved {saved:,.0f} labels, "
              f"{core_h:,.0f} core-h, {days:.2f} days @ {args.par} jobs")

    cols = ["PRISM\nlabels", "PRISM CPI\nMAPE (%)", "Concorde labels\nfor same MAPE",
            "Label\nreduction", "Labels\nsaved", "gem5\ncore-hours", f"Days on one server\n({args.par} jobs)"]
    widths = [0.10, 0.12, 0.17, 0.12, 0.11, 0.13, 0.20]
    fig, ax = plt.subplots(figsize=(9.6, 0.33 * len(rows) + 1.0), facecolor="white")
    ax.axis("off")
    tb = ax.table(cellText=rows, colLabels=cols, cellLoc="center", loc="center", colWidths=widths)
    tb.auto_set_font_size(False); tb.set_fontsize(9); tb.scale(1, 1.35)
    for (r, c), cell in tb.get_celld().items():
        cell.set_edgecolor("#BBBBBB"); cell.set_linewidth(0.6); cell.set_facecolor("white")
        if r == 0:
            cell.set_text_props(weight="bold"); cell.set_height(cell.get_height() * 1.7)
    fig.savefig(args.out, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(args.out.rsplit(".", 1)[0] + ".pdf", bbox_inches="tight", facecolor="white")
    print(f"Saved {args.out} (+ .pdf)")


if __name__ == "__main__":
    main()
