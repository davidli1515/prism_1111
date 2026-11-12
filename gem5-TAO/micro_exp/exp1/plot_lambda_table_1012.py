#!/usr/bin/env python3
"""
Draw the white-background loss-weight table from the JSON written by
collect_lambda_table_1012.py (no access to raw results).

Rows: lam_const = 1 with lam_task in TASKS x lam_mono in MONOS (configs with 3 seeds only),
then the lam_const = 0 control (italic). The selected setting is highlighted.

Usage:
  python plot_lambda_table_1012.py --json results/lambda_table_1012.json \
      --out results/lambda_table_1012.png [--tasks 0.1 0.3 1 3 10 30] [--monos 0 0.01 0.1] \
      [--chosen 10 1 0] [--control 1 0 0.01]
"""
import os, sys, json, argparse
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "/scratch/gem5-TAO/micro_exp/exp1/results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=f"{R}/lambda_table_1012.json")
    ap.add_argument("--out", default=f"{R}/lambda_table_1012.png")
    ap.add_argument("--tasks", type=float, nargs="+", default=[0.1, 0.3, 1, 3, 10, 30])
    ap.add_argument("--monos", type=float, nargs="+", default=[0, 0.01, 0.1])
    ap.add_argument("--chosen", type=float, nargs=3, default=[10, 1, 0], metavar=("TASK", "CONST", "MONO"))
    ap.add_argument("--control", type=float, nargs=3, default=[1, 0, 0.01], metavar=("TASK", "CONST", "MONO"))
    ap.add_argument("--min-seeds", type=int, default=3)
    args = ap.parse_args()

    data = json.load(open(args.json))["rows"]
    find = lambda t, c, m: next((r for r in data if (r["lam_task"], r["lam_const"], r["lam_mono"]) == (t, c, m)
                                 and r["n_seeds"] >= args.min_seeds), None)
    fmt = lambda r: [f"{r['lam_task']:g}", f"{r['lam_const']:g}", f"{r['lam_mono']:g}",
                     f"{r['cpi_mape']['mean']:.1f}", f"{r['component_mae']['mean']:.3f}",
                     f"{r['base_incons']['mean']:.2f}", f"{r['mono_viol']['mean']:.1f}"]

    rows, chosen = [], None
    for t in args.tasks:
        for m in args.monos:
            r = find(t, 1.0, m)
            if r is None:
                continue
            if [t, 1.0, m] == args.chosen:
                chosen = len(rows)
            rows.append(fmt(r))
    ctrl = find(*args.control)
    if ctrl is not None:
        rows.append(fmt(ctrl))

    cols = ["λ_task", "λ_const", "λ_mono", "CPI\nMAPE (%)", "Component\nerror (MAE)",
            "Base\ninconsistency (%)", "Monotonicity\nviolations (%)"]
    widths = [0.09, 0.09, 0.09, 0.13, 0.16, 0.19, 0.19]
    fig, ax = plt.subplots(figsize=(9.2, 0.33 * len(rows) + 1.0), facecolor="white")
    ax.axis("off")
    tb = ax.table(cellText=rows, colLabels=cols, cellLoc="center", loc="center", colWidths=widths)
    tb.auto_set_font_size(False); tb.set_fontsize(9); tb.scale(1, 1.35)
    # best (lowest) displayed value per metric column, over the sweep rows only
    # (the lam_const = 0 control row is excluded); ties are all highlighted
    n_sweep = len(rows) - (1 if ctrl is not None else 0)
    best = {c: min(float(rows[i][c]) for i in range(n_sweep)) for c in range(3, 7)}
    for (r, c), cell in tb.get_celld().items():
        cell.set_edgecolor("#BBBBBB"); cell.set_linewidth(0.6); cell.set_facecolor("white")
        if r == 0:
            cell.set_text_props(weight="bold"); cell.set_height(cell.get_height() * 1.7)
            continue
        if chosen is not None and r - 1 == chosen:
            cell.set_text_props(weight="bold"); cell.set_facecolor("#EEF3FB")
        elif ctrl is not None and r == len(rows):
            cell.set_text_props(style="italic")
        if c in best and r - 1 < n_sweep and float(rows[r - 1][c]) == best[c]:
            cell.set_facecolor("#CDEFD3")           # green: best value in this column
    fig.savefig(args.out, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(args.out.rsplit(".", 1)[0] + ".pdf", bbox_inches="tight", facecolor="white")
    print(f"Saved {args.out} (+ .pdf), {len(rows)} rows")


if __name__ == "__main__":
    main()
