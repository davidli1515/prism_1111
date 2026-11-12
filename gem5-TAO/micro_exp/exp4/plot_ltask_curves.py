"""
Plot train L_task and test MAPE curves comparing full vs no-mono.
Reads from training logs.

Usage:
    python plot_ltask_curves.py
"""

import re, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# Log paths — edit as needed
# ============================================================
LOGS = {
    "5% Full (with L_mono)":    "/scratch/gem5-TAO/micro_exp/exp4/logs_77/full_0.05.log",
    "5% No L_mono":             "/scratch/gem5-TAO/micro_exp/exp4/logs_77/no_mono_lab_0.05.log",
    # Add more if needed:
    # "15% Full":               "/scratch/gem5-TAO/micro_exp/exp4/logs_77/full_0.15.log",
    # "15% No L_mono":          "/scratch/gem5-TAO/micro_exp/exp4/logs_77/no_mono_lab_0.15.log",
}

SAVE_DIR = "/scratch/gem5-TAO/micro_exp/exp4/figures_final"

# ============================================================
# Parse log
# ============================================================
def parse_log(path):
    epochs, l_tasks, test_mapes, bests = [], [], [], []
    pattern = re.compile(
        r"Epoch\s+(\d+)\s+\|.*?L_task=([\d.]+).*?Test MAPE=([\d.]+)%\s+best=([\d.]+)%"
    )
    with open(path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                epochs.append(int(m.group(1)))
                l_tasks.append(float(m.group(2)))
                test_mapes.append(float(m.group(3)))
                bests.append(float(m.group(4)))
    return {
        "epochs": np.array(epochs),
        "l_task": np.array(l_tasks),
        "test_mape": np.array(test_mapes),
        "best_mape": np.array(bests),
    }


# ============================================================
# Plot
# ============================================================
def main():
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'legend.fontsize': 9,
        'figure.dpi': 600,
        'axes.linewidth': 1.0,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
    })

    os.makedirs(SAVE_DIR, exist_ok=True)

    # Parse all logs
    data = {}
    for name, path in LOGS.items():
        if os.path.exists(path):
            data[name] = parse_log(path)
            print(f"  {name}: {len(data[name]['epochs'])} epochs from {path}")
        else:
            print(f"  WARNING: {path} not found")

    if not data:
        print("No data loaded!")
        return

    # Colors
    colors = {
        "5% Full (with L_mono)":  "#1565C0",
        "5% No L_mono":           "#C62828",
        "15% Full (with L_mono)": "#1565C0",
        "15% No L_mono":          "#C62828",
    }
    linestyles = {
        "5% Full (with L_mono)":  "-",
        "5% No L_mono":           "--",
        "15% Full (with L_mono)": "-",
        "15% No L_mono":          "--",
    }

    # === Figure: 2 panels (Train L_task, Test MAPE) ===
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Train L_task
    for name, d in data.items():
        c = colors.get(name, "gray")
        ls = linestyles.get(name, "-")
        ax1.plot(d["epochs"], d["l_task"], color=c, linestyle=ls,
                linewidth=1.5, label=name)

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train L_task (MSE)")
    ax1.set_title("Train L_task over Training", fontweight="bold")
    ax1.yaxis.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(frameon=True, edgecolor='black', fancybox=False)

    # Panel 2: Test MAPE
    for name, d in data.items():
        c = colors.get(name, "gray")
        ls = linestyles.get(name, "-")
        ax2.plot(d["epochs"], d["test_mape"], color=c, linestyle=ls,
                linewidth=1.5, alpha=0.4, label=None)
        ax2.plot(d["epochs"], d["best_mape"], color=c, linestyle=ls,
                linewidth=1.5, label=name)

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Test CPI MAPE (%)")
    ax2.set_title("Test MAPE over Training", fontweight="bold")
    ax2.yaxis.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(frameon=True, edgecolor='black', fancybox=False)

    plt.tight_layout()
    path = os.path.join(SAVE_DIR, "ltask_curves.png")
    fig.savefig(path, dpi=600, bbox_inches="tight")
    fig.savefig(path.replace(".png", ".pdf"), format="pdf", bbox_inches="tight")
    plt.close()
    print(f"\n  Saved {path}")

    # Print key numbers
    print("\n=== Final values ===")
    for name, d in data.items():
        print(f"  {name}:")
        print(f"    Final L_task: {d['l_task'][-1]:.6f}")
        print(f"    Final test MAPE: {d['test_mape'][-1]:.2f}%")
        print(f"    Best test MAPE: {d['best_mape'][-1]:.2f}%")


if __name__ == "__main__":
    main()
