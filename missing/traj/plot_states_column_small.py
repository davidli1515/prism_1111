#!/usr/bin/env python3
"""plot_states_column.py (imported, unchanged) applied to the small-start reference states
(results_small/state_clusters_v1.npz, results_small/<bench>/part_*.npz).
Writes results_small/figs/fig_state_centroids_col.{png,pdf} and fig_state_pca_col.{png,pdf}; refuses to overwrite."""
import os
import plot_states_column as S
S.R = os.path.join(S.HERE, "results_small"); S.FIG = os.path.join(S.R, "figs")

if __name__ == "__main__":
    S.main()
