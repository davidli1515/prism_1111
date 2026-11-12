#!/usr/bin/env python3
"""cluster_states.py (imported, unchanged) applied to the SMALL-start greedy walks (results_small/), so that the
small starts become the reference bottleneck states of Section V. Writes results_small/state_clusters_v1.{npz,txt}
(never overwrites)."""
import os
import cluster_states as C
C.R = os.path.join(C.HERE, "results_small")

if __name__ == "__main__":
    C.main()
