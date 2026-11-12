#!/usr/bin/env python3
"""plot_entropy_gem5rep.py for the SMALL-start walks (results_small/, gem5_val4/ incl. plan_peak.json).
Writes results_small/figs/fig_{entropy,jain}_gem5rep_small.png|pdf (never overwrites)."""
import os
import plot_entropy_gem5rep as P
P.R = os.path.join(P.HERE, "results_small"); P.FIG = os.path.join(P.R, "figs"); P.V3 = os.path.join(P.HERE, "gem5_val4")
P.TAG, P.LABEL = "gem5rep_small", "small starts"
if __name__ == "__main__":
    P.main()
