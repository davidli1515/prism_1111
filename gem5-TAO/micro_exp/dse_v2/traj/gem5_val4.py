#!/usr/bin/env python3
"""gem5 round 4 = gem5_val3.py (imported, unchanged) applied to the SMALL-start greedy walks (results_small/):
one representative walk per benchmark (closest to the small-start median entropy curve), 12 evenly spaced
states. Writes gem5_val4/ (never overwrites). Reuse of earlier runs is disabled (walk ids refer to a different
start set).   python gem5_val4.py [--analyze-only]"""
import os
import gem5_val3 as G
HERE = os.path.dirname(os.path.abspath(__file__))
G.R, G.OUT, G.OLD = os.path.join(HERE, "results_small"), os.path.join(HERE, "gem5_val4"), os.path.join(HERE, "_no_reuse_")

if __name__ == "__main__":
    G.main()
