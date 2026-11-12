#!/bin/bash
cd /scratch/gem5-TAO/micro_exp/dse_v2
until grep -q "ALL DONE" run_reclaim_1005.log 2>/dev/null; do sleep 120; done
/scratch2/wl730/conda_envs/hybrid/bin/python summarize_reclaim.py > reclaim_summary_1005.txt 2>&1
