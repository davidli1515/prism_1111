#!/bin/bash
# Start the weight-fix grid only after the step-2 fill driver has finished.
cd /path/to/gem5/micro_exp/exp2
until grep -q "DRIVER DONE" run_ood_lowlabel_grid_1008_fill_driver.log 2>/dev/null; do sleep 120; done
echo "[$(date)] fill finished; starting weight-fix grid"
bash run_ood_lowlabel_grid_wfix_1010_queue.sh > run_ood_lowlabel_grid_wfix_1010.log 2>&1
echo "[$(date)] weight-fix grid finished"
