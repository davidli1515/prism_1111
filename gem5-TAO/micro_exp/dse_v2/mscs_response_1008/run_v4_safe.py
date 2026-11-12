#!/usr/bin/env python3
"""Run preserved v4 with isolated McPAT calls/cache. Original files stay intact.

Example (from repository root):
  python micro_exp/dse_v2/mscs_response_1008/run_v4_safe.py \
    --bench 602.gcc_s --method bo_v4_scalar --seed 42 \
    --out micro_exp/dse_v2/mscs_response_1008/results/v4_safe

The original v4 search, checkpoint, area XML translation, and stopping semantics
are retained. In particular, v4 checks the budget between steps and can overshoot
inside a step; compare archived prefixes at the SAME query count. This wrapper
does not claim to be a new MSCS optimizer or a replication of the paper's DSE.
"""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import run_dse_v4 as driver
from safe_area import SafeArea


def main():
    # Override the imported driver's runtime dependencies only. Restore them even
    # when argparse exits, so importing this module has no persistent side effects.
    original_area, original_prefetch = driver.Area, driver.prefetch_areas
    try:
        driver.Area = SafeArea
        driver.prefetch_areas = lambda area, configs, workers: area.prefetch(configs, workers)
        driver.main()
    finally:
        driver.Area, driver.prefetch_areas = original_area, original_prefetch


if __name__ == "__main__":
    main()
