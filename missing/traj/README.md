# Greedy optimisation trajectories (traj/)

Purpose: observe how the multi-stage CPI stack (MSCS) evolves while a design is optimised, from many
starts, for every SPEC benchmark and for the joint 17-benchmark objective. The optimiser is
scalar-driven (CPI and area only); MSCS is only recorded.

## Setup (fixed before running)
- Model: dse_v2 `SurrogateWL` = exp1 `lambda_sweep_1004/t10_c1_m0p01_s42.pt`, 18k normaliser, first 50
  even slices, CPI = slice mean of the 4-stage-mean CPI (the surrogate validated against gem5 in
  `gem5_bounds/`). Features: bit-exact fast feature builder (`coord_ablation/fast_infer.py`).
- Space: 17 parameters (`dse17/space17.py`); L2 1 MB/8-way, TAGE, fetch buffer 32 fixed.
- Starts: 1000 random designs, seed 2026, identical for every benchmark and for the joint run.
- Moves: one parameter up/down per step; list params one index (x2), integer params
  max(1, round(range/8/stride)) strides (widths 1, FUs 1, IQ 16, LQ/SQ 32, ROB 112); then validity
  projection (coupled parameter changes are flagged).
- Greedy: reclaim (down-move with CPI loss <= 0.2%, largest area saving) if any, else advance (up-move
  with largest gain/area-increase among gains >= 0.5%), else stop; no revisits; <= 60 steps.
- Area: dse17 fast area model (`dse17/area_model.pkl`); McPAT error on a sample in
  `results/area_check.json`.

## Files
`results/<bench>/part_XXX.npz` (50 starts per part), `results/joint_17/part_XXX.npz`, `meta.json`.
One row per visited state:

| key | shape | meaning |
|---|---|---|
| start, step | (n,) | start id (same ids across benchmarks), step index |
| cfg | (n,17) | parameter values, order = meta["params"] |
| area, cpi | (n,) | fast area (mm2), CPI (joint: geomean) |
| mscs | (n,16) | single benchmark: [Base,FP,BP,SP] x [Dec,Disp,Iss,Com], index 4*stage+cat |
| cpi_wl, mscs_wl | (n,17), (n,17,16) | joint only: per workload (order = meta["workloads"]) |
| kind, act | (n,) | 0 stop / 1 advance / 2 reclaim; chosen move slot (meta["moves"]) or -1 |
| stop | (n,) | "plateau" / "max_steps" on the last state of a walk |
| k_valid, k_coupled | (n,34) | candidate move exists; projection changed other parameters |
| k_cpi, k_area | (n,34) | CPI / area of every candidate move from this state |
| k_mscs | (n,34,16) | MSCS of every candidate (joint: mean over workloads) |
| k_cpi_wl | (n,34,17) | joint only: per-workload CPI of every candidate |

Load everything: `np.concatenate` over `part_*.npz` per tag.
