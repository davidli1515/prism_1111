# RpStacks check: does MSCS relief-vector similarity predict parameter interaction?

Code: `rpstacks_check.py` (new; imports dse_exp / exp1 / dse17 read-only). Raw: `results/rpstacks_raw.json`, report: `results/rpstacks.txt`.

## Hypotheses
- H_group (dse_exp/shapley_pruning.py): parameters with similar MSCS change vectors (high |cos|) interact.
- H_rpstack (RpStacks max-of-linear view): parameters relieving the same bottleneck are additive; interaction (masking / complements) is between parameters relieving different competing bottlenecks (low cos).

## Method
- Model: PRISM 100% labels, `exp1/results/exp1_v2/exp1_512_256/gem_full.pt`, original DSE pipeline (dse_exp normalizer cache, GEMPredictor 200-slice sampling, commit-stage CPI), bit-exact fast features. Checked: CPI identical to `GEMPredictor.predict_cpi` and Concorde identical to `ConcordePredictor.predict_cpi`.
- Reference model: Concorde 100% (`concorde.pt`), only to measure how model-dependent the "truth" is.
- Space: 17-param space (L2 1 MB / 8-way, TAGE, fetch buffer 32 fixed). 6 benchmarks (gcc, mcf, xalancbmk, imagick, leela, exchange2) x 6 random bases (seed 23).
- Players (10): W = fetch/decode/dispatch/commit width moved together (4 or 12); ROB, IQ, LQ, SQ, IntALU, FP_ALU, RdWrPort, l1_size, l1d_assoc, each at its min or max.
- Excluded: IntMultDiv, FP_MultDiv, wbWidth (not used by the analytical feature model), l1i_assoc (its distribution is frozen in the DSE precompute).
- Legality: every config goes through `space17.project17` (`_project_to_valid`). A pair is "coupled" if projection changed a player's value or any other parameter (ROB raised to 2*dispatch+IQ+LQ+SQ; l1d_assoc capped by l1_size). Reported for all pairs and for clean pairs.
- Truth: I_pq = [CPI(lo,lo) - CPI(hi,lo) - CPI(lo,hi) + CPI(hi,hi)] / CPI(base). I > 0 complements (masking), I < 0 substitutes (overlap). Strong = |I| >= 1%.
- Cheap signals (2 evaluations per player): v_p = MSCS(p_lo) - MSCS(p_hi) on the 12 FP/BP/SP x stage cells, as share (shapley_pruning vector) or CPI-weighted cycles; signed cos, |cos|, same dominant cell. Scalar-only baselines: |dCPI_p|*|dCPI_q|, min(|dCPI_p|, |dCPI_q|).
- Metrics: Spearman and AUC (AUC < 0.5 = opposite direction). Thresholds and signals fixed before running.

## Result (clean pairs, n=726)
| signal | AUC strong abs(I) | AUC complement | AUC substitute |
|---|---|---|---|
| cos (share), signed | 0.40 | 0.41 | 0.42 |
| abs(cos) (share) | 0.58 | 0.49 | 0.61 |
| abs(cos) (cycles) | 0.66 | 0.52 | 0.69 |
| same dominant cell | 0.47-0.52 | 0.46-0.50 | 0.46-0.54 |
| scalar abs(dCPI_p)*abs(dCPI_q) | **0.83** | 0.69 | **0.83** |
| scalar min | 0.77 | 0.61 | 0.78 |

- Neither hypothesis is supported for complements (masking): all MSCS signals ~0.5.
- Similar relief (|cos| on cycles) weakly predicts substitutes (overlap), but the scalar main-effect product is better; within main-effect terciles |cos| adds information only in the lowest tercile.
- Strong complements are rare (7% of clean pairs); strong substitutes 20%. Strongest interactions involve W (W x RdWrPort, W x IntALU) and IntALU x RdWrPort.
- Caveat: PRISM and Concorde disagree on interactions (Spearman 0.37, sign agreement 0.73 on |I| >= 1%), so the second-order "truth" is itself model-dependent. 55% of pairs are coupled by legality rules (ROB in almost every pair).
