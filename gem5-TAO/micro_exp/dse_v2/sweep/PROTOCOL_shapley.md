# MSCS-grouped pruned Shapley, re-tested with implementation fixes

Code: `shapley_check.py` (new). Imports `dse_exp/shapley_pruning.py` read-only and uses its `compute_cosine_matrix`, `group_params`, `increase_param`, `MSCS_12_INDICES` unchanged. Raw: `results/shapley_raw.json`, report: `results/shapley_check.txt`.

## Kept from shapley_pruning.py
- Value v(S): players in S at target (= base config value), others at min; marginal = v(before) - v(after) in CPI.
- Grouping vector: 12-dim MSCS share change, base -> p at min (increase direction if p already at min); |cos| matrix; `group_params(threshold=0.3)`.
- Pruned Shapley: per group, players outside the group at target, group players outside the coalition at min; exact within groups <= 8 players, MC 10 permutations otherwise.
- Baselines: matched-budget MC, one-at-a-time from the all-min config.

## Changed
1. Every config goes through `space17.project17` (`_project_to_valid` + fixed L2 / BP / fetch buffer). The original evaluated ~61% illegal coalition configs.
2. W = fetch/decode/dispatch/commit width as one player (target = base widths, min = all 4).
3. Players not modelled by the features dropped: IntMultDiv, FP_MultDiv, wbWidth, l1i_assoc. 10 players remain.
4. 17-param space; 6 benchmarks (gcc, mcf, xalancbmk, imagick, leela, exchange2) x 6 bases (DEFAULT_CONFIG projected + 5 random, seed 31).
5. Budget = distinct model evaluations (cache). Matched MC adds permutations until it has used at least the pruned method's distinct evaluations (grouping-vector evaluations included).
6. l1_size min = 16 (original overrode to 32 because 16 KB was not precomputed then).
- Model: PRISM 100%, `exp1_v2/exp1_512_256/gem_full.pt`, original DSE pipeline, commit-stage CPI (checked identical to `GEMPredictor.predict_cpi`).

## Added controls
- Scalar grouping: same group sizes as the MSCS grouping, filled with players in order of |main effect| (from the same evaluations, no MSCS).
- Random grouping: same group sizes, random members, 10 draws averaged.

## Truth
MC Shapley with 200 permutations (Concorde protocol). A second 200-permutation MC with another seed agrees: Spearman 0.992 (min 0.964), top-3 3.00/3, weighted error 0.027. Efficiency check: sum of Shapley values = v(all-min) - v(base).

## Result (36 benchmark x base cases)
| method | Spearman | top-3 overlap | weighted error | distinct evals |
|---|---|---|---|---|
| pruned, MSCS grouping | 0.900 | 2.61 | 0.164 | 112 |
| pruned, main-effect grouping | 0.900 | 2.64 | 0.101 | 107 |
| pruned, random grouping | 0.899 | 2.62 | 0.168 | 110 |
| matched-budget MC | **0.953** | **2.67** | **0.067** | 114 |
| one-at-a-time | 0.900 | 2.50 | 0.338 | 10 |

- MSCS grouping is indistinguishable from random grouping of the same sizes.
- Matched MC beats pruned-MSCS in 28/36 cases (Spearman) and 34/36 (weighted error).
- Groups still collapse: the largest group has 5-9 of the 10 players.
