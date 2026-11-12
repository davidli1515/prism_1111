# Excluding imagick: code distance between validation and training checkpoints

## Which benchmarks are affected

For four benchmarks, the checkpoint used by the DSE engine and by the gem5 validation is not one of the checkpoints that the training windows were simulated from: blender, deepsjeng, xalancbmk and imagick. Each of these benchmarks is therefore validated on a SimPoint that the model never saw during training. We decide whether such a benchmark can be validated by a criterion fixed in advance: how different the validation code is from the training code. We do not decide by the resulting error.

## How the distance is measured

Some feature columns depend only on the instruction trace, not on the microarchitecture. A column counts as trace-only when it meets both conditions:

- it is constant across all configurations within a training window;
- the DSE engine gives it the same value under two extreme configurations.

The steps are:

1. For each benchmark, take each training window's trace-only vector.
2. Normalise every column by its maximum magnitude.
3. Compute the distance from the training window to each validation slice, as the max over columns of the absolute difference.
4. Keep the nearest validation slice.

The reference scale is the typical distance between two different validation slices of the same benchmark. A ratio of about 1 means the validation code differs from the training code no more than one validation slice differs from another. A ratio far above 1 means the validation code is outside what training covered.

## Result

| Benchmark | Exact matches | Median nearest distance, training to validation | Median distance between validation slices | Ratio |
|---|---|---|---|---|
| blender | 0% | 0.20 | 0.20 | ≈1.0 |
| deepsjeng | 0% | 0.20 | 0.19 | ≈1.1 |
| xalancbmk | 0% | 0.16 | 0.13 | ≈1.2 |
| imagick | 0% | 1.00 | 0.11 | ≈9 |

"Exact matches" is the share of training windows with an exact match. For the 13 benchmarks whose training and validation checkpoints coincide, the exact-match rate is 50–100%, with distances around 1e-8.

blender, deepsjeng and xalancbmk run unseen code, but that code sits within the normal slice-to-slice variation. imagick's validation code is about nine times farther from its training code than validation slices are from each other. In effect, imagick is validated on a different program phase, so we exclude it. The other three benchmarks stay in.

The criterion uses only the instruction trace, so it can be checked without running the model or gem5. The source is `micro_exp/dse_v2/traj/calib/diag_ckpt_in_train.py`, and its output is `micro_exp/dse_v2/traj/results_calibrated/diag_ckpt_in_train.txt`.

## Source of the 2.52% figure

The 2.52% figure is a design-level error. It covers the 13 benchmarks whose training and validation checkpoints coincide, uses training-identical features, and is measured against windowed gem5 in cycles per micro-op. It averages the benchmarks with equal weight. It is not stored as a number in any file; it is computed from `trainfeat_vs_engine_designs.txt` (59-design snapshot) as follows:

1. Take the "design-level |err|: training" column, which is the value before the "/".
2. Read that column for the 13 benchmark rows other than blender, xalancbmk, deepsjeng and imagick.
3. Take the unweighted mean of those 13 values: 32.82% / 13 = 2.5246% ≈ 2.52%.

The same table's "seen checkpoint" row gives 2.60%. That row is the mean over the 50 designs, so it weights benchmarks by their design count. In this snapshot the counts are uneven: namd and perlbench have 9 designs each, gcc has 7, and the other benchmarks have 2–3 each. Both values are interim; the feature batch has not finished.

To reproduce the figure:

```
grep -vE "blender|xalancbmk|deepsjeng|imagick|^ALL|^seen|^unseen|^gem5|^bench" trainfeat_vs_engine_designs.txt \
  | awk -F'|' '{split($3,a,"/"); gsub(/[% ]/,"",a[1]); s+=a[1]; n++} END {printf "n=%d mean=%.4f%%\n", n, s/n}'
```
