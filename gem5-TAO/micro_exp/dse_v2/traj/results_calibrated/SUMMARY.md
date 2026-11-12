# Why PRISM and gem5 disagree in traj

## Conclusion

The per-benchmark biases in the traj validation (imagick +25%, xalancbmk +18%, wrf +12%, perlbench, roms and blender about −10%; 8 of 17 benchmarks with a mean bias above 8%) come mostly from a mismatch between how the validation measures CPI and how the training labels were produced, not from model error. They split into three parts:

1. **Model error against its own label protocol: 2–3%.** On benchmarks whose checkpoint is in the training set, gem5 run the same way as the training labels (cold-start 100K windows, cycles per micro-op) differs from the model by 2.3% (200 windows compared one by one, p90 5.3%).
2. **Cold-start gap.** Each training label is a 100K-instruction window simulated from a cold O3 start, whereas the traj validation simulates 18.7M instructions contiguously. At the start of each window the branch predictor is empty, so labels are higher: by 2–5% for most benchmarks and by 18% for exchange2.
3. **Unseen SimPoints.** For blender, xalancbmk, deepsjeng and imagick, the checkpoint used by the DSE and its validation is not in the training set. On the same windows, the model overestimates imagick by 40%. This part is genuine generalization error.

Two further inconsistencies in the validation itself amplify or cancel the parts above: the CPI definition (cycles per instruction vs. cycles per micro-op) and the slice coverage (Sections 4 and 5).

The model is the traj surrogate, `exp1/results/lambda_sweep_1004/t10_c1_m0p01_s42.pt`, with 4681 input features. None of the gem5 runs has an L2, and neither does the training data.

## 1. Four benchmarks with a trained checkpoint: error against the label protocol

For each benchmark, one validation design was taken: among its gem5_val* runs, the one with the median model error. gem5 was then run window by window, as for the training data, on the 50 slices the surrogate uses (-W 20M + k·100K, -I 100K). All CPIs are cycles per micro-op.

| benchmark | per-window MAPE | mean signed error | p90 | error of the 50-window mean |
|---|---|---|---|---|
| perlbench | 3.5% | −3.4% | 7.1% | −3.4% |
| exchange2 | 2.1% | −1.3% | 4.6% | −1.3% |
| roms | 3.2% | −1.8% | 6.2% | −1.8% |
| gcc | 0.5% | +0.2% | 1.1% | +0.2% |
| **all (200 windows)** | **2.3%** | | **5.3%** | mean absolute design-level error 1.7% |

The same four designs under the other comparisons:

| benchmark | model vs. windowed gem5 | windowed vs. contiguous gem5 | model vs. contiguous gem5 (cycles/uop) | traj as before (contiguous, cycles/inst) |
|---|---|---|---|---|
| perlbench | −3.4% | +4.7% | +1.2% | −11.7% |
| exchange2 | −1.3% | +18.3% | +16.8% | +9.9% |
| roms | −1.8% | +4.2% | +2.3% | −7.4% |
| gcc | +0.2% | +2.6% | +2.8% | +2.7% |
| mean absolute | 1.7% | | 5.8% | 7.9% |

**The training labels reproduce.** Seven training samples were rerun with their own configuration and window. Six match their label exactly, and one (roms, slice 203) differs by +0.35%. So the gem5 binary and command line are the same as when the training data was generated.

**The feature pipeline matches.** On the 13 benchmarks with a trained checkpoint, the CPI predicted from DSE-engine features differs from the CPI predicted from training features by −3.1% to +2.1% on average, so features are not the source of the bias (`diag_feature_parity.txt`).

## 2. Cold-start gap

The TimingSimple warmup warms the caches, but the branch predictor (TAGE) belongs to the O3 CPU and starts every window empty. Measured conditional mispredictions per kilo-instruction:

| benchmark | contiguous | windowed | extra | windowed vs. contiguous CPI |
|---|---|---|---|---|
| imagick | 0.82 | 1.68 | +0.9 | +1.7% |
| roms | 0.24 | 0.32 | +0.1 | +4.2% |
| perlbench | 1.92 | 4.09 | +2.2 | +4.7% |
| gcc | 0.41 | 7.14 | +6.7 | +2.6% |
| exchange2 | 2.74 | 9.07 | +6.3 | +18.3% |

The gap is roughly extra mispredictions × penalty ÷ total CPI:

- **imagick:** its branches are regular, so few extra mispredictions occur and the gap is small.
- **gcc:** it has as many extra mispredictions as exchange2, but its CPI of 5.4 is dominated by memory stalls, which dilute them.
- **exchange2:** its branches are many and data-dependent, and its CPI is only 0.46, so the gap is the largest.

This gap is part of the training-data protocol, and the model cannot remove it. Removing it requires an O3 detailed warmup before each labeled window and regenerating the data.

## 3. Unseen SimPoints

Trace-only feature columns of every training window were matched against the DSE precompute slices (`diag_ckpt_in_train.txt`):

- **No match (0%):** blender, xalancbmk, deepsjeng and imagick.
- **Partial match:** gcc and lbm have two training checkpoints, of which one matches (50% each); mcf matches 68%.
- **Full match:** about 100% for the others.

On imagick, the model overestimates windowed gem5 by +40.0%, while windowed and contiguous gem5 differ by only +1.7%. Almost all of imagick's bias is therefore model error on this new SimPoint.

## 4. CPI definition: cycles per instruction vs. cycles per micro-op

The training label 1/(Base_commit·W_min) equals cycles per micro-op, whereas the gem5 `cpi` used by the traj validation is cycles per instruction. ARM splits instructions into micro-ops. The micro-ops per instruction of each benchmark (identical for all configurations of a benchmark) are:

namd 1.114, blender 1.091, perlbench 1.147, gcc 1.001, mcf 1.024, cactuBSSN 1.006, lbm 1.043, omnetpp 1.171, wrf 1.081, xalancbmk 1.015, deepsjeng 1.091, imagick 1.150, leela 1.059, nab 1.023, exchange2 1.074, roms 1.105, xz 1.075.

This causes the bias of about −10% in perlbench, roms, blender and omnetpp. In exchange2, imagick and similar benchmarks, it happens to cancel part of the positive bias.

## 5. Slice coverage

The surrogate takes the first 50 even slice ids in lexicographic order, which are {0, 10–18, 100–186}. gem5 runs the contiguous range 0–186, so the surrogate never sees slices 19–99. The effect is largest for wrf: with the slices gem5 actually runs, the model's prediction drops by 21%. xalancbmk, leela, roms and blender are also affected; the rest barely change (`diag_slice_coverage.txt`). For mcf, slices are missing at some L1 sizes, and the surrogate uses slices up to 220.

## 6. Effect on all traj gem5 validations (1,324 designs, recomputed without new gem5 runs)

| comparison | MAPE | benchmarks with mean bias > 8% |
|---|---|---|
| as before: 50 slices, vs. cycles/inst | 9.3% | 8/17 |
| slices gem5 runs, vs. cycles/inst | 8.5% | 6/17 |
| 50 slices, vs. cycles/uop | 11.8% | 6/17 |
| slices gem5 runs, vs. cycles/uop | 10.2% | 4/17 |

With a consistent definition the error rises slightly. In the original 9.3%, the 1.0–1.17× inflation of cycles per instruction happened to hide the positive biases from the cold-start gap and the unseen SimPoints. Picking the most favorable comparison per benchmark gives 7.3%, but that number is not usable: it mixes CPI definitions and relies on errors cancelling.

## Limitations

- **Small sample:** Sections 1 and 2 use one design per benchmark (5 benchmarks, 257 gem5 runs).
- **Next step:** covering whole trajectories needs a windowed validation of gem5_val6 (263 designs × 10 slices, about 2,630 runs).

## Files

- `diag_slice_coverage.{txt,npz}`: slice coverage (`calib/diag_slice_coverage.py`).
- `diag_ckpt_in_train.txt`: whether each checkpoint is in the training set (`calib/diag_ckpt_in_train.py`).
- `diag_feature_parity.txt`: feature parity (`calib/diag_feature_parity.py`).
- `diag_cpi_definition.{txt,npz}`: cycles per instruction vs. cycles per micro-op (`calib/diag_cpi_definition.py`).
- `gem5_slice_pilot/`: windowed gem5 pilot (`calib/gem5_slice_pilot.py`). Use `summary_uop.txt`. In `summary.txt`, the column "windowed vs contiguous" compares cycles per micro-op with cycles per instruction and should not be used.
