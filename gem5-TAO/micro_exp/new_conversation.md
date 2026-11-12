# GEM Project Handoff Document
# Paste this at the start of a new conversation to restore context.

## Project
GEM: Physics-Informed ML for CPU Micro-architecture Performance Modeling
- Paper target: MICRO conference
- Core insight: micro-architectural invariants formalized as differentiable loss functions

## Data Layout

### Paths
- Concorde data: `/scratch2/wl730/micro_traindata_packed/concord/{benchmark}.npy`
- GEM data:      `/scratch2/wl730/micro_traindata_packed/gem/{benchmark}.npy`
- Example file:  `508.namd_r.npy`

### concord_data.npy — shape (N, 3874)
- `[0:3873]`  → Concorde input features (3873 dims)
- `[3873]`    → Golden CPI label

### gem_data.npy — shape (N, 4697)
- `[0:3873]`     → Concorde features (same as concord_data)
- `[3873:4681]`  → GEM D3 pressure features (808 dims)
- `[4681:4697]`  → MSCS target (16 dims, flattened 4×4)

### MSCS layout (16 dims, index within the 16-dim vector)
```
[0]=Base_dec  [1]=FP_dec   [2]=BP_dec   [3]=SP_dec
[4]=Base_disp [5]=FP_disp  [6]=BP_disp  [7]=SP_disp
[8]=Base_iss  [9]=FP_iss  [10]=BP_iss  [11]=SP_iss
[12]=Base_com [13]=FP_com [14]=BP_com  [15]=SP_com
```
Per-stage softmax: Base + FP + BP + SP = 1 for each stage.

### Pipeline widths (within gem_data)
```
gd[3850]=fetchWidth, gd[3851]=decodeWidth, gd[3852]=renameWidth,
gd[3853]=dispatchWidth, gd[3854]=issueWidth, gd[3855]=wbWidth(EXCLUDED),
gd[3856]=commitWidth
```
W_min = min(gd[[3850,3851,3852,3853,3854,3856]])  # exclude wbWidth

## CPI Formula
```
CPI_s = 1 / (Base_s × W_min)
```
Equivalent to: CPI_s = 1/W_min + (FP_s + BP_s + SP_s) / (N_window × W_min)

## Model Architectures

### GEM Model
- Input: 4681 dims (Concorde features + D3 features)
- Hidden: [1024, 512, 256] with BatchNorm + ReLU + Dropout(0.1)
- Output: 16 dims with per-stage softmax (4 stages × 4 states)
- Optimizer: Adam, lr=1e-3, weight_decay=1e-5
- Loss: L_total = L_task(MSE) + λ1·L_const + λ2·L_mono

### Concorde Baseline (from Concorde-ISCA'25 paper)
- Input: 3873 dims (Concorde features only)
- Hidden: [256, 128], no dropout
- Output: scalar CPI
- Optimizer: AdamW, lr=1e-3, weight_decay=0.3
- Loss: relative error |ŷ - y| / y

## Loss Functions

### L_task: MSE(predicted_MSCS, ground_truth_MSCS)

### L_const (throughput conservation):
CPI computed at each stage must equal golden CPI:
L_const = (1/4) × Σ_s (CPI_s − CPI_golden)²

### L_mono (bottleneck monotonicity):
From Decode → Dispatch → Issue → Commit:
- FP must decrease (hinge loss)
- SP must decrease (hinge loss)
- BP must increase (hinge loss)

## Experiment Plan

### Exp 1: In-Distribution Accuracy
- 19 benchmarks train, 19 test, 15:1 split within each benchmark
- GEM vs Concorde MAPE comparison
- Code: `/scratch/gem5-TAO/micro_exp/exp1/train_exp1.py`

### Exp 2: OOD Generalization (Leave-One-Out)
- 18 train, 1 held-out test, iterate all benchmarks
- Code: `/scratch/gem5-TAO/micro_exp/exp2/train_exp2.py`
- Checkpoints saved to: `exp2/results/checkpoints/{benchmark}/`

### Exp 3: Ablation Study
5 variants: Full GEM, w/o D2, w/o D2 + post-hoc projection, w/o D1 (scalar CPI), w/o D3
+ Constraint satisfaction metrics (CPI consistency error, monotonicity violation rate)

### Exp 4: Label Efficiency
Labeled data ratios: 5%, 10%, 20%, 50%, 100%
Two curves: Full GEM vs w/o D2

### Exp 5: Explainability & Differentiable Analysis
(a) Bottleneck attribution — MSCS stacked bar charts
(b) Gradient-based sensitivity — ∂CPI/∂(micro-arch params) heatmap
(c) Gradient-guided DSE — pareto front comparison

## Code Structure
```
/scratch/gem5-TAO/micro_exp/
├── exp1/
│   ├── config.py      # constants, hyperparams, paths
│   ├── model.py       # GEMModel, ConcordeModel, loss functions
│   ├── dataset.py     # data loading, normalization, dataloaders
│   └── train_exp1.py  # Exp1 training script
├── exp2/
│   └── train_exp2.py  # Exp2 LOO script (imports from exp1/)
├── exp3/ ... exp5/    # to be created
```

## Key Design Decisions
- Framework: pure PyTorch
- Per-stage softmax guarantees MSCS proportions sum to 1
- Normalizer fit on train only, applied to test
- Early stopping with patience=30
- Incremental JSON saving for crash recovery
- Exp2 saves model checkpoints + normalizers for reuse in Exp5