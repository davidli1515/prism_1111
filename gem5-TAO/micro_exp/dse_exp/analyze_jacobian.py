"""
Analyze Jacobian heatmap to set DSE hyperparameters.

Usage:
    python analyze_jacobian.py --file results/605.mcf_s/605.mcf_s_jacobian_heatmap.json
"""

import argparse
import json
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="Path to jacobian_heatmap.json")
    args = parser.parse_args()

    with open(args.file) as f:
        data = json.load(f)

    param_names = data["param_names"]
    bn_names = data["bottleneck_names"]
    jacobian = np.array(data["jacobian"])   # (n_params, 12)
    cpi_grad = data["cpi_gradient"]         # dict
    cpi_val = data["cpi_value"]

    n_params = len(param_names)

    print(f"{'='*70}")
    print(f"Jacobian Analysis")
    print(f"{'='*70}")
    print(f"CPI = {cpi_val:.4f}")
    print(f"Params: {n_params}, Bottleneck components: {len(bn_names)}")

    # ---- CPI gradient analysis → threshold ----
    print(f"\n{'='*70}")
    print(f"CPI Gradient (∂CPI/∂param) → determines 'threshold'")
    print(f"{'='*70}")

    grads = [(name, cpi_grad[name]) for name in param_names if name in cpi_grad]
    grads.sort(key=lambda x: abs(x[1]), reverse=True)
    abs_vals = [abs(g) for _, g in grads]

    print(f"\n  {'Param':<20s} {'∂CPI/∂param':>12s} {'|grad|':>10s}")
    print(f"  {'-'*44}")
    for name, g in grads:
        print(f"  {name:<20s} {g:>+12.6f} {abs(g):>10.6f}")

    print(f"\n  Statistics:")
    print(f"    max |grad|:    {max(abs_vals):.6f}")
    print(f"    median |grad|: {np.median(abs_vals):.6f}")
    print(f"    min |grad|:    {min(abs_vals):.6f}")
    print(f"    p25:           {np.percentile(abs_vals, 25):.6f}")
    print(f"    p75:           {np.percentile(abs_vals, 75):.6f}")

    # Suggest threshold: between p25 and median, so ~50-75% of params are "significant"
    suggested_threshold = np.percentile(abs_vals, 30)
    print(f"\n  >>> Suggested threshold: {suggested_threshold:.6f}")
    print(f"      (p30 of |grad|: ~70% of params will be INCREASE candidates)")

    # ---- Jacobian analysis → epsilon ----
    print(f"\n{'='*70}")
    print(f"Jacobian upper bounds (max|row|) → determines 'epsilon'")
    print(f"{'='*70}")

    upper_bounds = []
    print(f"\n  {'Param':<20s} {'max|jac_row|':>12s} {'Dominant BN':>15s}")
    print(f"  {'-'*50}")
    for i, name in enumerate(param_names):
        row = jacobian[i, :]
        ub = np.abs(row).max()
        dominant_idx = np.abs(row).argmax()
        dominant_name = bn_names[dominant_idx]
        upper_bounds.append(ub)
        print(f"  {name:<20s} {ub:>12.6f} {dominant_name:>15s}")

    print(f"\n  Statistics:")
    print(f"    max upper_bound:    {max(upper_bounds):.6f}")
    print(f"    median upper_bound: {np.median(upper_bounds):.6f}")
    print(f"    min upper_bound:    {min(upper_bounds):.6f}")
    print(f"    p10:                {np.percentile(upper_bounds, 10):.6f}")
    print(f"    p25:                {np.percentile(upper_bounds, 25):.6f}")

    # Suggest epsilon: below p10-p25, so only truly redundant params get DECREASE
    suggested_epsilon = np.percentile(upper_bounds, 15)
    print(f"\n  >>> Suggested epsilon: {suggested_epsilon:.6f}")
    print(f"      (p15 of upper bounds: only ~15% of params are 'truly redundant')")

    # ---- Jacobian heatmap preview (text) ----
    print(f"\n{'='*70}")
    print(f"Jacobian Heatmap (top effects)")
    print(f"{'='*70}")

    # Show which params most affect which bottleneck types
    for bn_type, cols in [("FP", [0,1,2,3]), ("BP", [4,5,6,7]), ("SP", [8,9,10,11])]:
        print(f"\n  {bn_type} sensitivity (max across 4 stages):")
        effects = []
        for i, name in enumerate(param_names):
            max_effect = np.abs(jacobian[i, cols]).max()
            effects.append((name, max_effect))
        effects.sort(key=lambda x: x[1], reverse=True)
        for name, eff in effects[:5]:
            stages = jacobian[i, cols]  # wrong i, let me fix
        # Redo properly
        for name, _ in effects[:5]:
            i = param_names.index(name)
            vals = jacobian[i, cols]
            stage_str = " ".join(f"{v:+.4f}" for v in vals)
            print(f"    {name:<20s} [{stage_str}]  max={np.abs(vals).max():.4f}")

    # ---- Coupling preview ----
    print(f"\n{'='*70}")
    print(f"Coupling Preview (cosine similarity of 4-stage BP profiles)")
    print(f"{'='*70}")

    # Focus on BP since it's most relevant for coupling
    bp_cols = [4, 5, 6, 7]
    active_params = [(i, name) for i, name in enumerate(param_names)
                     if np.abs(jacobian[i, bp_cols]).max() > suggested_epsilon]

    if len(active_params) >= 2:
        print(f"\n  {'Param A':<16s} {'Param B':<16s} {'Cosine':>8s} {'Coupled?':>10s}")
        print(f"  {'-'*52}")
        for ai in range(len(active_params)):
            for bi in range(ai + 1, len(active_params)):
                i, name_a = active_params[ai]
                j, name_b = active_params[bi]
                ga = jacobian[i, bp_cols]
                gb = jacobian[j, bp_cols]
                na, nb = np.linalg.norm(ga), np.linalg.norm(gb)
                if na > 1e-10 and nb > 1e-10:
                    cos = np.dot(ga, gb) / (na * nb)
                    coupled = "YES" if cos < 0.5 else "no"
                    print(f"  {name_a:<16s} {name_b:<16s} {cos:>8.3f} {coupled:>10s}")

    # ---- Summary ----
    print(f"\n{'='*70}")
    print(f"RECOMMENDED HYPERPARAMETERS")
    print(f"{'='*70}")
    print(f"  --threshold {suggested_threshold:.6f}")
    print(f"  --epsilon   {suggested_epsilon:.6f}")
    print(f"  --coupling-threshold {suggested_epsilon:.6f}  (same as epsilon)")
    print(f"  --cos-threshold 0.5")
    print(f"\n  Paste into run command:")
    print(f"  python run_gem_dse.py --benchmark 605.mcf_s "
          f"--threshold {suggested_threshold:.6f} --epsilon {suggested_epsilon:.6f}")


if __name__ == "__main__":
    main()
