"""
Exp 5c: Main DSE Script — Penetration-Guided Multi-Stage Bottleneck Analysis

Methods:
  1. concorde_bo         — Concorde + Bayesian Optimization (baseline)
  2. concorde_scalar     — Concorde scalar CPI gradient
  3. gem_scalar          — GEM scalar CPI gradient (better model)
  4. gem_penetration     — GEM Penetration-Guided (our contribution)

Usage:
    python run_gem_dse.py --benchmark 605.mcf_s
    python run_gem_dse.py --benchmark 623.xalancbmk_s
    python run_gem_dse.py --benchmark 605.mcf_s --methods gem_penetration --dry-run
"""

import argparse
import os
import sys
import json
import time
import numpy as np
import torch

EXP1_DIR = "/scratch/gem5-TAO/micro_exp/exp1"
DSE_DIR  = "/scratch/gem5-TAO/micro_exp/dse_exp"
sys.path.insert(0, EXP1_DIR)
sys.path.insert(0, DSE_DIR)

from config import GEM_DATA_DIR, CONCORD_DATA_DIR, BENCHMARK_NAMES
from model import GEMModel, ConcordeModel
from dataset import load_all_benchmarks, create_exp1_dataloaders, FeatureNormalizer
from dse_infer import DseEngine

GEM_INPUT_DIM        = 4681
CONCORDE_INPUT_DIM   = 3873
GEM_HIDDEN_DIMS      = [512, 256]
CONCORDE_HIDDEN_DIMS = [512, 256]
DROPOUT              = 0.1

from config_dse import DEFAULT_CONFIG, estimate_design_space_size
from area_model import compute_area_cached
from dse_engine import (
    GEMPredictor, ConcordePredictor, MultiWorkloadPredictor,
    run_bo_concorde, run_bo_gem, run_concorde_scalar, run_gem_scalar, run_gem_penetration,
    run_gem_attribution, run_gem_coordinated,
    run_bo_then_refine, _scalar_refine, _mscs_refine, _coordinated_refine,
    compare_methods, save_results,
    compute_pareto_front, compute_hypervolume_curve,
)

GEM_WEIGHT_PATH      = os.path.join(EXP1_DIR, "results/exp1_v2/exp1_512_256/gem_full.pt")
CONCORDE_WEIGHT_PATH = os.path.join(EXP1_DIR, "results/exp1_v2/exp1_512_256/concorde.pt")
RESULTS_DIR          = os.path.join(DSE_DIR, "results")

PRECOMPUTE_BASE = os.path.join(DSE_DIR, "precompute")

BENCHMARKS = [
    "508.namd_r", "526.blender_r", "600.perlbench_s", "602.gcc_s",
    "605.mcf_s", "607.cactuBSSN_s", "619.lbm_s", "620.omnetpp_s",
    "621.wrf_s", "623.xalancbmk_s", "631.deepsjeng_s", "638.imagick_s",
    "641.leela_s", "644.nab_s", "648.exchange2_s", "654.roms_s", "657.xz_s",
]

ALL_METHODS = ["gem_attribution", "gem_coordinated", "gem_scalar", "gem_scalar_noreclaim",
               "gem_bo", "gem_bo_scalar", "gem_bo_mscs", "gem_bo_coordinated"]

NORMALIZER_CACHE = os.path.join(DSE_DIR, "normalizer_cache")

def get_normalizers():
    paths = [os.path.join(NORMALIZER_CACHE, f"{p}.npy")
             for p in ["gem_mean", "gem_std", "con_mean", "con_std"]]
    if all(os.path.isfile(p) for p in paths):
        print("  Loading cached normalizers...")
        gem_norm = FeatureNormalizer()
        gem_norm.mean = np.load(paths[0]); gem_norm.std = np.load(paths[1])
        con_norm = FeatureNormalizer()
        con_norm.mean = np.load(paths[2]); con_norm.std = np.load(paths[3])
        return gem_norm, con_norm
    print("  Rebuilding normalizers from training data...")
    all_data = load_all_benchmarks(GEM_DATA_DIR, CONCORD_DATA_DIR, BENCHMARK_NAMES)
    (_, _, gem_norm, con_norm, _) = create_exp1_dataloaders(all_data)
    os.makedirs(NORMALIZER_CACHE, exist_ok=True)
    np.save(paths[0], gem_norm.mean); np.save(paths[1], gem_norm.std)
    np.save(paths[2], con_norm.mean); np.save(paths[3], con_norm.std)
    return gem_norm, con_norm


def main():
    parser = argparse.ArgumentParser(description="Exp 5c: MSCS-Guided DSE")
    parser.add_argument("--benchmark", nargs="+", required=True,
                        choices=BENCHMARKS,
                        help="One or more benchmarks (joint optimization if multiple)")
    parser.add_argument("--methods", nargs="*", default=ALL_METHODS)
    parser.add_argument("--n-starts", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--bo-budget", type=int, default=500)
    parser.add_argument("--threshold", type=float, default=0.01)
    parser.add_argument("--reclaim-threshold", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output dir (default: results_MMDD_HHMM)")
    parser.add_argument("--heatmap", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--bo-ckpt", type=str, default=None,
                        help="Path to saved BO results JSON (skip Phase 1 if exists)")
    parser.add_argument("--refine-rounds", type=int, default=3,
                        help="Number of iterative refine rounds (default: 3)")
    parser.add_argument("--refine-steps", type=int, default=15,
                        help="Steps per refine (default: 15)")
    args = parser.parse_args()

    benchmarks = args.benchmark  # list of benchmark names
    joint = len(benchmarks) > 1
    if joint:
        run_label = f"joint_{len(benchmarks)}"
    else:
        run_label = benchmarks[0]

    # Flat directory: results_MMDD_HHMM/ with all JSONs directly inside
    if args.output_dir is None:
        from datetime import datetime
        ts = datetime.now().strftime("%m%d_%H%M")
        output_dir = os.path.join(DSE_DIR, f"results_{ts}")
    else:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cpu")

    print(f"{'='*60}")
    print(f"MSCS-Guided DSE {'(joint)' if joint else ''}")
    print(f"{'='*60}")
    print(f"Benchmarks:   {benchmarks}")
    print(f"Design space: {estimate_design_space_size():.2e}")
    print(f"Methods:      {args.methods}")

    print(f"\n[1/4] Loading models...")
    gem_model = GEMModel(input_dim=GEM_INPUT_DIM, hidden_dims=GEM_HIDDEN_DIMS, dropout=DROPOUT)
    gem_model.load_state_dict(torch.load(GEM_WEIGHT_PATH, map_location=device))
    gem_model.eval()
    print(f"  GEM: {sum(p.numel() for p in gem_model.parameters()):,} params")

    concorde_model = ConcordeModel(input_dim=CONCORDE_INPUT_DIM,
                                     hidden_dims=CONCORDE_HIDDEN_DIMS, dropout=0.0)
    concorde_model.load_state_dict(torch.load(CONCORDE_WEIGHT_PATH, map_location=device))
    concorde_model.eval()
    print(f"  Concorde: {sum(p.numel() for p in concorde_model.parameters()):,} params")

    print(f"\n[2/4] Loading normalizers...")
    gem_norm, con_norm = get_normalizers()

    print(f"\n[3/4] Loading precomputed features...")
    dse_engine = DseEngine(PRECOMPUTE_BASE)

    # Create predictors — multi-workload wraps individual per-benchmark predictors
    if joint:
        gem_per_wl = [GEMPredictor(gem_model, gem_norm, dse_engine,
                                     workload=bm, device=device) for bm in benchmarks]
        gem_predictor = MultiWorkloadPredictor(gem_per_wl)
        # For concorde, use first benchmark (BO baseline doesn't use MSCS)
        concorde_predictor = ConcordePredictor(concorde_model, con_norm, dse_engine,
                                                workload=benchmarks[0], device=device)
    else:
        gem_predictor = GEMPredictor(gem_model, gem_norm, dse_engine,
                                      workload=benchmarks[0], device=device)
        concorde_predictor = ConcordePredictor(concorde_model, con_norm, dse_engine,
                                                workload=benchmarks[0], device=device)

    print(f"\n  Sanity check...")
    t_sc = time.time()
    gem_cpi = gem_predictor.predict_cpi(DEFAULT_CONFIG)
    area = compute_area_cached(DEFAULT_CONFIG)
    print(f"    GEM {'gmean ' if joint else ''}CPI={gem_cpi:.4f}, Area={area:.2f}")
    if not joint:
        con_cpi = concorde_predictor.predict_cpi(DEFAULT_CONFIG)
        mscs = gem_predictor.predict_mscs(DEFAULT_CONFIG)
        print(f"    Concorde CPI={con_cpi:.4f}")
        print(f"    MSCS FP: [{mscs[1]:.3f},{mscs[5]:.3f},{mscs[9]:.3f},{mscs[13]:.3f}]")
        print(f"    MSCS BP: [{mscs[2]:.3f},{mscs[6]:.3f},{mscs[10]:.3f},{mscs[14]:.3f}]")
        print(f"    MSCS SP: [{mscs[3]:.3f},{mscs[7]:.3f},{mscs[11]:.3f},{mscs[15]:.3f}]")
    else:
        print(f"    Per-workload CPI: ", end="")
        for p in gem_predictor.predictors:
            print(f"{p.workload}={p.predict_cpi(DEFAULT_CONFIG):.4f} ", end="")
        print()
    print(f"    Time: {time.time()-t_sc:.3f}s")

    if args.dry_run:
        print(f"\n  Dry run complete.")
        return

    benchmark = run_label  # for save_results compatibility

    print(f"\n[4/4] Running DSE methods...")
    all_results = []
    t0 = time.time()

    method_runners = {
        "gem_attribution": lambda: run_gem_attribution(
            gem_predictor, compute_area_cached,
            reclaim_threshold=args.reclaim_threshold,
            n_starts=args.n_starts, max_steps=args.max_steps,
            seed=args.seed, verbose=True),
        "gem_attribution_single": lambda: run_gem_attribution(
            gem_predictor, compute_area_cached,
            reclaim_threshold=args.reclaim_threshold,
            n_starts=args.n_starts, max_steps=args.max_steps,
            seed=args.seed, single_stage=True, verbose=True),
        "gem_penetration": lambda: run_gem_penetration(
            gem_predictor, compute_area_cached,
            reclaim_threshold=args.reclaim_threshold,
            n_starts=args.n_starts, max_steps=args.max_steps,
            seed=args.seed, verbose=True),
        "gem_scalar": lambda: run_gem_scalar(
            gem_predictor, compute_area_cached,
            threshold=args.threshold,
            n_starts=args.n_starts, max_steps=args.max_steps,
            seed=args.seed, no_reclaim=False, verbose=True),
        "gem_scalar_noreclaim": lambda: run_gem_scalar(
            gem_predictor, compute_area_cached,
            threshold=args.threshold,
            n_starts=args.n_starts, max_steps=args.max_steps,
            seed=args.seed, no_reclaim=True, verbose=True),
        "concorde_scalar": lambda: run_concorde_scalar(
            concorde_predictor, compute_area_cached,
            threshold=args.threshold,
            n_starts=args.n_starts, max_steps=args.max_steps,
            seed=args.seed, verbose=True),
        "concorde_bo": lambda: run_bo_concorde(
            concorde_predictor, compute_area_cached,
            budget=args.bo_budget, seed=args.seed, verbose=True),
        "gem_bo": lambda: run_bo_gem(
            gem_predictor, compute_area_cached,
            budget=args.bo_budget, seed=args.seed, verbose=True),
        "gem_bo_scalar": lambda: run_bo_then_refine(
            gem_predictor, compute_area_cached,
            budget=args.bo_budget, seed=args.seed, verbose=True,
            refine_fn=_scalar_refine, method_name="gem_bo_scalar"),
        "gem_bo_mscs": lambda: run_bo_then_refine(
            gem_predictor, compute_area_cached,
            budget=args.bo_budget, seed=args.seed, verbose=True,
            refine_fn=_mscs_refine, method_name="gem_bo_mscs"),
        "gem_coordinated": lambda: run_gem_coordinated(
            gem_predictor, compute_area_cached,
            n_starts=args.n_starts, max_steps=args.max_steps,
            seed=args.seed, verbose=True),
        "gem_bo_coordinated": lambda: run_bo_then_refine(
            gem_predictor, compute_area_cached,
            budget=args.bo_budget, seed=args.seed, verbose=True,
            refine_fn=_coordinated_refine, method_name="gem_bo_coordinated"),
    }

    # Special handling: if both gem_bo_scalar and gem_bo_mscs requested,
    # share Phase 1 (pure BO) to avoid running it twice
    bo_refine_methods = [m for m in args.methods if m in ("gem_bo_scalar", "gem_bo_mscs", "gem_bo_coordinated")]
    other_methods = [m for m in args.methods if m not in ("gem_bo_scalar", "gem_bo_mscs", "gem_bo_coordinated")]

    # Run non-BO-refine methods normally
    for method_name in other_methods:
        r = method_runners[method_name]()
        save_results(r, output_dir, prefix=f"{benchmark}_")
        all_results.append(r)

    # Run BO-refine methods with shared Phase 1
    if bo_refine_methods:
        print(f"\n{'='*60}")
        print(f"BO + Refine (iterative, {args.refine_rounds} rounds × {args.refine_steps} steps)")
        print(f"{'='*60}")

        # Phase 1: Pure BO (load checkpoint or run)
        bo_ckpt_path = args.bo_ckpt
        if bo_ckpt_path is None:
            bo_ckpt_path = os.path.join(output_dir, "bo_checkpoint.json")

        if os.path.exists(bo_ckpt_path):
            print(f"\n  Phase 1: Loading BO checkpoint from {bo_ckpt_path}")
            with open(bo_ckpt_path) as f:
                bo_base = json.load(f)
            print(f"    {len(bo_base['cpis'])} evals loaded")
        else:
            print(f"\n  Phase 1: Pure BO exploration (budget={args.bo_budget})...")
            bo_base = run_bo_gem(
                gem_predictor, compute_area_cached,
                budget=args.bo_budget, seed=args.seed, verbose=True)
            with open(bo_ckpt_path, "w") as f:
                json.dump(bo_base, f, indent=2, default=str)
            print(f"    Saved BO checkpoint: {bo_ckpt_path}")

        # Per-benchmark predictors for refine navigation
        refine_map = {
            "gem_bo_scalar": ("BO + Scalar refine", _scalar_refine),
            "gem_bo_mscs":   ("BO + MSCS refine",   _mscs_refine),
            "gem_bo_coordinated": ("BO + Coordinated", _coordinated_refine),
        }

        def safe_area(c):
            try: return compute_area_cached(c)
            except Exception: return None

        per_bm_predictors = {}
        if joint:
            for p in gem_predictor.predictors:
                per_bm_predictors[p.workload] = p
        else:
            per_bm_predictors[benchmarks[0]] = gem_predictor
        n_bm = len(per_bm_predictors)

        for method_name in bo_refine_methods:
            label, refine_fn = refine_map[method_name]
            print(f"\n  {'='*55}")
            print(f"  {label}: {args.refine_rounds} rounds × {args.refine_steps} steps × {n_bm} benchmarks")
            print(f"  {'='*55}")

            all_cpis = list(bo_base["cpis"])
            all_areas = list(bo_base["areas"])
            all_configs = list(bo_base["configs"])
            t0 = time.time()
            total_refined = 0

            for rnd in range(args.refine_rounds):
                pts = list(zip(all_cpis, all_areas))
                pidx = compute_pareto_front(pts)

                # Round 0: all Pareto; Round 1+: only new Pareto points
                if rnd == 0:
                    refine_idx = list(pidx.tolist())
                else:
                    refine_idx = [i for i in pidx if i >= prev_len]

                prev_len = len(all_cpis)

                if not refine_idx:
                    print(f"\n    Round {rnd+1}: no new Pareto points, stopping")
                    break

                print(f"\n    Round {rnd+1}/{args.refine_rounds}: "
                      f"refining {len(refine_idx)} points (Pareto={len(pidx)})")

                n_round = 0
                for i, pi in enumerate(refine_idx):
                    config = all_configs[pi]
                    gmean_before = all_cpis[pi]

                    for bm_name, bm_predictor in per_bm_predictors.items():
                        refined = refine_fn(config, bm_predictor, compute_area_cached,
                                             n_steps=args.refine_steps)
                        area = safe_area(refined)
                        if area is None:
                            continue
                        gmean_cpi = gem_predictor.predict_cpi(refined)
                        all_configs.append(refined)
                        all_cpis.append(gmean_cpi)
                        all_areas.append(area)
                        n_round += 1

                    elapsed = time.time() - t0
                    best_new = min(all_cpis[-n_bm:])
                    print(f"      [{i+1}/{len(refine_idx)}] gmean {gmean_before:.4f}→{best_new:.4f} [{elapsed:.0f}s]")

                total_refined += n_round
                pts = list(zip(all_cpis, all_areas))
                pidx_new = compute_pareto_front(pts)
                n_new_pareto = sum(1 for i in pidx_new if i >= prev_len)
                print(f"    Round {rnd+1} done: +{n_round} configs, +{n_new_pareto} new Pareto points")

            t_refine = time.time() - t0
            print(f"\n    Total: {total_refined} refined configs in {t_refine:.0f}s")

            pts = list(zip(all_cpis, all_areas))
            pidx_final = compute_pareto_front(pts)
            ref = (max(all_cpis) * 1.1, max(all_areas) * 1.1)
            r = {
                "method": method_name,
                "configs": all_configs,
                "cpis": all_cpis,
                "areas": all_areas,
                "pareto_indices": pidx_final.tolist(),
                "hv_curve": compute_hypervolume_curve(all_cpis, all_areas, ref),
                "ref_point": ref,
                "wall_time": bo_base.get("wall_time", 0) + t_refine,
                "n_bo_evals": len(bo_base["cpis"]),
                "n_refined": total_refined,
            }
            save_results(r, output_dir, prefix=f"{benchmark}_")
            all_results.append(r)

    if len(all_results) >= 2:
        print(f"\n{'='*60}\nComparison\n{'='*60}")
        comparison = compare_methods(all_results)
        for m in comparison["methods"]:
            wt = f"{m['wall_time']:.0f}s" if m['wall_time'] else "?"
            print(f"  {m['name']:25s} | evals={m['n_evals']:4d} | "
                  f"pareto={m['pareto_size']:3d} | HV={m['final_hv']:.4f} | {wt}")
        comp_path = os.path.join(output_dir, f"{benchmark}_comparison.json")
        with open(comp_path, "w") as f:
            json.dump(comparison, f, indent=2)
        print(f"\n  Saved: {comp_path}")

    total = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Done! Total: {total:.0f}s ({total/60:.1f}min)")
    print(f"Results: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()