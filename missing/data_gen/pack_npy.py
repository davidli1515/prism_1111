#!/usr/bin/env python3
"""
Collect and concatenate concord_data.npy / gem_data.npy from the nested
micro_traindata directory tree, grouped by benchmark name.

Directory structure:
  micro_traindata/
    <benchmark>/
      ckpt_<num>/
        slice_<num>/
          <datapoint_folder>/
            concord_data.npy
            gem_data.npy

Output:
  micro_traindata_packed/concord/<benchmark>.npy
  micro_traindata_packed/gem/<benchmark>.npy
"""

import os
import numpy as np
from pathlib import Path
from collections import defaultdict

SRC_ROOT = Path("/data/micro_traindata_321")
DST_ROOT = Path("/data/micro_train_data_321_packed")

CONCORD_DIR = DST_ROOT / "concord"
GEM_DIR = DST_ROOT / "gem"


def gather_npy_paths(benchmark_dir: Path):
    """Walk benchmark_dir and return lists of concord / gem .npy paths."""
    concord_paths = []
    gem_paths = []

    # Level 1: ckpt_* folders
    for ckpt in sorted(benchmark_dir.iterdir()):
        if not ckpt.is_dir() or not ckpt.name.startswith("ckpt_"):
            continue

        # Level 2: slice_* folders
        for slc in sorted(ckpt.iterdir()):
            if not slc.is_dir() or not slc.name.startswith("slice_"):
                continue

            # Level 3: datapoint folders (long hash names)
            for dp in sorted(slc.iterdir()):
                if not dp.is_dir():
                    continue

                c = dp / "concord_data.npy"
                g = dp / "gem_data.npy"
                if c.exists():
                    concord_paths.append(c)
                if g.exists():
                    gem_paths.append(g)

    return concord_paths, gem_paths


def concat_and_save(npy_paths: list, output_path: Path):
    """Load all .npy files, concatenate along axis 0, and save."""
    if not npy_paths:
        return 0

    arrays = []
    for p in npy_paths:
        arr = np.load(p, allow_pickle=True).astype(np.float32)
        # If the individual file is 1-D or a single sample, keep as-is;
        # np.concatenate will handle shape alignment along axis 0.
        arrays.append(arr if arr.ndim > 1 else arr[np.newaxis, ...])

    merged = np.concatenate(arrays, axis=0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, merged)
    return merged.shape[0]


def main():
    # Create output dirs
    CONCORD_DIR.mkdir(parents=True, exist_ok=True)
    GEM_DIR.mkdir(parents=True, exist_ok=True)

    # Discover benchmarks (top-level dirs)
    benchmarks = sorted(
        d for d in SRC_ROOT.iterdir()
        if d.is_dir()
    )

    print(f"Found {len(benchmarks)} benchmark(s) under {SRC_ROOT}\n")

    for bench in benchmarks:
        name = bench.name
        print(f"[{name}] Scanning ...")

        concord_paths, gem_paths = gather_npy_paths(bench)
        print(f"  concord_data.npy files: {len(concord_paths)}")
        print(f"  gem_data.npy     files: {len(gem_paths)}")

        # --- concord ---
        n_concord = concat_and_save(
            concord_paths, CONCORD_DIR / f"{name}.npy"
        )
        if n_concord:
            print(f"  -> saved concord/{name}.npy  ({n_concord} samples)")

        # --- gem ---
        n_gem = concat_and_save(
            gem_paths, GEM_DIR / f"{name}.npy"
        )
        if n_gem:
            print(f"  -> saved gem/{name}.npy  ({n_gem} samples)")

        print()

    print("All done!")


if __name__ == "__main__":
    main()
