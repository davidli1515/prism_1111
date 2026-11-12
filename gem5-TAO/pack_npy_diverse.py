#!/usr/bin/env python3
"""
Collect gem_data.npy from multiple source dirs, split train/test per slice,
output one file per (benchmark, ckpt) combo.

Output:
  <DST_ROOT>/<benchmark>_<ckpt>_train.npy
  <DST_ROOT>/<benchmark>_<ckpt>_test.npy
"""

import numpy as np
from pathlib import Path

SRC_ROOTS = [
    Path("/scratch2/wl730/micro_traindata"),
    Path("/scratch2/wl730/micro_traindata_321"),
]

DST_ROOT = Path("/scratch2/wl730/micro_traindata_diverse")

TRAIN_COUNT = 2  # first N per slice go to train


def gather_train_test_per_ckpt(benchmark_dir: Path):
    """
    Return dict: { ckpt_name: (train_paths, test_paths) }
    """
    result = {}

    for ckpt in sorted(benchmark_dir.iterdir()):
        if not ckpt.is_dir() or not ckpt.name.startswith("ckpt_"):
            continue

        train_paths = []
        test_paths = []

        for slc in sorted(ckpt.iterdir()):
            if not slc.is_dir() or not slc.name.startswith("slice_"):
                continue

            dp_dirs = sorted([d for d in slc.iterdir() if d.is_dir()])

            for i, dp in enumerate(dp_dirs):
                gem = dp / "gem_data.npy"
                if not gem.exists():
                    continue
                if i < TRAIN_COUNT:
                    train_paths.append(gem)
                else:
                    test_paths.append(gem)

        if train_paths or test_paths:
            result[ckpt.name] = (train_paths, test_paths)

    return result


def concat_and_save(npy_paths: list, output_path: Path):
    if not npy_paths:
        return 0
    arrays = []
    for p in npy_paths:
        arr = np.load(p, allow_pickle=True).astype(np.float32)
        if arr.ndim == 1:
            arr = arr[np.newaxis, ...]
        arrays.append(arr)
    merged = np.concatenate(arrays, axis=0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, merged)
    return merged.shape[0]


def main():
    DST_ROOT.mkdir(parents=True, exist_ok=True)

    # Collect all (benchmark, ckpt) pairs across all source dirs
    # Key: (bench_name, ckpt_name), avoid duplicates
    seen = set()
    dup_count = 0

    for src_root in SRC_ROOTS:
        if not src_root.exists():
            print(f"[WARN] Source not found, skipping: {src_root}")
            continue

        print(f"Scanning {src_root} ...")
        benchmarks = sorted(d for d in src_root.iterdir() if d.is_dir())

        for bench in benchmarks:
            bench_name = bench.name
            ckpt_data = gather_train_test_per_ckpt(bench)

            for ckpt_name, (train_paths, test_paths) in sorted(ckpt_data.items()):
                key = (bench_name, ckpt_name)
                if key in seen:
                    # Duplicate: all data goes to test
                    all_paths = train_paths + test_paths
                    print(f"  [DUP] {bench_name}/{ckpt_name}  -> all {len(all_paths)} files to test")
                    dup_count += 1

                    # Append to existing test file if present
                    out_test = DST_ROOT / f"{bench_name}_{ckpt_name}_test.npy"
                    if out_test.exists():
                        existing = np.load(out_test, allow_pickle=True)
                        extra = []
                        for p in all_paths:
                            arr = np.load(p, allow_pickle=True).astype(np.float32)
                            if arr.ndim == 1:
                                arr = arr[np.newaxis, ...]
                            extra.append(arr)
                        merged = np.concatenate([existing] + extra, axis=0)
                        np.save(out_test, merged)
                        print(f"    -> appended to test, now {merged.shape[0]} samples")
                    else:
                        n_test = concat_and_save(all_paths, out_test)
                        print(f"    -> test {n_test} samples")
                    continue

                seen.add(key)
                print(f"  [{bench_name} / {ckpt_name}]  train={len(train_paths)}, test={len(test_paths)}")

                n_train = concat_and_save(
                    train_paths,
                    DST_ROOT / f"{bench_name}_{ckpt_name}_train.npy"
                )
                n_test = concat_and_save(
                    test_paths,
                    DST_ROOT / f"{bench_name}_{ckpt_name}_test.npy"
                )
                print(f"    -> train {n_train} samples, test {n_test} samples")

        print()

    print(f"Unique (benchmark, ckpt) pairs: {len(seen)}")
    print(f"Duplicate (benchmark, ckpt) pairs: {dup_count}")
    print("All done!")


if __name__ == "__main__":
    main()