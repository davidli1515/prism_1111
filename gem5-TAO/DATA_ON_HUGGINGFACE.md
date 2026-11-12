# gem5-TAO: code here, data on Hugging Face

This directory is the browsable code part of the `gem5-TAO` working tree:

- the modified gem5 source tree (MSCS accounting in `src/cpu/o3/mscs.{cc,hh}`, trace hooks in the
  O3/Minor/Simple CPUs, branch predictor and caches)
- data-generation scripts at the top level (`gen_spec_traindata*.py`, `gen_simpletrace.py`,
  `spec17_benchmarks.py`, `pack_npy*.py`, `run_*.sh`, ...)
- `concord_analytical/`, `concord_analytical_onlyConcord/`, `external_simulator/`
- `micro_exp/`: all experiment code, plus figures (<= 5 MB) and small result files
  (json/csv/log/txt <= 100 KB)

Everything else, including large result files, traces, feature arrays, training data and model
checkpoints, is in the Hugging Face dataset
**[David1515/ml_cpu_prism](https://huggingface.co/datasets/David1515/ml_cpu_prism)** as
`tar.zst` shards. The dataset also contains a full copy of the code in this directory.

| Hugging Face shard | Contents |
|---|---|
| `code/gem5-TAO-src.tar.zst` | Same code as this directory (gem5 tree + scripts) |
| `micro_exp/dse_exp/traces/<benchmark>.tar.zst` | Per-slice gem5 O3 traces for 17 SPEC CPU2017 benchmarks |
| `micro_exp/dse_exp/precompute/*.tar.zst` | Precomputed DSE feature arrays |
| `micro_exp/<experiment>.tar.zst` | Complete experiment directories, including all results and `.pt` checkpoints (`exp2` is split into `exp2/exp2.tar.zst.part*`) |
| `train_data/micro_traindata_packed{,_all}.tar.zst` | Packed training arrays (scripts reference them at `/scratch2/wl730/micro_traindata_packed{,_all}`) |

Download and extract one shard:

```bash
hf download David1515/ml_cpu_prism --repo-type dataset \
    --include "micro_exp/exp1.tar.zst" --local-dir ./data
zstd -dc data/micro_exp/exp1.tar.zst | tar -xf - -C micro_exp/
```

Not distributed: SPEC CPU2017 binaries, sources and inputs (licensed), build outputs, temporary
files and third-party projects.
