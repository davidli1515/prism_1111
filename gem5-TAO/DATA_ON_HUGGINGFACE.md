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
| `micro_ckpt/<benchmark>.tar.zst` | Per-SimPoint gem5 run directories: `simpletrace.txt`, stats, configs, logs (without the checkpoints themselves) |

The gem5 SimPoint checkpoints (`cpt.<tick>/`, whose memory images contain SPEC binaries and inputs)
and the SPEC run directories needed to restart from them are in a separate **private** dataset,
shared only with holders of a SPEC CPU2017 license. Contact the author for access.

## Combining this code with the data

`fetch_data.sh` downloads shards and unpacks each one into its place in this directory, so the
result is the original working tree:

```bash
git clone https://github.com/davidli1515/prism_1111.git
cd prism_1111/gem5-TAO
pip install -U huggingface_hub        # provides the `hf` CLI; also needs zstd
./fetch_data.sh                       # everything, ~130 GB
./fetch_data.sh "micro_exp/exp1.tar.zst" "train_data/*"   # or only selected shards
```

Where each shard goes (paths relative to `gem5-TAO/`):

| Shard in the dataset | Unpacked into |
|---|---|
| `micro_exp/<experiment>.tar.zst`, `micro_exp/_files.tar.zst` | `micro_exp/` |
| `micro_exp/exp2/exp2.tar.zst.part00..03` (concatenate all 4) | `micro_exp/` |
| `micro_exp/dse_exp/_rest.tar.zst` | `micro_exp/dse_exp/` |
| `micro_exp/dse_exp/traces/<benchmark>.tar.zst` | `micro_exp/dse_exp/traces/` |
| `micro_exp/dse_exp/precompute/*.tar.zst` | `micro_exp/dse_exp/precompute/` |
| `train_data/*.tar.zst` | `$DATA/` (default `data/`) |
| `micro_ckpt/<benchmark>.tar.zst` | `$DATA/micro_ckpt/m5out_cpts/` |
| `code/gem5-TAO-src.tar.zst` | not needed (same code as this directory) |

Files that exist both here and in a shard are identical, so unpacking over the checkout is safe.
The scripts read training data and checkpoints from `/scratch2/wl730/...`; the script prints how to
point them at `$DATA` instead.

Not distributed: SPEC CPU2017 binaries, sources and inputs (licensed), build outputs, temporary
files and third-party projects.
