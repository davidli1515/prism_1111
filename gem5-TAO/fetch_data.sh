#!/usr/bin/env bash
# Download data from the Hugging Face dataset David1515/ml_cpu_prism and unpack every shard
# into its place inside this gem5-TAO checkout.
#
# Run from the gem5-TAO directory of the GitHub repo:
#   ./fetch_data.sh                                   # everything (~130 GB download)
#   ./fetch_data.sh "micro_exp/exp1.tar.zst"          # only some shards (glob patterns)
#   ./fetch_data.sh "micro_exp/dse_exp/traces/602*" "train_data/*"
#
# Environment:
#   DL=hf_data   where the .tar.zst shards are downloaded (can be deleted afterwards)
#   DATA=data    where train_data/ and micro_ckpt/ are unpacked
# Requires: huggingface_hub (`hf` CLI), zstd, tar.
set -euo pipefail
REPO=David1515/ml_cpu_prism
DL=${DL:-hf_data}
DATA=${DATA:-data}
[ -d micro_exp ] && [ -f SConstruct ] || { echo "run this from the gem5-TAO directory"; exit 1; }
if [ $# -gt 0 ]; then hf download $REPO --repo-type dataset --local-dir "$DL" --include "$@"
else hf download $REPO --repo-type dataset --local-dir "$DL"; fi

x() { echo "unpack $1 -> $2"; mkdir -p "$2"; zstd -dc "$1" | tar -xf - -C "$2"; }
shopt -s nullglob
# micro_exp shards unpack relative to the directory that matches their path in the dataset
for f in "$DL"/micro_exp/dse_exp/traces/*.tar.zst;     do x "$f" micro_exp/dse_exp/traces; done
for f in "$DL"/micro_exp/dse_exp/precompute/*.tar.zst; do x "$f" micro_exp/dse_exp/precompute; done
for f in "$DL"/micro_exp/dse_exp/*.tar.zst;            do x "$f" micro_exp/dse_exp; done
for f in "$DL"/micro_exp/*.tar.zst;                    do x "$f" micro_exp; done
parts=("$DL"/micro_exp/exp2/exp2.tar.zst.part*)
if [ ${#parts[@]} -gt 0 ]; then
  [ ${#parts[@]} -eq 4 ] || { echo "exp2 needs all 4 parts (micro_exp/exp2/*)"; exit 1; }
  echo "unpack exp2 (4 parts) -> micro_exp"; cat "${parts[@]}" | zstd -dc | tar -xf - -C micro_exp
fi
# data that lived outside gem5-TAO
for f in "$DL"/train_data/*.tar.zst; do x "$f" "$DATA"; done
for f in "$DL"/micro_ckpt/*.tar.zst; do x "$f" "$DATA/micro_ckpt/m5out_cpts"; done
# code/gem5-TAO-src.tar.zst is the same code as this checkout and is not unpacked.

cat <<EOF
Done. Training scripts read the packed arrays from /scratch2/wl730/micro_traindata_packed{,_all}
and checkpoints from /scratch2/wl730/micro_ckpt. Either symlink those paths to
  $(realpath -m "$DATA")/micro_traindata_packed{,_all}  and  $(realpath -m "$DATA")/micro_ckpt
or replace the prefix in the scripts, e.g.
  grep -rl /scratch2/wl730 micro_exp | xargs sed -i "s#/scratch2/wl730#$(realpath -m "$DATA")#g"
EOF
