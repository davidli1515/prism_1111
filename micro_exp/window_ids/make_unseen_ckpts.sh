#!/bin/bash
# ============================================================
# Create the missing SimPoint checkpoints (one unseen simpoint per benchmark)
# for the unseen-simpoint test set. Same command form as the original
# atomic_run.log of each ckpt, with fixes for the original failures:
#   mcf/roms/xz: OOM  -> larger --mem-size
#   gcc: page-table fault at 8GB -> 16GB + run-dir cwd (writes gcc-pp.opts... relative)
#   omnetpp/wrf/nab/exchange2/roms: run from their SPEC run dir (as BENCH_CWD in gen_spec_traindata.py)
# perlbench is NOT included: the program exits before 537.6B / 686.2B instructions
# with this input, so those simpoints are unreachable.
#
# Usage on the fast machine (edit the 4 paths first):
#   bash make_unseen_ckpts.sh            # all 9 in parallel
#   ONLY="605.mcf_s 620.omnetpp_s" bash make_unseen_ckpts.sh
# Afterwards copy each $OUT/<bench>/<tick>/ (must contain cpt.*) to
#   /data/micro_ckpt/m5out_cpts/<bench>/<tick>/
# ============================================================

GEM5=${GEM5:-/path/to/gem5/build/ARM/gem5.opt}         # must be the same gem5 build
SE_PY=${SE_PY:-/path/to/gem5/configs/example/se_new.py}
B=${SPEC:-/home/user/spec17_20_benchmarks_minimal}            # SPEC root (layout: <bench>/exe, <bench>/run/...)
OUT=${OUT:-/home/user/m5out_cpts_unseen}

export OMP_NUM_THREADS=1
mkdir -p "$OUT"

run() {   # run <bench> <tick> <mem> <cwd> <args...>
    local bench=$1 tick=$2 mem=$3 cwd=$4; shift 4
    if [ -n "$ONLY" ] && [[ " $ONLY " != *" $bench "* ]]; then return; fi
    local od="$OUT/$bench/$tick"; mkdir -p "$od"
    echo "[$(date +%H:%M:%S)] start $bench @ $tick (mem=$mem, cwd=$cwd)"
    ( cd "$cwd" && "$GEM5" --outdir="$od" "$SE_PY" --cpu-type=AtomicSimpleCPU --mem-size="$mem" "$@" \
        --take-checkpoints="$tick" --at-instruction --max-checkpoints=1 ) > "$od/atomic_run.log" 2>&1 &
}

R() { echo "$B/$1/run/run_base_refspeed_mytest-64.0000"; }

# 602.gcc_s   @ 1078.9B   (18k has 430.3B + 532.9B)
run 602.gcc_s 1078900000000 16GB "$(R 602.gcc_s)" \
    -c "$B/602.gcc_s/exe/sgcc_base.mytest-64" \
    -o "$(R 602.gcc_s)/gcc-pp.c -O5 -fipa-pta -o gcc-pp.opts-O5_-fipa-pta.s"

# 605.mcf_s   @ 1474.9B   (orig OOM at 8GB)
run 605.mcf_s 1474900000000 32GB "$(R 605.mcf_s)" \
    -c "$B/605.mcf_s/exe/mcf_s_base.mytest-64" -o "$(R 605.mcf_s)/inp.in"

# 619.lbm_s   @ 3904.4B   (18k has 1900.3B + 2151.9B; only remaining ckpt)
run 619.lbm_s 3904400000000 8GB "$(R 619.lbm_s)" \
    -c "$B/619.lbm_s/exe/lbm_s_base.mytest-64" \
    --options="2000 reference.out 0 0 $B/619.lbm_s/data/refspeed/input/200_200_260_ldc.of"

# 620.omnetpp_s @ 201.9B
run 620.omnetpp_s 201900000000 8GB "$(R 620.omnetpp_s)" \
    -c "$(R 620.omnetpp_s)/omnetpp_s_base.mytest-64" --options="-c General -r 0"

# 621.wrf_s   @ 705.3B
run 621.wrf_s 705300000000 8GB "$(R 621.wrf_s)" \
    -c "$(R 621.wrf_s)/wrf_s_base.mytest-64"

# 644.nab_s   @ 6481.6B   (12194.1B exited immediately -> cwd issue; use run dir)
run 644.nab_s 6481600000000 8GB "$(R 644.nab_s)" \
    -c "$(R 644.nab_s)/nab_s_base.mytest-64" -o "3j1n 20140317 220"

# 648.exchange2_s @ 1604.7B
run 648.exchange2_s 1604700000000 8GB "$(R 648.exchange2_s)" \
    -c "$B/648.exchange2_s/exe/exchange2_s_base.mytest-64" -o "6"

# 654.roms_s  @ 3746.9B   (orig OOM at 8GB; the working 306.3B ckpt used 16GB)
run 654.roms_s 3746900000000 32GB "$(R 654.roms_s)" \
    -c "$(R 654.roms_s)/sroms_base.mytest-64" --input "$(R 654.roms_s)/ocean_benchmark3.in.x"

# 657.xz_s    @ 613.5B    (32GB, as the working 108.2B ckpt)
run 657.xz_s 613500000000 32GB "$(R 657.xz_s)" \
    -c "$(R 657.xz_s)/xz_s_base.mytest-64" \
    -o "$(R 657.xz_s)/cpu2006docs.tar.xz 6643 055ce243071129412e9dd0b3b69a21654033a9b723d874b2015c774fac1553d9713be561ca86f74e4f16f22e664fc17a79f30caa5ad2c04fbc447549c2810fae 1036078272 1111795472 1"

wait
echo "=== done $(date) ==="
for d in "$OUT"/*/*; do
    if ls -d "$d"/cpt.* >/dev/null 2>&1; then echo "OK      $d"; else echo "FAILED  $d  ($(grep -m1 -oE 'fatal.*|panic.*|Exiting @ tick [0-9]+ because .*' "$d/atomic_run.log" | head -c 120))"; fi
done
