#!/bin/bash
# ============================================================
# gem5 SE smoke test for all 4 benchmarks × 3 traces
# Runs 1M instructions each, continues on error, reports at end
# ============================================================

GEM5="/scratch/gem5-TAO/build/ARM/gem5.opt"
SE="/scratch/gem5-TAO/configs/example/se_new.py"
BIN="/scratch/gem5-TAO/micro_exp/new_bench/new_benchmarks/binaries"
INPUT="/scratch/gem5-TAO/micro_exp/new_bench/new_benchmarks/inputs"
OUTDIR="./gem5_smoke_results"

mkdir -p "${OUTDIR}" ./tmp_leveldb

PASS=0
FAIL=0
RESULTS=()

run_test() {
    local name="$1"
    shift
    local outdir="${OUTDIR}/${name}"
    mkdir -p "${outdir}"

    echo "--------------------------------------------"
    echo "[RUN] ${name}"
    echo "--------------------------------------------"

    ${GEM5} --outdir="${outdir}" "$@" > "${outdir}/stdout.log" 2> "${outdir}/stderr.log"
    local rc=$?

    if grep -q "a thread reached the max instruction count" "${outdir}/stdout.log" "${outdir}/stderr.log" 2>/dev/null; then
        local insts=$(grep "sim_insts" "${outdir}/stats.txt" 2>/dev/null | head -1 | awk '{print $2}')
        echo "  PASS (${insts} instructions)"
        RESULTS+=("PASS  ${name}  insts=${insts}")
        PASS=$((PASS+1))
    else
        echo "  FAIL (rc=${rc})"
        echo "  stderr tail:"
        tail -5 "${outdir}/stderr.log" 2>/dev/null | sed 's/^/    /'
        RESULTS+=("FAIL  ${name}  rc=${rc}")
        FAIL=$((FAIL+1))
    fi
    echo ""
}

# === Snappy × 3 ===
run_test "snappy_html" \
    ${SE} -c ${BIN}/snappy_bench -o "${INPUT}/snappy_html.dat 500" \
    --cpu-type=O3CPU --caches --l2cache -I 1000000

run_test "snappy_proto" \
    ${SE} -c ${BIN}/snappy_bench -o "${INPUT}/snappy_proto.dat 500" \
    --cpu-type=O3CPU --caches --l2cache -I 1000000

run_test "snappy_random" \
    ${SE} -c ${BIN}/snappy_bench -o "${INPUT}/snappy_random.dat 500" \
    --cpu-type=O3CPU --caches --l2cache -I 1000000

# === re2 × 3 ===
run_test "re2_literal" \
    ${SE} -c ${BIN}/re2_bench -o "literal 2000" \
    --cpu-type=O3CPU --caches --l2cache -I 1000000

run_test "re2_alternation" \
    ${SE} -c ${BIN}/re2_bench -o "alternation 2000" \
    --cpu-type=O3CPU --caches --l2cache -I 1000000

run_test "re2_backtrack" \
    ${SE} -c ${BIN}/re2_bench -o "backtrack 1500" \
    --cpu-type=O3CPU --caches --l2cache -I 1000000

# === LevelDB × 3 ===
# fillrandom first to populate db, then read from it
run_test "leveldb_fillrandom" \
    ${SE} -c ${BIN}/leveldb_bench -o "fillrandom 5000" \
    --cpu-type=O3CPU --caches --l2cache -I 1000000

run_test "leveldb_readrandom" \
    ${SE} -c ${BIN}/leveldb_bench -o "readrandom 5000" \
    --cpu-type=O3CPU --caches --l2cache -I 1000000

run_test "leveldb_readseq" \
    ${SE} -c ${BIN}/leveldb_bench -o "readseq 5000" \
    --cpu-type=O3CPU --caches --l2cache -I 1000000

# === zstd × 3 ===
run_test "zstd_level1" \
    ${SE} -c ${BIN}/zstd_bench -o "1 ${INPUT}/zstd_mixed.dat 200" \
    --cpu-type=O3CPU --caches --l2cache -I 1000000

run_test "zstd_level9" \
    ${SE} -c ${BIN}/zstd_bench -o "9 ${INPUT}/zstd_mixed.dat 200" \
    --cpu-type=O3CPU --caches --l2cache -I 1000000

run_test "zstd_level19" \
    ${SE} -c ${BIN}/zstd_bench -o "19 ${INPUT}/zstd_mixed.dat 200" \
    --cpu-type=O3CPU --caches --l2cache -I 1000000

# === Summary ===
echo "============================================"
echo " SUMMARY: ${PASS} passed, ${FAIL} failed"
echo "============================================"
for r in "${RESULTS[@]}"; do
    echo "  $r"
done
echo ""
echo "Logs in ${OUTDIR}/<test_name>/stderr.log"
echo "Stats in ${OUTDIR}/<test_name>/stats.txt"