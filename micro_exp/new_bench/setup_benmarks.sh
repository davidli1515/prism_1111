#!/bin/bash
# ============================================================
# Download, cross-compile, and prepare 4 benchmarks × 3 traces
# for gem5 SE mode (AArch64, static, single-threaded)
#
# Benchmarks:
#   1. Snappy   (3 inputs: html, protobuf, random)
#   2. re2      (3 patterns: literal, alternation, backtrack)
#   3. LevelDB  (3 ops: fillrandom, readrandom, readseq)
#   4. zstd     (3 levels: 1, 9, 19)
#
# Usage: bash setup_benchmarks.sh [GEM5_DIR]
# ============================================================

set -e

GEM5_DIR="${1:-/path/to/gem5}"
WORK_DIR="$(pwd)/new_benchmarks"
BIN_DIR="${WORK_DIR}/binaries"
INPUT_DIR="${WORK_DIR}/inputs"
NPROC=$(nproc)

# --- Toolchain paths ---
ARM_TOOLCHAIN="/opt/arm-gnu/arm-gnu-toolchain-14.3.rel1-x86_64-aarch64-none-linux-gnu"
CC="${ARM_TOOLCHAIN}/bin/aarch64-none-linux-gnu-gcc"
CXX="${ARM_TOOLCHAIN}/bin/aarch64-none-linux-gnu-g++"
SYSROOT="${ARM_TOOLCHAIN}/aarch64-none-linux-gnu/libc"

mkdir -p "$WORK_DIR" "$BIN_DIR" "$INPUT_DIR"
cd "$WORK_DIR"

# ============================================================
# Create CMake toolchain file
# ============================================================
TOOLCHAIN_FILE="${WORK_DIR}/aarch64-toolchain.cmake"
cat > "${TOOLCHAIN_FILE}" << TCEOF
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)
set(CMAKE_C_COMPILER ${CC})
set(CMAKE_CXX_COMPILER ${CXX})
set(CMAKE_SYSROOT ${SYSROOT})
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
TCEOF
echo "Toolchain file: ${TOOLCHAIN_FILE}"

# ============================================================
# Verify toolchain
# ============================================================
echo ""
echo "============================================"
echo " Checking toolchain"
echo "============================================"
if [ ! -x "${CC}" ]; then
    echo "ERROR: ${CC} not found"
    exit 1
fi
echo "CC:  $(${CC} --version | head -1)"
echo "CXX: $(${CXX} --version | head -1)"

# Quick test
echo 'int main(){return 0;}' > /tmp/test_arm.c
${CC} --sysroot=${SYSROOT} -static /tmp/test_arm.c -o /tmp/test_arm 2>&1
file /tmp/test_arm
rm -f /tmp/test_arm /tmp/test_arm.c
echo "Toolchain OK"

# ============================================================
# 1. Snappy
# ============================================================
echo ""
echo "============================================"
echo " [1/4] Snappy"
echo "============================================"
if [ ! -d snappy ]; then
    git clone --depth 1 https://github.com/google/snappy.git
    cd snappy && git submodule update --init && cd ..
fi

cd snappy
rm -rf build_arm && mkdir build_arm && cd build_arm
cmake .. \
    -DCMAKE_TOOLCHAIN_FILE="${TOOLCHAIN_FILE}" \
    -DCMAKE_EXE_LINKER_FLAGS="-static" \
    -DCMAKE_CXX_FLAGS="-O2" \
    -DSNAPPY_BUILD_TESTS=OFF \
    -DSNAPPY_BUILD_BENCHMARKS=OFF \
    -DBUILD_SHARED_LIBS=OFF
make -j${NPROC} 2>&1 | tail -5
cd "$WORK_DIR"

# Build standalone snappy driver
cat > snappy_bench.cc << 'EOF'
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include "snappy.h"

int main(int argc, char* argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <input_file> [iters]\n", argv[0]);
        return 1;
    }
    FILE* f = fopen(argv[1], "rb");
    if (!f) { perror("fopen"); return 1; }
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<char> input(fsize);
    fread(input.data(), 1, fsize, f);
    fclose(f);

    int iters = (argc > 2) ? atoi(argv[2]) : 500;
    std::string compressed, decompressed;

    for (int i = 0; i < iters; i++) {
        snappy::Compress(input.data(), input.size(), &compressed);
        snappy::Uncompress(compressed.data(), compressed.size(), &decompressed);
    }
    printf("Snappy: %ld bytes -> %zu bytes, %d iters\n",
           fsize, compressed.size(), iters);
    return 0;
}
EOF

${CXX} -O2 -static --sysroot=${SYSROOT} \
    -I snappy -I snappy/build_arm \
    snappy_bench.cc snappy/build_arm/libsnappy.a \
    -o "${BIN_DIR}/snappy_bench"
file "${BIN_DIR}/snappy_bench"
echo "-> snappy_bench OK"

# Generate 3 inputs
python3 -c "
import random, string
html = '<html><body>\n'
for i in range(2000):
    tag = random.choice(['div','span','p','a','table','tr','td'])
    text = ''.join(random.choices(string.ascii_lowercase + ' ', k=80))
    html += f'<{tag} class=\"item-{i}\">{text}</{tag}>\n'
html += '</body></html>'
with open('${INPUT_DIR}/snappy_html.dat', 'w') as f:
    f.write(html)
print(f'  snappy_html.dat: {len(html)} bytes')
"

python3 -c "
import struct, random
data = bytearray()
for i in range(50000):
    data += struct.pack('<BHI', random.randint(8,120), random.randint(0,65535), random.randint(0,2**32-1))
    data += bytes(random.choices(range(32,127), k=random.randint(4,20)))
with open('${INPUT_DIR}/snappy_proto.dat', 'wb') as f:
    f.write(data)
print(f'  snappy_proto.dat: {len(data)} bytes')
"

python3 -c "
import os
data = os.urandom(500000)
with open('${INPUT_DIR}/snappy_random.dat', 'wb') as f:
    f.write(data)
print(f'  snappy_random.dat: {len(data)} bytes')
"
echo "-> 3 snappy inputs generated"

# ============================================================
# 2. re2 (needs abseil-cpp first)
# ============================================================
echo ""
echo "============================================"
echo " [2/4] re2 (building abseil-cpp first)"
echo "============================================"

# --- Build abseil-cpp ---
if [ ! -d abseil-cpp ]; then
    git clone --depth 1 https://github.com/abseil/abseil-cpp.git
fi
cd abseil-cpp
rm -rf build_arm && mkdir build_arm && cd build_arm
cmake .. \
    -DCMAKE_TOOLCHAIN_FILE="${TOOLCHAIN_FILE}" \
    -DCMAKE_CXX_FLAGS="-O2" \
    -DCMAKE_EXE_LINKER_FLAGS="-static" \
    -DCMAKE_INSTALL_PREFIX="${WORK_DIR}/arm_install" \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DBUILD_SHARED_LIBS=OFF \
    -DBUILD_TESTING=OFF \
    -DABSL_BUILD_TESTING=OFF
make -j${NPROC} 2>&1 | tail -5
make install 2>&1 | tail -3
cd "$WORK_DIR"
echo "-> abseil-cpp installed to arm_install/"

# --- Build re2 ---
if [ ! -d re2 ]; then
    git clone --depth 1 https://github.com/google/re2.git
fi
cd re2
rm -rf build_arm && mkdir build_arm && cd build_arm
cmake .. \
    -DCMAKE_TOOLCHAIN_FILE="${TOOLCHAIN_FILE}" \
    -DCMAKE_CXX_FLAGS="-O2" \
    -DCMAKE_EXE_LINKER_FLAGS="-static" \
    -DCMAKE_PREFIX_PATH="${WORK_DIR}/arm_install" \
    -DRE2_BUILD_TESTING=OFF \
    -DBUILD_SHARED_LIBS=OFF
make -j${NPROC} 2>&1 | tail -5
cd "$WORK_DIR"

# --- Write re2 benchmark driver ---
cat > re2_bench.cc << 'EOF'
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include "re2/re2.h"

std::string gen_text(int size) {
    std::string text;
    text.reserve(size);
    unsigned int seed = 42;
    for (int i = 0; i < size; i++) {
        seed = seed * 1103515245 + 12345;
        char c = 'a' + (seed >> 16) % 26;
        if ((seed >> 8) % 20 == 0) c = ' ';
        if ((seed >> 12) % 100 == 0) c = '\n';
        text.push_back(c);
    }
    for (int i = 0; i < size; i += size / 50) {
        if (i + 20 < size)
            text.replace(i, 10, "foo123bar ");
    }
    return text;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <literal|alternation|backtrack> [iters]\n", argv[0]);
        return 1;
    }
    const char* mode = argv[1];
    int iters = (argc > 2) ? atoi(argv[2]) : 2000;

    std::string text = gen_text(200000);
    std::string pattern;

    if (strcmp(mode, "literal") == 0) {
        pattern = "foo123bar";
    } else if (strcmp(mode, "alternation") == 0) {
        pattern = "(foo|bar|baz|qux)(123|456|789)(abc|def|ghi)";
    } else if (strcmp(mode, "backtrack") == 0) {
        pattern = "([a-z]+)([0-9]+)([a-z]+)\\s";
    } else {
        fprintf(stderr, "Unknown mode: %s\n", mode);
        return 1;
    }

    RE2 re(pattern);
    if (!re.ok()) {
        fprintf(stderr, "Bad pattern: %s\n", re.error().c_str());
        return 1;
    }

    int total_matches = 0;
    for (int i = 0; i < iters; i++) {
        re2::StringPiece input(text);
        std::string match;
        while (RE2::FindAndConsume(&input, re, &match)) {
            total_matches++;
        }
    }
    printf("re2 [%s]: %d matches, %d iters\n", mode, total_matches, iters);
    return 0;
}
EOF

# --- Link re2_bench with libre2 + all absl libs ---
# Use --start-group/--end-group to handle circular deps in abseil
ABSL_LIBS=$(find "${WORK_DIR}/arm_install/lib" -name "libabsl_*.a" | tr '\n' ' ')

${CXX} -O2 -static --sysroot=${SYSROOT} \
    -I re2 -I "${WORK_DIR}/arm_install/include" \
    re2_bench.cc \
    re2/build_arm/libre2.a \
    -Wl,--start-group ${ABSL_LIBS} -Wl,--end-group \
    -lpthread \
    -o "${BIN_DIR}/re2_bench"
file "${BIN_DIR}/re2_bench"
echo "-> re2_bench OK"

# ============================================================
# 3. LevelDB
# ============================================================
echo ""
echo "============================================"
echo " [3/4] LevelDB"
echo "============================================"
if [ ! -d leveldb ]; then
    git clone --depth 1 https://github.com/google/leveldb.git
    cd leveldb && git submodule update --init && cd ..
fi

# Build libleveldb.a only (skip db_bench which needs Google Benchmark)
cd leveldb
rm -rf build_arm && mkdir build_arm && cd build_arm
cmake .. \
    -DCMAKE_TOOLCHAIN_FILE="${TOOLCHAIN_FILE}" \
    -DCMAKE_CXX_FLAGS="-O2" \
    -DCMAKE_EXE_LINKER_FLAGS="-static" \
    -DLEVELDB_BUILD_TESTS=OFF \
    -DLEVELDB_BUILD_BENCHMARKS=OFF \
    -DBUILD_SHARED_LIBS=OFF
make -j${NPROC} 2>&1 | tail -5
cd "$WORK_DIR"

# Write standalone leveldb benchmark driver
cat > leveldb_bench.cc << 'EOF'
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include "leveldb/db.h"
#include "leveldb/write_batch.h"

// Simple LCG random
static unsigned int rng_state = 12345;
unsigned int rng_next() {
    rng_state = rng_state * 1103515245 + 12345;
    return rng_state;
}

std::string make_key(int k) {
    char buf[32];
    snprintf(buf, sizeof(buf), "key%010d", k);
    return std::string(buf);
}

std::string make_value(int size) {
    std::string v(size, 'x');
    for (int i = 0; i < size; i++)
        v[i] = 'a' + (rng_next() % 26);
    return v;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <fillrandom|readrandom|readseq> <num_ops> [db_path]\n", argv[0]);
        return 1;
    }
    const char* mode = argv[1];
    int num = atoi(argv[2]);
    const char* db_path = (argc > 3) ? argv[3] : "/tmp/leveldb_bench";

    leveldb::DB* db;
    leveldb::Options options;
    options.create_if_missing = true;
    // Disable compression for more predictable behavior
    options.compression = leveldb::kNoCompression;

    leveldb::Status status = leveldb::DB::Open(options, db_path, &db);
    if (!status.ok()) {
        fprintf(stderr, "Open failed: %s\n", status.ToString().c_str());
        return 1;
    }

    leveldb::WriteOptions wopts;
    leveldb::ReadOptions ropts;
    std::string value;
    int found = 0;

    if (strcmp(mode, "fillrandom") == 0) {
        for (int i = 0; i < num; i++) {
            int k = rng_next() % num;
            db->Put(wopts, make_key(k), make_value(100));
        }
        printf("leveldb fillrandom: %d ops\n", num);

    } else if (strcmp(mode, "readrandom") == 0) {
        for (int i = 0; i < num; i++) {
            int k = rng_next() % num;
            status = db->Get(ropts, make_key(k), &value);
            if (status.ok()) found++;
        }
        printf("leveldb readrandom: %d ops, %d found\n", num, found);

    } else if (strcmp(mode, "readseq") == 0) {
        leveldb::Iterator* it = db->NewIterator(ropts);
        int count = 0;
        for (it->SeekToFirst(); it->Valid() && count < num; it->Next()) {
            // Touch key and value to prevent optimization
            volatile size_t len = it->key().size() + it->value().size();
            (void)len;
            count++;
        }
        delete it;
        printf("leveldb readseq: %d entries scanned\n", count);

    } else {
        fprintf(stderr, "Unknown mode: %s\n", mode);
        delete db;
        return 1;
    }

    delete db;
    return 0;
}
EOF

${CXX} -O2 -static --sysroot=${SYSROOT} \
    -I leveldb/include \
    leveldb_bench.cc \
    leveldb/build_arm/libleveldb.a \
    -lpthread \
    -o "${BIN_DIR}/leveldb_bench"
file "${BIN_DIR}/leveldb_bench"
echo "-> leveldb_bench OK"

# ============================================================
# 4. zstd
# ============================================================
echo ""
echo "============================================"
echo " [4/4] zstd"
echo "============================================"
if [ ! -d zstd ]; then
    git clone --depth 1 https://github.com/facebook/zstd.git
fi

cd zstd
make clean 2>/dev/null || true

# Build libzstd.a first
CC="${CC}" AR="${ARM_TOOLCHAIN}/bin/aarch64-none-linux-gnu-ar" \
    CFLAGS="-O2 --sysroot=${SYSROOT}" \
    make -C lib libzstd.a -j${NPROC} 2>&1 | tail -3

# Build fullbench (pure in-memory benchmark)
CC="${CC}" AR="${ARM_TOOLCHAIN}/bin/aarch64-none-linux-gnu-ar" \
    CFLAGS="-O2 --sysroot=${SYSROOT}" \
    LDFLAGS="-static --sysroot=${SYSROOT}" \
    make -C tests fullbench -j${NPROC} 2>&1 | tail -5

if [ -f tests/fullbench ]; then
    cp tests/fullbench "${BIN_DIR}/zstd_fullbench"
    file "${BIN_DIR}/zstd_fullbench"
    echo "-> zstd_fullbench OK"
fi
cd "$WORK_DIR"

# Build standalone zstd driver for level comparison
cat > zstd_bench.c << 'EOF'
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define ZSTD_STATIC_LINKING_ONLY
#include "zstd.h"

int main(int argc, char* argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <level 1-19> <input_file> [iters]\n", argv[0]);
        return 1;
    }
    int level = atoi(argv[1]);
    int iters = (argc > 3) ? atoi(argv[3]) : 200;

    FILE* f = fopen(argv[2], "rb");
    if (!f) { perror("fopen"); return 1; }
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);
    char* src = (char*)malloc(fsize);
    fread(src, 1, fsize, f);
    fclose(f);

    size_t cbound = ZSTD_compressBound(fsize);
    char* dst = (char*)malloc(cbound);
    char* dec = (char*)malloc(fsize);

    for (int i = 0; i < iters; i++) {
        size_t csize = ZSTD_compress(dst, cbound, src, fsize, level);
        if (ZSTD_isError(csize)) {
            fprintf(stderr, "Compress error: %s\n", ZSTD_getErrorName(csize));
            return 1;
        }
        size_t dsize = ZSTD_decompress(dec, fsize, dst, csize);
        if (ZSTD_isError(dsize)) {
            fprintf(stderr, "Decompress error: %s\n", ZSTD_getErrorName(dsize));
            return 1;
        }
    }
    printf("zstd level %d: %ld -> %zu bytes, %d iters\n",
           level, fsize, ZSTD_compress(dst, cbound, src, fsize, level), iters);
    free(src); free(dst); free(dec);
    return 0;
}
EOF

${CC} -O2 -static --sysroot=${SYSROOT} \
    -I zstd/lib -I zstd/lib/common \
    zstd_bench.c zstd/lib/libzstd.a \
    -o "${BIN_DIR}/zstd_bench"
file "${BIN_DIR}/zstd_bench"
echo "-> zstd_bench OK"

# Generate zstd test input
python3 -c "
import os, random, string
data = bytearray()
for _ in range(100):
    text = ''.join(random.choices(string.ascii_lowercase + ' \n', k=2000))
    data += text.encode()
    data += os.urandom(500)
    pattern = os.urandom(20)
    data += pattern * 50
with open('${INPUT_DIR}/zstd_mixed.dat', 'wb') as f:
    f.write(data)
print(f'  zstd_mixed.dat: {len(data)} bytes')
"
echo "-> zstd input generated"

# ============================================================
# Summary
# ============================================================
echo ""
echo "============================================"
echo " Build Summary"
echo "============================================"
echo ""
echo "Binaries:"
for bin in "${BIN_DIR}"/*; do
    info=$(file "$bin" 2>/dev/null | grep -oE 'ARM aarch64|statically linked|ELF 64-bit' | tr '\n' ', ')
    printf "  %-25s %s\n" "$(basename $bin)" "$info"
done

echo ""
echo "Inputs:"
ls -lh "${INPUT_DIR}/"

# ============================================================
# gem5 SE smoke test commands (1M instructions each)
# ============================================================
echo ""
echo "============================================"
echo " gem5 SE smoke test commands"
echo "============================================"

GEM5="${GEM5_DIR}/build/ARM/gem5.opt"
SE="${GEM5_DIR}/configs/example/se.py"

cat << CMDS

# === Snappy × 3 traces ===
${GEM5} ${SE} -c ${BIN_DIR}/snappy_bench -o "${INPUT_DIR}/snappy_html.dat 500" --cpu-type=O3CPU --caches --l2cache -I 1000000
${GEM5} ${SE} -c ${BIN_DIR}/snappy_bench -o "${INPUT_DIR}/snappy_proto.dat 500" --cpu-type=O3CPU --caches --l2cache -I 1000000
${GEM5} ${SE} -c ${BIN_DIR}/snappy_bench -o "${INPUT_DIR}/snappy_random.dat 500" --cpu-type=O3CPU --caches --l2cache -I 1000000

# === re2 × 3 traces ===
${GEM5} ${SE} -c ${BIN_DIR}/re2_bench -o "literal 2000" --cpu-type=O3CPU --caches --l2cache -I 1000000
${GEM5} ${SE} -c ${BIN_DIR}/re2_bench -o "alternation 2000" --cpu-type=O3CPU --caches --l2cache -I 1000000
${GEM5} ${SE} -c ${BIN_DIR}/re2_bench -o "backtrack 1500" --cpu-type=O3CPU --caches --l2cache -I 1000000

# === LevelDB × 3 traces ===
# NOTE: needs file I/O, may fail in SE mode
mkdir -p ./tmp_leveldb
${GEM5} ${SE} -c ${BIN_DIR}/leveldb_bench -o "fillrandom 5000 ./tmp_leveldb/t1" --cpu-type=O3CPU --caches --l2cache -I 1000000
${GEM5} ${SE} -c ${BIN_DIR}/leveldb_bench -o "readrandom 5000 ./tmp_leveldb/t1" --cpu-type=O3CPU --caches --l2cache -I 1000000
${GEM5} ${SE} -c ${BIN_DIR}/leveldb_bench -o "readseq 5000 ./tmp_leveldb/t1" --cpu-type=O3CPU --caches --l2cache -I 1000000

# === zstd × 3 traces (different compression levels) ===
${GEM5} ${SE} -c ${BIN_DIR}/zstd_bench -o "1 ${INPUT_DIR}/zstd_mixed.dat 200" --cpu-type=O3CPU --caches --l2cache -I 1000000
${GEM5} ${SE} -c ${BIN_DIR}/zstd_bench -o "9 ${INPUT_DIR}/zstd_mixed.dat 200" --cpu-type=O3CPU --caches --l2cache -I 1000000
${GEM5} ${SE} -c ${BIN_DIR}/zstd_bench -o "19 ${INPUT_DIR}/zstd_mixed.dat 200" --cpu-type=O3CPU --caches --l2cache -I 1000000

CMDS

echo ""
echo "============================================"
echo " Instruction count calibration"
echo "============================================"
echo "After smoke test, calibrate each trace:"
echo "  1. Run with -I 0, let it finish"
echo "  2. grep sim_insts m5out/stats.txt"
echo "  3. Adjust iters so total ~ 40M instructions"
echo "     (400 windows × 100K/window)"
echo ""
echo "Total target: 4 bench × 3 traces × 24k dp = 96k datapoints"
echo ""
echo "Done!"