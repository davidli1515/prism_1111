/*
 * Concorde Feature Extraction
 * g++ -O3 -std=c++17 trace_analysis.cpp analytical_models.cpp main.cpp -o concord_analytical

 * Usage:
 *   ./concord_analytical <dp_dir> --ckpt-dir <ckpt_dir> --concord_warmup <N>
 *                        [--sim-path PATH]
 *
 * Input files:
 *   <dp_dir>/arch_params.npy    — float32 (23,), sampled micro-arch params
 *   <dp_dir>/o3trace.txt        — O3 execution trace
 *   <dp_dir>/stats.txt          — gem5 stats (for CPI)
 *   <ckpt_dir>/simpletrace.txt  — warmup trace (large, only first N lines used)
 *
 * Output files (in dp_dir):
 *   3873_vector.npy  — feature vector
 *   cpi.npy          — CPI
 *   mscs.npy         — 4×4 MSCS
 *
 * Pipeline:
 *   1. Load arch_params.npy → get L1/L2/BP for ext sim + all model params
 *   2. Extract first N lines from simpletrace → warmup ext sim
 *   3. Run ext sim on o3trace (appends Cache_Latency + BPred_Dir)
 *   4. Parse augmented o3trace
 *   5. Compute 3873-dim feature vector (params drive all analytical models)
 *   6. Save outputs; delete o3trace.txt
 */

#include "trace_analysis.h"
#include "analytical_models.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <string>
#include <fstream>

static std::string SIM_PATH = "/scratch/gem5-TAO/external_simulator/simple_trace_analysis";

static int run_cmd(const char* cmd) {
    printf("  $ %s\n", cmd);
    return system(cmd);
}

static int extract_head(const std::string& src, const std::string& dst, int n) {
    std::ifstream fin(src);
    if (!fin.is_open()) { fprintf(stderr, "Error: cannot open %s\n", src.c_str()); return -1; }
    std::ofstream fout(dst);
    if (!fout.is_open()) { fprintf(stderr, "Error: cannot write %s\n", dst.c_str()); return -1; }
    std::string line; int c = 0;
    while (c < n && std::getline(fin, line)) { fout << line << '\n'; c++; }
    return c;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        fprintf(stderr,
            "Usage: %s <dp_dir> --ckpt-dir <ckpt_dir> --concord_warmup <N> [--sim-path PATH]\n",
            argv[0]);
        return 1;
    }

    std::string dp_path = argv[1];
    while (!dp_path.empty() && dp_path.back() == '/') dp_path.pop_back();

    std::string ckpt_dir;
    int warmup_n = 0;

    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--ckpt-dir") == 0 && i+1 < argc) {
            ckpt_dir = argv[++i];
            while (!ckpt_dir.empty() && ckpt_dir.back() == '/') ckpt_dir.pop_back();
        }
        else if (strcmp(argv[i], "--sim-path") == 0 && i+1 < argc) SIM_PATH = argv[++i];
        else if (strcmp(argv[i], "--concord_warmup") == 0 && i+1 < argc) warmup_n = atoi(argv[++i]);
    }

    if (ckpt_dir.empty()) { fprintf(stderr, "Error: --ckpt-dir required\n"); return 1; }
    if (warmup_n <= 0)    { fprintf(stderr, "Error: --concord_warmup <N> required (N>0)\n"); return 1; }

    // === Step 1: Load arch params ===
    printf("=== Step 1: Load Arch Params ===\n");
    ArchConfig cfg;
    std::string params_path = dp_path + "/arch_params.npy";
    if (!cfg.load(params_path)) return 1;

    // === Step 2: Extract warmup ===
    printf("\n=== Step 2: Extract Warmup (%d instructions) ===\n", warmup_n);
    auto t0 = std::chrono::high_resolution_clock::now();

    std::string simpletrace = ckpt_dir + "/simpletrace.txt";
    std::string warmup_tmp  = dp_path + "/warmup_trace.tmp";
    int extracted = extract_head(simpletrace, warmup_tmp, warmup_n);
    if (extracted < 0) return 1;
    printf("  Extracted %d lines\n", extracted);

    // === Step 3: Run external simulator ===
    printf("\n=== Step 3: External Simulator ===\n");

    std::string o3trace    = dp_path + "/o3trace.txt";
    std::string stats_file = dp_path + "/stats.txt";
    char cmd[4096];

    // Warmup with params from arch_params.npy
    snprintf(cmd, sizeof(cmd), "%s warmup %s test %d %d %s",
             SIM_PATH.c_str(), warmup_tmp.c_str(),
             cfg.l1_size_kb, cfg.l2_size_kb, cfg.bp_str().c_str());
    int ret = run_cmd(cmd);
    if (ret != 0) { fprintf(stderr, "Error: warmup failed (%d)\n", ret); return 1; }

    // Sim on o3trace
    snprintf(cmd, sizeof(cmd), "%s sim %s test %d %d %s",
             SIM_PATH.c_str(), o3trace.c_str(),
             cfg.l1_size_kb, cfg.l2_size_kb, cfg.bp_str().c_str());
    ret = run_cmd(cmd);
    if (ret != 0) { fprintf(stderr, "Error: sim failed (%d)\n", ret); return 1; }

    auto t1 = std::chrono::high_resolution_clock::now();
    printf("  Ext sim: %.1f ms\n", std::chrono::duration<double,std::milli>(t1-t0).count());

    std::remove(warmup_tmp.c_str());

    // === Step 4: Parse trace ===
    printf("\n=== Step 4: Parse Trace ===\n");
    auto t2 = std::chrono::high_resolution_clock::now();
    ConcordeTrace ct = parse_trace(o3trace);
    auto t3 = std::chrono::high_resolution_clock::now();
    print_summary(ct);
    printf("  Parse: %.1f ms\n", std::chrono::duration<double,std::milli>(t3-t2).count());

    if (ct.n_windows == 0) { fprintf(stderr, "Error: need >= %d instructions\n", WINDOW_SIZE); return 1; }

    // === Step 5: Compute features ===
    printf("\n=== Step 5: Compute Features ===\n");
    double features[N_FEATURES];
    auto t4 = std::chrono::high_resolution_clock::now();
    compute_features(ct, cfg, features);
    auto t5 = std::chrono::high_resolution_clock::now();
    printf("  Features (%d dims): %.1f ms\n", N_FEATURES,
           std::chrono::duration<double,std::milli>(t5-t4).count());

    // === Step 6: Save outputs ===
    printf("\n=== Step 6: Save Outputs ===\n");

    std::string vec_path = dp_path + "/3873_vector.npy";
    write_npy(vec_path, features, N_FEATURES);
    printf("  Saved %s\n", vec_path.c_str());

    double cpi = parse_cpi(stats_file);
    std::string cpi_path = dp_path + "/cpi.npy";
    write_npy(cpi_path, &cpi, 1);
    printf("  Saved %s  (CPI=%.4f)\n", cpi_path.c_str(), cpi);

    double mscs[16];
    compute_mscs(ct, mscs);
    std::string mscs_path = dp_path + "/mscs.npy";
    write_npy_2d(mscs_path, mscs, 4, 4);
    printf("  Saved %s\n", mscs_path.c_str());

    // Cleanup
    std::remove(o3trace.c_str());
    printf("  Deleted %s\n", o3trace.c_str());

    double total_ms = std::chrono::duration<double,std::milli>(t5-t0).count();
    printf("\n=== Done (%.1f ms) ===\n", total_ms);
    return 0;
}