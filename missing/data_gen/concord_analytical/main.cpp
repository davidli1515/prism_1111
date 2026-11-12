/*
 * Concorde + GEM Feature Extraction
 *
 * Usage:
 *   ./concord_analytical <dp_dir> --ckpt-dir <ckpt_dir> --concord_warmup <N>
 *                        [--sim-path PATH]
 *
 * Outputs (in dp_dir):
 *   concord_data.npy  (3874,) = [concorde_features(3873), CPI(1)]
 *   gem_data.npy      (4697,) = [concorde_features(3873), gem_pressure(808), MSCS(16)]
 */

#include "trace_analysis.h"
#include "analytical_models.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <string>
#include <fstream>

static std::string SIM_PATH = "/path/to/gem5/external_simulator/simple_trace_analysis";

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
    if (warmup_n <= 0)    { fprintf(stderr, "Error: --concord_warmup required\n"); return 1; }

    // === Step 1: Load arch params ===
    printf("=== Step 1: Load Arch Params ===\n");
    ArchConfig cfg;
    if (!cfg.load(dp_path + "/arch_params.npy")) return 1;

    // === Step 2: Extract warmup ===
    printf("\n=== Step 2: Extract Warmup (%d instructions) ===\n", warmup_n);
    auto t0 = std::chrono::high_resolution_clock::now();
    std::string warmup_tmp = dp_path + "/warmup_trace.tmp";
    int extracted = extract_head(ckpt_dir + "/simpletrace.txt", warmup_tmp, warmup_n);
    if (extracted < 0) return 1;
    printf("  Extracted %d lines\n", extracted);

    // === Step 3: External simulator ===
    printf("\n=== Step 3: External Simulator ===\n");
    std::string o3trace = dp_path + "/o3trace.txt";
    char cmd[4096];
    // Use dp_path as unique checkpoint name to avoid parallel conflicts
    std::string ckpt_name = dp_path + "/extsim_ckpt";

    snprintf(cmd, sizeof(cmd), "%s warmup %s %s %d %d %s",
             SIM_PATH.c_str(), warmup_tmp.c_str(), ckpt_name.c_str(),
             cfg.l1_size_kb, cfg.l2_size_kb, cfg.bp_str().c_str());
    if (run_cmd(cmd) != 0) { fprintf(stderr, "Error: warmup failed\n"); return 1; }

    snprintf(cmd, sizeof(cmd), "%s sim %s %s %d %d %s",
             SIM_PATH.c_str(), o3trace.c_str(), ckpt_name.c_str(),
             cfg.l1_size_kb, cfg.l2_size_kb, cfg.bp_str().c_str());
    if (run_cmd(cmd) != 0) { fprintf(stderr, "Error: sim failed\n"); return 1; }

    // Clean up checkpoint files
    std::remove((ckpt_name + "_cache.bin").c_str());
    std::remove((ckpt_name + "_bp.bin").c_str());

    std::remove(warmup_tmp.c_str());

    // === Step 4: Parse trace ===
    printf("\n=== Step 4: Parse Trace ===\n");
    auto t1 = std::chrono::high_resolution_clock::now();
    ConcordeTrace ct = parse_trace(o3trace);
    auto t2 = std::chrono::high_resolution_clock::now();
    print_summary(ct);
    printf("  Parse: %.1f ms\n", std::chrono::duration<double,std::milli>(t2-t1).count());

    if (ct.n_windows == 0) { fprintf(stderr, "Error: need >= %d instructions\n", WINDOW_SIZE); return 1; }

    // === Step 5: Compute all features ===
    printf("\n=== Step 5: Compute Features ===\n");
    auto t3 = std::chrono::high_resolution_clock::now();

    // Concorde features (3873)
    double concorde[N_FEATURES];
    compute_features(ct, cfg, concorde);

    // GEM D3 pressure features (808)
    double gem_pressure[N_GEM_PRESSURE];
    compute_gem_features(ct, cfg, gem_pressure);

    // MSCS (16)
    double mscs[N_MSCS];
    compute_mscs(ct, mscs);
    printf("  MSCS: [%.4f %.4f %.4f %.4f | %.4f %.4f %.4f %.4f | %.4f %.4f %.4f %.4f | %.4f %.4f %.4f %.4f]\n",
           mscs[0],mscs[1],mscs[2],mscs[3], mscs[4],mscs[5],mscs[6],mscs[7],
           mscs[8],mscs[9],mscs[10],mscs[11], mscs[12],mscs[13],mscs[14],mscs[15]);

    // CPI
    double cpi = parse_cpi(dp_path + "/stats.txt");

    auto t4 = std::chrono::high_resolution_clock::now();
    printf("  Concorde (%d) + GEM pressure (%d) + MSCS (%d): %.1f ms\n",
           N_FEATURES, N_GEM_PRESSURE, N_MSCS,
           std::chrono::duration<double,std::milli>(t4-t3).count());

    // === Step 6: Assemble and save ===
    printf("\n=== Step 6: Save Outputs ===\n");

    // concord_data.npy: [features(3873), CPI(1)] = 3874
    double concord_data[N_CONCORD_DATA];
    std::memcpy(concord_data, concorde, sizeof(double) * N_FEATURES);
    concord_data[N_FEATURES] = cpi;

    std::string concord_path = dp_path + "/concord_data.npy";
    write_npy(concord_path, concord_data, N_CONCORD_DATA);
    printf("  Saved %s  (%d dims, CPI=%.4f)\n", concord_path.c_str(), N_CONCORD_DATA, cpi);

    // gem_data.npy: [features(3873), gem_pressure(808), MSCS(16)] = 4697
    double gem_data[N_GEM_DATA];
    std::memcpy(gem_data,                            concorde,     sizeof(double) * N_FEATURES);
    std::memcpy(gem_data + N_FEATURES,               gem_pressure, sizeof(double) * N_GEM_PRESSURE);
    std::memcpy(gem_data + N_FEATURES + N_GEM_PRESSURE, mscs,      sizeof(double) * N_MSCS);

    std::string gem_path = dp_path + "/gem_data.npy";
    write_npy(gem_path, gem_data, N_GEM_DATA);
    printf("  Saved %s  (%d dims = %d+%d+%d)\n",
           gem_path.c_str(), N_GEM_DATA, N_FEATURES, N_GEM_PRESSURE, N_MSCS);

    // // Cleanup - disabled for debugging
    // std::remove(o3trace.c_str());
    // printf("  Deleted %s\n", o3trace.c_str());

    double total = std::chrono::duration<double,std::milli>(t4-t0).count();
    printf("\n=== Done (%.1f ms) ===\n", total);
    return 0;
}