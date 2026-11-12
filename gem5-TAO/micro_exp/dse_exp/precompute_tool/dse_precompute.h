/*
 * Complete DSE Precompute: Concorde 3873 + GEM 808 = 4681 dims
 *
 * Precomputes per-window timeseries for ALL swept resources so that
 * the full 4681-dim feature vector can be assembled on-the-fly
 * for any CPU configuration in ~100μs (exact, zero approximation).
 *
 * Memory config (L1i/L2) is fixed at precompute time.
 * CPU params (ROB, queues, widths, pipes, etc.) vary at DSE time.
 */

#pragma once
#include "trace_analysis.h"
#include "analytical_models.h"
#include <string>
#include <vector>
#include <random>

// ── Sweep ranges (step=1 for exact lookup, no interpolation) ──
constexpr int DSE_ROB_START = 72,  DSE_ROB_END = 1024;   // 953 values
constexpr int DSE_LQ_START  = 16,  DSE_LQ_END  = 256;    // 241 values
constexpr int DSE_SQ_START  = 16,  DSE_SQ_END  = 256;    // 241 values
constexpr int DSE_N_ROB = DSE_ROB_END - DSE_ROB_START + 1;  // 953
constexpr int DSE_N_LQ  = DSE_LQ_END  - DSE_LQ_START  + 1;  // 241
constexpr int DSE_N_SQ  = DSE_SQ_END  - DSE_SQ_START  + 1;  // 241
constexpr int DSE_FW_START = 4, DSE_FW_END = 12;
constexpr int DSE_N_FW  = DSE_FW_END - DSE_FW_START + 1;    // 9
constexpr int DSE_NB_START = 1, DSE_NB_END = 8;             // fetch bufs
constexpr int DSE_DW_START = 4, DSE_DW_END = 12;            // decode width (for fetch bufs)
constexpr int DSE_N_NB = DSE_NB_END - DSE_NB_START + 1;     // 8
constexpr int DSE_N_DW = DSE_DW_END - DSE_DW_START + 1;     // 9
constexpr int DSE_N_FB = DSE_N_NB * DSE_N_DW;               // 72

struct DsePrecomputed {
    int W = 0;  // n_windows (typically 250 for 100K instructions)

    // ── Section A: Swept resource throughput (per-window) ──
    // Index by [param_value - START][window]
    std::vector<std::vector<double>> rob_thr;      // [DSE_N_ROB][W]
    std::vector<std::vector<double>> lq_thr;       // [DSE_N_LQ][W]
    std::vector<std::vector<double>> sq_thr;       // [DSE_N_SQ][W]
    std::vector<std::vector<double>> fetchw_thr;   // [DSE_N_FW][W]
    std::vector<std::vector<double>> fetchbuf_thr; // [DSE_N_FB][W] row-major: [nb_idx*DSE_N_DW + dw_idx][w]

    // ── Section B: Per-window instruction counts (for bandwidth/pipes on-the-fly) ──
    std::vector<int> n_alu, n_fp, n_ls;   // per-window category counts
    std::vector<int> n_load, n_store;     // per-window for pipes model

    // ── Section C: Fixed throughput (per memory config) ──
    std::vector<double> icache_thr;       // [W]

    // ── Section D: Program-dependent features (architecture-independent CDFs) ──
    double auxiliary[416];                // ISB dist(101), branch dists(3×101), mispred(1), ROB avg_thr(11)
    double exec_dist[DIST_DIM];          // [101]
    double iss_dist[N_ROB_SIZES][DIST_DIM];  // [11][101]
    double com_dist[N_ROB_SIZES][DIST_DIM];  // [11][101]
    double rob_avg_thr[N_ROB_SIZES];     // scalar avg throughput at 11 Concorde ROB sizes

    // ── Section E: GEM-specific ──
    std::vector<double> T_crit;          // [W] critical path span per window

    // Helper: flat size for serialization
    int flat_size() const;
};

// Precompute everything for one slice.
void dse_precompute(const ConcordeTrace& ct, const ArchConfig& icache_cfg,
                    DsePrecomputed& out);

// Serialize / deserialize
void dse_save(const std::string& path, const DsePrecomputed& data);
bool dse_load(const std::string& path, DsePrecomputed& data);

// Reconstruct full 4681-dim feature vector for a target architecture.
// Output: features[0:3873] = Concorde, features[3873:4681] = GEM D3
// Exact match with compute_features() + compute_gem_features().
constexpr int DSE_TOTAL_FEATURES = N_FEATURES + N_GEM_PRESSURE;  // 4681
void dse_infer(const DsePrecomputed& pre, const ArchConfig& cfg,
               double features[DSE_TOTAL_FEATURES]);

// Random arch for verification
ArchConfig random_arch_cpu_only(const ArchConfig& base, std::mt19937& rng);
