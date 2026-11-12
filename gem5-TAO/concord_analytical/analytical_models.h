/*
 * Concorde + GEM Analytical Models
 *
 * Concorde features: 3873 dims
 * GEM pressure features: 808 dims (4 stages × 2 streams × 101)
 * MSCS: 16 dims (4 stages × 4 states)
 *
 * Output files:
 *   concord_data.npy: [concorde_features(3873), CPI(1)] = 3874
 *   gem_data.npy:     [concorde_features(3873), gem_pressure(808), MSCS(16)] = 4697
 */

#pragma once
#include "trace_analysis.h"
#include <string>

constexpr int N_FEATURES      = 3873;
constexpr int N_GEM_PRESSURE  = 808;   // 4 stages × 2 streams × 101
constexpr int N_MSCS          = 16;
constexpr int N_CONCORD_DATA  = 3874;  // 3873 + CPI
constexpr int N_GEM_DATA      = 4697;  // 3873 + 808 + 16

constexpr int DIST_DIM        = 101;
constexpr int N_ROB_SIZES     = 11;
constexpr int N_ARCH_PARAMS   = 23;
static const int ROB_SIZES[N_ROB_SIZES] = {1,2,4,8,16,32,64,128,256,512,1024};

// 23-dim arch params from arch_params.npy (float32):
// [0]  fetchWidth   [12] IntALU
// [1]  decodeWidth  [13] IntMultDiv
// [2]  renameWidth  [14] FP_ALU
// [3]  dispatchWidth[15] FP_MultDiv
// [4]  issueWidth   [16] RdWrPort
// [5]  wbWidth      [17] l1_size (KB)
// [6]  commitWidth  [18] l2_size (KB)
// [7]  numROBEntries[19] l1d_assoc
// [8]  numIQEntries [20] l1i_assoc
// [9]  LQEntries    [21] l2_assoc
// [10] SQEntries    [22] bp_type (0=BiMode, 1=TAGE)
// [11] fetchBufferSize

struct ArchConfig {
    float raw[N_ARCH_PARAMS] = {};

    int fetch_width, decode_width, rename_width, commit_width;
    int dispatch_width, issue_width;           // for GEM queue capacities
    int rob_size, load_queue_size, store_queue_size;
    int n_fetch_bufs;
    int iq_entries;                             // for GEM queue capacities
    int alu_issue_width, fp_issue_width, ls_issue_width;
    int n_ls_pipes, n_l_pipes, max_icache_fills;
    int l1_size_kb, l2_size_kb;
    int l1d_assoc, l1i_assoc, l2_assoc;
    int branch_predictor;

    bool load(const std::string& path);
    std::string bp_str() const { return (branch_predictor == 1) ? "tage" : "bimode"; }

    // GEM: W_min = min of all pipeline stage widths
    int w_min() const;
};

// Concorde 3873-dim features
void compute_features(const ConcordeTrace& ct, const ArchConfig& cfg,
                      double features[N_FEATURES]);

// GEM D3: 808-dim stage-interaction pressure features
void compute_gem_features(const ConcordeTrace& ct, const ArchConfig& cfg,
                          double gem_features[N_GEM_PRESSURE]);

// MSCS from stage stats delta (last - first instruction)
void compute_mscs(const ConcordeTrace& ct, double mscs[N_MSCS]);

// CPI from stats.txt
double parse_cpi(const std::string& stats_path);

// NPY I/O
void write_npy(const std::string& path, const double* data, int size);