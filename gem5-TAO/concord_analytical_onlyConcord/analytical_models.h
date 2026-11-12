/*
 * Concorde Analytical Models + Output
 *
 * Feature vector: 3873 dims (Table 3 in paper)
 *   [0..1110]     11 throughput distributions × 101
 *   [1111..1526]  4 stall distributions × 101 + mispred_rate + 11 ROB avg thr
 *   [1527..3849]  23 latency distributions × 101
 *   [3850..3872]  Architecture parameters (23)
 */

#pragma once
#include "trace_analysis.h"
#include <string>

constexpr int N_FEATURES     = 3873;
constexpr int DIST_DIM       = 101;
constexpr int N_ROB_SIZES    = 11;
constexpr int N_ARCH_PARAMS  = 23;
static const int ROB_SIZES[N_ROB_SIZES] = {1,2,4,8,16,32,64,128,256,512,1024};

// ─── 23-dim arch params (from arch_params.npy, float32) ─────
// [0]  fetchWidth      [12] IntALU
// [1]  decodeWidth     [13] IntMultDiv
// [2]  renameWidth     [14] FP_ALU
// [3]  dispatchWidth   [15] FP_MultDiv
// [4]  issueWidth      [16] RdWrPort
// [5]  wbWidth         [17] l1_size (KB)
// [6]  commitWidth     [18] l2_size (KB)
// [7]  numROBEntries   [19] l1d_assoc
// [8]  numIQEntries    [20] l1i_assoc
// [9]  LQEntries       [21] l2_assoc
// [10] SQEntries       [22] bp_type (0=BiMode, 1=TAGE)
// [11] fetchBufferSize

struct ArchConfig {
    float raw[N_ARCH_PARAMS] = {};  // raw params (goes into feature[3850..3872])

    // Derived fields used by analytical models
    int fetch_width;
    int decode_width;
    int rename_width;
    int commit_width;
    int rob_size;
    int load_queue_size;
    int store_queue_size;
    int n_fetch_bufs;
    int alu_issue_width;    // IntALU FU count
    int fp_issue_width;     // FP_ALU FU count
    int ls_issue_width;     // RdWrPort
    int n_ls_pipes;         // RdWrPort (same as ls_issue_width)
    int n_l_pipes;          // 0 (not in params)
    int max_icache_fills;   // default 8 (not in params)
    int l1_size_kb;
    int l2_size_kb;
    int l1d_assoc;
    int l1i_assoc;
    int l2_assoc;
    int branch_predictor;   // 0=BiMode/none, 1=TAGE

    // Load from arch_params.npy (float32, shape (23,))
    bool load(const std::string& path);

    // BP type as string for external simulator
    std::string bp_str() const { return (branch_predictor == 1) ? "tage" : "bimode"; }
};

void compute_features(const ConcordeTrace& ct, const ArchConfig& cfg,
                      double features[N_FEATURES]);

double parse_cpi(const std::string& stats_path);

void compute_mscs(const ConcordeTrace& ct, double mscs[16]);

void write_npy(const std::string& path, const double* data, int size);
void write_npy_2d(const std::string& path, const double* data, int rows, int cols);