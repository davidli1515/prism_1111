/*
 * Concorde Trace Analysis
 *
 * Parses gem5 O3 trace augmented by external simulator.
 *
 * Full CSV layout (72 fields):
 *   [0..41]   42 instruction fields (field[41] = storeTick+"START")
 *   [42..69]  28 pipeline stage stats (cumulative)
 *   [70]      Cache_Latency: 2=L1, 20=L2, 200=RAM, 0=non-memory
 *   [71]      BPred_Dir:     1=predicted taken, 0=not-taken / not-branch
 *
 * Stage stats a[0..27] = fields[42..69]:
 *   a[0..3]   decode:   run, idle, blocked, squash
 *   a[4..9]   rename:   run, idle, block, otherstall_reg, otherstall_serialize, squash
 *   a[10..15] iew:      run, idle, block, block_iq, block_lsq, squash
 *   a[16..21] iq:       run, idle, block_depend, block_mem, block_fu, squash
 *   a[22..27] commit:   run, idle, block_fu, block_mem, block_depend, squash
 */

#pragma once
#include <cstdint>
#include <string>
#include <vector>

constexpr int WINDOW_SIZE     = 400;
constexpr int CACHE_LINE_BITS = 6;
constexpr int N_STAGE_STATS   = 28;

constexpr int LAT_ALU   = 3;
constexpr int LAT_STORE = 1;

enum InstrCategory : int8_t { CAT_ALU = 0, CAT_FP = 1, CAT_LS = 2, CAT_OTHER = 3 };
enum BranchType    : int8_t { BR_NONE = 0, BR_DIRECT_COND = 1, BR_DIRECT_UNCOND = 2, BR_INDIRECT = 3 };

struct Instruction {
    int64_t  seq_num;
    uint64_t pc;

    bool    is_load, is_store;
    int64_t data_cache_line;   // addr >> 6, or -1
    int64_t icache_line;       // pc >> 6

    int32_t       exec_latency;
    InstrCategory category;

    bool       is_branch;
    BranchType branch_type;
    bool       is_mispredicted;
    bool       is_taken;

    bool is_nop, is_isb;

    std::vector<int> deps;

    int64_t fetch_tick, issue_tick, complete_tick, commit_tick;
};

struct ConcordeTrace {
    std::vector<Instruction> instrs;
    int    n_windows;
    double branch_mispred_rate;

    // Stage stats from first and last committed instruction (cumulative counters).
    // MSCS = last - first.
    int64_t first_stage_stats[N_STAGE_STATS];
    int64_t last_stage_stats[N_STAGE_STATS];
    bool    has_stage_stats;

    int size() const { return (int)instrs.size(); }
};

ConcordeTrace parse_trace(const std::string& filepath);
void print_summary(const ConcordeTrace& ct);