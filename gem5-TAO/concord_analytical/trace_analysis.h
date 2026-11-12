/*
 * Concorde Trace Analysis
 *
 * Parses gem5 O3 trace augmented by external simulator.
 * Stage stats (after START) are normalized MSCS proportions (floats, sum=1 per stage).
 * Ext sim appends Cache_Latency and BPred_Dir as the LAST two fields.
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
    int64_t data_cache_line, icache_line;
    int32_t       exec_latency;
    InstrCategory category;
    bool       is_branch;
    BranchType branch_type;
    bool       is_mispredicted, is_taken;
    bool is_nop, is_isb;
    std::vector<int> deps;
    int64_t fetch_tick, issue_tick, complete_tick, commit_tick;
};

struct ConcordeTrace {
    std::vector<Instruction> instrs;
    int    n_windows;
    double branch_mispred_rate;
    double last_stage_stats[N_STAGE_STATS];  // normalized floats from last valid line
    bool   has_stage_stats;
    int size() const { return (int)instrs.size(); }
};

ConcordeTrace parse_trace(const std::string& filepath);
void print_summary(const ConcordeTrace& ct);