/*
 * Concorde Trace Analysis - Implementation
 */

#include "trace_analysis.h"
#include <fstream>
#include <unordered_map>
#include <algorithm>
#include <cstdio>
#include <cstring>

// ═══════════════════ CSV Parsing ═══════════════════

static std::vector<std::string> parse_full_line(const std::string& line) {
    std::vector<std::string> fields;
    std::string field;
    bool in_quotes = false;

    for (char c : line) {
        if (c == '"') { in_quotes = !in_quotes; }
        else if (c == ',' && !in_quotes) { fields.push_back(field); field.clear(); }
        else { field += c; }
    }
    if (!field.empty()) fields.push_back(field);

    // field[41] = "NNNSTARTxxx" → split at START
    if (fields.size() > 41) {
        auto& f41 = fields[41];
        auto pos = f41.find("START");
        if (pos != std::string::npos)
            f41 = f41.substr(0, pos);
    }
    return fields;
}

static inline int64_t to_i64(const std::string& s) {
    if (s.empty()) return 0;
    try { return std::stoll(s); }
    catch (...) { return 0; }
}
static inline int to_int(const std::string& s) {
    if (s.empty()) return 0;
    try { return std::stoi(s); }
    catch (...) {
        // stoi overflow — try stoll then truncate (e.g. large tick in wrong field)
        try { return (int)std::stoll(s); }
        catch (...) { return 0; }
    }
}
static inline bool to_bool(const std::string& s) { return to_int(s) != 0; }

// ═══════════════════ Classification ═══════════════════

static void classify_load_store(const std::string& name, bool is_mem_ref,
                                bool& is_load, bool& is_store) {
    is_load = is_store = false;
    if (!is_mem_ref) return;
    if (name.size() >= 2 && name[0] == 'l' && name[1] == 'd') { is_load = true; return; }
    if (name.size() >= 2 && name[0] == 's' && name[1] == 't') { is_store = true; return; }
    if (name.find("ld") != std::string::npos) { is_load = true; return; }
    if (name.find("st") != std::string::npos) { is_store = true; return; }
    is_load = true;
}

static BranchType classify_branch(bool is_ctrl, bool is_cond, bool is_indirect) {
    if (!is_ctrl)    return BR_NONE;
    if (is_indirect) return BR_INDIRECT;
    if (is_cond)     return BR_DIRECT_COND;
    return BR_DIRECT_UNCOND;
}

static InstrCategory classify_category(const std::string& name,
                                       bool is_load, bool is_store,
                                       bool is_ctrl, bool is_nop) {
    if (is_load || is_store) return CAT_LS;
    if (is_ctrl || is_nop)   return CAT_OTHER;
    if (!name.empty() && name[0] == 'f') return CAT_FP;
    if (name == "isb" || name == "dsb" || name == "dmb" ||
        name == "mrs" || name == "msr" || name == "nop" ||
        name == "wfe" || name == "wfi" || name == "sev" ||
        name == "yield" || name == "clrex")
        return CAT_OTHER;
    return CAT_ALU;
}

// ═══════════════════ Dependency Computation ═══════════════════

static void compute_dependencies(std::vector<Instruction>& instrs,
                                 const std::vector<std::vector<int>>& src_regs,
                                 const std::vector<std::vector<int>>& dst_regs) {
    int n = (int)instrs.size();
    std::unordered_map<int, int>     last_writer;
    std::unordered_map<int64_t, int> last_store;

    for (int i = 0; i < n; i++) {
        std::vector<int> dep_set;
        for (int r : src_regs[i]) {
            auto it = last_writer.find(r);
            if (it != last_writer.end()) dep_set.push_back(it->second);
        }
        if (instrs[i].is_load && instrs[i].data_cache_line >= 0) {
            auto it = last_store.find(instrs[i].data_cache_line);
            if (it != last_store.end()) dep_set.push_back(it->second);
        }
        std::sort(dep_set.begin(), dep_set.end());
        dep_set.erase(std::unique(dep_set.begin(), dep_set.end()), dep_set.end());
        instrs[i].deps = std::move(dep_set);

        for (int r : dst_regs[i]) last_writer[r] = i;
        if (instrs[i].is_store && instrs[i].data_cache_line >= 0)
            last_store[instrs[i].data_cache_line] = i;
    }
}

// ═══════════════════ Main Parser ═══════════════════

ConcordeTrace parse_trace(const std::string& filepath) {
    std::ifstream fin(filepath);
    if (!fin.is_open()) {
        fprintf(stderr, "Error: cannot open %s\n", filepath.c_str());
        return {};
    }

    std::vector<Instruction> instrs;
    std::vector<std::vector<int>> all_src_regs, all_dst_regs;

    int64_t first_stats[N_STAGE_STATS] = {};
    int64_t last_stats[N_STAGE_STATS]  = {};
    bool got_first_stats = false;

    std::string line;
    while (std::getline(fin, line)) {
        if (line.empty()) continue;

        auto f = parse_full_line(line);
        if (f.size() < 72) continue;

        if (to_bool(f[17])) continue;  // skip squashed

        Instruction inst{};
        inst.seq_num = to_i64(f[0]);
        inst.pc      = (uint64_t)to_i64(f[2]);

        std::string name = f[4];
        bool is_mem_ref  = to_bool(f[5]);
        inst.is_nop      = to_bool(f[7]);
        bool is_ctrl     = to_bool(f[9]);
        bool is_indirect = to_bool(f[11]);
        bool is_cond     = to_bool(f[12]);
        bool actual_taken = to_bool(f[15]);
        int num_srcs     = to_int(f[20]);
        int num_dests    = to_int(f[21]);

        int ext_cache_lat = to_int(f[70]);
        int ext_bpred_dir = to_int(f[71]);

        classify_load_store(name, is_mem_ref, inst.is_load, inst.is_store);

        uint64_t eff_addr = (uint64_t)to_i64(f[3]);
        inst.data_cache_line = (inst.is_load || inst.is_store)
                               ? (int64_t)(eff_addr >> CACHE_LINE_BITS) : -1;
        inst.icache_line = (int64_t)(inst.pc >> CACHE_LINE_BITS);

        inst.is_branch   = is_ctrl;
        inst.branch_type = classify_branch(is_ctrl, is_cond, is_indirect);
        inst.category    = classify_category(name, inst.is_load, inst.is_store, is_ctrl, inst.is_nop);
        inst.is_isb      = (name == "isb");

        // Exec latency from ext sim
        if (inst.is_load && ext_cache_lat > 0)
            inst.exec_latency = ext_cache_lat;
        else if (inst.is_store)
            inst.exec_latency = LAT_STORE;
        else
            inst.exec_latency = LAT_ALU;

        // Branch misprediction
        inst.is_taken = actual_taken;
        inst.is_mispredicted = inst.is_branch && ((ext_bpred_dir != 0) != actual_taken);

        // Registers
        std::vector<int> srcs, dsts;
        for (int i = 0; i < std::min(num_srcs, 8); i++)
            srcs.push_back(to_int(f[23 + i]));
        for (int i = 0; i < std::min(num_dests, 6); i++)
            dsts.push_back(to_int(f[31 + i]));

        inst.fetch_tick    = to_i64(f[37]);
        inst.issue_tick    = to_i64(f[38]);
        inst.complete_tick = to_i64(f[39]);
        inst.commit_tick   = to_i64(f[40]);

        // ── Parse stage stats: a[0..27] = fields[42..69] ──
        int64_t stats[N_STAGE_STATS];
        for (int i = 0; i < N_STAGE_STATS; i++)
            stats[i] = to_i64(f[42 + i]);

        if (!got_first_stats) {
            std::memcpy(first_stats, stats, sizeof(first_stats));
            got_first_stats = true;
        }
        std::memcpy(last_stats, stats, sizeof(last_stats));

        instrs.push_back(std::move(inst));
        all_src_regs.push_back(std::move(srcs));
        all_dst_regs.push_back(std::move(dsts));
    }

    compute_dependencies(instrs, all_src_regs, all_dst_regs);

    int n_br = 0, n_mis = 0;
    for (auto& inst : instrs) {
        if (inst.is_branch) { n_br++; if (inst.is_mispredicted) n_mis++; }
    }

    ConcordeTrace ct;
    ct.instrs = std::move(instrs);
    ct.n_windows = ct.size() / WINDOW_SIZE;
    ct.branch_mispred_rate = (n_br > 0) ? (double)n_mis / n_br : 0.0;
    ct.has_stage_stats = got_first_stats;
    std::memcpy(ct.first_stage_stats, first_stats, sizeof(first_stats));
    std::memcpy(ct.last_stage_stats, last_stats, sizeof(last_stats));
    return ct;
}

void print_summary(const ConcordeTrace& ct) {
    int n = ct.size();
    int n_load = 0, n_store = 0, n_br = 0, n_mis = 0;
    int n_alu = 0, n_fp = 0, n_other = 0;
    for (int i = 0; i < n; i++) {
        auto& inst = ct.instrs[i];
        n_load += inst.is_load;  n_store += inst.is_store;
        n_br += inst.is_branch;  n_mis += (inst.is_branch && inst.is_mispredicted);
        if (inst.category == CAT_ALU) n_alu++;
        if (inst.category == CAT_FP)  n_fp++;
        if (inst.category == CAT_OTHER) n_other++;
    }
    printf("  Instructions:  %d  (%d windows x %d)\n", n, ct.n_windows, WINDOW_SIZE);
    printf("  Load / Store:  %d / %d\n", n_load, n_store);
    printf("  Branches:      %d  (mispred rate: %.2f%%)\n", n_br, ct.branch_mispred_rate * 100);
    printf("  ALU / FP / LS / OTHER:  %d / %d / %d / %d\n",
           n_alu, n_fp, n_load + n_store, n_other);
}