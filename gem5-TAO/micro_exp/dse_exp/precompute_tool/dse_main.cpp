/*
 * DSE Precompute + Verify: full 4681-dim (Concorde 3873 + GEM 808)
 *
 * Usage:
 *   ./dse_precompute <trace_dir> <output_dir> [options]
 *
 * Each slice needs only: o3trace.txt
 *
 * Compile:
 *   cat dse_precompute_impl.cpp >> analytical_models.cpp
 *   g++ -O2 -std=c++17 -o dse_precompute dse_main.cpp analytical_models.cpp trace_analysis.cpp
 */

#include "trace_analysis.h"
#include "analytical_models.h"
#include "dse_precompute.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <string>
#include <sys/stat.h>
#include <random>
#include <cmath>

static bool file_exists(const std::string& p) {
    struct stat st; return stat(p.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}
static void mkdir_p(const std::string& p) {
    std::string cmd = "mkdir -p \"" + p + "\""; system(cmd.c_str());
}

static ArchConfig make_icache_config(int l1i, int l2, int l1i_a, int l2_a) {
    ArchConfig c{}; std::memset(&c, 0, sizeof(c));
    c.l1_size_kb = l1i; c.l2_size_kb = l2; c.l1i_assoc = l1i_a; c.l1d_assoc = 4; c.l2_assoc = l2_a;
    c.max_icache_fills = 8;
    c.fetch_width=c.decode_width=c.rename_width=8;
    c.dispatch_width=c.issue_width=c.commit_width=8;
    c.rob_size=256; c.iq_entries=64; c.load_queue_size=32; c.store_queue_size=32;
    c.n_fetch_bufs=4; c.alu_issue_width=c.fp_issue_width=c.ls_issue_width=4;
    c.n_ls_pipes=2; c.n_l_pipes=0; c.branch_predictor=1;
    c.raw[0]=c.raw[1]=c.raw[2]=c.raw[3]=c.raw[4]=8.f; c.raw[5]=8.f; c.raw[6]=8.f;
    c.raw[7]=256.f; c.raw[8]=64.f; c.raw[9]=32.f; c.raw[10]=32.f; c.raw[11]=4.f;
    c.raw[12]=4.f; c.raw[13]=4.f; c.raw[14]=4.f; c.raw[15]=4.f; c.raw[16]=4.f;
    c.raw[17]=(float)l1i; c.raw[18]=(float)l2; c.raw[19]=4.f;
    c.raw[20]=(float)l1i_a; c.raw[21]=(float)l2_a; c.raw[22]=1.f;
    return c;
}

static int process_slice(const std::string& trace_dir, const std::string& out_dir,
                         int sid, const ArchConfig& icfg, int n_verify) {
    char buf[32]; snprintf(buf, sizeof(buf), "slice_%d", sid);
    std::string sname = buf;
    std::string trace_path = trace_dir + "/" + sname + "/o3trace.txt";
    std::string out_path   = out_dir   + "/" + sname;

    if (!file_exists(trace_path)) {
        fprintf(stderr, "[%s] SKIP: trace not found\n", sname.c_str());
        return -1;
    }

    auto t0 = std::chrono::high_resolution_clock::now();

    // Parse trace
    ConcordeTrace ct = parse_trace(trace_path);
    if (ct.n_windows == 0) {
        fprintf(stderr, "[%s] ERROR: empty trace\n", sname.c_str());
        return -1;
    }

    auto t1 = std::chrono::high_resolution_clock::now();

    // Precompute
    DsePrecomputed pre;
    dse_precompute(ct, icfg, pre);

    auto t2 = std::chrono::high_resolution_clock::now();

    // Save
    mkdir_p(out_path);
    dse_save(out_path + "/dse_precomputed.npy", pre);

    double parse_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double precomp_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
    printf("[%s] parse=%.0fms precompute=%.0fms windows=%d\n",
           sname.c_str(), parse_ms, precomp_ms, pre.W);

    // Verify
    if (n_verify > 0) {
        std::mt19937 rng(42 + sid);
        double max_err_all = 0;

        for (int v = 0; v < n_verify; v++) {
            ArchConfig test = random_arch_cpu_only(icfg, rng);

            // Direct computation from trace
            double direct_c[N_FEATURES], direct_g[N_GEM_PRESSURE];
            compute_features(ct, test, direct_c);
            compute_gem_features(ct, test, direct_g);

            // On-the-fly from precomputed
            double onthefly[DSE_TOTAL_FEATURES];
            dse_infer(pre, test, onthefly);

            // Compare Concorde 3873
            double max_c = 0;
            for (int i = 0; i < N_FEATURES; i++) {
                double e = std::abs(direct_c[i] - onthefly[i]);
                if (e > max_c) max_c = e;
            }

            // Compare GEM 808
            double max_g = 0;
            for (int i = 0; i < N_GEM_PRESSURE; i++) {
                double e = std::abs(direct_g[i] - onthefly[N_FEATURES + i]);
                if (e > max_g) max_g = e;
            }

            double max_err = std::max(max_c, max_g);
            if (max_err > max_err_all) max_err_all = max_err;

            if (max_err > 0.001) {
                printf("  [v%d] ROB=%d Wm=%d concorde_err=%.6f gem_err=%.6f\n",
                       v, test.rob_size, test.w_min(), max_c, max_g);
            }
        }

        printf("[%s] verify: %d tests, max_err=%.9f %s\n",
               sname.c_str(), n_verify, max_err_all,
               max_err_all < 0.001 ? "EXACT" : "CHECK");
    }

    return 0;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        fprintf(stderr,
            "Usage: %s <trace_dir> <output_dir> [options]\n"
            "  --verify N          Verify N random configs per slice\n"
            "  --slice-range S E   (default: 0 399)\n"
            "  --l1i KB            L1i cache size (default: 64)\n"
            "  --l2 KB             L2 cache size (default: 1024)\n"
            "  --l1i-assoc N       (default: 4)\n"
            "  --l2-assoc N        (default: 16)\n", argv[0]);
        return 1;
    }

    std::string trace_dir = argv[1], out_dir = argv[2];
    int n_verify = 0, s0 = 0, s1 = 399;
    int l1i = 64, l2 = 1024, l1i_a = 4, l2_a = 16;

    for (int i = 3; i < argc; i++) {
        if      (!strcmp(argv[i],"--verify")     && i+1<argc) n_verify = atoi(argv[++i]);
        else if (!strcmp(argv[i],"--slice-range") && i+2<argc) { s0=atoi(argv[++i]); s1=atoi(argv[++i]); }
        else if (!strcmp(argv[i],"--l1i")         && i+1<argc) l1i   = atoi(argv[++i]);
        else if (!strcmp(argv[i],"--l2")          && i+1<argc) l2    = atoi(argv[++i]);
        else if (!strcmp(argv[i],"--l1i-assoc")   && i+1<argc) l1i_a = atoi(argv[++i]);
        else if (!strcmp(argv[i],"--l2-assoc")    && i+1<argc) l2_a  = atoi(argv[++i]);
    }

    while (!trace_dir.empty() && trace_dir.back()=='/') trace_dir.pop_back();
    while (!out_dir.empty()   && out_dir.back()=='/') out_dir.pop_back();

    ArchConfig icfg = make_icache_config(l1i, l2, l1i_a, l2_a);

    printf("=== DSE Precompute (4681-dim: Concorde 3873 + GEM 808) ===\n");
    printf("  Traces: %s\n  Output: %s\n  Slices: %d-%d\n",
           trace_dir.c_str(), out_dir.c_str(), s0, s1);
    printf("  Icache: L1i=%dKB(a=%d) L2=%dKB(a=%d)\n", l1i, l1i_a, l2, l2_a);
    printf("  Sweeps: ROB[%d-%d] LQ[%d-%d] SQ[%d-%d] FW[%d-%d] FB[%d×%d]\n",
           DSE_ROB_START, DSE_ROB_END, DSE_LQ_START, DSE_LQ_END,
           DSE_SQ_START, DSE_SQ_END, DSE_FW_START, DSE_FW_END, DSE_N_NB, DSE_N_DW);
    if (n_verify) printf("  Verify: %d random configs/slice\n", n_verify);
    printf("\n");

    mkdir_p(out_dir);
    auto t = std::chrono::high_resolution_clock::now();
    int ok = 0, skip = 0;
    for (int s = s0; s <= s1; s++) {
        if (process_slice(trace_dir, out_dir, s, icfg, n_verify) == 0) ok++; else skip++;
    }
    double sec = std::chrono::duration<double>(std::chrono::high_resolution_clock::now()-t).count();
    printf("\n=== Done: %d OK, %d skip, %.1fs ===\n", ok, skip, sec);
    return 0;
}
