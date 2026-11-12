/*
 * Concorde Analytical Models — Implementation
 */

#include "analytical_models.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <unordered_map>
#include <vector>
#include <cassert>
#include <cstdio>
#include <fstream>
#include <sstream>

// ═══════════════════ NPY Reader (float32) ═══════════════════

bool ArchConfig::load(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "Error: cannot open %s\n", path.c_str()); return false; }

    // Read npy header: magic(6) + version(2) + header_len(2)
    char magic[8];
    if (fread(magic, 1, 8, f) != 8 || magic[0] != '\x93' || magic[1] != 'N') {
        fprintf(stderr, "Error: %s is not a valid npy file\n", path.c_str());
        fclose(f); return false;
    }
    uint16_t hlen;
    if (fread(&hlen, 2, 1, f) != 1) { fclose(f); return false; }

    // Skip header string
    fseek(f, hlen, SEEK_CUR);

    // Read 23 float32 values
    if (fread(raw, sizeof(float), N_ARCH_PARAMS, f) != N_ARCH_PARAMS) {
        fprintf(stderr, "Error: failed to read 23 floats from %s\n", path.c_str());
        fclose(f); return false;
    }
    fclose(f);

    // Map raw params to derived fields
    fetch_width      = (int)raw[0];
    decode_width     = (int)raw[1];
    rename_width     = (int)raw[2];
    // raw[3] = dispatchWidth, raw[4] = issueWidth, raw[5] = wbWidth — not directly used by models
    commit_width     = (int)raw[6];
    rob_size         = (int)raw[7];
    // raw[8] = numIQEntries — not directly modeled as throughput
    load_queue_size  = (int)raw[9];
    store_queue_size = (int)raw[10];
    n_fetch_bufs     = (int)raw[11];
    alu_issue_width  = (int)raw[12];  // IntALU FU count
    // raw[13] = IntMultDiv — not separately modeled
    fp_issue_width   = (int)raw[14];  // FP_ALU FU count
    // raw[15] = FP_MultDiv — not separately modeled
    ls_issue_width   = (int)raw[16];  // RdWrPort
    n_ls_pipes       = (int)raw[16];  // same as RdWrPort
    n_l_pipes        = 0;
    max_icache_fills = 8;             // not in params, use default
    l1_size_kb       = (int)raw[17];
    l2_size_kb       = (int)raw[18];
    l1d_assoc        = (int)raw[19];
    l1i_assoc        = (int)raw[20];
    l2_assoc         = (int)raw[21];
    branch_predictor = (int)raw[22];

    printf("  Loaded arch params from %s:\n", path.c_str());
    printf("    fetch=%d decode=%d rename=%d commit=%d ROB=%d LQ=%d SQ=%d\n",
           fetch_width, decode_width, rename_width, commit_width,
           rob_size, load_queue_size, store_queue_size);
    printf("    IntALU=%d FP_ALU=%d RdWrPort=%d fetchBuf=%d\n",
           alu_issue_width, fp_issue_width, ls_issue_width, n_fetch_bufs);
    printf("    L1=%dKB(d_assoc=%d,i_assoc=%d) L2=%dKB(assoc=%d) BP=%s\n",
           l1_size_kb, l1d_assoc, l1i_assoc, l2_size_kb, l2_assoc, bp_str().c_str());
    return true;
}

// ═══════════════════ NPY Writer ═══════════════════

void write_npy(const std::string& path, const double* data, int size) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) { fprintf(stderr, "Error: cannot write %s\n", path.c_str()); return; }
    char header[256];
    int hlen = snprintf(header, sizeof(header),
        "{'descr': '<f8', 'fortran_order': False, 'shape': (%d,), }", size);
    int total = 10 + hlen;
    int padding = 64 - (total % 64); if (padding == 64) padding = 0;
    for (int i = 0; i < padding - 1; i++) header[hlen++] = ' ';
    header[hlen++] = '\n';
    uint16_t hl = (uint16_t)hlen;
    fwrite("\x93NUMPY\x01\x00", 1, 8, f);
    fwrite(&hl, 2, 1, f);
    fwrite(header, 1, hlen, f);
    fwrite(data, sizeof(double), size, f);
    fclose(f);
}

void write_npy_2d(const std::string& path, const double* data, int rows, int cols) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) { fprintf(stderr, "Error: cannot write %s\n", path.c_str()); return; }
    char header[256];
    int hlen = snprintf(header, sizeof(header),
        "{'descr': '<f8', 'fortran_order': False, 'shape': (%d, %d), }", rows, cols);
    int total = 10 + hlen;
    int padding = 64 - (total % 64); if (padding == 64) padding = 0;
    for (int i = 0; i < padding - 1; i++) header[hlen++] = ' ';
    header[hlen++] = '\n';
    uint16_t hl = (uint16_t)hlen;
    fwrite("\x93NUMPY\x01\x00", 1, 8, f);
    fwrite(&hl, 2, 1, f);
    fwrite(header, 1, hlen, f);
    fwrite(data, sizeof(double), rows * cols, f);
    fclose(f);
}

// ═══════════════════ CPI Parser ═══════════════════

double parse_cpi(const std::string& stats_path) {
    std::ifstream fin(stats_path);
    if (!fin.is_open()) {
        fprintf(stderr, "Warning: cannot open %s\n", stats_path.c_str());
        return -1.0;
    }
    std::string line;
    while (std::getline(fin, line)) {
        if (line.find("system.switch_cpus_1.commitStats0.cpi") != std::string::npos) {
            std::istringstream ss(line);
            std::string key; double val;
            ss >> key >> val;
            return val;
        }
    }
    fprintf(stderr, "Warning: CPI not found in %s\n", stats_path.c_str());
    return -1.0;
}

// ═══════════════════ MSCS Computation ═══════════════════

void compute_mscs(const ConcordeTrace& ct, double mscs[16]) {
    std::memset(mscs, 0, sizeof(double) * 16);
    if (!ct.has_stage_stats) return;

    int64_t a[N_STAGE_STATS];
    for (int i = 0; i < N_STAGE_STATS; i++)
        a[i] = ct.last_stage_stats[i] - ct.first_stage_stats[i];

    mscs[0]  = (double)a[0];  mscs[1]  = (double)a[1];
    mscs[2]  = (double)a[2];  mscs[3]  = (double)a[3];
    mscs[4]  = (double)a[4];  mscs[5]  = (double)a[5];
    mscs[6]  = (double)(a[6] + a[7] + a[8]);   mscs[7]  = (double)a[9];
    mscs[8]  = (double)a[16]; mscs[9]  = (double)a[17];
    mscs[10] = (double)(a[18] + a[19] + a[20]); mscs[11] = (double)a[21];
    mscs[12] = (double)a[22]; mscs[13] = (double)a[23];
    mscs[14] = (double)(a[24] + a[25] + a[26]); mscs[15] = (double)a[27];
}

// ═══════════════════ Distribution Encoding ═══════════════════

static void encode_distribution(const double* vals, int N, double out[DIST_DIM]) {
    if (N == 0) { std::fill(out, out + DIST_DIM, 0.0); return; }
    std::vector<double> sorted(vals, vals + N);
    std::sort(sorted.begin(), sorted.end());
    for (int i = 0; i < 50; i++) {
        double p = (N == 1) ? 0.0 : (double)i / 49.0;
        double idx = p * (N - 1);
        int lo = (int)idx, hi = std::min(lo + 1, N - 1);
        out[i] = sorted[lo] * (1.0 - (idx - lo)) + sorted[hi] * (idx - lo);
    }
    double total_w = 0;
    for (int i = 0; i < N; i++) total_w += sorted[i];
    if (total_w > 1e-12) {
        std::vector<double> cum(N);
        cum[0] = sorted[0];
        for (int i = 1; i < N; i++) cum[i] = cum[i-1] + sorted[i];
        for (int i = 0; i < 50; i++) {
            double target = ((double)i / 49.0) * total_w;
            int idx = (int)(std::lower_bound(cum.begin(), cum.end(), target) - cum.begin());
            out[50 + i] = sorted[std::min(idx, N-1)];
        }
    } else {
        std::fill(out + 50, out + 100, 0.0);
    }
    double sum = 0;
    for (int i = 0; i < N; i++) sum += vals[i];
    out[100] = sum / N;
}

// ═══════════════════ Memory State Machine ═══════════════════

struct MemoryModel {
    std::unordered_map<int64_t, std::vector<int32_t>> line_exec_times;
    std::unordered_map<int64_t, int>     access_counter;
    std::unordered_map<int64_t, int64_t> last_req_cycle;
    std::unordered_map<int64_t, int64_t> last_resp_cycle;

    void init(const ConcordeTrace& ct) {
        line_exec_times.clear();
        for (int i = 0; i < ct.size(); i++) {
            auto& inst = ct.instrs[i];
            if (inst.is_load && inst.data_cache_line >= 0)
                line_exec_times[inst.data_cache_line].push_back(inst.exec_latency);
        }
    }
    void reset() { access_counter.clear(); last_req_cycle.clear(); last_resp_cycle.clear(); }
    int64_t resp_cycle(int64_t req_cycle, const Instruction& inst) {
        if (inst.is_load && inst.data_cache_line >= 0) {
            int64_t cl = inst.data_cache_line;
            int64_t& lr = last_req_cycle[cl];
            req_cycle = std::max(req_cycle, lr); lr = req_cycle;
            int& cnt = access_counter[cl];
            auto& times = line_exec_times[cl];
            int32_t et = (cnt < (int)times.size()) ? times[cnt] : inst.exec_latency;
            int64_t resp = std::max(req_cycle + et, last_resp_cycle[cl]);
            last_resp_cycle[cl] = resp; cnt++;
            return resp;
        }
        return req_cycle + inst.exec_latency;
    }
};

// ═══════════════════ Window throughput ═══════════════════

static void finish_to_throughput(const int64_t* finish, int /*N*/, int W, int K,
                                 double* throughput) {
    int64_t running_max = 0;
    for (int j = 0; j < W; j++) {
        int64_t prev = running_max;
        for (int i = j*K; i < (j+1)*K; i++)
            if (finish[i] > running_max) running_max = finish[i];
        int64_t delta = running_max - prev;
        throughput[j] = (delta > 0) ? (double)K / delta : (double)K;
    }
}

// ═══════════════════ ROB Model ═══════════════════

static void rob_model(const ConcordeTrace& ct, int rob_size, MemoryModel& mem,
                      double* thr, double* iss_lat, double* exe_lat, double* com_lat) {
    int N = ct.size(), W = ct.n_windows, K = WINDOW_SIZE;
    std::vector<int64_t> a(N), s(N), f(N), c(N);
    mem.reset();
    for (int i = 0; i < N; i++) {
        a[i] = (i >= rob_size) ? c[i - rob_size] : 0;
        s[i] = a[i];
        for (int d : ct.instrs[i].deps) if (f[d] > s[i]) s[i] = f[d];
        f[i] = mem.resp_cycle(s[i], ct.instrs[i]);
        c[i] = (i > 0) ? std::max(f[i], c[i-1]) : f[i];
        if (iss_lat) iss_lat[i] = (double)(s[i] - a[i]);
        if (exe_lat) exe_lat[i] = (double)(f[i] - s[i]);
        if (com_lat) com_lat[i] = (double)(c[i] - f[i]);
    }
    if (thr) {
        for (int j = 0; j < W; j++) {
            int64_t prev_c = (j > 0) ? c[j*K-1] : 0;
            thr[j] = (double)K / std::max((int64_t)1, c[(j+1)*K-1] - prev_c);
        }
    }
}

// ═══════════════════ LQ / SQ / Bandwidth / Fetch / Pipes ═══════════════════

static void lq_model(const ConcordeTrace& ct, int lq_size, MemoryModel& mem, double* thr) {
    int N = ct.size(), W = ct.n_windows, K = WINDOW_SIZE;
    std::vector<int> li; for (int i=0;i<N;i++) if (ct.instrs[i].is_load) li.push_back(i);
    int nL = (int)li.size(); std::vector<int64_t> al(nL),fl(nL),cl(nL); mem.reset();
    for (int i=0;i<nL;i++) {
        al[i]=(i>=lq_size)?cl[i-lq_size]:0;
        fl[i]=mem.resp_cycle(al[i],ct.instrs[li[i]]);
        cl[i]=(i>0)?std::max(fl[i],cl[i-1]):fl[i];
    }
    std::vector<int64_t> fin(N,0); for (int i=0;i<nL;i++) fin[li[i]]=cl[i];
    finish_to_throughput(fin.data(),N,W,K,thr);
}

static void sq_model(const ConcordeTrace& ct, int sq_size, double* thr) {
    int N = ct.size(), W = ct.n_windows, K = WINDOW_SIZE;
    std::vector<int> si; for (int i=0;i<N;i++) if (ct.instrs[i].is_store) si.push_back(i);
    int nS = (int)si.size(); std::vector<int64_t> as(nS),fs(nS),cs(nS);
    for (int i=0;i<nS;i++) {
        as[i]=(i>=sq_size)?cs[i-sq_size]:0;
        fs[i]=as[i]+ct.instrs[si[i]].exec_latency;
        cs[i]=(i>0)?std::max(fs[i],cs[i-1]):fs[i];
    }
    std::vector<int64_t> fin(N,0); for (int i=0;i<nS;i++) fin[si[i]]=cs[i];
    finish_to_throughput(fin.data(),N,W,K,thr);
}

static void bandwidth_model(const ConcordeTrace& ct, InstrCategory cat, int width, double* thr) {
    int W=ct.n_windows, K=WINDOW_SIZE;
    for (int j=0;j<W;j++) { int b=j*K,c=0; for (int i=0;i<K;i++) if(ct.instrs[b+i].category==cat) c++;
        thr[j]=(c>0)?(double)K*width/c:(double)K; }
}

static void fetch_width_model(const ConcordeTrace& ct, int width, double* thr) {
    int W=ct.n_windows, K=WINDOW_SIZE;
    for (int j=0;j<W;j++) { int b=j*K, cyc=0, f2=0;
        while (f2<K) { cyc++; for (int i=0;i<std::min(width,K-f2);i++) {
            if (ct.instrs[b+f2].is_branch && ct.instrs[b+f2].is_taken) { f2++; break; } f2++; } }
        thr[j]=(double)K/cyc; }
}

static void pipes_model(const ConcordeTrace& ct, int nls, int nl, double* lo, double* hi) {
    int W=ct.n_windows, K=WINDOW_SIZE, tot=nls+nl;
    if (!tot) { for (int j=0;j<W;j++) lo[j]=hi[j]=0; return; }
    for (int j=0;j<W;j++) { int b=j*K,nld=0,nst=0;
        for (int i=0;i<K;i++) { nld+=ct.instrs[b+i].is_load; nst+=ct.instrs[b+i].is_store; }
        double tm=(double)nld/tot+(nls>0?(double)nst/nls:0); lo[j]=(tm>0)?K/tm:(double)K;
        double ts=nls>0?(double)nst/nls:0, rem=std::max(0.0,(double)nld-(nl>0?ts*nl:0));
        double tn=ts+rem/tot; hi[j]=(tn>0)?K/tn:(double)K; }
}

// ═══════════════════ I-cache LRU Simulation ═══════════════════
// Uses l1i_assoc, l2_assoc from ArchConfig

static void icache_sim(const ConcordeTrace& ct, const ArchConfig& cfg,
                       std::vector<int32_t>& lat) {
    int N = ct.size();
    lat.resize(N, 0);

    int l1i_ways = std::max(cfg.l1i_assoc, 1);
    int l2_ways  = std::max(cfg.l2_assoc, 1);
    int l1i_sets = (cfg.l1_size_kb * 1024) / (64 * l1i_ways);
    int l2_sets  = (cfg.l2_size_kb * 1024) / (64 * l2_ways);
    if (l1i_sets < 1) l1i_sets = 1;
    if (l2_sets < 1)  l2_sets = 1;

    std::vector<std::vector<int64_t>> l1(l1i_sets, std::vector<int64_t>(l1i_ways, -1));
    std::vector<std::vector<int>>     l1a(l1i_sets, std::vector<int>(l1i_ways, 0));
    std::vector<std::vector<int64_t>> l2(l2_sets, std::vector<int64_t>(l2_ways, -1));
    std::vector<std::vector<int>>     l2a(l2_sets, std::vector<int>(l2_ways, 0));

    auto lru = [](auto& t, auto& a, int w, int64_t tg) -> bool {
        for (int i=0;i<w;i++) if (t[i]==tg) {
            a[i]=0; for (int j=0;j<w;j++) if(j!=i) a[j]++; return true;
        }
        int v=0; for (int i=1;i<w;i++) if(a[i]>a[v]) v=i;
        t[v]=tg; a[v]=0; for (int i=0;i<w;i++) if(i!=v) a[i]++;
        return false;
    };

    for (int i = 0; i < N; i++) {
        int64_t cl = ct.instrs[i].icache_line;
        if (lru(l1[cl % l1i_sets], l1a[cl % l1i_sets], l1i_ways, cl)) { lat[i]=0; continue; }
        if (lru(l2[cl % l2_sets],  l2a[cl % l2_sets],  l2_ways,  cl)) { lat[i]=10; continue; }
        lat[i] = 30;
    }
}

// ═══════════════════ I-cache Fills / Fetch Buffers ═══════════════════

static void icache_fills_model(const ConcordeTrace& ct, int mf, const std::vector<int32_t>& il, double* thr) {
    int N=ct.size(),W=ct.n_windows,K=WINDOW_SIZE;
    struct F{int64_t l,d;}; std::vector<F> inf; inf.reserve(mf);
    std::vector<int64_t> av(N); int64_t cyc=0;
    for(int i=0;i<N;i++){int64_t cl=ct.instrs[i].icache_line;
        if(!il[i]){av[i]=cyc;continue;}
        bool fo=false;for(auto&f:inf)if(f.l==cl){av[i]=f.d;fo=true;break;}if(fo)continue;
        if((int)inf.size()>=mf){auto e=std::min_element(inf.begin(),inf.end(),
            [](auto&a,auto&b){return a.d<b.d;});if(e->d>cyc)cyc=e->d;inf.erase(e);}
        int64_t d=cyc+il[i];inf.push_back({cl,d});av[i]=d;
        inf.erase(std::remove_if(inf.begin(),inf.end(),[cyc](auto&f){return f.d<=cyc;}),inf.end());}
    finish_to_throughput(av.data(),N,W,K,thr);
}

static void fetch_bufs_model(const ConcordeTrace& ct, int nb, int dw, const std::vector<int32_t>& il, double* thr) {
    int N=ct.size(),W=ct.n_windows,K=WINDOW_SIZE;
    std::vector<int64_t> d(N); int64_t fc=0,dc=0; int bu=0;
    for(int i=0;i<N;i++){if(bu>=nb){dc++;bu-=std::min(dw,bu);}
        fc=std::max(fc,dc);fc+=(int64_t)il[i];d[i]=fc;bu++;}
    finish_to_throughput(d.data(),N,W,K,thr);
}

// ═══════════════════ ROB-derived features ═══════════════════

struct RobRes {
    double avg_thr[N_ROB_SIZES], exec_dist[DIST_DIM];
    double iss_dist[N_ROB_SIZES][DIST_DIM], com_dist[N_ROB_SIZES][DIST_DIM];
};

static void compute_rob_features(const ConcordeTrace& ct, MemoryModel& mem, RobRes& r) {
    int N=ct.size(),W=ct.n_windows;
    std::vector<double> thr(W),iss(N),exe(N),com(N);
    for (int ri=0;ri<N_ROB_SIZES;ri++){
        rob_model(ct,ROB_SIZES[ri],mem,thr.data(),iss.data(),exe.data(),com.data());
        double s=0;for(int j=0;j<W;j++)s+=thr[j]; r.avg_thr[ri]=s/W;
        encode_distribution(iss.data(),N,r.iss_dist[ri]);
        encode_distribution(com.data(),N,r.com_dist[ri]);
        if(!ri) encode_distribution(exe.data(),N,r.exec_dist);
    }
}

// ═══════════════════ Main Feature Assembly ═══════════════════

void compute_features(const ConcordeTrace& ct, const ArchConfig& cfg, double features[N_FEATURES]) {
    std::memset(features,0,sizeof(double)*N_FEATURES);
    int W=ct.n_windows, K=WINDOW_SIZE; if(!W) return;
    double* p = features;

    std::vector<int32_t> il;
    icache_sim(ct, cfg, il);

    MemoryModel mem; mem.init(ct);
    RobRes rr; compute_rob_features(ct,mem,rr);

    // Group 1: 11 throughput distributions (1111)
    std::vector<double> thr(W), thr2(W);

    rob_model(ct,cfg.rob_size,mem,thr.data(),nullptr,nullptr,nullptr);
    encode_distribution(thr.data(),W,p); p+=DIST_DIM;
    lq_model(ct,cfg.load_queue_size,mem,thr.data());
    encode_distribution(thr.data(),W,p); p+=DIST_DIM;
    sq_model(ct,cfg.store_queue_size,thr.data());
    encode_distribution(thr.data(),W,p); p+=DIST_DIM;
    bandwidth_model(ct,CAT_ALU,cfg.alu_issue_width,thr.data());
    encode_distribution(thr.data(),W,p); p+=DIST_DIM;
    bandwidth_model(ct,CAT_FP,cfg.fp_issue_width,thr.data());
    encode_distribution(thr.data(),W,p); p+=DIST_DIM;
    bandwidth_model(ct,CAT_LS,cfg.ls_issue_width,thr.data());
    encode_distribution(thr.data(),W,p); p+=DIST_DIM;
    fetch_width_model(ct,cfg.fetch_width,thr.data());
    encode_distribution(thr.data(),W,p); p+=DIST_DIM;
    pipes_model(ct,cfg.n_ls_pipes,cfg.n_l_pipes,thr.data(),thr2.data());
    encode_distribution(thr.data(),W,p); p+=DIST_DIM;
    encode_distribution(thr2.data(),W,p); p+=DIST_DIM;
    icache_fills_model(ct,cfg.max_icache_fills,il,thr.data());
    encode_distribution(thr.data(),W,p); p+=DIST_DIM;
    fetch_bufs_model(ct,cfg.n_fetch_bufs,cfg.decode_width,il,thr.data());
    encode_distribution(thr.data(),W,p); p+=DIST_DIM;
    assert(p-features==1111);

    // Group 2: Pipeline stalls (416)
    std::vector<double> cnt(W);
    for(int j=0;j<W;j++){int b=j*K,c=0;for(int i=0;i<K;i++)c+=ct.instrs[b+i].is_isb;cnt[j]=c;}
    encode_distribution(cnt.data(),W,p); p+=DIST_DIM;
    for(int bt=BR_DIRECT_COND;bt<=BR_INDIRECT;bt++){
        for(int j=0;j<W;j++){int b=j*K,c=0;for(int i=0;i<K;i++)if(ct.instrs[b+i].branch_type==bt)c++;cnt[j]=c;}
        encode_distribution(cnt.data(),W,p); p+=DIST_DIM;}
    *p++=ct.branch_mispred_rate;
    for(int ri=0;ri<N_ROB_SIZES;ri++)*p++=rr.avg_thr[ri];
    assert(p-features==1527);

    // Group 3: Latency distributions (2323)
    std::memcpy(p,rr.exec_dist,sizeof(double)*DIST_DIM); p+=DIST_DIM;
    for(int ri=0;ri<N_ROB_SIZES;ri++){std::memcpy(p,rr.iss_dist[ri],sizeof(double)*DIST_DIM);p+=DIST_DIM;}
    for(int ri=0;ri<N_ROB_SIZES;ri++){std::memcpy(p,rr.com_dist[ri],sizeof(double)*DIST_DIM);p+=DIST_DIM;}
    assert(p-features==3850);

    // Group 4: Architecture parameters (23) — from arch_params.npy
    for (int i = 0; i < N_ARCH_PARAMS; i++)
        *p++ = (double)cfg.raw[i];
    assert(p-features==N_FEATURES);
}