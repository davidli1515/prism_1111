/*
 * Concorde + GEM Analytical Models — Implementation
 *
 * Concorde models (§3.2 of Concorde paper):
 *   ROB, LQ, SQ, ALU/FP/LS bandwidth, Fetch width,
 *   Pipes, I-cache fills, Fetch buffers
 *
 * GEM D3 Stage-Interaction Pressure Model:
 *   Raw pressures (P_FE, P_ROB) computed from Concorde's T_resource
 *   Propagated through pipeline queues → per-stage FE/BE pressures
 *   T_crit from dependency DAG critical path (no ROB/commit constraint)
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

// ═══════════════════ ArchConfig ═══════════════════

bool ArchConfig::load(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "Error: cannot open %s\n", path.c_str()); return false; }
    char magic[8];
    if (fread(magic, 1, 8, f) != 8 || magic[0] != '\x93') { fclose(f); return false; }
    uint16_t hlen;
    if (fread(&hlen, 2, 1, f) != 1) { fclose(f); return false; }
    fseek(f, hlen, SEEK_CUR);
    if (fread(raw, sizeof(float), N_ARCH_PARAMS, f) != N_ARCH_PARAMS) { fclose(f); return false; }
    fclose(f);

    fetch_width      = (int)raw[0];
    decode_width     = (int)raw[1];
    rename_width     = (int)raw[2];
    dispatch_width   = (int)raw[3];
    issue_width      = (int)raw[4];
    commit_width     = (int)raw[6];
    rob_size         = (int)raw[7];
    iq_entries       = (int)raw[8];
    load_queue_size  = (int)raw[9];
    store_queue_size = (int)raw[10];
    n_fetch_bufs     = (int)raw[11];
    alu_issue_width  = (int)raw[12];
    fp_issue_width   = (int)raw[14];
    ls_issue_width   = (int)raw[16];
    n_ls_pipes       = (int)raw[16];
    n_l_pipes        = 0;
    max_icache_fills = 8;
    l1_size_kb       = (int)raw[17];
    l2_size_kb       = (int)raw[18];
    l1d_assoc        = (int)raw[19];
    l1i_assoc        = (int)raw[20];
    l2_assoc         = (int)raw[21];
    branch_predictor = (int)raw[22];

    printf("  Arch: fetch=%d decode=%d rename=%d dispatch=%d issue=%d commit=%d\n",
           fetch_width, decode_width, rename_width, dispatch_width, issue_width, commit_width);
    printf("        ROB=%d IQ=%d LQ=%d SQ=%d fetchBuf=%d W_min=%d\n",
           rob_size, iq_entries, load_queue_size, store_queue_size, n_fetch_bufs, w_min());
    printf("        IntALU=%d FP_ALU=%d RdWrPort=%d L1=%dKB L2=%dKB BP=%s\n",
           alu_issue_width, fp_issue_width, ls_issue_width, l1_size_kb, l2_size_kb, bp_str().c_str());
    return true;
}

int ArchConfig::w_min() const {
    return std::min({fetch_width, decode_width, rename_width,
                     dispatch_width, issue_width, commit_width});
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

// ═══════════════════ CPI Parser ═══════════════════

double parse_cpi(const std::string& stats_path) {
    std::ifstream fin(stats_path);
    if (!fin.is_open()) { fprintf(stderr, "Warning: cannot open %s\n", stats_path.c_str()); return -1.0; }
    std::string line;
    while (std::getline(fin, line)) {
        if (line.find("system.switch_cpus_1.commitStats0.cpi") != std::string::npos) {
            std::istringstream ss(line); std::string key; double val;
            ss >> key >> val; return val;
        }
    }
    fprintf(stderr, "Warning: CPI not found in %s\n", stats_path.c_str());
    return -1.0;
}

// ═══════════════════ MSCS ═══════════════════

void compute_mscs(const ConcordeTrace& ct, double mscs[N_MSCS]) {
    std::memset(mscs, 0, sizeof(double) * N_MSCS);
    if (!ct.has_stage_stats) return;
    const double* a = ct.last_stage_stats;  // normalized floats from last line
    mscs[0]=a[0]; mscs[1]=a[1]; mscs[2]=a[2]; mscs[3]=a[3];
    mscs[4]=a[4]; mscs[5]=a[5]; mscs[6]=a[6]+a[7]+a[8]; mscs[7]=a[9];
    mscs[8]=a[16]; mscs[9]=a[17]; mscs[10]=a[18]+a[19]+a[20]; mscs[11]=a[21];
    mscs[12]=a[22]; mscs[13]=a[23]; mscs[14]=a[24]+a[25]+a[26]; mscs[15]=a[27];
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
    } else { std::fill(out + 50, out + 100, 0.0); }
    double sum = 0; for (int i = 0; i < N; i++) sum += vals[i];
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

static void icache_sim(const ConcordeTrace& ct, const ArchConfig& cfg,
                       std::vector<int32_t>& lat) {
    int N = ct.size(); lat.resize(N, 0);
    int l1w = std::max(cfg.l1i_assoc, 1), l2w = std::max(cfg.l2_assoc, 1);
    int s1 = (cfg.l1_size_kb*1024)/(64*l1w), s2 = (cfg.l2_size_kb*1024)/(64*l2w);
    if (s1<1) s1=1;
    if (s2<1) s2=1;
    std::vector<std::vector<int64_t>> l1(s1,std::vector<int64_t>(l1w,-1));
    std::vector<std::vector<int>>     l1a(s1,std::vector<int>(l1w,0));
    std::vector<std::vector<int64_t>> l2(s2,std::vector<int64_t>(l2w,-1));
    std::vector<std::vector<int>>     l2a(s2,std::vector<int>(l2w,0));
    auto lru=[](auto& t,auto& a,int w,int64_t tg)->bool{
        for(int i=0;i<w;i++) if(t[i]==tg){a[i]=0;for(int j=0;j<w;j++)if(j!=i)a[j]++;return true;}
        int v=0;for(int i=1;i<w;i++)if(a[i]>a[v])v=i;
        t[v]=tg;a[v]=0;for(int i=0;i<w;i++)if(i!=v)a[i]++;return false;};
    for (int i=0;i<N;i++) { int64_t cl=ct.instrs[i].icache_line;
        if(lru(l1[cl%s1],l1a[cl%s1],l1w,cl)){lat[i]=0;continue;}
        if(lru(l2[cl%s2],l2a[cl%s2],l2w,cl)){lat[i]=10;continue;} lat[i]=30; }
}

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

// ═══════════════════ Concorde Feature Assembly (3873) ═══════════════════

void compute_features(const ConcordeTrace& ct, const ArchConfig& cfg, double features[N_FEATURES]) {
    std::memset(features,0,sizeof(double)*N_FEATURES);
    int W=ct.n_windows, K=WINDOW_SIZE; if(!W) return;
    double* p = features;

    std::vector<int32_t> il; icache_sim(ct,cfg,il);
    MemoryModel mem; mem.init(ct);
    RobRes rr; compute_rob_features(ct,mem,rr);

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

    std::vector<double> cnt(W);
    for(int j=0;j<W;j++){int b=j*K,c=0;for(int i=0;i<K;i++)c+=ct.instrs[b+i].is_isb;cnt[j]=c;}
    encode_distribution(cnt.data(),W,p); p+=DIST_DIM;
    for(int bt=BR_DIRECT_COND;bt<=BR_INDIRECT;bt++){
        for(int j=0;j<W;j++){int b=j*K,c=0;for(int i=0;i<K;i++)if(ct.instrs[b+i].branch_type==bt)c++;cnt[j]=c;}
        encode_distribution(cnt.data(),W,p); p+=DIST_DIM;}
    *p++=ct.branch_mispred_rate;
    for(int ri=0;ri<N_ROB_SIZES;ri++)*p++=rr.avg_thr[ri];
    assert(p-features==1527);

    std::memcpy(p,rr.exec_dist,sizeof(double)*DIST_DIM); p+=DIST_DIM;
    for(int ri=0;ri<N_ROB_SIZES;ri++){std::memcpy(p,rr.iss_dist[ri],sizeof(double)*DIST_DIM);p+=DIST_DIM;}
    for(int ri=0;ri<N_ROB_SIZES;ri++){std::memcpy(p,rr.com_dist[ri],sizeof(double)*DIST_DIM);p+=DIST_DIM;}
    assert(p-features==3850);

    for (int i = 0; i < N_ARCH_PARAMS; i++) *p++ = (double)cfg.raw[i];
    assert(p-features==N_FEATURES);
}

// ═══════════════════ GEM D3: Stage-Interaction Pressure (808) ═══════════════════
//
// Per window:
//   1. Raw pressures from Concorde's T_resource:
//      P_FE,raw  = max(0, T_icache × W_min - K)
//      P_ROB,raw = max(0, T_ROB × W_min - K)
//   2. Critical path (dependency DAG only, no ROB/commit constraint):
//      f_crit[i] = max(f_crit[d] for d in Dep(i)) + exec_latency[i]
//      T_crit[j] = critical path span for window j
//      C_ILP[j]  = max(0, (T_ROB[j] - T_crit[j]) × W_min)
//   3. Queue capacities (from arch params):
//      C_FDQ = 2×decodeWidth,  C_DQ = 2×dispatchWidth
//      C_DIQ = 2×issueWidth,   C_IQ = numIQEntries
//      C_LQ  = LQEntries,      C_ROB = numROBEntries
//   4. Forward propagation (FE):
//      Decode   = max(0, P_FE,raw - C_FDQ)
//      Dispatch = max(0, FE_Decode - C_DQ)
//      Issue    = max(0, FE_Dispatch - C_DIQ - C_IQ - C_LQ)    [parallel path]
//      Commit   = max(0, FE_Dispatch - C_ROB)                   [parallel path]
//   5. Backward propagation (BE):
//      Commit   = P_ROB,raw
//      Issue    = max(0, BE_Commit - C_ILP)
//      Dispatch = max(0, BE_Commit - min(C_ROB, C_IQ + C_LQ))
//      Decode   = max(0, BE_Dispatch - C_DQ)
//
// Output: 8 time series × 101-dim distribution = 808 features

void compute_gem_features(const ConcordeTrace& ct, const ArchConfig& cfg,
                          double gem_features[N_GEM_PRESSURE]) {
    std::memset(gem_features, 0, sizeof(double) * N_GEM_PRESSURE);
    int W = ct.n_windows, K = WINDOW_SIZE, N = ct.size();
    if (W == 0) return;

    int Wm = cfg.w_min();
    if (Wm < 1) Wm = 1;

    // Queue capacities
    int C_FDQ = 2 * cfg.decode_width;
    int C_DQ  = 2 * cfg.dispatch_width;
    int C_DIQ = 2 * cfg.issue_width;
    int C_IQ  = cfg.iq_entries;
    int C_LQ  = cfg.load_queue_size;
    int C_ROB = cfg.rob_size;

    // ── T_ROB per window (from ROB model at target config) ──
    std::vector<double> thr_rob(W);
    MemoryModel mem; mem.init(ct);
    rob_model(ct, cfg.rob_size, mem, thr_rob.data(), nullptr, nullptr, nullptr);

    // ── T_icache per window (from icache fills model) ──
    std::vector<int32_t> icache_lat;
    icache_sim(ct, cfg, icache_lat);
    std::vector<double> thr_icache(W);
    icache_fills_model(ct, cfg.max_icache_fills, icache_lat, thr_icache.data());

    // ── T_crit per window: critical path (deps only, no ROB/commit) ──
    std::vector<int64_t> f_crit(N, 0);
    for (int i = 0; i < N; i++) {
        int64_t s = 0;
        for (int d : ct.instrs[i].deps)
            if (f_crit[d] > s) s = f_crit[d];
        f_crit[i] = s + ct.instrs[i].exec_latency;
    }
    // Per-window T_crit via running max (same pattern as finish_to_throughput)
    std::vector<double> T_crit_w(W);
    {
        int64_t running = 0;
        for (int j = 0; j < W; j++) {
            int64_t prev = running;
            for (int i = j*K; i < (j+1)*K; i++)
                if (f_crit[i] > running) running = f_crit[i];
            T_crit_w[j] = (double)(running - prev);
        }
    }

    // ── Per-window raw pressures and propagation ──
    std::vector<double> fe_dec(W), fe_dis(W), fe_iss(W), fe_com(W);
    std::vector<double> be_dec(W), be_dis(W), be_iss(W), be_com(W);

    for (int j = 0; j < W; j++) {
        // T_resource = K / throughput (total cycles for this window)
        double T_rob = (thr_rob[j] > 1e-9) ? (double)K / thr_rob[j] : 0;
        double T_fe  = (thr_icache[j] > 1e-9) ? (double)K / thr_icache[j] : 0;

        // Raw pressures (Eq 4: P = T × W_min - N)
        double P_FE_raw  = std::max(0.0, T_fe * Wm - K);
        double P_ROB_raw = std::max(0.0, T_rob * Wm - K);

        // ILP slack (Eq 13)
        double C_ilp = std::max(0.0, (T_rob - T_crit_w[j]) * Wm);

        // Forward pass: FE pressure through pipeline (Eq 5-8)
        fe_dec[j] = std::max(0.0, P_FE_raw - C_FDQ);
        fe_dis[j] = std::max(0.0, fe_dec[j] - C_DQ);
        fe_iss[j] = std::max(0.0, fe_dis[j] - C_DIQ - C_IQ - C_LQ);  // parallel
        fe_com[j] = std::max(0.0, fe_dis[j] - C_ROB);                  // parallel

        // Backward pass: BE pressure through pipeline (Eq 9, 14-16)
        be_com[j] = P_ROB_raw;
        be_iss[j] = std::max(0.0, be_com[j] - C_ilp);
        be_dis[j] = std::max(0.0, be_com[j] - (double)std::min(C_ROB, C_IQ + C_LQ));
        be_dec[j] = std::max(0.0, be_dis[j] - C_DQ);
    }

    // Encode 8 distributions × 101 = 808 features
    double* p = gem_features;
    encode_distribution(fe_dec.data(), W, p); p += DIST_DIM;
    encode_distribution(fe_dis.data(), W, p); p += DIST_DIM;
    encode_distribution(fe_iss.data(), W, p); p += DIST_DIM;
    encode_distribution(fe_com.data(), W, p); p += DIST_DIM;
    encode_distribution(be_dec.data(), W, p); p += DIST_DIM;
    encode_distribution(be_dis.data(), W, p); p += DIST_DIM;
    encode_distribution(be_iss.data(), W, p); p += DIST_DIM;
    encode_distribution(be_com.data(), W, p); p += DIST_DIM;
    assert(p - gem_features == N_GEM_PRESSURE);
}