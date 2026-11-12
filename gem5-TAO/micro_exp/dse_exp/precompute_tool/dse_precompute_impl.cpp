// ═══════════════════════════════════════════════════════════════════════════
// DSE Precompute Implementation — APPEND TO END OF analytical_models.cpp
// ═══════════════════════════════════════════════════════════════════════════

#include "dse_precompute.h"
#include <cassert>

// ─────────────────────────── Precompute ───────────────────────────

void dse_precompute(const ConcordeTrace& ct, const ArchConfig& icache_cfg,
                    DsePrecomputed& out) {
    int N = ct.size(), W = ct.n_windows, K = WINDOW_SIZE;
    if (W == 0) return;
    out.W = W;

    printf("  DSE precompute: %d instructions, %d windows\n", N, W);
    MemoryModel mem; mem.init(ct);

    // ═══ A1: ROB sweep ═══
    printf("    ROB sweep: %d sizes [%d..%d]...", DSE_N_ROB, DSE_ROB_START, DSE_ROB_END);
    fflush(stdout);
    out.rob_thr.resize(DSE_N_ROB, std::vector<double>(W));
    {
        std::vector<double> thr(W);
        for (int ri = 0; ri < DSE_N_ROB; ri++) {
            int rob_sz = DSE_ROB_START + ri;
            rob_model(ct, rob_sz, mem, thr.data(), nullptr, nullptr, nullptr);
            out.rob_thr[ri] = thr;
        }
    }
    printf(" done\n");

    // ═══ A2: LQ sweep ═══
    printf("    LQ sweep: %d sizes [%d..%d]...", DSE_N_LQ, DSE_LQ_START, DSE_LQ_END);
    fflush(stdout);
    out.lq_thr.resize(DSE_N_LQ, std::vector<double>(W));
    {
        std::vector<double> thr(W);
        for (int li = 0; li < DSE_N_LQ; li++) {
            lq_model(ct, DSE_LQ_START + li, mem, thr.data());
            out.lq_thr[li] = thr;
        }
    }
    printf(" done\n");

    // ═══ A3: SQ sweep ═══
    printf("    SQ sweep: %d sizes [%d..%d]...", DSE_N_SQ, DSE_SQ_START, DSE_SQ_END);
    fflush(stdout);
    out.sq_thr.resize(DSE_N_SQ, std::vector<double>(W));
    {
        std::vector<double> thr(W);
        for (int si = 0; si < DSE_N_SQ; si++) {
            sq_model(ct, DSE_SQ_START + si, thr.data());
            out.sq_thr[si] = thr;
        }
    }
    printf(" done\n");

    // ═══ A4: Fetch width sweep ═══
    out.fetchw_thr.resize(DSE_N_FW, std::vector<double>(W));
    {
        std::vector<double> thr(W);
        for (int fi = 0; fi < DSE_N_FW; fi++) {
            fetch_width_model(ct, DSE_FW_START + fi, thr.data());
            out.fetchw_thr[fi] = thr;
        }
    }

    // ═══ A5: Fetch buffers sweep ═══
    std::vector<int32_t> icache_lat;
    icache_sim(ct, icache_cfg, icache_lat);

    out.fetchbuf_thr.resize(DSE_N_FB, std::vector<double>(W));
    {
        std::vector<double> thr(W);
        for (int nb = 0; nb < DSE_N_NB; nb++)
            for (int dw = 0; dw < DSE_N_DW; dw++) {
                fetch_bufs_model(ct, DSE_NB_START + nb, DSE_DW_START + dw, icache_lat, thr.data());
                out.fetchbuf_thr[nb * DSE_N_DW + dw] = thr;
            }
    }

    // ═══ B: Per-window instruction counts ═══
    out.n_alu.resize(W); out.n_fp.resize(W); out.n_ls.resize(W);
    out.n_load.resize(W); out.n_store.resize(W);
    for (int j = 0; j < W; j++) {
        int b = j * K, na = 0, nf = 0, nl = 0, nld = 0, nst = 0;
        for (int i = 0; i < K; i++) {
            auto& inst = ct.instrs[b + i];
            if (inst.category == CAT_ALU) na++;
            if (inst.category == CAT_FP)  nf++;
            if (inst.category == CAT_LS)  nl++;
            if (inst.is_load) nld++;
            if (inst.is_store) nst++;
        }
        out.n_alu[j] = na; out.n_fp[j] = nf; out.n_ls[j] = nl;
        out.n_load[j] = nld; out.n_store[j] = nst;
    }

    // ═══ C: Icache throughput ═══
    out.icache_thr.resize(W);
    icache_fills_model(ct, icache_cfg.max_icache_fills, icache_lat, out.icache_thr.data());

    // ═══ D: Program-dependent features ═══

    // D1: Auxiliary features [416 dims]
    // Layout: ISB_dist(101) + branch_dists(3×101) + mispred(1) + ROB_avg(11) = 416
    {
        double* p = out.auxiliary;
        std::vector<double> cnt(W);

        // ISB count distribution
        for (int j = 0; j < W; j++) {
            int b = j * K, c = 0;
            for (int i = 0; i < K; i++) c += ct.instrs[b + i].is_isb;
            cnt[j] = c;
        }
        encode_distribution(cnt.data(), W, p); p += DIST_DIM;

        // Branch type distributions (3 types)
        for (int bt = BR_DIRECT_COND; bt <= BR_INDIRECT; bt++) {
            for (int j = 0; j < W; j++) {
                int b = j * K, c = 0;
                for (int i = 0; i < K; i++)
                    if (ct.instrs[b + i].branch_type == bt) c++;
                cnt[j] = c;
            }
            encode_distribution(cnt.data(), W, p); p += DIST_DIM;
        }

        // Branch mispred rate
        *p++ = ct.branch_mispred_rate;

        assert(p - out.auxiliary == 405);
    }

    // D2: ROB features at 11 Concorde sweep points
    {
        RobRes rr;
        compute_rob_features(ct, mem, rr);

        // ROB avg throughput (11 values) → goes into auxiliary[405:416]
        for (int ri = 0; ri < N_ROB_SIZES; ri++)
            out.auxiliary[405 + ri] = rr.avg_thr[ri];
        std::memcpy(out.rob_avg_thr, rr.avg_thr, sizeof(out.rob_avg_thr));

        // Latency distributions
        std::memcpy(out.exec_dist, rr.exec_dist, sizeof(out.exec_dist));
        std::memcpy(out.iss_dist, rr.iss_dist, sizeof(out.iss_dist));
        std::memcpy(out.com_dist, rr.com_dist, sizeof(out.com_dist));
    }

    // ═══ E: GEM T_crit ═══
    out.T_crit.resize(W);
    {
        std::vector<int64_t> f_crit(N, 0);
        for (int i = 0; i < N; i++) {
            int64_t s = 0;
            for (int d : ct.instrs[i].deps)
                if (f_crit[d] > s) s = f_crit[d];
            f_crit[i] = s + ct.instrs[i].exec_latency;
        }
        int64_t running = 0;
        for (int j = 0; j < W; j++) {
            int64_t prev = running;
            for (int i = j * K; i < (j + 1) * K; i++)
                if (f_crit[i] > running) running = f_crit[i];
            out.T_crit[j] = (double)(running - prev);
        }
    }

    printf("    Precompute complete.\n");
}

// ─────────────────────────── Inference ───────────────────────────

void dse_infer(const DsePrecomputed& pre, const ArchConfig& cfg,
               double features[DSE_TOTAL_FEATURES]) {
    std::memset(features, 0, sizeof(double) * DSE_TOTAL_FEATURES);
    int W = pre.W, K = WINDOW_SIZE;
    if (W == 0) return;

    double* p = features;
    std::vector<double> thr(W), thr2(W);

    // ════════ Concorde 3873 dims ════════

    // [0:101] ROB throughput at target ROB size
    {
        int idx = cfg.rob_size - DSE_ROB_START;
        idx = std::max(0, std::min(idx, DSE_N_ROB - 1));
        encode_distribution(pre.rob_thr[idx].data(), W, p);
    }
    p += DIST_DIM;

    // [101:202] LQ throughput
    {
        int idx = cfg.load_queue_size - DSE_LQ_START;
        idx = std::max(0, std::min(idx, DSE_N_LQ - 1));
        encode_distribution(pre.lq_thr[idx].data(), W, p);
    }
    p += DIST_DIM;

    // [202:303] SQ throughput
    {
        int idx = cfg.store_queue_size - DSE_SQ_START;
        idx = std::max(0, std::min(idx, DSE_N_SQ - 1));
        encode_distribution(pre.sq_thr[idx].data(), W, p);
    }
    p += DIST_DIM;

    // [303:404] ALU bandwidth
    for (int j = 0; j < W; j++) {
        int c = pre.n_alu[j];
        thr[j] = (c > 0) ? (double)K * cfg.alu_issue_width / c : (double)K;
    }
    encode_distribution(thr.data(), W, p); p += DIST_DIM;

    // [404:505] FP bandwidth
    for (int j = 0; j < W; j++) {
        int c = pre.n_fp[j];
        thr[j] = (c > 0) ? (double)K * cfg.fp_issue_width / c : (double)K;
    }
    encode_distribution(thr.data(), W, p); p += DIST_DIM;

    // [505:606] LS bandwidth
    for (int j = 0; j < W; j++) {
        int c = pre.n_ls[j];
        thr[j] = (c > 0) ? (double)K * cfg.ls_issue_width / c : (double)K;
    }
    encode_distribution(thr.data(), W, p); p += DIST_DIM;

    // [606:707] Fetch width
    {
        int idx = cfg.fetch_width - DSE_FW_START;
        idx = std::max(0, std::min(idx, DSE_N_FW - 1));
        encode_distribution(pre.fetchw_thr[idx].data(), W, p);
    }
    p += DIST_DIM;

    // [707:808] [808:909] Pipes lower/upper
    {
        int nls = cfg.n_ls_pipes, nl = cfg.n_l_pipes, tot = nls + nl;
        for (int j = 0; j < W; j++) {
            int nld = pre.n_load[j], nst = pre.n_store[j];
            if (!tot) { thr[j] = thr2[j] = 0; continue; }
            // Lower bound (worst case)
            double tm = (double)nld / tot + (nls > 0 ? (double)nst / nls : 0);
            thr[j] = (tm > 0) ? K / tm : (double)K;
            // Upper bound (best case)
            double ts = nls > 0 ? (double)nst / nls : 0;
            double rem = std::max(0.0, (double)nld - (nl > 0 ? ts * nl : 0));
            double tn = ts + rem / tot;
            thr2[j] = (tn > 0) ? K / tn : (double)K;
        }
        encode_distribution(thr.data(), W, p);  p += DIST_DIM;
        encode_distribution(thr2.data(), W, p); p += DIST_DIM;
    }

    // [909:1010] Icache fills (fixed per memory config)
    encode_distribution(pre.icache_thr.data(), W, p); p += DIST_DIM;

    // [1010:1111] Fetch buffers
    {
        int nb_idx = cfg.n_fetch_bufs - DSE_NB_START;
        int dw_idx = cfg.decode_width - DSE_DW_START;
        nb_idx = std::max(0, std::min(nb_idx, DSE_N_NB - 1));
        dw_idx = std::max(0, std::min(dw_idx, DSE_N_DW - 1));
        int fb_idx = nb_idx * DSE_N_DW + dw_idx;
        encode_distribution(pre.fetchbuf_thr[fb_idx].data(), W, p);
    }
    p += DIST_DIM;

    assert(p - features == 1111);

    // [1111:1527] Auxiliary features (program-dependent, fixed)
    std::memcpy(p, pre.auxiliary, sizeof(pre.auxiliary)); p += 416;
    assert(p - features == 1527);

    // [1527:1628] exec_dist
    std::memcpy(p, pre.exec_dist, sizeof(pre.exec_dist)); p += DIST_DIM;

    // [1628:2739] iss_dist × 11 ROB sizes
    for (int ri = 0; ri < N_ROB_SIZES; ri++) {
        std::memcpy(p, pre.iss_dist[ri], sizeof(double) * DIST_DIM);
        p += DIST_DIM;
    }

    // [2739:3850] com_dist × 11 ROB sizes
    for (int ri = 0; ri < N_ROB_SIZES; ri++) {
        std::memcpy(p, pre.com_dist[ri], sizeof(double) * DIST_DIM);
        p += DIST_DIM;
    }
    assert(p - features == 3850);

    // [3850:3873] Arch params
    for (int i = 0; i < N_ARCH_PARAMS; i++) *p++ = (double)cfg.raw[i];
    assert(p - features == N_FEATURES);

    // ════════ GEM D3 808 dims ════════

    double* gp = features + N_FEATURES;
    int Wm = cfg.w_min(); if (Wm < 1) Wm = 1;
    int C_FDQ = 2 * cfg.decode_width, C_DQ = 2 * cfg.dispatch_width;
    int C_DIQ = 2 * cfg.issue_width,  C_IQ = cfg.iq_entries;
    int C_LQ  = cfg.load_queue_size,  C_ROB = cfg.rob_size;

    // Look up T_rob at target ROB size
    int rob_idx = std::max(0, std::min(cfg.rob_size - DSE_ROB_START, DSE_N_ROB - 1));

    std::vector<double> fe_dec(W), fe_dis(W), fe_iss(W), fe_com(W);
    std::vector<double> be_dec(W), be_dis(W), be_iss(W), be_com(W);

    for (int j = 0; j < W; j++) {
        double T_rob = (pre.rob_thr[rob_idx][j] > 1e-9) ? (double)K / pre.rob_thr[rob_idx][j] : 0;
        double T_fe  = (pre.icache_thr[j] > 1e-9) ? (double)K / pre.icache_thr[j] : 0;
        double P_FE  = std::max(0.0, T_fe * Wm - K);
        double P_ROB = std::max(0.0, T_rob * Wm - K);
        double C_ilp = std::max(0.0, (T_rob - pre.T_crit[j]) * Wm);

        fe_dec[j] = std::max(0.0, P_FE - C_FDQ);
        fe_dis[j] = std::max(0.0, fe_dec[j] - C_DQ);
        fe_iss[j] = std::max(0.0, fe_dis[j] - C_DIQ - C_IQ - C_LQ);
        fe_com[j] = std::max(0.0, fe_dis[j] - C_ROB);
        be_com[j] = P_ROB;
        be_iss[j] = std::max(0.0, be_com[j] - C_ilp);
        be_dis[j] = std::max(0.0, be_com[j] - (double)std::min(C_ROB, C_IQ + C_LQ));
        be_dec[j] = std::max(0.0, be_dis[j] - C_DQ);
    }

    encode_distribution(fe_dec.data(), W, gp); gp += DIST_DIM;
    encode_distribution(fe_dis.data(), W, gp); gp += DIST_DIM;
    encode_distribution(fe_iss.data(), W, gp); gp += DIST_DIM;
    encode_distribution(fe_com.data(), W, gp); gp += DIST_DIM;
    encode_distribution(be_dec.data(), W, gp); gp += DIST_DIM;
    encode_distribution(be_dis.data(), W, gp); gp += DIST_DIM;
    encode_distribution(be_iss.data(), W, gp); gp += DIST_DIM;
    encode_distribution(be_com.data(), W, gp); gp += DIST_DIM;
    assert(gp - features == DSE_TOTAL_FEATURES);
}

// ─────────────────────────── Save / Load ───────────────────────────

int DsePrecomputed::flat_size() const {
    // header(1) + rob + lq + sq + fw + fb + counts(5) + icache + aux + latency + T_crit
    return 1 + DSE_N_ROB*W + DSE_N_LQ*W + DSE_N_SQ*W + DSE_N_FW*W + DSE_N_FB*W
           + 5*W + W + 416 + DIST_DIM + N_ROB_SIZES*DIST_DIM*2 + W;
}

void dse_save(const std::string& path, const DsePrecomputed& data) {
    int W = data.W;
    int total = data.flat_size();
    std::vector<double> buf(total);
    double* p = buf.data();

    *p++ = (double)W;

    for (int i = 0; i < DSE_N_ROB; i++) for (int j = 0; j < W; j++) *p++ = data.rob_thr[i][j];
    for (int i = 0; i < DSE_N_LQ;  i++) for (int j = 0; j < W; j++) *p++ = data.lq_thr[i][j];
    for (int i = 0; i < DSE_N_SQ;  i++) for (int j = 0; j < W; j++) *p++ = data.sq_thr[i][j];
    for (int i = 0; i < DSE_N_FW;  i++) for (int j = 0; j < W; j++) *p++ = data.fetchw_thr[i][j];
    for (int i = 0; i < DSE_N_FB;  i++) for (int j = 0; j < W; j++) *p++ = data.fetchbuf_thr[i][j];
    for (int j = 0; j < W; j++) *p++ = (double)data.n_alu[j];
    for (int j = 0; j < W; j++) *p++ = (double)data.n_fp[j];
    for (int j = 0; j < W; j++) *p++ = (double)data.n_ls[j];
    for (int j = 0; j < W; j++) *p++ = (double)data.n_load[j];
    for (int j = 0; j < W; j++) *p++ = (double)data.n_store[j];
    for (int j = 0; j < W; j++) *p++ = data.icache_thr[j];
    std::memcpy(p, data.auxiliary, 416 * sizeof(double)); p += 416;
    std::memcpy(p, data.exec_dist, DIST_DIM * sizeof(double)); p += DIST_DIM;
    for (int r = 0; r < N_ROB_SIZES; r++) { std::memcpy(p, data.iss_dist[r], DIST_DIM*sizeof(double)); p += DIST_DIM; }
    for (int r = 0; r < N_ROB_SIZES; r++) { std::memcpy(p, data.com_dist[r], DIST_DIM*sizeof(double)); p += DIST_DIM; }
    for (int j = 0; j < W; j++) *p++ = data.T_crit[j];

    assert(p - buf.data() == total);
    write_npy(path, buf.data(), total);
    printf("    Saved %s (%.1f MB)\n", path.c_str(), total * 8.0 / 1024 / 1024);
}

bool dse_load(const std::string& path, DsePrecomputed& data) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;
    char magic[8];
    if (fread(magic, 1, 8, f) != 8 || magic[0] != '\x93') { fclose(f); return false; }
    uint16_t hlen;
    if (fread(&hlen, 2, 1, f) != 1) { fclose(f); return false; }
    fseek(f, hlen, SEEK_CUR);

    double wv;
    if (fread(&wv, sizeof(double), 1, f) != 1) { fclose(f); return false; }
    int W = (int)wv;
    data.W = W;

    auto read_vv = [&](std::vector<std::vector<double>>& vv, int N) {
        vv.resize(N, std::vector<double>(W));
        for (int i = 0; i < N; i++)
            if ((int)fread(vv[i].data(), sizeof(double), W, f) != W) return false;
        return true;
    };
    auto read_vi = [&](std::vector<int>& v) {
        v.resize(W);
        std::vector<double> tmp(W);
        if ((int)fread(tmp.data(), sizeof(double), W, f) != W) return false;
        for (int j = 0; j < W; j++) v[j] = (int)tmp[j];
        return true;
    };

    if (!read_vv(data.rob_thr, DSE_N_ROB)) { fclose(f); return false; }
    if (!read_vv(data.lq_thr, DSE_N_LQ))   { fclose(f); return false; }
    if (!read_vv(data.sq_thr, DSE_N_SQ))    { fclose(f); return false; }
    if (!read_vv(data.fetchw_thr, DSE_N_FW)){ fclose(f); return false; }
    if (!read_vv(data.fetchbuf_thr, DSE_N_FB)){fclose(f); return false; }
    if (!read_vi(data.n_alu))   { fclose(f); return false; }
    if (!read_vi(data.n_fp))    { fclose(f); return false; }
    if (!read_vi(data.n_ls))    { fclose(f); return false; }
    if (!read_vi(data.n_load))  { fclose(f); return false; }
    if (!read_vi(data.n_store)) { fclose(f); return false; }
    data.icache_thr.resize(W);
    if ((int)fread(data.icache_thr.data(), sizeof(double), W, f) != W) { fclose(f); return false; }
    if (fread(data.auxiliary, sizeof(double), 416, f) != 416) { fclose(f); return false; }
    if (fread(data.exec_dist, sizeof(double), DIST_DIM, f) != DIST_DIM) { fclose(f); return false; }
    for (int r = 0; r < N_ROB_SIZES; r++)
        if (fread(data.iss_dist[r], sizeof(double), DIST_DIM, f) != DIST_DIM) { fclose(f); return false; }
    for (int r = 0; r < N_ROB_SIZES; r++)
        if (fread(data.com_dist[r], sizeof(double), DIST_DIM, f) != DIST_DIM) { fclose(f); return false; }
    data.T_crit.resize(W);
    if ((int)fread(data.T_crit.data(), sizeof(double), W, f) != W) { fclose(f); return false; }
    fclose(f);
    return true;
}

// ─────────────────────────── Random Arch ───────────────────────────

ArchConfig random_arch_cpu_only(const ArchConfig& base, std::mt19937& rng) {
    ArchConfig c = base;
    auto ri = [&](int lo, int hi) { return std::uniform_int_distribution<int>(lo, hi)(rng); };

    c.fetch_width    = ri(4, 12);
    c.decode_width   = ri(4, std::min(12, c.fetch_width));
    c.rename_width   = ri(4, std::min(12, c.decode_width));
    c.dispatch_width = ri(4, std::min(12, c.rename_width));
    c.issue_width    = ri(std::max(4, c.dispatch_width), 12);
    c.commit_width   = ri(std::max(4, c.dispatch_width), 12);
    c.iq_entries       = ri(32, 256);
    c.load_queue_size  = ri(16, 256);
    c.store_queue_size = ri(16, 256);
    int diq = 2 * c.issue_width;
    int rob_floor = std::max(72, diq + c.iq_entries + c.load_queue_size + c.store_queue_size);
    c.rob_size = (rob_floor <= 1024) ? ri(rob_floor, 1024) : 1024;
    c.alu_issue_width = ri(1, 8);
    c.fp_issue_width  = ri(1, 8);
    c.ls_issue_width  = ri(1, 8);
    c.n_ls_pipes      = c.ls_issue_width;
    c.n_l_pipes       = 0;
    c.n_fetch_bufs    = ri(1, 8);
    c.max_icache_fills = 8;

    c.raw[0]=(float)c.fetch_width; c.raw[1]=(float)c.decode_width;
    c.raw[2]=(float)c.rename_width; c.raw[3]=(float)c.dispatch_width;
    c.raw[4]=(float)c.issue_width; c.raw[5]=(float)11; // wbWidth dummy
    c.raw[6]=(float)c.commit_width; c.raw[7]=(float)c.rob_size;
    c.raw[8]=(float)c.iq_entries; c.raw[9]=(float)c.load_queue_size;
    c.raw[10]=(float)c.store_queue_size; c.raw[11]=(float)c.n_fetch_bufs;
    c.raw[12]=(float)c.alu_issue_width; c.raw[13]=4.0f;
    c.raw[14]=(float)c.fp_issue_width; c.raw[15]=4.0f;
    c.raw[16]=(float)c.ls_issue_width;
    return c;
}
