#include "cpu/o3/mscs.hh"

#include <algorithm>
#include <limits>

namespace gem5
{

namespace o3
{

void
MSCSAccountant::init(int w_min, const int delay_cfg[4], FILE *trace,
                     const std::string &diag_path)
{
    W = std::max(1, w_min);
    for (int i = 0; i < 4; i++)
        delayCfg[i] = delay_cfg[i];
    tracePtr = trace;
    diagPath = diag_path;
}

void
MSCSAccountant::tick(Cycles now)
{
    if (!started) {
        started = true;
        c0 = (int64_t)(uint64_t)now;
    }
    int64_t a = (int64_t)(uint64_t)now - c0;
    if ((int64_t)ticked.size() <= a)
        ticked.resize(a + 1, 0);
    ticked[a] = 1;
}

int
MSCSAccountant::rowOf(int stage)
{
    switch (stage) {
      case DEC: return 0;
      case DISP: return 1;
      case ISS: return 2;
      case COM: return 3;
      case IEWD: return IEW_ROW;
      default: return -1;
    }
}

void
MSCSAccountant::blocked(int stage, Cycles now, InstSeqNum held_seq,
                        uint8_t sub)
{
    const int k = rowOf(stage);
    if (!started || k < 0 || held_seq == 0 || sub >= NumSub)
        return;
    int64_t a = (int64_t)(uint64_t)now - c0;
    if (a < 0)
        return;
    auto &v = heldSeq[k];
    auto &u = heldSub[k];
    if ((int64_t)v.size() <= a) {
        v.resize(a + 1, 0);
        u.resize(a + 1, 0);
    }
    if (v[a] == 0 || held_seq < v[a]) {
        v[a] = held_seq;
        u[a] = sub;
    }
}

void
MSCSAccountant::stallReason(int stage, Cycles now, uint8_t sub)
{
    const int k = rowOf(stage);
    if (!started || k < 0 || sub >= NumSub)
        return;
    int64_t a = (int64_t)(uint64_t)now - c0;
    if (a < 0)
        return;
    auto &v = stallSub[k];
    if ((int64_t)v.size() <= a)
        v.resize(a + 1, 0);
    v[a] = sub + 1;
}

void
MSCSAccountant::decodeHandled(Cycles now, int n)
{
    if (!started || n <= 0)
        return;
    int64_t a = (int64_t)(uint64_t)now - c0;
    if (a < 0)
        return;
    if ((int64_t)decProc.size() <= a)
        decProc.resize(a + 1, 0);
    decProc[a] += n;
}

void
MSCSAccountant::fetchMiss(Cycles start, Cycles end, InstSeqNum next_seq)
{
    if (!started)
        return;
    misses.push_back({(int64_t)(uint64_t)start, (int64_t)(uint64_t)end,
                      next_seq});
}

void
MSCSAccountant::commit(const InstRec &r)
{
    if (!started)
        return;
    committed.push_back(r);
}

void
MSCSAccountant::emitLine(std::string prefix)
{
    if (!tracePtr)
        return;
    if (havePending) {
        Result placeholder;          // all zeros: only the last line counts
        writeStack(pendingLine, placeholder);
    }
    pendingLine = std::move(prefix);
    havePending = true;
}

void
MSCSAccountant::writeStack(const std::string &prefix, const Result &r)
{
    const double (*v)[NumClasses] = r.v;
    const double (*b)[NumSub] = r.sub;
    // 28 fields (see mscs.hh): Decode, Dispatch, IEW, Issue, Commit.
    fprintf(tracePtr, "%sSTART,%.6f,%.6f,%.6f,%.6f,", prefix.c_str(),
            v[0][BASE], v[0][FP], v[0][BP], v[0][SP]);
    static const int rows[4] = {1, IEW_ROW, 2, 3};
    for (int i = 0; i < 4; i++) {
        const int k = rows[i];
        fprintf(tracePtr, "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f%s",
                v[k][BASE], v[k][FP], b[k][0], b[k][1], b[k][2], v[k][SP],
                i == 3 ? "\n" : ",");
    }
}

void
MSCSAccountant::finish(std::vector<InstRec> live)
{
    if (finished || !started)
        return;
    finished = true;

    Result r = compute(live);
    if (tracePtr) {
        if (havePending)
            writeStack(pendingLine, r);
        fflush(tracePtr);
    }
}

MSCSAccountant::Result
MSCSAccountant::compute(std::vector<InstRec> &live)
{
    Result res;
    const int64_t n = (int64_t)committed.size();
    const int64_t INF = std::numeric_limits<int64_t>::max() / 4;

    // Diagnostics
    int64_t missing_stamps = 0, base_outside = 0;
    int64_t iew_missing = 0, iew_base_outside = 0;
    int64_t fallback[NumRows] = {}, blocked_cycles[NumRows] = {};
    int64_t inherited_bp[NumRows] = {};
    int64_t violations[NumStamps] = {};
    int64_t emp_min[NumStamps];
    std::fill(emp_min, emp_min + NumStamps, INF);
    int64_t miss_total = (int64_t)misses.size(), miss_valid = 0;

    if (n == 0)
        return res;

    // ---- 1. correct-path sequence: committed, then live (in seq order) ----
    std::sort(live.begin(), live.end(),
              [](const InstRec &a, const InstRec &b) {
                  return a.seq < b.seq; });
    const InstSeqNum last_com_seq = committed.back().seq;
    live.erase(std::remove_if(live.begin(), live.end(),
                              [&](const InstRec &x) {
                                  return x.seq <= last_com_seq; }),
               live.end());

    std::vector<InstRec> all;
    all.reserve(committed.size() + live.size());
    for (const auto &c : committed) {
        InstRec r = c;
        for (int s = 0; s < NumStamps; s++)
            r.t[s] = r.t[s] < 0 ? -1 : r.t[s] - c0;
        // A committed instruction passed every stage; fill a missing
        // stamp from the next stage (counted in the diagnostics).
        for (int s = COM - 1; s >= FETCH; s--) {
            if (r.t[s] < 0) {
                r.t[s] = r.t[s + 1];
                missing_stamps++;
            }
        }
        if (r.t[IEWD] < 0) {
            r.t[IEWD] = r.t[ISS];
            iew_missing++;
        }
        all.push_back(r);
    }
    for (const auto &l : live) {
        InstRec r = l;
        for (int s = 0; s < NumStamps; s++)
            r.t[s] = r.t[s] < 0 ? -1 : r.t[s] - c0;
        all.push_back(r);
    }
    const int64_t n_all = (int64_t)all.size();

    // ---- 2. d_s: minimum latency from s- to s ----
    // (IEW: from Dispatch, always the observed minimum)
    int64_t d[NumStamps] = {};
    auto pred_of = [](int s) { return s == IEWD ? (int)DISP : s - 1; };
    for (int s = DEC; s <= IEWD; s++) {
        const int ps = pred_of(s);
        for (const auto &r : all) {
            if (r.t[s] >= 0 && r.t[ps] >= 0)
                emp_min[s] = std::min(emp_min[s], r.t[s] - r.t[ps]);
        }
        if (s <= COM && delayCfg[s - 1] >= 0)
            d[s] = delayCfg[s - 1];
        else
            d[s] = emp_min[s] == INF ? 1 : std::max<int64_t>(emp_min[s], 0);
        for (const auto &r : all) {
            if (r.t[s] >= 0 && r.t[ps] >= 0 && r.t[s] - r.t[ps] < d[s])
                violations[s]++;
        }
    }

    // ---- 3. per-cycle counts ----
    int64_t max_t = 0;
    for (const auto &r : all)
        for (int s = 0; s < NumStamps; s++)
            max_t = std::max(max_t, r.t[s]);
    int64_t H = std::max<int64_t>(max_t + 1, (int64_t)decProc.size());
    H = std::max<int64_t>(H, (int64_t)ticked.size());
    for (int k = 0; k < NumRows; k++) {
        H = std::max<int64_t>(H, (int64_t)heldSeq[k].size());
        H = std::max<int64_t>(H, (int64_t)stallSub[k].size());
    }

    std::vector<std::vector<uint16_t>> cnt_all(NumStamps,
                                               std::vector<uint16_t>(H, 0));
    std::vector<std::vector<uint16_t>> cnt_com(NumStamps,
                                               std::vector<uint16_t>(H, 0));
    for (int64_t i = 0; i < n_all; i++) {
        const auto &r = all[i];
        for (int s = 0; s < NumStamps; s++) {
            if (r.t[s] < 0)
                continue;
            cnt_all[s][r.t[s]]++;
            if (i < n)
                cnt_com[s][r.t[s]]++;
        }
    }
    decProc.resize(H, 0);
    ticked.resize(H, 0);
    for (int k = 0; k < NumRows; k++) {
        heldSeq[k].resize(H, 0);
        heldSub[k].resize(H, 0);
        stallSub[k].resize(H, 0);
    }

    // M_c: decode cycles in [t_dec(mispredicted branch),
    //                        t_dec(next correct-path instruction))
    std::vector<uint8_t> M(H, 0);
    int64_t m_open_from = INF;
    for (int64_t i = 0; i < n_all; i++) {
        if (!all[i].mispred || all[i].t[DEC] < 0)
            continue;
        int64_t start = all[i].t[DEC];
        int64_t end = (i + 1 < n_all && all[i + 1].t[DEC] >= 0)
                          ? all[i + 1].t[DEC] : INF;
        if (end == INF)
            m_open_from = std::min(m_open_from, start);
        for (int64_t a = start; a < std::min(end, H); a++)
            M[a] = 1;
    }
    auto m_at = [&](int64_t a) {
        return (a >= 0 && a < H && M[a]) || a >= m_open_from;
    };

    // I_c: fetch cycles stalled on an I-side miss whose line delivers a
    // correct-path instruction (next allocated seq is correct-path).
    std::vector<InstSeqNum> com_seqs, live_seqs;
    com_seqs.reserve(n);
    for (const auto &c : committed)
        com_seqs.push_back(c.seq);
    for (const auto &l : live)
        live_seqs.push_back(l.seq);
    InstSeqNum max_seq = all.back().seq;
    std::vector<uint8_t> I(H, 0);
    for (const auto &m : misses) {
        bool valid = std::binary_search(com_seqs.begin(), com_seqs.end(),
                                        m.nextSeq) ||
                     std::binary_search(live_seqs.begin(), live_seqs.end(),
                                        m.nextSeq) ||
                     m.nextSeq > max_seq;
        if (!valid)
            continue;
        miss_valid++;
        int64_t s = std::max<int64_t>(m.start - c0, 0);
        int64_t e = std::min<int64_t>(m.end - c0, H - 1);
        for (int64_t a = s; a <= e; a++)
            I[a] = 1;
    }

    // B_c^s (Table 1): structural condition sampled in cycle a (idle cycles
    // keep the last sampled state) and the stuck instruction is on the
    // correct path (committed, or still in flight at the end).
    auto correct = [&](InstSeqNum q) {
        return std::binary_search(com_seqs.begin(), com_seqs.end(), q) ||
               std::binary_search(live_seqs.begin(), live_seqs.end(), q) ||
               q > max_seq;
    };
    // Bc holds 0 (not blocked) or 1 + BPSub; Sc holds the BPSub of a
    // back-end stall outside Table 1 (default 0), same idle-cycle rule.
    std::vector<std::vector<uint8_t>> Bc(NumRows, std::vector<uint8_t>(H, 0));
    std::vector<std::vector<uint8_t>> Sc(NumRows, std::vector<uint8_t>(H, 0));
    uint8_t B_last[NumRows] = {}, S_last[NumRows] = {};
    for (int k = 0; k < NumRows; k++) {
        uint8_t last = 0, slast = 0;
        for (int64_t a = 0; a < H; a++) {
            if (ticked[a]) {
                const InstSeqNum q = heldSeq[k][a];
                last = (q != 0 && correct(q)) ? 1 + heldSub[k][a] : 0;
                slast = stallSub[k][a] ? stallSub[k][a] - 1 : 0;
            }
            Bc[k][a] = last;
            Sc[k][a] = slast;
        }
        B_last[k] = last;
        S_last[k] = slast;
    }
    auto blocked_at = [&](int k, int64_t a) -> uint8_t {
        if (a < 0)
            return 0;
        return a < H ? Bc[k][a] : B_last[k];
    };
    auto stall_at = [&](int k, int64_t a) -> uint8_t {
        if (a < 0)
            return 0;
        return a < H ? Sc[k][a] : S_last[k];
    };

    // ---- 4. window offsets and length C ----
    int64_t O[NumStamps];
    O[DEC] = 0;
    O[DISP] = O[DEC] + d[DISP];
    O[ISS] = O[DISP] + d[ISS];
    O[COM] = O[ISS] + d[COM];
    O[FETCH] = O[DEC] - d[DEC];
    O[IEWD] = O[DISP] + d[IEWD];

    int64_t C = 1;
    for (int s = DEC; s <= COM; s++) {
        int64_t carry = 0, placed = 0, last = -1;
        for (int64_t a = 0; placed < n; a++) {
            int64_t c = a < H ? cnt_com[s][a] : 0;
            if (a >= H && carry == 0)
                break;               // cannot happen: all n are stamped
            int64_t tot = c + carry;
            int64_t b = std::min<int64_t>(tot, W);
            carry = tot - b;
            placed += b;
            if (b > 0) {
                last = a;
                if (a < O[s])
                    base_outside += b;
            }
        }
        C = std::max(C, last - O[s] + 1);
    }

    // ---- 5. classify every slot ----
    int64_t counts[NumRows][NumClasses] = {};
    int64_t subcounts[NumRows][NumSub] = {};
    int64_t carry[NumStamps] = {};
    int64_t iew_base_in = 0;
    std::vector<uint8_t> prev(W), cur(W), disp_row(W);

    // Classify the W slots of stage s (row k) at window position j, given
    // the slots of its predecessor at the same position (Eq. 4).
    auto classify = [&](int s, int k, int64_t j,
                        const std::vector<uint8_t> &pred,
                        std::vector<uint8_t> &out) -> int64_t {
        const int64_t a = O[s] + j;
        int64_t c = (a >= 0 && a < H) ? cnt_com[s][a] : 0;
        int64_t tot = c + carry[s];
        int64_t b = std::min<int64_t>(tot, W);
        carry[s] = tot - b;

        // B_c^s: Table 1 condition on a correct-path instruction
        const uint8_t blk = blocked_at(k, a);
        const bool blocked = blk != 0;
        if (blocked)
            blocked_cycles[k]++;

        uint8_t dec_cls = BP;
        if (s == DEC && !blocked) {
            const int64_t proc = (a >= 0 && a < H) ? decProc[a] : 0;
            const int64_t corr = (a >= 0 && a < H) ? cnt_all[DEC][a] : 0;
            if (proc > corr) {
                dec_cls = SP;                             // X_c
            } else {
                const int64_t fa = a - d[DEC];
                const bool ic = fa >= 0 && fa < H && I[fa];
                dec_cls = (ic || !m_at(a)) ? FP : SP;
            }
        }

        for (int p = 0; p < W; p++) {
            uint8_t cls;
            uint8_t sub = 0;
            if (p < b) {
                cls = BASE;                               // R1
            } else if (blocked) {
                cls = BP;
                sub = blk - 1;
            } else if (s == DEC) {
                cls = dec_cls;                            // Eq. 3
            } else {
                cls = pred[p];                            // Eq. 4
                if (cls == BASE) {
                    fallback[k]++;
                    cls = BP;
                    sub = stall_at(k, a);
                } else if (cls == BP) {
                    inherited_bp[k]++;
                }
            }
            out[p] = cls;
            counts[k][cls]++;
            if (cls == BP)
                subcounts[k][sub]++;
        }
        return b;
    };

    for (int64_t j = 0; j < C; j++) {
        for (int s = DEC; s <= COM; s++) {
            classify(s, rowOf(s), j, prev, cur);
            std::swap(prev, cur);
            if (s == DISP)
                disp_row = prev;
        }
        iew_base_in += classify(IEWD, IEW_ROW, j, disp_row, cur);
    }
    iew_base_outside = n - iew_base_in;

    const double denom = (double)C * (double)W;
    for (int k = 0; k < NumRows; k++) {
        for (int c = 0; c < NumClasses; c++)
            res.v[k][c] = counts[k][c] / denom;
        for (int u = 0; u < NumSub; u++)
            res.sub[k][u] = subcounts[k][u] / denom;
    }
    res.valid = true;

    // ---- 6. diagnostics ----
    if (!diagPath.empty()) {
        FILE *f = fopen(diagPath.c_str(), "w");
        if (f) {
            static const char *sn[4] = {"decode", "dispatch", "issue",
                                        "commit"};
            fprintf(f, "window_start_cycle=%ld\n", (long)c0);
            fprintf(f, "n_committed=%ld\nn_live_at_end=%ld\n", (long)n,
                    (long)live.size());
            fprintf(f, "W_min=%d\nC=%ld\nCPI=%.6f\n", W, (long)C,
                    (double)C / n);
            for (int s = DEC; s <= COM; s++) {
                fprintf(f, "d_%s=%ld min_latency_%s=%ld violations_%s=%ld\n",
                        sn[s - DEC], (long)d[s], sn[s - DEC],
                        (long)(emp_min[s] == INF ? -1 : emp_min[s]),
                        sn[s - DEC], (long)violations[s]);
            }
            fprintf(f, "missing_stamps=%ld\nfallback_slots=%ld\n"
                    "base_outside_window=%ld\n", (long)missing_stamps,
                    (long)(fallback[0] + fallback[1] + fallback[2] +
                           fallback[3]),
                    (long)base_outside);
            for (int k = 0; k < 4; k++) {
                fprintf(f, "%s_blocked_cycles=%ld %s_fallback_slots=%ld\n",
                        sn[k], (long)blocked_cycles[k], sn[k],
                        (long)fallback[k]);
            }
            fprintf(f, "fetch_misses=%ld fetch_misses_correct_path=%ld\n",
                    (long)miss_total, (long)miss_valid);
            for (int k = 0; k < 4; k++) {
                fprintf(f, "%s_slots base=%ld fp=%ld bp=%ld sp=%ld\n",
                        sn[k], (long)counts[k][BASE], (long)counts[k][FP],
                        (long)counts[k][BP], (long)counts[k][SP]);
            }
            // BP breakdown (trace sub-fields) and the IEW row
            static const char *rn[NumRows] = {"decode", "dispatch", "issue",
                                              "commit", "iew"};
            for (int k = 1; k < NumRows; k++) {
                fprintf(f, "%s_bp_sub=%ld,%ld,%ld %s_bp_inherited=%ld\n",
                        rn[k], (long)subcounts[k][0], (long)subcounts[k][1],
                        (long)subcounts[k][2], rn[k], (long)inherited_bp[k]);
            }
            fprintf(f, "d_iew=%ld min_latency_iew=%ld violations_iew=%ld\n",
                    (long)d[IEWD],
                    (long)(emp_min[IEWD] == INF ? -1 : emp_min[IEWD]),
                    (long)violations[IEWD]);
            fprintf(f, "iew_missing_stamps=%ld iew_base_outside_window=%ld\n",
                    (long)iew_missing, (long)iew_base_outside);
            fprintf(f, "iew_blocked_cycles=%ld iew_fallback_slots=%ld\n",
                    (long)blocked_cycles[IEW_ROW], (long)fallback[IEW_ROW]);
            fprintf(f, "iew_slots base=%ld fp=%ld bp=%ld sp=%ld\n",
                    (long)counts[IEW_ROW][BASE], (long)counts[IEW_ROW][FP],
                    (long)counts[IEW_ROW][BP], (long)counts[IEW_ROW][SP]);
            fclose(f);
        }
    }
    return res;
}

} // namespace o3
} // namespace gem5
