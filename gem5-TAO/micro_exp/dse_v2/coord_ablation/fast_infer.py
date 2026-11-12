"""
Bit-exact, faster replacement for dse_exp/dse_infer.dse_infer (the original is not modified).

Same operations on the same inputs, so outputs are identical (verify_fast.py checks np.array_equal
against the original on every slice).  Speed comes from:
  1. per-slice template: blocks that do not depend on the config (Icache fills, auxiliary,
     exec_dist, iss/com dists) are encoded / copied once
  2. per-slice memo of every config-dependent block, keyed by exactly the config values it reads
     (a refine step evaluates single-parameter neighbours, so most blocks are reused)
  3. encode_distribution with its constant linspace grids precomputed
"""
import numpy as np
from dse_infer import (WINDOW_SIZE, DIST_DIM, DSE_TOTAL, N_ARCH_PARAMS, N_FEATURES, N_ROB_SIZES,
                       DSE_ROB_START, DSE_N_ROB, DSE_LQ_START, DSE_N_LQ, DSE_SQ_START, DSE_N_SQ,
                       DSE_FW_START, DSE_N_FW, DSE_NB_START, DSE_N_NB, DSE_DW_START, DSE_N_DW)

_P50 = np.linspace(0.0, 1.0, 50)
_GRID = {}
CACHE_CAP = 400_000          # memo entries per worker before the memo is cleared


def encode_distribution(vals):
    """Identical arithmetic to dse_infer.encode_distribution."""
    N = len(vals)
    out = np.zeros(DIST_DIM)
    if N == 0:
        return out
    sorted_v = np.sort(vals)
    if N == 1:
        out[:50] = sorted_v[0]
    else:
        g = _GRID.get(N)
        if g is None:
            g = _GRID[N] = np.linspace(0.0, 1.0, N)
        out[:50] = np.interp(_P50, g, sorted_v)
    total_w = sorted_v.sum()
    if total_w > 1e-12:
        cum = np.cumsum(sorted_v)
        targets = np.linspace(0.0, total_w, 50)
        indices = np.searchsorted(cum, targets)
        np.clip(indices, 0, N - 1, out=indices)
        out[50:100] = sorted_v[indices]
    out[100] = sorted_v.mean()
    return out


class _Slice:
    def __init__(self, pre):
        self.pre = pre
        K = WINDOW_SIZE
        t = np.zeros(DSE_TOTAL)
        t[909:1010] = encode_distribution(pre['icache_thr'])
        t[1111:1527] = pre['auxiliary']
        t[1527:1628] = pre['exec_dist']
        p = 1628
        for r in range(N_ROB_SIZES):
            t[p:p + DIST_DIM] = pre['iss_dist'][r]; p += DIST_DIM
        for r in range(N_ROB_SIZES):
            t[p:p + DIST_DIM] = pre['com_dist'][r]; p += DIST_DIM
        self.template = t
        self.T_fe = np.where(pre['icache_thr'] > 1e-9, K / pre['icache_thr'], 0.0)
        self.nld = pre['n_load'].astype(float)
        self.nst = pre['n_store'].astype(float)
        self.memo = {}


class FastSliceInfer:
    def __init__(self):
        self.n_memo = 0

    def _get(self, s, key, fn):
        v = s.memo.get(key)
        if v is None:
            v = fn()
            s.memo[key] = v
            self.n_memo += 1
        return v

    def __call__(self, s, config):
        pre, K, W = s.pre, WINDOW_SIZE, s.pre['W']
        fetch_w, decode_w = config["fetchWidth"], config["decodeWidth"]
        rename_w = dispatch_w = issue_w = config["dispatchWidth"]
        commit_w = config["commitWidth"]
        rob_size, iq_entries = config["numROBEntries"], config["numIQEntries"]
        lq_entries, sq_entries = config["LQEntries"], config["SQEntries"]
        n_fetch_bufs = config["fetchBufferSize"]
        alu_w, fp_w, ls_w = config["IntALU"], config["FP_ALU"], config["RdWrPort"]
        n_ls_pipes, n_l_pipes = config["RdWrPort"], 0
        Wm = min(fetch_w, decode_w, rename_w, dispatch_w, issue_w, commit_w)
        if Wm < 1:
            Wm = 1

        f = s.template.copy()
        rob_idx = max(0, min(rob_size - DSE_ROB_START, DSE_N_ROB - 1))
        f[0:101] = self._get(s, ("rob", rob_idx), lambda: encode_distribution(pre['rob_thr'][rob_idx]))
        lq_idx = max(0, min(lq_entries - DSE_LQ_START, DSE_N_LQ - 1))
        f[101:202] = self._get(s, ("lq", lq_idx), lambda: encode_distribution(pre['lq_thr'][lq_idx]))
        sq_idx = max(0, min(sq_entries - DSE_SQ_START, DSE_N_SQ - 1))
        f[202:303] = self._get(s, ("sq", sq_idx), lambda: encode_distribution(pre['sq_thr'][sq_idx]))
        f[303:404] = self._get(s, ("alu", alu_w), lambda: encode_distribution(
            np.where(pre['n_alu'] > 0, K * alu_w / pre['n_alu'], float(K))))
        f[404:505] = self._get(s, ("fp", fp_w), lambda: encode_distribution(
            np.where(pre['n_fp'] > 0, K * fp_w / pre['n_fp'], float(K))))
        f[505:606] = self._get(s, ("ls", ls_w), lambda: encode_distribution(
            np.where(pre['n_ls'] > 0, K * ls_w / pre['n_ls'], float(K))))
        fw_idx = max(0, min(fetch_w - DSE_FW_START, DSE_N_FW - 1))
        f[606:707] = self._get(s, ("fw", fw_idx), lambda: encode_distribution(pre['fw_thr'][fw_idx]))

        def pipes():
            nls, nl = n_ls_pipes, n_l_pipes
            tot = nls + nl
            nld, nst = s.nld, s.nst
            if tot == 0:
                thr_lo = np.zeros(W); thr_hi = np.zeros(W)
            else:
                tm = nld / tot + np.where(nls > 0, nst / nls, 0.0)
                thr_lo = np.where(tm > 0, K / tm, float(K))
                ts = np.where(nls > 0, nst / nls, 0.0)
                rem = np.maximum(0.0, nld - (ts * nl if nl > 0 else 0.0))
                tn = ts + rem / tot
                thr_hi = np.where(tn > 0, K / tn, float(K))
            return np.concatenate([encode_distribution(thr_lo), encode_distribution(thr_hi)])
        f[707:909] = self._get(s, ("pipes", n_ls_pipes), pipes)

        nb_idx = max(0, min(n_fetch_bufs - DSE_NB_START, DSE_N_NB - 1))
        dw_idx = max(0, min(decode_w - DSE_DW_START, DSE_N_DW - 1))
        fb_idx = nb_idx * DSE_N_DW + dw_idx
        f[1010:1111] = self._get(s, ("fb", fb_idx), lambda: encode_distribution(pre['fb_thr'][fb_idx]))

        f[3850:3873] = [fetch_w, decode_w, rename_w, dispatch_w, issue_w, config["wbWidth"], commit_w,
                        rob_size, iq_entries, lq_entries, sq_entries, n_fetch_bufs,
                        config["IntALU"], config["IntMultDiv"], config["FP_ALU"], config["FP_MultDiv"],
                        config["RdWrPort"], config["l1_size"], config["l2_size"], config["l1d_assoc"],
                        config["l1i_assoc"], config["l2_assoc"], config["bp_type"]]

        def d3():
            C_FDQ, C_DQ, C_DIQ = 2 * decode_w, 2 * dispatch_w, 2 * issue_w
            C_IQ, C_LQ, C_ROB = iq_entries, lq_entries, rob_size
            T_rob = np.where(pre['rob_thr'][rob_idx] > 1e-9, K / pre['rob_thr'][rob_idx], 0.0)
            P_FE = np.maximum(0.0, s.T_fe * Wm - K)
            P_ROB = np.maximum(0.0, T_rob * Wm - K)
            C_ilp = np.maximum(0.0, (T_rob - pre['T_crit']) * Wm)
            fe_dec = np.maximum(0.0, P_FE - C_FDQ)
            fe_dis = np.maximum(0.0, fe_dec - C_DQ)
            fe_iss = np.maximum(0.0, fe_dis - C_DIQ - C_IQ - C_LQ)
            fe_com = np.maximum(0.0, fe_dis - C_ROB)
            be_com = P_ROB
            be_iss = np.maximum(0.0, be_com - C_ilp)
            be_dis = np.maximum(0.0, be_com - min(C_ROB, C_IQ + C_LQ))
            be_dec = np.maximum(0.0, be_dis - C_DQ)
            return np.concatenate([encode_distribution(x) for x in
                                   [fe_dec, fe_dis, fe_iss, fe_com, be_dec, be_dis, be_iss, be_com]])
        f[N_FEATURES:DSE_TOTAL] = self._get(
            s, ("d3", rob_idx, Wm, decode_w, dispatch_w, issue_w, iq_entries, lq_entries, rob_size), d3)
        return f


class FastEngine:
    """Drop-in for a (single-workload) DseEngine: get_features(config, workload) -> list of (4681,)."""
    def __init__(self, base):
        self.max_slices = base.max_slices
        self.workload_names = set(base.workload_names)
        self.slices = {l1: {wl: [_Slice(pre) for _, pre in sl] for wl, sl in d.items()}
                       for l1, d in base.data.items()}
        self.infer = FastSliceInfer()

    def get_features(self, config, workload):
        l1 = config["l1_size"]
        if l1 not in self.slices or workload not in self.slices[l1]:
            raise ValueError(f"no precompute for L1={l1} / {workload}")
        if self.infer.n_memo > CACHE_CAP:
            for d in self.slices.values():
                for sl in d.values():
                    for s in sl:
                        s.memo.clear()
            self.infer.n_memo = 0
        return [self.infer(s, config) for s in self.slices[l1][workload]]
