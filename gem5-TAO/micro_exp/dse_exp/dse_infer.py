"""
GEM DSE Inference: Load precomputed timeseries, generate 4681-dim features.

Usage:
    from dse_infer import DseEngine

    engine = DseEngine("/scratch/gem5-TAO/micro_exp/dse_exp/precompute", max_slices=200)

    config = {
        "fetchWidth": 8, "decodeWidth": 8, "dispatchWidth": 8,
        "commitWidth": 8, "wbWidth": 8,
        "numROBEntries": 192, "numIQEntries": 64,
        "LQEntries": 32, "SQEntries": 32, "fetchBufferSize": 32,
        "IntALU": 2, "IntMultDiv": 1, "FP_ALU": 2, "FP_MultDiv": 1,
        "RdWrPort": 2,
        "l1_size": 64, "l2_size": 1024,
        "l1d_assoc": 2, "l1i_assoc": 2, "l2_assoc": 8, "bp_type": 0,
    }

    # config["l1_size"] selects which L1 precompute to use
    features_mcf   = engine.get_features(config, "605.mcf_s")       # list of (4681,)
    features_xalan = engine.get_features(config, "623.xalancbmk_s") # list of (4681,)
"""

import numpy as np
import os
import glob


# ═══ Constants (must match dse_precompute.h) ═══
WINDOW_SIZE = 400
DIST_DIM = 101

DSE_ROB_START, DSE_ROB_END = 72, 1024
DSE_LQ_START, DSE_LQ_END = 16, 256
DSE_SQ_START, DSE_SQ_END = 16, 256
DSE_FW_START, DSE_FW_END = 4, 12
DSE_NB_START, DSE_NB_END = 1, 8
DSE_DW_START, DSE_DW_END = 4, 12

DSE_N_ROB = DSE_ROB_END - DSE_ROB_START + 1   # 953
DSE_N_LQ  = DSE_LQ_END  - DSE_LQ_START  + 1   # 241
DSE_N_SQ  = DSE_SQ_END  - DSE_SQ_START  + 1   # 241
DSE_N_FW  = DSE_FW_END  - DSE_FW_START  + 1   # 9
DSE_N_NB  = DSE_NB_END  - DSE_NB_START  + 1   # 8
DSE_N_DW  = DSE_DW_END  - DSE_DW_START  + 1   # 9
DSE_N_FB  = DSE_N_NB * DSE_N_DW                # 72

N_ROB_SIZES = 11
ROB_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

N_FEATURES = 3873
N_GEM_PRESSURE = 808
N_ARCH_PARAMS = 23
DSE_TOTAL = N_FEATURES + N_GEM_PRESSURE  # 4681


# ═══ Distribution encoding (exact port of C++ encode_distribution) ═══

def encode_distribution(vals):
    """Encode a timeseries into 101-dim CDF: 50 regular + 50 size-weighted + mean."""
    N = len(vals)
    out = np.zeros(DIST_DIM)
    if N == 0:
        return out
    sorted_v = np.sort(vals)
    # Regular percentiles [0:50]
    if N == 1:
        out[:50] = sorted_v[0]
    else:
        p = np.linspace(0.0, 1.0, 50)
        out[:50] = np.interp(p, np.linspace(0.0, 1.0, N), sorted_v)
    # Size-weighted percentiles [50:100]
    total_w = sorted_v.sum()
    if total_w > 1e-12:
        cum = np.cumsum(sorted_v)
        targets = np.linspace(0.0, total_w, 50)
        indices = np.searchsorted(cum, targets)
        np.clip(indices, 0, N - 1, out=indices)
        out[50:100] = sorted_v[indices]
    # Mean [100]
    out[100] = sorted_v.mean()
    return out


# ═══ Load precomputed .npy ═══

def load_precomputed(path):
    """Load dse_precomputed.npy, return dict of arrays."""
    raw = np.load(path)
    p = 0

    W = int(raw[p]); p += 1

    rob_thr = raw[p:p + DSE_N_ROB * W].reshape(DSE_N_ROB, W); p += DSE_N_ROB * W
    lq_thr  = raw[p:p + DSE_N_LQ * W].reshape(DSE_N_LQ, W);   p += DSE_N_LQ * W
    sq_thr  = raw[p:p + DSE_N_SQ * W].reshape(DSE_N_SQ, W);    p += DSE_N_SQ * W
    fw_thr  = raw[p:p + DSE_N_FW * W].reshape(DSE_N_FW, W);    p += DSE_N_FW * W
    fb_thr  = raw[p:p + DSE_N_FB * W].reshape(DSE_N_FB, W);    p += DSE_N_FB * W

    n_alu   = raw[p:p + W].astype(int); p += W
    n_fp    = raw[p:p + W].astype(int); p += W
    n_ls    = raw[p:p + W].astype(int); p += W
    n_load  = raw[p:p + W].astype(int); p += W
    n_store = raw[p:p + W].astype(int); p += W

    icache_thr = raw[p:p + W]; p += W
    auxiliary  = raw[p:p + 416]; p += 416
    exec_dist  = raw[p:p + DIST_DIM]; p += DIST_DIM

    iss_dist = np.zeros((N_ROB_SIZES, DIST_DIM))
    for r in range(N_ROB_SIZES):
        iss_dist[r] = raw[p:p + DIST_DIM]; p += DIST_DIM

    com_dist = np.zeros((N_ROB_SIZES, DIST_DIM))
    for r in range(N_ROB_SIZES):
        com_dist[r] = raw[p:p + DIST_DIM]; p += DIST_DIM

    T_crit = raw[p:p + W]; p += W

    return {
        'W': W,
        'rob_thr': rob_thr, 'lq_thr': lq_thr, 'sq_thr': sq_thr,
        'fw_thr': fw_thr, 'fb_thr': fb_thr,
        'n_alu': n_alu, 'n_fp': n_fp, 'n_ls': n_ls,
        'n_load': n_load, 'n_store': n_store,
        'icache_thr': icache_thr,
        'auxiliary': auxiliary, 'exec_dist': exec_dist,
        'iss_dist': iss_dist, 'com_dist': com_dist,
        'T_crit': T_crit,
    }


# ═══ Infer 4681-dim features ═══

def dse_infer(pre, config):
    """
    Compute 4681-dim feature vector from precomputed timeseries + config dict.
    Exact match with C++ dse_infer().
    """
    W = pre['W']
    K = WINDOW_SIZE
    features = np.zeros(DSE_TOTAL)

    # ── Parse config ──
    fetch_w    = config["fetchWidth"]
    decode_w   = config["decodeWidth"]
    rename_w   = config["dispatchWidth"]   # dispatchWidth = renameWidth = issueWidth
    dispatch_w = config["dispatchWidth"]
    issue_w    = config["dispatchWidth"]
    wb_w       = config["wbWidth"]
    commit_w   = config["commitWidth"]
    rob_size   = config["numROBEntries"]
    iq_entries = config["numIQEntries"]
    lq_entries = config["LQEntries"]
    sq_entries = config["SQEntries"]
    n_fetch_bufs = config["fetchBufferSize"]
    alu_w      = config["IntALU"]
    fp_w       = config["FP_ALU"]
    ls_w       = config["RdWrPort"]
    n_ls_pipes = config["RdWrPort"]
    n_l_pipes  = 0

    Wm = min(fetch_w, decode_w, rename_w, dispatch_w, issue_w, commit_w)
    if Wm < 1:
        Wm = 1

    # ── Build 23-dim arch params (matching ArchConfig::raw[] layout) ──
    arch_raw = np.zeros(N_ARCH_PARAMS)
    arch_raw[0]  = fetch_w
    arch_raw[1]  = decode_w
    arch_raw[2]  = rename_w
    arch_raw[3]  = dispatch_w
    arch_raw[4]  = issue_w
    arch_raw[5]  = wb_w
    arch_raw[6]  = commit_w
    arch_raw[7]  = rob_size
    arch_raw[8]  = iq_entries
    arch_raw[9]  = lq_entries
    arch_raw[10] = sq_entries
    arch_raw[11] = n_fetch_bufs
    arch_raw[12] = config["IntALU"]
    arch_raw[13] = config["IntMultDiv"]
    arch_raw[14] = config["FP_ALU"]
    arch_raw[15] = config["FP_MultDiv"]
    arch_raw[16] = config["RdWrPort"]
    arch_raw[17] = config["l1_size"]
    arch_raw[18] = config["l2_size"]
    arch_raw[19] = config["l1d_assoc"]
    arch_raw[20] = config["l1i_assoc"]
    arch_raw[21] = config["l2_assoc"]
    arch_raw[22] = config["bp_type"]

    p = 0  # output pointer

    # ════════ Concorde 3873 dims ════════

    # [0:101] ROB throughput
    rob_idx = max(0, min(rob_size - DSE_ROB_START, DSE_N_ROB - 1))
    features[p:p+DIST_DIM] = encode_distribution(pre['rob_thr'][rob_idx])
    p += DIST_DIM

    # [101:202] LQ throughput
    lq_idx = max(0, min(lq_entries - DSE_LQ_START, DSE_N_LQ - 1))
    features[p:p+DIST_DIM] = encode_distribution(pre['lq_thr'][lq_idx])
    p += DIST_DIM

    # [202:303] SQ throughput
    sq_idx = max(0, min(sq_entries - DSE_SQ_START, DSE_N_SQ - 1))
    features[p:p+DIST_DIM] = encode_distribution(pre['sq_thr'][sq_idx])
    p += DIST_DIM

    # [303:404] ALU bandwidth
    thr = np.where(pre['n_alu'] > 0, K * alu_w / pre['n_alu'], float(K))
    features[p:p+DIST_DIM] = encode_distribution(thr)
    p += DIST_DIM

    # [404:505] FP bandwidth
    thr = np.where(pre['n_fp'] > 0, K * fp_w / pre['n_fp'], float(K))
    features[p:p+DIST_DIM] = encode_distribution(thr)
    p += DIST_DIM

    # [505:606] LS bandwidth
    thr = np.where(pre['n_ls'] > 0, K * ls_w / pre['n_ls'], float(K))
    features[p:p+DIST_DIM] = encode_distribution(thr)
    p += DIST_DIM

    # [606:707] Fetch width
    fw_idx = max(0, min(fetch_w - DSE_FW_START, DSE_N_FW - 1))
    features[p:p+DIST_DIM] = encode_distribution(pre['fw_thr'][fw_idx])
    p += DIST_DIM

    # [707:909] Pipes lower/upper
    nls = n_ls_pipes
    nl = n_l_pipes
    tot = nls + nl
    nld = pre['n_load'].astype(float)
    nst = pre['n_store'].astype(float)

    if tot == 0:
        thr_lo = np.zeros(W)
        thr_hi = np.zeros(W)
    else:
        # Lower bound
        tm = nld / tot + np.where(nls > 0, nst / nls, 0.0)
        thr_lo = np.where(tm > 0, K / tm, float(K))
        # Upper bound
        ts = np.where(nls > 0, nst / nls, 0.0)
        rem = np.maximum(0.0, nld - (ts * nl if nl > 0 else 0.0))
        tn = ts + rem / tot
        thr_hi = np.where(tn > 0, K / tn, float(K))

    features[p:p+DIST_DIM] = encode_distribution(thr_lo)
    p += DIST_DIM
    features[p:p+DIST_DIM] = encode_distribution(thr_hi)
    p += DIST_DIM

    # [909:1010] Icache fills (fixed)
    features[p:p+DIST_DIM] = encode_distribution(pre['icache_thr'])
    p += DIST_DIM

    # [1010:1111] Fetch buffers
    nb_idx = max(0, min(n_fetch_bufs - DSE_NB_START, DSE_N_NB - 1))
    dw_idx = max(0, min(decode_w - DSE_DW_START, DSE_N_DW - 1))
    fb_idx = nb_idx * DSE_N_DW + dw_idx
    features[p:p+DIST_DIM] = encode_distribution(pre['fb_thr'][fb_idx])
    p += DIST_DIM

    assert p == 1111

    # [1111:1527] Auxiliary (fixed)
    features[p:p+416] = pre['auxiliary']
    p += 416
    assert p == 1527

    # [1527:1628] exec_dist (fixed)
    features[p:p+DIST_DIM] = pre['exec_dist']
    p += DIST_DIM

    # [1628:2739] iss_dist × 11
    for r in range(N_ROB_SIZES):
        features[p:p+DIST_DIM] = pre['iss_dist'][r]
        p += DIST_DIM

    # [2739:3850] com_dist × 11
    for r in range(N_ROB_SIZES):
        features[p:p+DIST_DIM] = pre['com_dist'][r]
        p += DIST_DIM
    assert p == 3850

    # [3850:3873] Arch params
    features[p:p+N_ARCH_PARAMS] = arch_raw
    p += N_ARCH_PARAMS
    assert p == N_FEATURES  # 3873

    # ════════ GEM D3 808 dims ════════

    C_FDQ = 2 * decode_w
    C_DQ  = 2 * dispatch_w
    C_DIQ = 2 * issue_w
    C_IQ  = iq_entries
    C_LQ  = lq_entries
    C_ROB = rob_size

    # T_rob at target ROB size
    T_rob = np.where(pre['rob_thr'][rob_idx] > 1e-9,
                     K / pre['rob_thr'][rob_idx], 0.0)
    T_fe  = np.where(pre['icache_thr'] > 1e-9,
                     K / pre['icache_thr'], 0.0)

    P_FE  = np.maximum(0.0, T_fe * Wm - K)
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

    gp = p
    for stream in [fe_dec, fe_dis, fe_iss, fe_com, be_dec, be_dis, be_iss, be_com]:
        features[gp:gp+DIST_DIM] = encode_distribution(stream)
        gp += DIST_DIM
    assert gp == DSE_TOTAL  # 4681

    return features


# ═══ Engine class ═══

class DseEngine:
    """
    Load precomputed slices for multiple L1 configs.

    Directory structure:
        precompute/l1_32/605.mcf_s/slice_0/dse_precomputed.npy
        precompute/l1_64/605.mcf_s/slice_0/dse_precomputed.npy
        precompute/l1_128/...
        precompute/l1_256/...

    Usage:
        engine = DseEngine("/scratch/.../precompute", max_slices=200)
        features = engine.get_features(config, "605.mcf_s")
        # config["l1_size"] selects which L1 precompute to use
    """

    def __init__(self, precompute_base, max_slices=200):
        """
        Args:
            precompute_base: root dir containing l1_32/, l1_64/, l1_128/, l1_256/
            max_slices: max slices per (workload, L1) combination
        """
        self.max_slices = max_slices
        # Structure: {l1_size: {workload: [(slice_id, pre), ...]}}
        self.data = {}
        self.workload_names = set()

        l1_dirs = sorted(glob.glob(os.path.join(precompute_base, "l1_*")))
        for l1_dir in l1_dirs:
            l1_val = int(os.path.basename(l1_dir).replace("l1_", ""))
            self.data[l1_val] = {}

            bench_dirs = sorted(glob.glob(os.path.join(l1_dir, "*")))
            for bench_dir in bench_dirs:
                wl_name = os.path.basename(bench_dir)
                if not os.path.isdir(bench_dir):
                    continue
                pattern = os.path.join(bench_dir, "slice_*", "dse_precomputed.npy")
                files = sorted(glob.glob(pattern))
                slices = []
                for f in files:
                    slice_dir = os.path.dirname(f)
                    slice_id = int(os.path.basename(slice_dir).replace("slice_", ""))
                    if slice_id % 2 != 0:
                        continue
                    if len(slices) >= max_slices:
                        break
                    pre = load_precomputed(f)
                    slices.append((slice_id, pre))
                if slices:
                    self.data[l1_val][wl_name] = slices
                    self.workload_names.add(wl_name)

            n_wl = len(self.data[l1_val])
            n_sl = sum(len(s) for s in self.data[l1_val].values())
            print(f"  L1={l1_val}KB: {n_wl} workloads, {n_sl} slices")

        print(f"DseEngine: L1 configs={sorted(self.data.keys())}, "
              f"workloads={sorted(self.workload_names)}")

    def get_features(self, config, workload):
        """
        Args:
            config: dict with "l1_size" key to select L1 precompute
            workload: str, e.g. "605.mcf_s"

        Returns:
            list of np.ndarray, each shape (4681,)
        """
        l1 = config["l1_size"]
        if l1 not in self.data:
            raise ValueError(f"L1={l1} not precomputed. Available: {sorted(self.data.keys())}")
        if workload not in self.data[l1]:
            raise ValueError(f"Workload '{workload}' not found for L1={l1}")
        return [dse_infer(pre, config) for _, pre in self.data[l1][workload]]

    def get_workloads(self):
        return sorted(self.workload_names)

    def get_l1_configs(self):
        return sorted(self.data.keys())

    def num_slices(self, workload=None, l1=None):
        if l1 and workload:
            return len(self.data.get(l1, {}).get(workload, []))
        if l1:
            return sum(len(s) for s in self.data.get(l1, {}).values())
        return sum(sum(len(s) for s in wls.values()) for wls in self.data.values())