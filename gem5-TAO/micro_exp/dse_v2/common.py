"""
DSE v2 shared pieces: surrogate predictor with query accounting, area with a
persistent cache, Pareto / hypervolume helpers.

Nothing in micro_exp/dse_exp is modified; its modules are only imported.
"""
import os, sys, json, time, hashlib
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
MICRO = os.path.dirname(HERE)
DSE_DIR = os.path.join(MICRO, "dse_exp")
EXP1_DIR = os.path.join(MICRO, "exp1")
for p in (EXP1_DIR, DSE_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from model import GEMModel                                   # exp1
from dataset import load_all_benchmarks, create_exp1_dataloaders, FeatureNormalizer
from config_dse import (ALL_PARAMS, GRADIENT_PARAMS, get_param_values,   # dse_exp
                        compute_w_min, sample_random_config, PARAM_TO_GD_INDEX,
                        GEM_DATA_ARCH_START, GEM_DATA_ARCH_END)
from dse_infer import DseEngine
from area_model import compute_area

GEM_INPUT_DIM = 4681
BASE_IDX = [0, 4, 8, 12]
PRECOMPUTE = os.path.join(DSE_DIR, "precompute")
DEFAULT_CKPT = os.path.join(EXP1_DIR, "results/lambda_sweep_1004/t10_c1_m0p01_s42.pt")
NORM_CACHE = os.path.join(HERE, "normalizer_cache")
AREA_CACHE = os.path.join(HERE, "area_cache")
MCPAT_WORK = os.path.join(HERE, "mcpat_work")

SPEC17 = ["508.namd_r", "526.blender_r", "600.perlbench_s", "602.gcc_s",
          "605.mcf_s", "607.cactuBSSN_s", "619.lbm_s", "620.omnetpp_s",
          "621.wrf_s", "623.xalancbmk_s", "631.deepsjeng_s", "638.imagick_s",
          "641.leela_s", "644.nab_s", "648.exchange2_s", "654.roms_s", "657.xz_s"]


def config_key(c):
    return tuple((k, c[k]) for k in sorted(c))


# raw arch slots (offset from 3850) driven by each parameter (dispatch = rename = issue)
ARCH_SLOTS = {n: [i - 3850] for n, i in PARAM_TO_GD_INDEX.items()}
ARCH_SLOTS["dispatchWidth"] = [2, 3, 4]


def arch_delta(c_from, c_to):
    """(23,) raw change of the arch slots between two configs."""
    d = np.zeros(23)
    for n, slots in ARCH_SLOTS.items():
        if n in c_from and c_to[n] != c_from[n]:
            for s in slots:
                d[s] += c_to[n] - c_from[n]
    return d


# ============================================================
# Normalizer matching the surrogate's training split
# ============================================================

def get_normalizer(train_data="18k", seed=42):
    """exp1 split (seed) on the given dataset; cached under dse_v2/normalizer_cache."""
    d = {"18k": "/scratch2/wl730/micro_traindata_packed",
         "40k": "/scratch2/wl730/micro_traindata_packed_all"}[train_data]
    tag = f"{train_data}_s{seed}"
    mp, sp = (os.path.join(NORM_CACHE, f"{tag}_{x}.npy") for x in ("mean", "std"))
    norm = FeatureNormalizer()
    if os.path.exists(mp) and os.path.exists(sp):
        norm.mean, norm.std = np.load(mp), np.load(sp)
        return norm
    data = load_all_benchmarks(f"{d}/gem", f"{d}/concord", None)
    _, _, gem_norm, _, _ = create_exp1_dataloaders(data, seed=seed)
    os.makedirs(NORM_CACHE, exist_ok=True)
    np.save(mp, gem_norm.mean); np.save(sp, gem_norm.std)
    return gem_norm


# ============================================================
# Surrogate with query accounting
# ============================================================

class Surrogate:
    """
    Joint predictor over workloads. One *query* = one distinct configuration
    evaluated on all workloads (cached; re-evaluating a config is free).
      cpi(c)  -> geometric-mean CPI over workloads
      mscs(c) -> MSCS (16) averaged over slices, then over workloads
    Per-workload CPI = mean over slices of the per-sample CPI, where the per-sample
    CPI is the 4-stage mean (as in training evaluation) or the commit stage.
    """
    def __init__(self, ckpt=DEFAULT_CKPT, workloads=SPEC17, train_data="18k",
                 hidden=(512, 256), cpi_stage="mean", max_slices=200, device="cpu"):
        self.workloads = list(workloads)
        self.cpi_stage = cpi_stage
        self.device = torch.device(device)
        self.model = GEMModel(GEM_INPUT_DIM, list(hidden), 0.0).to(self.device)
        self.model.load_state_dict(torch.load(ckpt, map_location=self.device))
        self.model.eval()
        self.norm = get_normalizer(train_data)
        self.engine = DseEngine(PRECOMPUTE, max_slices=max_slices)
        self.cache, self.jcache = {}, {}
        self.n_queries = 0
        self.n_jac = 0
        self.t_query = 0.0
        self.ckpt = ckpt

    @torch.no_grad()
    def _eval(self, c):
        t0 = time.time()
        w = float(compute_w_min(c))
        cpis, mscs = [], []
        for wl in self.workloads:
            X = np.stack([f[:GEM_INPUT_DIM] for f in self.engine.get_features(c, workload=wl)])
            x = torch.from_numpy(self.norm.transform(X)).float().to(self.device)
            m = self.model(x)
            base = torch.clamp(m[:, BASE_IDX], min=0.01)
            cs = torch.clamp(1.0 / (base * w + 1e-8), max=100.0)
            per = cs.mean(1) if self.cpi_stage == "mean" else cs[:, 3]
            cpis.append(per.mean().item())
            mscs.append(m.mean(0).cpu().numpy())
        self.n_queries += 1
        self.t_query += time.time() - t0
        return {"cpi": float(np.exp(np.mean(np.log(cpis)))),
                "per_wl_cpi": cpis, "mscs": np.mean(mscs, 0)}

    def jac(self, c):
        """Linearization at c (costs one query, also fills the query cache):
        J_cpi (23,) and J_mscs (16,23) = d[geomean CPI, mean MSCS] / d(raw arch slots
        3850:3873). NOTE: only the explicit arch-parameter inputs are differentiated;
        the per-resource throughput distributions that also depend on the config are
        held fixed, so this is a cheap *screen*, verified by exact queries."""
        k = config_key(c)
        if k in self.jcache:
            return self.jcache[k]
        t0 = time.time()
        a0, a1 = GEM_DATA_ARCH_START, GEM_DATA_ARCH_END
        mu = torch.tensor(self.norm.mean[a0:a1], dtype=torch.float32, device=self.device)
        sd = torch.tensor(self.norm.std[a0:a1], dtype=torch.float32, device=self.device)
        Xs = []
        for wl in self.workloads:
            X = np.stack([f[:GEM_INPUT_DIM] for f in self.engine.get_features(c, workload=wl)])
            Xs.append(torch.from_numpy(self.norm.transform(X)).float().to(self.device))
        araw = torch.tensor(np.asarray(Xs[0][0, a0:a1].cpu().numpy()) * self.norm.std[a0:a1]
                            + self.norm.mean[a0:a1], dtype=torch.float32, device=self.device)
        w = float(compute_w_min(c))

        def f(a):
            cpis, ms = [], []
            for x in Xs:
                xa = torch.cat([x[:, :a0], ((a - mu) / sd).expand(len(x), -1), x[:, a1:]], 1)
                m = self.model(xa)
                base = torch.clamp(m[:, BASE_IDX], min=0.01)
                cs = torch.clamp(1.0 / (base * w + 1e-8), max=100.0)
                per = cs.mean(1) if self.cpi_stage == "mean" else cs[:, 3]
                cpis.append(per.mean()); ms.append(m.mean(0))
            gcpi = torch.exp(torch.log(torch.stack(cpis)).mean())
            return torch.cat([gcpi.view(1), torch.stack(ms).mean(0)]), torch.stack(cpis), torch.stack(ms)

        J = torch.autograd.functional.jacobian(lambda a: f(a)[0], araw).detach().cpu().numpy()
        with torch.no_grad():
            out, cpis, ms = f(araw)
        if k not in self.cache:                                   # same numbers as _eval
            self.cache[k] = {"cpi": float(out[0]), "per_wl_cpi": cpis.cpu().tolist(),
                             "mscs": ms.mean(0).cpu().numpy()}
        self.n_queries += 1          # a Jacobian always costs one query (features + fwd/bwd)
        self.n_jac += 1
        self.t_query += time.time() - t0
        self.jcache[k] = (J[0], J[1:])
        return self.jcache[k]

    def query(self, c):
        k = config_key(c)
        if k not in self.cache:
            self.cache[k] = self._eval(c)
        return self.cache[k]

    def cpi(self, c):
        return self.query(c)["cpi"]

    def mscs(self, c):
        return self.query(c)["mscs"]


# ============================================================
# Area (McPAT) with a persistent, per-process-safe cache
# ============================================================

class Area:
    def __init__(self, cache_dir=AREA_CACHE):
        os.makedirs(cache_dir, exist_ok=True)
        self.path = os.path.join(cache_dir, f"area_{os.getpid()}.json")
        self.cache = {}
        for f in os.listdir(cache_dir):                       # merge earlier runs
            if f.endswith(".json"):
                try:
                    self.cache.update(json.load(open(os.path.join(cache_dir, f))))
                except Exception:
                    pass
        self.n_fail = 0
        self._dirty = 0

    @staticmethod
    def _k(c):
        return hashlib.md5(json.dumps(config_key(c)).encode()).hexdigest()

    def __call__(self, c):
        k = self._k(c)
        if k not in self.cache:
            try:
                self.cache[k] = compute_area(c, work_dir=MCPAT_WORK)
            except Exception:
                self.n_fail += 1
                self.cache[k] = None
            self._dirty += 1
            if self._dirty >= 50:
                self.flush()
        return self.cache[k]

    def flush(self):
        json.dump(self.cache, open(self.path, "w"))
        self._dirty = 0


# ============================================================
# Pareto / HV
# ============================================================

def pareto_mask(P):
    """P: (n,2) minimize both. Returns boolean mask of non-dominated points."""
    P = np.asarray(P, float)
    order = np.lexsort((P[:, 1], P[:, 0]))
    mask = np.zeros(len(P), bool)
    best_a = np.inf
    for i in order:
        if P[i, 1] < best_a:
            mask[i] = True
            best_a = P[i, 1]
    return mask


def hypervolume(P, ref):
    P = np.asarray(P, float)
    if len(P) == 0:
        return 0.0
    P = P[(P[:, 0] < ref[0]) & (P[:, 1] < ref[1])]
    if len(P) == 0:
        return 0.0
    F = P[pareto_mask(P)]
    F = F[np.argsort(F[:, 0])]
    hv, prev = 0.0, ref[1]
    for cpi, area in F:
        hv += (ref[0] - cpi) * (prev - area)
        prev = area
    return float(hv)


class Archive:
    """All configurations a method 'evaluated' (kept as candidate designs),
    with the cumulative surrogate-query count when each was added."""
    def __init__(self, surrogate, area_fn):
        self.S, self.A = surrogate, area_fn
        self.configs, self.cpis, self.areas, self.queries, self.tags = [], [], [], [], []
        self._seen = set()

    def add(self, c, tag=""):
        k = config_key(c)
        if k in self._seen:
            return None
        a = self.A(c)
        if a is None:
            return None
        self._seen.add(k)
        self.configs.append(dict(c)); self.cpis.append(self.S.cpi(c)); self.areas.append(a)
        self.queries.append(self.S.n_queries); self.tags.append(tag)
        return self.cpis[-1], a

    def to_json(self, **extra):
        P = np.c_[self.cpis, self.areas] if self.cpis else np.zeros((0, 2))
        pm = pareto_mask(P) if len(P) else np.zeros(0, bool)
        return {"configs": self.configs, "cpis": self.cpis, "areas": self.areas,
                "queries": self.queries, "tags": self.tags,
                "pareto_indices": np.where(pm)[0].tolist(),
                "n_queries": self.S.n_queries, **extra}
