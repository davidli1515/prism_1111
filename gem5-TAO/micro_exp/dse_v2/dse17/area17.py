"""
Race-free McPAT area for dse17: every call runs in its own fresh directory (compute_area writes and
deletes <work_dir>/mcpat_temp_<pid>.xml, so a shared directory is NOT safe under threads), parallel
calls use processes, and the cache lives only in dse17/area_cache (no older cache is read).
Also a fast learned area model (fit_area_model.py) used ONLY for in-step decisions of the
iso-area refine; archived areas always come from McPAT.
"""
import os, json, glob, shutil, tempfile, hashlib, pickle
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import space17                                      # noqa: F401  (sets sys.path)
from area_model import compute_area

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(HERE, "mcpat_work")
CACHE = os.path.join(HERE, "area_cache")
MODEL = os.path.join(HERE, "area_model.pkl")


def area_isolated(c):
    os.makedirs(WORK, exist_ok=True)
    d = tempfile.mkdtemp(dir=WORK)
    try:
        return compute_area(c, work_dir=d)
    except Exception:
        return None
    finally:
        shutil.rmtree(d, ignore_errors=True)


def key(c):
    return hashlib.md5(json.dumps(tuple((k, c[k]) for k in sorted(c))).encode()).hexdigest()


class AreaStore:
    def __init__(self, workers=16):
        os.makedirs(CACHE, exist_ok=True)
        self.cache = {}
        for f in glob.glob(os.path.join(CACHE, "*.json")):
            self.cache.update(json.load(open(f)))
        self.path = os.path.join(CACHE, f"area_{os.getpid()}.json")
        self.new, self.workers = {}, workers

    def prefetch(self, cs):
        todo, seen = [], set()
        for c in cs:
            k = key(c)
            if k not in self.cache and k not in seen:
                seen.add(k); todo.append(c)
        if todo:
            with ProcessPoolExecutor(self.workers) as ex:
                for c, a in zip(todo, ex.map(area_isolated, todo, chunksize=2)):
                    self.cache[key(c)] = self.new[key(c)] = a

    def __call__(self, c):
        k = key(c)
        if k not in self.cache:
            self.cache[k] = self.new[k] = area_isolated(c)
        return self.cache[k]

    def flush(self):
        json.dump(self.new, open(self.path, "w"))


# ---------------------------------------------------------------- fast area model
def featurize(cs):
    from space17 import SEARCH
    X = np.array([[c[p] for p in SEARCH] for c in cs], float)
    return X


class FastArea:
    def __init__(self, path=MODEL):
        self.m = pickle.load(open(path, "rb"))

    def __call__(self, c):
        return float(np.exp(self.m.predict(featurize([c]))[0]))

    def many(self, cs):
        return np.exp(self.m.predict(featurize(cs)))
