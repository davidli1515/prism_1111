"""Isolated McPAT adapter for NEW runs; never reads/writes legacy area caches."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "dse_exp"))
from area_model import compute_area


class SafeArea:
    def __init__(self, root=HERE / "area_v1"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.cache, self.errors = {}, {}
        self.n_fail = 0
        for p in self.root.glob("area_*.json"):
            self.cache.update(json.loads(p.read_text()))

    @staticmethod
    def key(c):
        return hashlib.sha256(json.dumps(c, sort_keys=True).encode()).hexdigest()

    def _compute(self, c):
        key = self.key(c)
        try:
            with tempfile.TemporaryDirectory(prefix="mcpat_", dir=self.root) as work:
                value = float(compute_area(c, work_dir=work))
            if not 0 < value < float("inf"):
                raise ValueError(f"Invalid area: {value}")
            return key, value, None
        except Exception as e:
            return key, None, str(e)

    def _record(self, result):
        key, value, error = result
        if error is not None:
            self.n_fail += 1
            self.errors[key] = error
        else:
            self.cache[key] = value
            self.errors.pop(key, None)
        return value

    def __call__(self, c):
        key = self.key(c)
        return self.cache[key] if key in self.cache else self._record(self._compute(c))

    def prefetch(self, configs, workers=4):
        todo = {self.key(c): c for c in configs if self.key(c) not in self.cache}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for result in pool.map(self._compute, todo.values()):
                self._record(result)

    def flush(self):
        # Cache files are disjoint per process; replace makes reads atomic.
        path = self.root / f"area_{os.getpid()}.json"
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(self.cache, sort_keys=True))
        os.replace(temp, path)

