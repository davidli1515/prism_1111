"""Exact McPAT caching across runs, with private XML files and a SQLite store.

Only XML-equivalent configurations share entries. No area approximation, no
legacy cache import. The original wrapper, template and executable salt keys.
"""
import hashlib
import json
from pathlib import Path
import sqlite3
import threading

from safe_area import SafeArea
import area_model


def area_config(c):
    # These are exactly the fields read by modify_xml_for_config. Its FPU input
    # is max(FP_ALU, FP_MultDiv), and its decode input is max(decode, dispatch).
    d = {k: v for k, v in c.items() if k not in ("wbWidth", "bp_type", "FP_MultDiv")}
    d["FP_ALU"] = max(c["FP_ALU"], c["FP_MultDiv"])
    d["decodeWidth"] = max(c["decodeWidth"], c["dispatchWidth"])
    return d


class AreaStore(SafeArea):
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.cache, self.errors, self.n_fail = {}, {}, 0
        paths = (area_model.__file__, area_model.MCPAT_TEMPLATE, area_model.MCPAT_BIN)
        self.sources = {str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}
        self.salt = json.dumps(self.sources, sort_keys=True)
        self.local = threading.local()
        db = self.connection()
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("CREATE TABLE IF NOT EXISTS areas (key TEXT PRIMARY KEY, value REAL NOT NULL)")
        db.commit()

    def connection(self):
        if not hasattr(self.local, "db"):
            self.local.db = sqlite3.connect(self.root / "areas.sqlite", timeout=120)
        return self.local.db

    def key(self, c):
        return hashlib.sha256((self.salt + json.dumps(area_config(c), sort_keys=True)).encode()).hexdigest()

    def lookup(self, key):
        if key not in self.cache:
            row = self.connection().execute("SELECT value FROM areas WHERE key=?", (key,)).fetchone()
            if row is not None:
                self.cache[key] = row[0]
        return self.cache.get(key)

    def _compute(self, c):
        key = self.key(c)
        row = self.connection().execute("SELECT value FROM areas WHERE key=?", (key,)).fetchone()
        return (key, row[0], None) if row is not None else super()._compute(c)

    def _record(self, result):
        value = super()._record(result)
        if value is not None:
            db = self.connection()
            old = db.execute("SELECT value FROM areas WHERE key=?", (result[0],)).fetchone()
            if old is not None and abs(old[0]-value) > 1e-6:
                raise ValueError(f"Inconsistent exact area cache: {old[0]} vs {value}")
            db.execute("INSERT OR IGNORE INTO areas VALUES (?, ?)", (result[0], value))
            db.commit()
        return value

    def __call__(self, c):
        v = self.lookup(self.key(c))
        return v if v is not None else self._record(self._compute(c))

    def prefetch(self, configs, workers=4):
        missing = [c for c in configs if self.lookup(self.key(c)) is None]
        if missing:
            super().prefetch(missing, workers)

    def flush(self):
        self.connection().commit()
