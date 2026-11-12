import unittest
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np

from engine import (BudgetExhausted, BudgetOracle, AtomicArchive, risk_score,
                    select_verified_options, pressure_composition, config_key)
from config_dse import DEFAULT_CONFIG
from engine import PARAMS, _shift
from area_store import area_config, AreaStore
from area_model import modify_xml_for_config, MCPAT_TEMPLATE


class FakeOracle:
    def __init__(self):
        self.cache, self.calls = {}, 0

    def query(self, c):
        key = config_key(c)
        if key not in self.cache:
            self.calls += 1
            self.cache[key] = (np.tile([.25, .2, .25, .3], (5, 4)), 1., tuple(range(5)))
        return self.cache[key]


class EngineTest(unittest.TestCase):
    def test_area_store_threads_are_isolated_and_persist_exact_results(self):
        paths = []
        def fake(c, work_dir):
            paths.append(work_dir)
            return 20. + c["IntALU"]
        ca, cb = dict(DEFAULT_CONFIG), dict(DEFAULT_CONFIG)
        cb["IntALU"] += 1
        with tempfile.TemporaryDirectory() as tmp, patch("safe_area.compute_area", side_effect=fake):
            first = AreaStore(tmp)
            first.prefetch([ca, cb, ca], workers=2)
            self.assertEqual(len(paths), 2)
            self.assertEqual(len(set(paths)), 2)
            second = AreaStore(tmp)
            self.assertEqual(second(ca), 20. + ca["IntALU"])
            self.assertEqual(second(cb), 20. + cb["IntALU"])
            self.assertEqual(len(paths), 2)

    def test_area_cache_equivalence_matches_actual_xml(self):
        configs = [dict(DEFAULT_CONFIG)]
        for p in PARAMS:
            for k in (-4, -1, 1, 4):
                c = _shift(DEFAULT_CONFIG, p, k)
                if c is not None:
                    configs.append(c)
        keys = {}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"config.xml"
            for c in configs:
                modify_xml_for_config(MCPAT_TEMPLATE, c, path)
                key = tuple(sorted(area_config(c).items()))
                xml = path.read_bytes()
                if key in keys:
                    self.assertEqual(keys[key], xml)
                keys[key] = xml

    def test_method_cache_reuse_is_charged_and_hard_cap_is_atomic(self):
        physical = FakeOracle()
        a, b = {"x": 1}, {"x": 2}
        physical.query(a)
        S = BudgetOracle(physical, 1)
        arch = AtomicArchive(S, lambda c: 10.)
        arch.add(a); arch.add(a)
        self.assertEqual((S.n_queries, physical.calls), (1, 1))
        with self.assertRaises(BudgetExhausted):
            arch.add(b)
        self.assertEqual([len(arch.configs), len(arch.cpis), len(arch.areas), len(arch.queries)], [1]*4)
        self.assertEqual(S.n_queries, 1)

    def test_area_failure_still_charges_surrogate(self):
        arch = AtomicArchive(BudgetOracle(FakeOracle(), 1), lambda c: None)
        self.assertIsNone(arch.add({"x": 1}))
        self.assertEqual(arch.S.n_queries, 1)
        self.assertEqual(arch.configs, [])
        self.assertEqual(len(arch.failed_area), 1)

    def test_protection_binds_to_specific_partner(self):
        futures = [{"param": "decodeWidth", "area": 12}, {"param": "IntALU", "area": 13}]
        checks = [dict(j="numIQEntries", k="decodeWidth", future_cost=.019, area=10, reference_area=12),
                  dict(j="numIQEntries", k="IntALU", future_cost=.001, area=11, reference_area=13)]
        choices = select_verified_options(futures, checks)
        joints = [r for r in choices if r["choice"] == "joint"]
        self.assertEqual(len(joints), 1)
        self.assertEqual(joints[0]["k"], "IntALU")
        # Unknown reclaims are never implicitly treated as verified-safe joints.
        self.assertEqual(len(select_verified_options(futures, [])), 2)

    def test_composition_strips_base_and_scalar_control_ignores_pressure(self):
        m = np.tile([.25, .2, .25, .3], (5, 4))
        altered = m.reshape(5, 4, 4).copy()
        altered[..., 0] = .5
        altered[..., 1:] *= 2/3
        np.testing.assert_allclose(pressure_composition(m), pressure_composition(altered), rtol=1e-12)
        c, d = dict(DEFAULT_CONFIG), dict(DEFAULT_CONFIG)
        d["IntALU"] += 1
        physical = FakeOracle()
        physical.query(c); physical.query(d)
        changed = m.reshape(5, 4, 4).copy()
        changed[..., 1:] = changed[..., 1:][..., ::-1]
        physical.cache[config_key(d)] = (changed.reshape(5, 16), 1., tuple(range(5)))
        S = BudgetOracle(physical, 2)
        self.assertAlmostEqual(risk_score(S, c, d, "scalar_guard"), 0.)
        self.assertGreater(risk_score(S, c, d, "mscs_guard"), 0.)


if __name__ == "__main__":
    unittest.main()
