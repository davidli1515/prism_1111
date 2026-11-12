"""Checks for information separation and equal query accounting, without inference."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from diagnose import (Evaluator, LIMIT, assess, diversity_pair, merged_stream,
                      near_quality, pressure_residual)


class DiagnosticChecks(unittest.TestCase):
    def test_residual_removes_stage_total(self):
        rng = np.random.RandomState(42)
        m = rng.rand(50, 4, 4)
        changed = m.copy()
        changed[..., 1:] += rng.rand(50, 4, 1)
        changed[..., 0] = rng.rand(50, 4)
        np.testing.assert_allclose(pressure_residual(m), pressure_residual(changed), atol=1e-15)
        np.testing.assert_allclose(pressure_residual(m).sum(-1), 0., atol=1e-15)

    def test_direction_is_retained_at_equal_norm(self):
        v = np.array([[1., -1., 0.], [1., -1., 0.], [-1., 1., 0.], [0., 1., -1.]])
        self.assertTrue(np.all(np.abs(v).sum(1) == 2.))
        pair, distances = diversity_pair(v)
        self.assertEqual(pair, [0, 2])
        self.assertEqual(distances[1], 0.)

    def test_quality_gate(self):
        self.assertTrue(near_quality([[1., 10.], [1.005, 10.1]]))
        self.assertFalse(near_quality([[1., 10.], [1.02, 10.1]]))

    def test_interleave_deduplicates_without_future_reordering(self):
        row = lambda i: {"config": {"i": i}}
        x = merged_stream([row(1), row(2), row(3)], [row(1), row(4), row(3)])
        self.assertEqual([r["config"]["i"] for r in x], [1, 2, 4, 3])

    def test_all_pairs_use_common_distinct_budget(self):
        row = lambda i: {"config": {"i": i}, "cpi": 2.-i*.01, "area": 2.-i*.01}
        branches = [[row(1), row(2)], [row(1), row(3)],
                    [row(4), row(5)], [row(6), row(7)]]
        ctx = {"cutoff": 1, "id": "test", "bench": "test", "reference": [3., 3.]}
        data = {"queries": [1], "configs": [{"i": 0}], "cpis": [2.], "areas": [2.]}
        decision = {"pairs": {"scalar_quality": [0, 1], "mscs_diversity": [0, 2]}}
        result = assess(ctx, data, branches, decision)
        self.assertEqual(result["query_budget"], 3)
        self.assertEqual(len(result["pair_hvi"]), 6)
        # Last point of the disjoint pair (6,7) is cut off at the common budget.
        self.assertAlmostEqual(result["pair_hvi"]["2,3"], 1.06**2-1.)

    def test_limit_precedes_any_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            evaluator = Evaluator.__new__(Evaluator)
            evaluator.directory = Path(tmp)
            evaluator.n_requests = LIMIT
            with self.assertRaisesRegex(RuntimeError, "limit"):
                evaluator.query("test", {"i": 0})
            self.assertEqual(evaluator.n_requests, LIMIT)


if __name__ == "__main__":
    unittest.main()
