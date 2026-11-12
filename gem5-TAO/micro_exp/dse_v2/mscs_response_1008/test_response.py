import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np
from response import compose, config_additive, cpi_samples, evaluate_top_pairs, extrapolate
from safe_area import SafeArea


def softmax(z):
    e = np.exp(z-z.max(axis=-1, keepdims=True))
    return (e/e.sum(axis=-1, keepdims=True)).reshape(-1, 16)


class ResponseTests(unittest.TestCase):
    def test_additive_logits(self):
        rng = np.random.RandomState(42)
        z, a, b = [rng.normal(size=(8, 4, 4)) for _ in range(3)]
        np.testing.assert_allclose(compose(softmax(z), softmax(z+a), softmax(z+b)),
                                   softmax(z+a+b), rtol=1e-12)

    def test_equal_single_cpis_can_have_opposite_pair_outcomes(self):
        # Four identical stages satisfy the paper's weak monotonicity constraints.
        m = lambda v: np.tile(np.array(v)/sum(v), (1, 4))
        base, a, b = m([1, 1.5, 1.5, 1]), m([1, 2.5, .5, 1]), m([1, .5, 2.5, 1])
        np.testing.assert_allclose(cpi_samples(base, 4), cpi_samples(a, 4))
        np.testing.assert_allclose(cpi_samples(base, 4), cpi_samples(b, 4))
        self.assertLess(cpi_samples(compose(base, a, b), 4)[0], cpi_samples(base, 4)[0])
        self.assertGreater(cpi_samples(compose(base, a, a), 4)[0], cpi_samples(base, 4)[0])

    def test_larger_step_in_affine_region(self):
        rng = np.random.RandomState(3)
        z, a = [rng.normal(size=(8, 4, 4)) for _ in range(2)]
        np.testing.assert_allclose(extrapolate(softmax(z), softmax(z+a), 4),
                                   softmax(z+4*a), rtol=1e-12)

    def test_projection_and_exact_query_budget(self):
        self.assertFalse(config_additive({"r": 4}, {"r": 5}, {"r": 5}, {"r": 5}))
        candidates = [dict(config=i, key=i, predicted_cpi=float(i)) for i in (2, 1, 1, 3)]
        queried = []
        evaluate_top_pairs(candidates, queried.append, 2)
        self.assertEqual(queried, [1, 2])
        evaluate_top_pairs(candidates, queried.append, 0)
        self.assertEqual(queried, [1, 2])

    def test_mcpat_threads_have_disjoint_xml_files(self):
        barrier = threading.Barrier(4)
        def fake_compute(c, work_dir):
            path = Path(work_dir) / f"mcpat_temp_{os.getpid()}.xml"
            path.write_text(str(c["n"]))
            barrier.wait(timeout=5)
            return float(path.read_text())
        with tempfile.TemporaryDirectory() as root:
            area = SafeArea(root)
            configs = [{"n": i} for i in range(1, 5)]
            with patch("safe_area.compute_area", fake_compute):
                area.prefetch(configs + configs, workers=4)
            self.assertEqual([area(c) for c in configs], [1., 2., 3., 4.])
            self.assertEqual(area.n_fail, 0)
            self.assertEqual(list(Path(root).glob("mcpat_*")), [])


if __name__ == "__main__":
    unittest.main()
