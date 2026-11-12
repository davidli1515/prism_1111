"""Counterexamples guarding the experiment's semantics, not model accuracy."""
import unittest
import numpy as np
from audit import region, root_bounds, certificate, decide
from config_dse import DEFAULT_CONFIG


class IntentTests(unittest.TestCase):
    def test_complete_cartesian_set_no_collateral_changes(self):
        c = dict(DEFAULT_CONFIG, fetchBufferSize=8, l1i_assoc=2, l1_size=16)
        cs, rejected = region(c)
        self.assertEqual(len(cs), 4*4-1)  # assoc=32 violates min cache-set count
        self.assertEqual(len(rejected), 4)
        self.assertEqual({(x['fetchBufferSize'], x['l1i_assoc']) for x in cs},
                         {(f, a) for f in (8, 16, 32, 64) for a in (2, 4, 8, 16)}-{(8, 2)})
        self.assertTrue(all(all(x[k] == c[k] for k in c if k not in ('fetchBufferSize', 'l1i_assoc')) for x in cs))

    def test_empty_region_not_counted_as_pruned(self):
        cs, _ = region(dict(DEFAULT_CONFIG, fetchBufferSize=64, l1i_assoc=32))
        self.assertEqual(cs, [])
        d = decide([[.1, 1.]], dict.fromkeys(('scalar_width', 'mscs_decode', 'mscs_stage_max'), .2), [])
        self.assertFalse(any(m['prune'] for m in d['methods'].values()))

    def test_slice_weighting_and_stage_order(self):
        m = np.array([[.25, .5, .2, .05]*4, [.125, .1, .7, .075]*4], dtype=np.float32)
        b = root_bounds(m, 4)
        self.assertAlmostEqual(b['mscs_decode'], (1*.5+2*.9)/2, places=6)
        self.assertGreater(abs(b['mscs_decode']-b['naive_product_bound']), .05)
        m[0, 5] += .1; m[0, 6] -= .1
        b = root_bounds(m, 4)
        self.assertLess(b['mscs_stage_max'], b['mscs_decode'])
        self.assertEqual(b['fp_stage_order_violation_slices'], 1)

    def test_area_minimum_includes_zero_and_negative_changes(self):
        bounds = dict.fromkeys(('scalar_width', 'mscs_decode', 'mscs_stage_max'), .98)
        self.assertFalse(decide([[.97, 11]], bounds, [10., 12.])['methods']['mscs_decode']['prune'])
        self.assertEqual(decide([[.97, 11]], bounds, [9., 12.])['area_lower'], 9.)
        self.assertTrue(decide([[.97, 11]], bounds, [12., 14.])['methods']['mscs_decode']['prune'])
        self.assertFalse(decide([[.97, 11]], bounds, [None, 14.])['methods']['mscs_decode']['prune'])

    def test_pareto_root_obstruction_and_directional_advantage(self):
        f = [[1., 10.], [.97, 11.]]
        self.assertFalse(certificate(f, .98, 10.)['prune'])
        self.assertTrue(certificate(f, .98, 12.)['prune'])
        self.assertFalse(certificate(f, .25, 12.)['prune'])


if __name__ == '__main__':
    unittest.main()
