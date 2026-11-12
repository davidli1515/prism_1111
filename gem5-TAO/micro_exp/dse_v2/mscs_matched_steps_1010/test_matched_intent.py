"""Meaningful edge checks: budgets, step clipping and unresolved feasibility."""
import unittest
from matched_support import outcome, pool_indices, get_param_values


class Intent(unittest.TestCase):
    def test_infeasible_winner_cannot_be_selected(self):
        r = outcome(10, 20, [dict(cpi=8, area=21), dict(cpi=9, area=19)])
        self.assertEqual(r['best_cpi'], 9)

    def test_unknown_improving_area_is_not_silently_rejected(self):
        r = outcome(10, 20, [dict(cpi=8, area=None), dict(cpi=9, area=19)])
        self.assertFalse(r['resolved']); self.assertIsNone(r['best_cpi'])
        self.assertEqual((r['cpi_lower'], r['cpi_upper']), (8, 9))

    def test_unknown_area_irrelevant_if_candidate_cannot_win(self):
        for cpi in (9.5, 10, 11):
            r = outcome(10, 20, [dict(cpi=cpi, area=None), dict(cpi=9, area=19)])
            self.assertTrue(r['resolved']); self.assertEqual(r['best_cpi'], 9)

    def test_acceptance_tolerance_and_stay_put(self):
        self.assertEqual(outcome(10, 20, [dict(cpi=10-1e-7, area=19)])['best_cpi'], 10)
        self.assertEqual(outcome(10, 20, [dict(cpi=11, area=19)])['best_cpi'], 10)

    def test_exact_stride_excludes_clipped_moves(self):
        values = get_param_values('numIQEntries'); at = len(values)-4
        d = dict(root=dict(numIQEntries=values[at]), candidates=[
            dict(parameter='numIQEntries', sign=1, stride=1, config=dict(numIQEntries=values[-1])),
            dict(parameter='numIQEntries', sign=-1, stride=1, config=dict(numIQEntries=values[at-8]))])
        self.assertEqual(pool_indices(d, 'stride1'), [1])
        self.assertEqual(pool_indices(d, 'full'), [0, 1])


if __name__ == '__main__':
    unittest.main()
