import unittest
import numpy as np
import torch
from adaptive_support import RankAllocator, rank_feedback, LocalModel
from core import LocalModel as OriginalLocalModel
from test_core import Affine


class Intent(unittest.TestCase):
    def test_initial_tie_and_choice_are_label_free_and_do_not_mutate(self):
        p = RankAllocator(); before = p.snapshot()
        choice = p.choose([9., 8., 7.], [7., 8., 9.])
        self.assertEqual(choice[0], [2, 0]); self.assertEqual(p.snapshot(), before)
        # Batch remains frozen after labels arrive; only the next choice changes.
        p.observe(10., [9., 8.], [1., 100.], [9., 11.])
        self.assertEqual(choice[0], [2, 0])
        self.assertEqual(p.mode, 'mscs_logit')
        self.assertEqual(p.choose([9.,8.,7.], [7.,8.,9.])[0], [0,1])

    def test_ranking_not_magnitude(self):
        r = rank_feedback(10., [9.1, 8.], [1., 100.], [9., 11.])
        self.assertEqual(r['loss2'], {'scalar': 4, 'mscs': 0})
        self.assertEqual(r['discriminating_comparisons'], 2)

    def test_expert_swap_symmetry_after_informative_feedback(self):
        p, q = RankAllocator(), RankAllocator()
        p.observe(10., [9.1,8.], [1.,100.], [9.,11.])
        q.observe(10., [1.,100.], [9.1,8.], [9.,11.])
        self.assertEqual(p.state['loss2']['scalar'], q.state['loss2']['mscs'])
        self.assertEqual(p.mode, 'mscs_logit'); self.assertEqual(q.mode, 'scalar_excess')
        self.assertEqual(p.choose([9.,8.,7.], [7.,8.,9.])[0],
                         q.choose([7.,8.,9.], [9.,8.,7.])[0])

    def test_actual_ties_ignored_predicted_ties_half_loss(self):
        r = rank_feedback(10., [10.], [9.], [9.])
        self.assertEqual(r['loss2'], {'scalar':1, 'mscs':0})
        r = rank_feedback(10., [1.], [100.], [10.+1e-7])
        self.assertEqual(r['comparisons'], [])

    def test_leadership_can_tie_and_reverse(self):
        p = RankAllocator()
        p.observe(10., [11.], [9.], [9.]); self.assertEqual(p.mode, 'mscs_logit')
        p.observe(10., [9.], [11.], [9.]); self.assertEqual(p.mode, 'hybrid')
        p.observe(10., [9.], [11.], [9.]); self.assertEqual(p.mode, 'scalar_excess')

    def test_stable_dedup_empty_and_single_candidate(self):
        p = RankAllocator()
        self.assertEqual(p.choose([1.,1.,1.], [1.,1.,1.])[0], [0,1])
        self.assertEqual(p.choose([1.], [2.])[0], [0])
        self.assertEqual(p.choose([], [])[0], [])
        p.observe(10., [11.], [9.], [9.])
        self.assertEqual(p.choose([1.,1.,1.], [1.,1.,1.])[0], [0,1])

    def test_state_is_private_and_snapshot_is_detached(self):
        p, q = RankAllocator(), RankAllocator()
        snap = p.snapshot(); snap['loss2']['scalar'] = 900
        p.observe(10., [11.], [9.], [9.])
        self.assertEqual(q.snapshot()['completed_rounds'], 0)
        self.assertEqual(p.snapshot()['loss2']['scalar'], 2)

    def test_unchanged_scalar_and_mscs_predictors(self):
        model = Affine(); x = torch.zeros((3,2),dtype=torch.float64)
        xs = torch.stack([x+torch.tensor([1.,0.]), x+torch.tensor([0.,1.])])
        adaptive = LocalModel(model,x,4,'adaptive'); pred = adaptive.predict(xs)
        for arm, key in [('scalar_excess','scalar_cpi'), ('mscs_logit','mscs_cpi')]:
            old = OriginalLocalModel(model,x,4,arm)
            torch.testing.assert_close(pred[key],old.predict(xs)['cpi'])
        self.assertEqual(adaptive.backward_outputs, 12)


if __name__ == '__main__':
    unittest.main()
