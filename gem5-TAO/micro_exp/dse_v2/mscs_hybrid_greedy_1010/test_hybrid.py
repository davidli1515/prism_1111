import unittest
import numpy as np
import torch
from hybrid_support import choose, LocalModel, OriginalLocalModel
from test_core import Affine


class Intent(unittest.TestCase):
    def test_disagreement_reserves_both_sources_without_a_third_query(self):
        ids, sources = choose('hybrid', [1.,2.,3.], [3.,2.,1.])
        self.assertEqual(ids, [0,2]); self.assertEqual(sources, ['scalar_anchor','mscs_alternative'])
        self.assertEqual(choose('scalar_excess',[1.,2.,3.],None)[0], [0,1])
        self.assertEqual(choose('mscs_logit',[1.,2.,3.],[3.,2.,1.])[0], [2,1])

    def test_agreement_and_ties_preserve_distinct_candidates_and_generator_order(self):
        self.assertEqual(choose('hybrid',[1.,1.,2.],[1.,1.,2.])[0], [0,1])
        self.assertEqual(choose('hybrid',[3.,2.,1.],[3.,2.,1.])[0], [2,1])
        self.assertEqual(choose('hybrid',[1.],[2.])[0], [0])
        self.assertEqual(choose('hybrid',[],[])[0], [])

    def test_hybrid_uses_identical_scalar_and_mscs_numerics(self):
        model = Affine(); x = torch.zeros((3,2),dtype=torch.float64)
        xs = torch.stack([x+torch.tensor([1.,0.]),x+torch.tensor([0.,1.])])
        h = LocalModel(model,x,4,'hybrid'); hp = h.predict(xs)
        sc = OriginalLocalModel(model,x,4,'scalar_excess'); sp = sc.predict(xs)
        ms = OriginalLocalModel(model,x,4,'mscs_logit'); mp = ms.predict(xs)
        torch.testing.assert_close(hp['scalar_cpi'],sp['cpi'])
        torch.testing.assert_close(hp['mscs_cpi'],mp['cpi'])
        torch.testing.assert_close(h.gc,sc.gc)
        ids, _ = choose('hybrid',hp['scalar_cpi'],hp['mscs_cpi'])
        self.assertEqual(ids,[0,1])
        self.assertEqual(h.backward_outputs,12)


if __name__ == '__main__':
    unittest.main()
