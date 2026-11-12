import unittest
import torch
from local_support import LocalAllocator,LocalModel
from core import LocalModel as OriginalLocalModel
from test_core import Affine


def action(p='LQEntries',sign=1):
    return dict(parameter=p,sign=sign)


class Intent(unittest.TestCase):
    def test_initial_ties_are_fixed_hybrid_and_distinct(self):
        p=LocalAllocator();a=[action(),action('SQEntries'),action('numIQEntries')]
        self.assertEqual(p.choose(a,[1,2,3],[3,2,1])[0],[0,2])
        self.assertEqual(p.choose(a,[1,1,1],[1,1,1])[0],[0,1])
        self.assertEqual(p.choose([],[],[])[0],[])
        self.assertEqual(p.choose(a[:1],[1],[1])[0],[0])

    def test_resource_and_direction_isolation(self):
        p=LocalAllocator();p.observe(10,[action()],[9],[11],[11])
        a=[action(),action(sign=-1),action('SQEntries')]
        before=p.snapshot();_,_,info=p.choose(a,[9,8,7],[11,20,30])
        self.assertEqual(info['anchor_models'],['mscs','scalar','scalar'])
        self.assertEqual(info['anchor_scores'],[11,8,7]);self.assertEqual(p.snapshot(),before)

    def test_cross_resource_comparison_does_not_assign_credit(self):
        p=LocalAllocator()
        # Both experts get improvement signs right but reverse cross-resource order.
        r=p.observe(10,[action(),action('SQEntries')],[8,9],[9,8],[9,8])
        self.assertTrue(all(v['feedback']['loss2']=={'scalar':0,'mscs':0} for v in r.values()))

    def test_same_bucket_relative_ranking_is_scored(self):
        p=LocalAllocator();r=p.observe(10,[action(),action()],[8,9],[9,8],[9,8])
        self.assertEqual(r['LQEntries:+1']['feedback']['loss2'],{'scalar':2,'mscs':0})

    def test_update_affects_next_batch_and_can_reverse(self):
        p=LocalAllocator();a=[action(),action('SQEntries')]
        first=p.choose(a,[8,9],[11,9]);self.assertEqual(first[0],[0,1])
        p.observe(10,[action()],[8],[11],[11])
        self.assertEqual(first[0],[0,1]);self.assertEqual(p.choose(a,[8,9],[11,9])[0],[1,0])
        p.observe(10,[action()],[9],[11],[9])
        self.assertEqual(p.choose(a,[8,9],[11,9])[2]['anchor_models'],['scalar','scalar'])

    def test_observed_ties_do_not_manufacture_evidence(self):
        p=LocalAllocator();r=p.observe(10,[action()],[1],[100],[10+1e-7])
        self.assertEqual(r['LQEntries:+1']['feedback']['comparisons'],[])

    def test_predictors_are_unchanged(self):
        model=Affine();x=torch.zeros((3,2),dtype=torch.float64)
        xs=torch.stack([x+torch.tensor([1.,0.]),x+torch.tensor([0.,1.])])
        local=LocalModel(model,x,4,'local');pred=local.predict(xs)
        for arm,key in [('scalar_excess','scalar_cpi'),('mscs_logit','mscs_cpi')]:
            old=OriginalLocalModel(model,x,4,arm)
            torch.testing.assert_close(pred[key],old.predict(xs)['cpi'])


if __name__=='__main__':unittest.main()
