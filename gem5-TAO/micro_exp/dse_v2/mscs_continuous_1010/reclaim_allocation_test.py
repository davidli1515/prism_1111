"""At zero scalar slope, finite component curvature still constrains shrinkage."""
import unittest
import numpy as np
from reclaim_allocation_support import allocation,objective,TOLERANCE


class ReclaimTest(unittest.TestCase):
    def test_hidden_finite_cost(self):
        ell=np.log(np.array([[[.5,.5,1e-12]]]))
        odds=np.array([[[[-2.,2.,0.]]]]);total=1+np.exp(ell).sum(-1);cs=total/(1+1e-8*total)
        stage=np.zeros((1,1,1))
        root=dict(numROBEntries=100,dispatchWidth=4,numIQEntries=10,LQEntries=10,SQEntries=10,IntALU=2)
        basis=[dict(parameter='IntALU',config=dict(root,IntALU=1))]
        answers={k:allocation(k,root,basis,[-1.],10.,cs,ell,stage,odds,1) for k in ('scalar','mscs')}
        for r in answers.values():self.assertTrue(r['success'])
        self.assertAlmostEqual(answers['scalar']['fractions'][0],1.,places=6)
        expected=np.arccosh(1+2*TOLERANCE)/2
        self.assertAlmostEqual(answers['mscs']['fractions'][0],expected,places=6)
        self.assertGreater(objective(np.array(answers['scalar']['fractions']),'mscs',cs,ell,stage,odds,1)[0],1+TOLERANCE)
        self.assertLessEqual(answers['mscs']['normalized_predicted_cpi'],1+TOLERANCE+1e-8)


if __name__=='__main__':unittest.main()
