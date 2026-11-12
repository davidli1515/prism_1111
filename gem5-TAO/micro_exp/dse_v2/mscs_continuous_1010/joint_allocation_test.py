"""Analytically solvable cancellation case; scalar prefers the wrong corner."""
import unittest
import numpy as np
from scipy.optimize import minimize
from joint_allocation_support import objective


class AllocationTest(unittest.TestCase):
    def test_complementary_allocation(self):
        ell=np.log(np.array([[[.5,.5,1e-12]]]))
        odds=np.array([[[[-2.4,1.,0.]]],[[[.8,-1.6,0.]]]])
        total=1+np.exp(ell).sum(-1);cs=total/(1+1e-8*total)
        # Exact root derivative of physical CPI, independent of scalar objective.
        stage=(np.exp(ell)[None]*odds).sum(-1)/(1+1e-8*total[None])**2
        results={}
        for kind in ('scalar','mscs'):
            fn=lambda w:objective(w,kind,cs,ell,stage,odds,1)
            r=minimize(fn,[0.,0.],jac=True,method='SLSQP',bounds=[(0,1)]*2,
                       constraints=[dict(type='ineq',fun=lambda x:1-x.sum(),jac=lambda x:-np.ones(2))],
                       options=dict(ftol=1e-12,maxiter=100))
            self.assertTrue(r.success);results[kind]=r.x
        np.testing.assert_allclose(results['scalar'],[1,0],atol=1e-6)
        a=(2.4+np.log(3.2/2.6))/5.8
        np.testing.assert_allclose(results['mscs'],[a,1-a],atol=2e-6)
        true=lambda x:objective(x,'mscs',cs,ell,stage,odds,1)[0]
        self.assertLess(true(results['mscs']),1.)
        self.assertGreater(true(results['scalar']),1.)


if __name__=='__main__':unittest.main()
