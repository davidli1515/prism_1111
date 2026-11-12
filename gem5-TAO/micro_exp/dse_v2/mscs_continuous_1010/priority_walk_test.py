import unittest
import torch
from types import SimpleNamespace
from priority_walk_support import scalar_response,scalar_quadratic
from width_priority_support import predict


class ScalarTests(unittest.TestCase):
    def test_zero_residual_matches_scalar_with_variable_width(self):
        torch.manual_seed(42)
        x=torch.randn(2,5,dtype=torch.float64);ell=torch.randn(2,4,3,dtype=torch.float64)
        g=torch.randn(2,4,3,5,dtype=torch.float64)*.01;local=SimpleNamespace(x=x,ell=ell,gell=g)
        xs=x[None]+torch.randn(3,2,5,dtype=torch.float64);widths=[4,7,12]
        hx=x[None];hy=torch.logsumexp(ell,-1)[None]
        scores,info=scalar_quadratic(local,xs,widths,hx,hy)
        de=torch.einsum('nskd,cnd->cnsk',g,xs-x[None]);expected=predict(ell,de,widths)['scalar_cpi']
        torch.testing.assert_close(scores,expected);torch.testing.assert_close(info['correction'],torch.zeros_like(info['correction']))

    def test_response_is_base_only(self):
        m=torch.tensor([[[.2,.1,.3,.4]]],dtype=torch.float64)
        p=m.clone();p[...,1:]=torch.tensor([.4,.1,.3],dtype=torch.float64)
        torch.testing.assert_close(scalar_response(m),scalar_response(p))
        self.assertAlmostEqual(float(scalar_response(m)),__import__('math').log(4.))


if __name__=='__main__':unittest.main()
