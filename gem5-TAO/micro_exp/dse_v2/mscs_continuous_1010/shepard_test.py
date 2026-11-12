import unittest
import torch
from shepard_support import blend_responses,normalized_weights


class ShepardTest(unittest.TestCase):
    def test_affine_response(self):
        torch.manual_seed(42);d=torch.float64
        centers=torch.randn(3,2,4,dtype=d);g=torch.randn(2,3,4,dtype=d);b=torch.randn(2,3,dtype=d)
        ys=[b+torch.einsum('nvd,nd->nv',g,x) for x in centers]
        xs=torch.randn(7,2,4,dtype=d);value,_=blend_responses(xs,list(centers),ys,[g]*3)
        torch.testing.assert_close(value,b[None]+torch.einsum('nvd,cnd->cnv',g,xs),rtol=1e-12,atol=1e-12)

    def test_exact_root_derivative(self):
        d=torch.float64;centers=[torch.tensor([[0.,0.]],dtype=d),torch.tensor([[1.,2.]],dtype=d)]
        ys=[torch.tensor([[3.]],dtype=d),torch.tensor([[10.]],dtype=d)]
        gs=[torch.tensor([[[2.,-1.]]],dtype=d),torch.tensor([[[-4.,3.]]],dtype=d)]
        x=centers[0][None];value,_=blend_responses(x,centers,ys,gs)
        torch.testing.assert_close(value,ys[0][None])
        for i in range(2):
            delta=torch.zeros_like(x);delta[...,i]=1e-6
            hi,_=blend_responses(x+delta,centers,ys,gs);lo,_=blend_responses(x-delta,centers,ys,gs)
            torch.testing.assert_close((hi-lo)/2e-6,gs[0][...,i][None],rtol=1e-5,atol=1e-6)

    def test_coincident_centers(self):
        weights=normalized_weights(torch.tensor([[[0.,4.]],[[0.,16.]],[[9.,0.]]],dtype=torch.float64))
        torch.testing.assert_close(weights[:,0,0],torch.tensor([.5,.5,0.],dtype=torch.float64))
        torch.testing.assert_close(weights[:,0,1],torch.tensor([0.,0.,1.],dtype=torch.float64))
        self.assertTrue(torch.isfinite(weights).all())


if __name__=='__main__':unittest.main()
