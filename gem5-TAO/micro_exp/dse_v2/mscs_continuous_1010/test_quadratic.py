import unittest
import torch
from quadratic_support import quadratic_fit
from secant_support import fit_correction


class Intent(unittest.TestCase):
    def test_exact_quadratic_residual(self):
        h=torch.tensor([[[2.]]],dtype=torch.float64);r=h.square()
        c=torch.tensor([[[-1.]],[[0.]],[[.5]],[[3.]]],dtype=torch.float64)
        out,_=quadratic_fit(c,h,r);torch.testing.assert_close(out,c.square())

    def test_derivative_preserved_at_root(self):
        h=torch.tensor([[[1.,0.]],[[0.,1.]]],dtype=torch.float64)
        r=torch.tensor([[[2.]],[[3.]]],dtype=torch.float64)
        c=torch.zeros((1,1,2),dtype=torch.float64,requires_grad=True)
        out,_=quadratic_fit(c,h,r);grad=torch.autograd.grad(out.sum(),c)[0]
        torch.testing.assert_close(grad,torch.zeros_like(c))
        old,_=fit_correction(c,h,r);oldgrad=torch.autograd.grad(old.sum(),c)[0]
        self.assertGreater(float(oldgrad.abs().sum()),0.)

    def test_zero_history_displacement_stable(self):
        c=torch.ones((2,1,3));h=torch.zeros((1,1,3));r=torch.ones((1,1,4))
        out,_=quadratic_fit(c,h,r);self.assertTrue(torch.equal(out,torch.zeros_like(out)))


if __name__=='__main__':unittest.main()
