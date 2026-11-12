import unittest
import torch
from secant_support import fit_correction,response,predict,LocalModel
from test_core import Affine


class Intent(unittest.TestCase):
    def test_minimum_norm_secants_interpolate_independent_points(self):
        x=torch.tensor([[[1.,0.]],[[0.,2.]]]);r=torch.tensor([[[3.,-1.]],[[2.,7.]]])
        out,info=fit_correction(x,x,r)
        torch.testing.assert_close(out,r)
        torch.testing.assert_close(info['weights'],torch.eye(2,dtype=torch.float64)[None])

    def test_zero_displacement_is_stable(self):
        x=torch.randn(3,2,4);h=torch.zeros(1,2,4);r=torch.ones(1,2,5)
        out,_=fit_correction(x,h,r);self.assertTrue(torch.equal(out,torch.zeros_like(out)))

    def test_scalar_does_not_receive_pressure_composition(self):
        a=torch.tensor([[[.4,.1,.2,.3]]]);b=torch.tensor([[[.4,.3,.1,.2]]])
        torch.testing.assert_close(response(a,4,'scalar'),response(b,4,'scalar'))
        self.assertFalse(torch.allclose(response(a,4,'mscs'),response(b,4,'mscs')))

    def test_no_history_recovers_original_predictors(self):
        model=Affine();x=torch.zeros((3,2),dtype=torch.float64)
        xs=torch.stack([x+torch.tensor([1.,0.]),x+torch.tensor([0.,1.])])
        local=LocalModel(model,x,4,'local');base=local.predict(xs)
        for space in ('scalar','mscs'):
            actual=response(local.m,4,space)
            pred,_,_=predict(local,xs,base,space,x[None][:0],actual[None][:0])
            torch.testing.assert_close(pred,base[space+'_cpi'])


if __name__=='__main__':unittest.main()
