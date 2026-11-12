import unittest
import numpy as np
from exchange_support import weights,response,rounding
from config_dse import DEFAULT_CONFIG


class ExchangeTests(unittest.TestCase):
    def test_no_false_cancellation_and_gradients(self):
        names=['a','b'];basis=[dict(parameter=k,sign=s) for k in names for s in (1,-1)]
        ell=np.array([[[.2,-.3,.7]]*4]);dirs=np.arange(48,dtype=float).reshape(4,1,4,3)/100-.2
        for v in (np.array([.3,-.4]),np.array([-.2,.5])):
            w,j=weights(v,basis,names);self.assertEqual(np.count_nonzero(w),2)
            for k in range(2):self.assertEqual(w[2*k]*w[2*k+1],0.)
            for kind in ('scalar','mscs'):
                f,g=response(v,kind,basis,names,ell,dirs,8);num=[]
                for i in range(2):
                    d=np.zeros(2);d[i]=1e-6
                    num.append((response(v+d,kind,basis,names,ell,dirs,8)[0]-response(v-d,kind,basis,names,ell,dirs,8)[0])/2e-6)
                np.testing.assert_allclose(g,num,rtol=2e-5,atol=1e-8)
        w,j=weights(np.zeros(2),basis,names);np.testing.assert_array_equal(w,0)
        for kind in ('scalar','mscs'):
            f0=response(np.zeros(2),kind,basis,names,ell,dirs,8)[0]
            for i in range(2):
                for s in (-1,1):
                    d=np.zeros(2);d[i]=s*1e-9
                    self.assertLess(abs(response(d,kind,basis,names,ell,dirs,8)[0]-f0),1e-7)

    def test_minimal_rob_repair(self):
        root=dict(DEFAULT_CONFIG,numROBEntries=144)
        basis=[dict(parameter='numROBEntries',sign=1,config=dict(root,numROBEntries=176)),
               dict(parameter='numIQEntries',sign=1,config=dict(root,numIQEntries=80))]
        rr=rounding(root,basis,['numROBEntries','numIQEntries'],[0.,1.])
        self.assertEqual(len(rr),1);self.assertEqual(rr[0]['config']['numROBEntries'],160)


if __name__=='__main__':unittest.main()
