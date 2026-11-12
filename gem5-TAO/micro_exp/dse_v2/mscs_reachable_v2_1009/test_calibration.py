import unittest
import numpy as np
from calibration import controls,model,within_benchmark_permutation
from run import features,group_ids
from config_dse import DEFAULT_CONFIG


class Intent(unittest.TestCase):
    def test_known_signal_and_scalar_null(self):
        c=controls()
        self.assertTrue(c['positive_pass'],c['positive_mae'])
        self.assertTrue(c['negative_pass'])

    def test_base_information_is_identical(self):
        c=dict(DEFAULT_CONFIG,fetchBufferSize=32,l1i_assoc=16)
        m=np.tile([.2,.5,.2,.1]*4,(50,1)).astype(np.float32)
        changed=m.reshape(50,4,4).copy();changed[:,:,[1,2]]=changed[:,:,[2,1]]
        a,b=features(c,m),features(c,changed)
        np.testing.assert_array_equal(a[0],b[0]);np.testing.assert_array_equal(a[1],b[1])
        self.assertFalse(np.array_equal(a[2],b[2]))
        self.assertEqual(len(a[2]),4)

    def test_training_scaler_does_not_see_future(self):
        x=np.array([[1.,10.],[2.,20.],[3.,30.]])
        fit=model(.1).fit(x,[0.,1.,2.]);before=fit[0].mean_.copy()
        fit.predict([[1e8,-1e8]])
        np.testing.assert_array_equal(before,fit[0].mean_)
        np.testing.assert_allclose(before,[2.,20.])

    def test_connected_neighbors_stay_together(self):
        c=dict(DEFAULT_CONFIG)
        rows=[{'root':c},{'root':dict(c,numROBEntries=193)},
              {'root':dict(c,numROBEntries=193,numIQEntries=65)},
              {'root':dict(c,numROBEntries=400,numIQEntries=100,LQEntries=128)}]
        ids=group_ids(rows)
        self.assertEqual(ids[0],ids[2]);self.assertNotEqual(ids[0],ids[3])
        benches=np.array(['a','b','a','b','a','b'])
        perm=within_benchmark_permutation(benches)
        np.testing.assert_array_equal(benches,benches[perm])


if __name__=='__main__':unittest.main()
