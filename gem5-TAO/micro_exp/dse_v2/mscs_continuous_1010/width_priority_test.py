import unittest
import torch
from width_priority_support import predict,width_moves,close_width
from core import legal
from config_dse import DEFAULT_CONFIG


class WidthTests(unittest.TestCase):
    def test_known_width_and_equal_slopes(self):
        ell=torch.zeros((2,4,3),dtype=torch.float64);slope=torch.zeros((2,2,4,3),dtype=torch.float64)
        p=predict(ell,slope,[4,8])
        torch.testing.assert_close(p['scalar_cpi'],p['mscs_cpi'])
        self.assertAlmostEqual(float(p['scalar_cpi'][0]/p['scalar_cpi'][1]),2.,places=7)
        slope.fill_(.3);p=predict(ell,slope,[4,8]);torch.testing.assert_close(p['scalar_cpi'],p['mscs_cpi'])

    def test_jensen_and_gradient(self):
        ell=torch.tensor([[[.2,-.7,.8]]*4],dtype=torch.float64)
        slope=torch.tensor([[[[1.,-1.,.3]]*4]],dtype=torch.float64)
        p=predict(ell,slope,[8]);self.assertGreater(float(p['mscs_cpi'][0]),float(p['scalar_cpi'][0]))
        e=1e-5
        plus=predict(ell,e*slope,[8]);minus=predict(ell,-e*slope,[8])
        self.assertAlmostEqual(float((plus['mscs_cpi']-minus['mscs_cpi'])/(2*e)),float((plus['scalar_cpi']-minus['scalar_cpi'])/(2*e)),places=7)

    def test_explicit_closure(self):
        root=dict(DEFAULT_CONFIG,numROBEntries=144)
        up=close_width(root,'dispatchWidth',12)
        self.assertEqual({up[k] for k in ('fetchWidth','decodeWidth','dispatchWidth','commitWidth','wbWidth')},{12})
        self.assertEqual(up['numROBEntries'],152);self.assertTrue(legal(up))
        low=close_width(root,'fetchWidth',4)
        self.assertEqual([low[k] for k in ('fetchWidth','decodeWidth','dispatchWidth')],[4,4,4])
        self.assertEqual(low['commitWidth'],8);self.assertEqual(low['wbWidth'],8)
        pool=width_moves(root,4)
        self.assertEqual(len(pool),len({tuple(sorted(a['config'].items())) for a in pool}))
        for a in pool:
            self.assertTrue(legal(a['config']))
            self.assertEqual(set(a['edits']),{k for k in root if a['config'][k]!=root[k]})
            self.assertEqual(a['config']['l1_size'],root['l1_size']);self.assertEqual(a['config']['bp_type'],root['bp_type'])


if __name__=='__main__':unittest.main()
