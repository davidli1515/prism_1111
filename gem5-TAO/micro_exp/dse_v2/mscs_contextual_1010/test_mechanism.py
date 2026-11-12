import unittest
import numpy as np
import torch
from torch import nn
from mechanism import (LocalModel, candidates, context_rows, select, witnesses,
                       compute_w_min, config_key, legal, MODES)
from config_dse import DEFAULT_CONFIG


def affine(weights):
    m = nn.Module(); m.backbone = nn.Identity(); m.head = nn.Linear(2, 16, dtype=torch.float64)
    with torch.no_grad():
        m.head.weight.zero_(); m.head.bias.zero_()
        for s in range(4):
            m.head.weight[4*s+1:4*s+4] = torch.tensor(weights, dtype=torch.float64)
    return m.eval()


class Intent(unittest.TestCase):
    def test_conditional_decoupling_has_a_counterfactual_witness(self):
        model = affine([[-2, 1], [1, -2], [0, 0]])
        x = torch.zeros((3, 2), dtype=torch.float64)
        pred = LocalModel(model, x, 4, 'mscs_logit').predict(torch.stack([x+torch.tensor([1,0]), x+torch.tensor([0,1]), x+1]))['cpi'].numpy()
        pool = [{'kind':'single'}, {'kind':'single'}, {'kind':'up_up','parts':[0,1]}]
        row = context_rows(pool, pred, 1.)[2]
        self.assertGreater(pred[0], 1.); self.assertGreater(pred[1], 1.)
        self.assertLess(pred[2], 1.); self.assertLess(row['interaction'], 0.)

    def test_protect_requires_future_cost_not_just_large_response(self):
        model = affine([[.5, .1], [-2, -.11], [0, 0]])
        x = torch.zeros((3, 2), dtype=torch.float64)
        pred = LocalModel(model, x, 4, 'mscs_logit').predict(torch.stack([x+torch.tensor([1,0]), x+torch.tensor([0,1]), x+1]))['cpi'].numpy()
        pool = [{'kind':'single'}, {'kind':'single'}, {'kind':'up_down','parts':[0,1]}]
        r = context_rows(pool, pred, 1.)[2]
        self.assertGreater(r['investment_gain'], 0.)
        self.assertTrue(0 <= r['alone_cost'] < .003)
        self.assertGreater(r['after_investment_cost'], .003)
        self.assertTrue(r['protect'])
        quiet = context_rows(pool, np.array([.9, 1.001, .901]), 1.)[2]
        self.assertFalse(quiet['protect'])

    def test_candidates_are_unique_legal_and_have_exactly_their_two_edits(self):
        root = dict(DEFAULT_CONFIG); pool = candidates(root)
        self.assertEqual(len(pool), len({config_key(c['config']) for c in pool}))
        for c in pool:
            self.assertTrue(legal(c['config']))
            self.assertEqual(compute_w_min(c['config']), compute_w_min(root))
            changed = {p for p in root if root[p] != c['config'][p]}
            if c['kind'] == 'single':
                self.assertEqual(changed, {c['parameter']})
            else:
                self.assertEqual(changed, {pool[i]['parameter'] for i in c['parts']})
                self.assertEqual(len(changed), 2)

    def test_ablation_and_shared_scalar_anchor_do_not_veto_each_other(self):
        c = dict(DEFAULT_CONFIG)
        pool = [{'kind':'single','parts':[], 'config':c}, {'kind':'single','parts':[], 'config':c},
                {'kind':'up_up','parts':[0,1], 'config':c}, {'kind':'up_down','parts':[0,1], 'config':c}]
        sc = np.array([.9, 1.001, .88, .92]); mm = np.array([.95, 1.001, .8, .97])
        out = {mode: select(pool, sc, mm, 1., mode) for mode in MODES}
        self.assertEqual(len({r['anchor'] for r in out.values()}), 1)
        self.assertTrue(all(len(set(r['chosen'])) == 2 for r in out.values()))
        self.assertFalse(out['mscs']['allowed'][3])
        self.assertTrue(out['no_protect']['allowed'][3])
        self.assertAlmostEqual(out['no_decouple']['scores'][2], .951)
        self.assertAlmostEqual(out['no_context']['scores'][3], .951)
        self.assertEqual(witnesses(pool, sc, mm, 1., out['mscs'])['protect'], 3)


if __name__ == '__main__':
    unittest.main()
