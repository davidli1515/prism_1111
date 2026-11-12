import unittest
import numpy as np
import torch
from torch import nn
from core import (ARMS, GEMModel, LocalModel, probabilities, cpis, exact, contributions,
                  legal, moves, config_key)
from config_dse import DEFAULT_CONFIG


class Affine(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Identity()
        self.head = nn.Linear(2, 16, dtype=torch.float64)
        with torch.no_grad():
            self.head.weight.zero_(); self.head.bias.zero_()
            for stage in range(4):
                self.head.weight[4*stage+1] = torch.tensor([-1., -.005])
                self.head.weight[4*stage+2] = torch.tensor([.99, 0.])
        self.eval()


class Intent(unittest.TestCase):
    def test_affine_components_predict_finite_move_exactly(self):
        model = Affine(); x = torch.zeros((3, 2), dtype=torch.float64)
        xs = torch.stack([torch.tensor([[1., 0.]]*3), torch.tensor([[0., 1.]]*3)]).double()
        local = LocalModel(model, x, 4, 'mscs_logit')
        p = local.predict(xs)
        truth = torch.tensor([exact(model, xx, 4)[0] for xx in xs], dtype=torch.float64)
        torch.testing.assert_close(p['cpi'], truth)
        self.assertEqual(int(torch.argmin(p['cpi'])), 1)
        for arm in ARMS[:2]:
            other = LocalModel(model, x, 4, arm).predict(xs)
            torch.testing.assert_close(other['cpi_slope'], p['cpi_slope'])
            self.assertEqual(int(torch.argmin(other['cpi'])), 0)
        self.assertGreater(float(truth[0]), local.cpi)
        self.assertLess(float(truth[1]), local.cpi)

    def test_full_direction_jacobian_against_autograd_jvp_and_finite_difference(self):
        torch.manual_seed(42)
        model = GEMModel(11, [7, 6], 0.).double().eval()
        x = torch.randn(3, 11, dtype=torch.float64)
        delta = torch.randn_like(x)*.1
        m = LocalModel(model, x, 4, 'mscs_logit')
        s = LocalModel(model, x, 4, 'scalar_linear')
        torch.testing.assert_close(m.gc, s.gc, rtol=1e-9, atol=1e-10)
        def fun(xx):
            z, pp = probabilities(model, xx)
            return torch.cat([cpis(pp, 4), (z[..., 1:]-z[..., :1]).reshape(len(xx), 12)], 1)
        _, jvp = torch.autograd.functional.jvp(fun, x, delta)
        dc = torch.einsum('nsd,nd->ns', s.gc, delta)
        dl = torch.einsum('nskd,nd->nsk', m.gell, delta).reshape(3, 12)
        torch.testing.assert_close(torch.cat([dc, dl], 1), jvp, rtol=1e-9, atol=1e-10)
        finite = (fun(x+1e-5*delta)-fun(x-1e-5*delta))/2e-5
        torch.testing.assert_close(finite, jvp, rtol=1e-6, atol=1e-9)

    def test_uses_non_arch_features(self):
        model = Affine(); x = torch.zeros((2, 2), dtype=torch.float64)
        local = LocalModel(model, x, 4, 'mscs_logit')
        only_second_feature = x.clone(); only_second_feature[:, 1] = 1
        p = local.predict(only_second_feature[None])
        self.assertLess(float(p['cpi'][0]), local.cpi)

    def test_clamps_and_softmax_are_preserved(self):
        model = Affine()
        with torch.no_grad():
            for stage in range(4):
                model.head.bias[4*stage] = -12
        x = torch.zeros((3, 2), dtype=torch.float64)
        a = LocalModel(model, x, 4, 'mscs_logit')
        b = LocalModel(model, x, 4, 'scalar_linear')
        torch.testing.assert_close(a.gc, b.gc)
        self.assertTrue(torch.equal(a.gc, torch.zeros_like(a.gc)))
        p = a.predict(torch.stack([x, x+1000]))
        self.assertTrue(torch.isfinite(p['cpi']).all())
        torch.testing.assert_close(p['mscs'].sum(-1), torch.ones_like(p['mscs'][..., 0]))
        self.assertTrue((p['cpi'] <= 25.).all())

    def test_explanations_account_for_cpi_without_summing_stages_twice(self):
        model = Affine(); x = torch.zeros((3, 2), dtype=torch.float64)
        _, a = exact(model, x, 4); _, b = exact(model, x+torch.tensor([0., 1.]), 4)
        da = contributions(b, 4)-contributions(a, 4)
        self.assertEqual(tuple(da.shape), (4, 3))
        self.assertAlmostEqual(float(da.sum(-1).mean()), float(cpis(b, 4).mean()-cpis(a, 4).mean()), places=8)

    def test_moves_have_no_collateral_edits_or_duplicate_costs(self):
        c = dict(DEFAULT_CONFIG)
        mm = moves(c)
        self.assertTrue(mm)
        self.assertEqual(len(mm), len({config_key(m['config']) for m in mm}))
        for m in mm:
            changed = [p for p in c if c[p] != m['config'][p]]
            self.assertEqual(changed, [m['parameter']]); self.assertTrue(legal(m['config']))
        seen = {config_key(m['config']) for m in mm[:5]}
        self.assertFalse(seen & {config_key(m['config']) for m in moves(c, seen=seen)})
        self.assertTrue(all(m['stride'] == 1 for m in moves(c, radius=1)))

    def test_root_zero_displacement_is_identity_and_weights_unchanged(self):
        torch.manual_seed(42)
        model = GEMModel(11, [7, 6], 0.).double().eval()
        snapshot = {k: v.clone() for k, v in model.state_dict().items()}
        x = torch.randn(3, 11, dtype=torch.float64)
        truth, _ = exact(model, x, 4)
        for arm in ARMS:
            local = LocalModel(model, x, 4, arm)
            self.assertAlmostEqual(float(local.predict(x[None])['cpi'][0]), truth, places=9)
            for k, v in model.state_dict().items():
                torch.testing.assert_close(v, snapshot[k])


if __name__ == '__main__':
    unittest.main()
