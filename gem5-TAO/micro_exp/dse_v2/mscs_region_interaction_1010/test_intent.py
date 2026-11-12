"""Semantic tests: region identity, group isolation, actual interaction detectability."""
import copy
import unittest
import numpy as np
from region_data import (FE, axes, enumerate_set, features, expand, scopes, connected_groups,
                         legal, make_arrays, load_saved, root_features)
from config_dse import DEFAULT_CONFIG, sample_random_config
from calibration import design, fit_predict, folds, gate, ridge_fit, predict, shuffle_pressure, weights


def toy():
    rng = np.random.RandomState(42)
    n = 36
    rows, direct = [], []
    pressure = .25+1e-4*rng.uniform(-1, 1, (n, 4))
    z = (pressure-.25)/1e-4
    target = []
    for r in range(n):
        for j, kind in enumerate(('buffer', 'assoc', 'joint')):
            rows.append({'root_id': r, 'group': r, 'bench': 'toy', 'pattern': kind})
            db, da = float(j != 1), float(j != 0)
            direct.append([db, da, db*da, db, da, db*da])
            target.append(.01+.003*z[r, 0]*db-.002*z[r, 0]*da+.002*z[r, 1]*db*da)
    dd = np.array(direct)
    data = {'compact': np.c_[dd, np.ones(len(rows))], 'arch': np.ones((len(rows), 2)),
            'detail': np.ones((len(rows), 5)), 'pressure': np.repeat(pressure, 3, axis=0),
            'direction': dd, 'cap': np.full(len(rows), .2)}
    return data, rows, np.array(target), np.arange(72), np.arange(72, 108)


def controls():
    d, rows, y, tr, va = toy()
    p, _ = fit_predict(d, rows, y, tr, va)
    errs = {a: float(np.mean(abs(v-y[va]))) for a, v in p.items()}
    yn = .01+.003*d['direction'][:, 0]-.002*d['direction'][:, 1]
    pn, _ = fit_predict(d, rows, yn, tr, va)
    null = float(np.max(abs(pn['mscs']-pn['scalar'])))
    return {'positive_pass': errs['mscs'] < .1*min(errs['scalar'], errs['shuffled']),
            'positive_mae': errs, 'negative_pass': null < 1e-4,
            'negative_max_difference': null,
            'scope': 'Specified synthetic interaction signal, not evidence of real MSCS value'}


class Intent(unittest.TestCase):
    def test_cartesian_scope_and_missing_label_rejection(self):
        c = dict(DEFAULT_CONFIG, fetchBufferSize=16, l1i_assoc=8)
        whole = enumerate_set(c, axes(c))
        self.assertEqual(len(whole), 8)
        self.assertEqual([len(v) for _, _, v in scopes(c)], [2, 2, 8])
        bad = dict(c, l1_size=16, l1i_assoc=32)
        self.assertFalse(legal(bad))
        r = {'root': c, 'bench': 'test', 'root_cpi': 2., 'candidates': whole[:-1],
             'descendant_cpis': [1.9]*7}
        with self.assertRaisesRegex(AssertionError, 'Incomplete'):
            expand([r])

    def test_identical_root_different_scope_changes_features(self):
        c = dict(DEFAULT_CONFIG, fetchBufferSize=32, l1i_assoc=16)
        m = np.tile([.2, .5, .2, .1]*4, (50, 1))
        ff = [features(c, m, domain) for _, domain, _ in scopes(c)]
        self.assertEqual(len(ff), 3)
        np.testing.assert_array_equal(ff[0]['pressure'], ff[1]['pressure'])
        self.assertFalse(np.array_equal(ff[0]['compact'], ff[1]['compact']))
        self.assertFalse(np.array_equal(ff[0]['direction'], ff[1]['direction']))

    def test_pressure_change_preserves_all_scalar_features(self):
        c = dict(DEFAULT_CONFIG, fetchBufferSize=32, l1i_assoc=16)
        m = np.tile([.2, .5, .2, .1]*4, (50, 1)).astype(np.float32)
        n = m.reshape(50, 4, 4).copy(); n[:, :, [1, 2]] = n[:, :, [2, 1]]
        a, b = features(c, m, axes(c)), features(c, n, axes(c))
        for k in ('compact', 'arch', 'detail', 'direction', 'cap'):
            np.testing.assert_array_equal(a[k], b[k])
        self.assertFalse(np.array_equal(a['pressure'], b['pressure']))

    def test_connected_groups_and_root_equal_weights(self):
        c = dict(DEFAULT_CONFIG)
        rr = [{'root': c}, {'root': dict(c, numROBEntries=193)},
              {'root': dict(c, numROBEntries=193, numIQEntries=65)},
              {'root': dict(c, numROBEntries=400, numIQEntries=100, LQEntries=128)}]
        g = connected_groups(rr)
        self.assertEqual(g[0], g[2]); self.assertNotEqual(g[0], g[3])
        rows = [{'root_id': 0, 'group': 0}]*3+[{'root_id': 1, 'group': 0}]+[{'root_id': 2, 'group': 1}]*3
        w = weights(rows, np.arange(7))
        np.testing.assert_allclose([sum(w[:3]), w[3], sum(w[4:])], [.25, .25, .5])

    def test_shuffle_respects_root_partition_and_benchmark(self):
        d, rows, y, tr, va = toy()
        for i in range(len(rows)):
            rows[i]['bench'] = 'a' if rows[i]['root_id'] % 2 else 'b'
        pp, donors = shuffle_pressure(d, rows, tr)
        for a, b in donors.items():
            self.assertEqual(a % 2, b % 2); self.assertLess(b, 24)
        for r in range(24):
            np.testing.assert_array_equal(pp[3*r], pp[3*r+2])

    def test_heldout_labels_cannot_change_predictions_or_model_selection(self):
        d, rows, y, tr, va = toy()
        a, info = fit_predict(d, rows, y, tr, va)
        changed = y.copy(); changed[va] = 1e9
        b, infob = fit_predict(d, rows, changed, tr, va)
        self.assertEqual(info, infob)
        for k in a:
            np.testing.assert_array_equal(a[k], b[k])

    def test_train_only_scaling_and_dual_ridge_against_primal(self):
        rng = np.random.RandomState(42)
        x, y = rng.normal(size=(9, 4)), rng.uniform(.02, .03, 9)
        w = np.arange(1, 10, dtype=float); w /= w.sum()
        fit = ridge_fit(x, y, w, [2, 2], .1)
        z = (x-fit['mean'])/fit['scale']/fit['blockscale']
        coef = np.linalg.solve(z.T @ ((9*w)[:, None]*z)+.1*np.eye(4), z.T @ (9*w*(y-w@y)))
        np.testing.assert_allclose(fit['coef'], coef, atol=1e-12)
        mean = fit['mean'].copy(); predict(fit, np.full((2, 4), 1e8), np.ones(2))
        np.testing.assert_array_equal(mean, fit['mean'])

    def test_interaction_signal_and_scalar_only_null(self):
        r = controls()
        self.assertTrue(r['positive_pass'], r)
        self.assertTrue(r['negative_pass'], r)

    def test_independent_primal_replay_and_no_pressure_scaler_leakage(self):
        from validate_result import replay_fold
        d, rows, y, tr, va = toy()
        predictions, report = fit_predict(d, rows, y, tr, va)
        replay = replay_fold(d, rows, y, dict(report, train=tr.tolist(), valid=va.tolist()))
        for arm in predictions:
            np.testing.assert_allclose(predictions[arm], replay[arm], atol=2e-9, rtol=1e-7)
        _, _, _, stats = design(d, rows, tr, 'compact', 'mscs')
        unchanged = [x.copy() for x in stats]
        modified = {k: v.copy() for k, v in d.items()}; modified['pressure'][va] += 1000
        design(modified, rows, va, 'compact', 'mscs', stats)
        for a, b in zip(stats, unchanged):
            np.testing.assert_array_equal(a, b)

    def test_gates_require_all_conditions(self):
        good = {'scalar': {'mae': .01, 'max_underprediction': .02},
                'mscs': {'mae': .008, 'max_underprediction': .019},
                'shuffled': {'mae': .011, 'max_underprediction': .023},
                'constant': {'mae': .02, 'max_underprediction': .025}}
        benches = {b: copy.deepcopy(good) for b in ('602.gcc_s', '605.mcf_s')}
        self.assertTrue(gate(good, benches)['pass'])
        bad = copy.deepcopy(good); bad['mscs']['max_underprediction'] = .021
        self.assertFalse(gate(bad, benches)['pass'])
        bad = copy.deepcopy(good); bad['constant']['mae'] = .009
        self.assertFalse(gate(bad, benches)['pass'])
        benches['605.mcf_s']['mscs']['mae'] = .012
        self.assertFalse(gate(good, benches)['pass'])

    def test_saved_complete_sets_and_legacy_projection_agree(self):
        roots, rows = load_saved()
        self.assertEqual(len(rows), 40)
        self.assertEqual(len(set(r['group'] for r in rows)), 21)
        from dse_engine import _project_to_valid
        for r in roots:
            for c in [r['root']]+r['candidates']:
                self.assertEqual(_project_to_valid(c), c)
        rng = np.random.RandomState(42)
        for _ in range(30):
            c = sample_random_config(rng)
            self.assertEqual(legal(c), _project_to_valid(c) == c)
        make_arrays(roots, rows)
        for tr, va in folds(rows, np.arange(len(rows)), 5):
            self.assertFalse({rows[i]['root_id'] for i in tr} & {rows[i]['root_id'] for i in va})


if __name__ == '__main__':
    unittest.main()
