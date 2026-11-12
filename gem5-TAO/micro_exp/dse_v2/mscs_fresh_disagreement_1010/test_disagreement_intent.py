import unittest
from fresh_disagreement_support import configs_in, config_key, ALL_PARAMS, choices


class Intent(unittest.TestCase):
    def test_inventory_detects_config_without_special_container_name(self):
        c = {k:1 for k in ALL_PARAMS}
        known, unknown = configs_in(dict(bench='602.gcc_s', arbitrary=[c]))
        self.assertEqual(known,{('602.gcc_s',config_key(c))}); self.assertFalse(unknown)

    def test_unknown_benchmark_is_preserved_as_unknown(self):
        c = {k:1 for k in ALL_PARAMS}
        known, unknown = configs_in(dict(arbitrary=c))
        self.assertFalse(known); self.assertEqual(unknown,{config_key(c)})

    def test_other_benchmark_is_not_mislabeled_as_target(self):
        c = {k:1 for k in ALL_PARAMS}
        known, unknown = configs_in(dict(benchmark='623.xalancbmk_s', whatever=c))
        self.assertEqual(known,{('623.xalancbmk_s',config_key(c))}); self.assertFalse(unknown)

    def test_cold_local_equals_hybrid_on_disagreement_and_ties(self):
        d = dict(candidates=[dict(parameter=p,sign=1) for p in ('LQEntries','SQEntries','numIQEntries')],
                 scalar_predicted_cpis=[1,2,3],mscs_predicted_cpis=[3,2,1],
                 allocator_before=dict(completed_rounds=0,buckets={}))
        for ids in ([0,1,2],[1,2]):
            p = choices(d,ids); self.assertEqual(p['local'],p['hybrid'])
            self.assertTrue(all(len(x)==len(set(x))==2 for x in p.values()))
        d['mscs_predicted_cpis']=[1,1,1]; d['scalar_predicted_cpis']=[1,1,1]
        self.assertEqual(choices(d,[0,1,2])['local'],[0,1])


if __name__ == '__main__':
    unittest.main()
