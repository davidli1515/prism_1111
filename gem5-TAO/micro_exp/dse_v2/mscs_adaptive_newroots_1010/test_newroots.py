import unittest
from pathlib import Path
from newroots_support import (nearest_unexposed, collect_configs, config_key,
                              bind_runners, ADAPTIVE, HYBRID, RankAllocator)


class SelectionAndReuse(unittest.TestCase):
    def test_area_rank_target_and_exclusion_ties(self):
        configs=[{'x':i} for i in range(9)]
        choose=lambda used:nearest_unexposed(configs,'b',used,legal_fn=lambda c:True)
        self.assertEqual(choose(set())[:2],(6,6))
        used={('b',config_key(configs[6]))}
        self.assertEqual(choose(used)[:2],(5,6))
        used.add(('b',config_key(configs[5])))
        self.assertEqual(choose(used)[:2],(7,6))

    def test_no_valid_root_fails_without_silent_substitution(self):
        with self.assertRaises(ValueError):
            nearest_unexposed([{'x':1}],'b',set(),legal_fn=lambda c:False)

    def test_excludes_root_trials_and_predicted_only_configs(self):
        a={'fetchWidth':4,'LQEntries':20,'bp_type':1}; b=dict(a,LQEntries=28)
        obj={'contexts':[{'bench':'602.gcc_s','root':a,'candidates':[{'config':b}]}]}
        found=collect_configs(obj)
        self.assertEqual(found,{('602.gcc_s',config_key(a)),('602.gcc_s',config_key(b))})
        # Feedback values do not affect the exposure extraction.
        obj['contexts'][0]['cpi']=1e9
        self.assertEqual(collect_configs(obj),found)

    def test_exact_policy_reuse_and_output_isolation(self):
        h,a=bind_runners('/tmp/newroots_unit_output',123.)
        self.assertEqual(Path(h.walk.__code__.co_filename).resolve(),HYBRID/'run_hybrid.py')
        self.assertEqual(Path(a.walk.__code__.co_filename).resolve(),ADAPTIVE/'run_adaptive.py')
        self.assertIs(a.RankAllocator,RankAllocator)
        self.assertEqual(a.OUT,h.OUT)
        self.assertEqual(a.OUT,Path('/tmp/newroots_unit_output'))
        self.assertEqual(a.START,h.START)


if __name__ == '__main__':
    unittest.main()
