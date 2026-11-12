import unittest
from pair_priority_support import dense_singles,pair_universe
from config_dse import DEFAULT_CONFIG
from secant_support import config_key


class PairTests(unittest.TestCase):
    def test_stride_three_and_blind_pair(self):
        root=dict(DEFAULT_CONFIG,numROBEntries=320,LQEntries=40,SQEntries=64)
        singles=dense_singles(root,{config_key(root)})
        li=next(i for i,a in enumerate(singles) if a['parameter']=='LQEntries' and a['config']['LQEntries']==64)
        iq=next(i for i,a in enumerate(singles) if a['parameter']=='numIQEntries' and a['config']['numIQEntries']==96)
        self.assertEqual(singles[li]['stride'],3)
        eligible=[i for i,a in enumerate(singles) if a['kind']=='single_resource' and a['sign']==1]
        pairs=pair_universe(root,singles,eligible,{config_key(root)})
        matches=[a for a in pairs if set(a['indices'])=={li,iq}];self.assertEqual(len(matches),1)
        self.assertEqual(matches[0]['config'],dict(root,LQEntries=64,numIQEntries=96))
        self.assertEqual(len(pairs),len({config_key(a['config']) for a in pairs}))
        for a in pairs:self.assertEqual(len(a['edits']),2)


if __name__=='__main__':unittest.main()
