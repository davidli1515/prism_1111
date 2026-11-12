import unittest
import numpy as np
from run import feature_blocks, nearby, predict
from config_dse import DEFAULT_CONFIG


class Intent(unittest.TestCase):
    def test_extra_features_do_not_change_scalar_or_base(self):
        c = dict(DEFAULT_CONFIG, fetchBufferSize=64, l1i_assoc=4)
        m = np.tile([.2, .5, .2, .1]*4, (50, 1)).astype(np.float32)
        changed = m.reshape(50, 4, 4).copy()
        changed[:, :, [1, 2]] = changed[:, :, [2, 1]]
        a, b = feature_blocks(c, m), feature_blocks(c, changed)
        for x, y in zip(a[:3], b[:3]):
            np.testing.assert_array_equal(x, y)
        self.assertFalse(np.array_equal(a[3], b[3]))

    def test_scalar_retains_slice_distribution(self):
        c = dict(DEFAULT_CONFIG, fetchBufferSize=64, l1i_assoc=4)
        m = np.tile([.2, .5, .2, .1]*4, (50, 1)).astype(np.float32)
        a = feature_blocks(c, m)
        m[:25, ::4] = .1; m[25:, ::4] = .3
        b = feature_blocks(c, m)
        self.assertFalse(np.array_equal(a[2], b[2]))

    def test_group_purge_includes_neighboring_resources(self):
        c = dict(DEFAULT_CONFIG)
        self.assertTrue(nearby(c, dict(c, fetchBufferSize=64, l1i_assoc=32)))
        self.assertTrue(nearby(c, dict(c, numROBEntries=193)))
        self.assertFalse(nearby(c, dict(c, numROBEntries=193, numIQEntries=65)))


if __name__ == '__main__':
    unittest.main()
