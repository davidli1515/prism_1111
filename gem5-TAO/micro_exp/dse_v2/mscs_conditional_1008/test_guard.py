import unittest
from lookahead_guard import inspect_reclaims


class GuardTests(unittest.TestCase):
    def test_large_mscs_score_does_not_imply_protect(self):
        actions = [{"id": "large_mscs", "score": 1., "checks": [
            {"key": "a", "config": 1.001, "reference_cpi": 1., "partner": "k"}]}]
        result = inspect_reclaims(actions, float, 1)
        self.assertEqual(result["decisions"][0]["status"], "verified_safe")

    def test_small_current_cost_can_hide_future_harm(self):
        # The observed xalancbmk four-point example; reference is AFTER decode grows.
        checks = [{"key": "joint", "config": 1.5822672843933105,
                   "reference_cpi": 1.5526820421218872, "partner": "decodeWidth"}]
        result = inspect_reclaims([{"id": "IQ", "score": .053, "checks": checks}], float, 1)
        self.assertEqual(result["decisions"][0]["status"], "protect")
        self.assertGreater(result["decisions"][0]["evidence"][0]["cost"], .019)

    def test_budget_exhaustion_does_not_mean_safe_and_duplicates_are_free(self):
        a = {"key": "a", "config": 1., "reference_cpi": 1., "partner": "k1"}
        b = {"key": "b", "config": 1., "reference_cpi": 1., "partner": "k2"}
        actions = [{"id": 1, "score": 2, "checks": [a, b]},
                   {"id": 2, "score": 1, "checks": [a]}]
        calls = []
        def query(c):
            calls.append(c)
            return c
        result = inspect_reclaims(actions, query, 1)
        self.assertEqual(len(calls), 1)
        self.assertEqual(result["queries"], 1)
        self.assertEqual([d["status"] for d in result["decisions"]], ["unverified", "verified_safe"])


if __name__ == "__main__":
    unittest.main()
