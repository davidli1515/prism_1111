import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from run_control import skip_cancelled_run


class RunControlTest(unittest.TestCase):
    def test_only_pending_disallowed_seeds_are_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "run_control.json").write_text(json.dumps(
                {"allowed_seeds": [42], "reason": "single-seed test"}))
            for seed in (42, 43, 44):
                with contextlib.redirect_stdout(io.StringIO()):
                    skipped = skip_cancelled_run([
                        "--bench", "602.gcc_s", "--seed", str(seed), "--out", tmp,
                        "--total-queries", "5000"])
                self.assertEqual(skipped, seed != 42)
            records = list(root.glob("cancelled_*.json"))
            self.assertEqual(len(records), 2)
            self.assertTrue(all(json.loads(p.read_text())["surrogate_queries"] == 0 for p in records))
            self.assertFalse((root / "602.gcc_s").exists())

    def test_no_control_keeps_original_behavior(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(skip_cancelled_run(["--seed", "44", "--out", tmp]))


if __name__ == "__main__":
    unittest.main()
