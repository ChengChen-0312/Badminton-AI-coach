import os
import tempfile
import unittest
from pathlib import Path

from src.vision.court_env_profile import (
    apply_court_env_overrides,
    parse_env_file,
    resolve_and_apply_court_env_overrides,
)


class CourtEnvProfileTests(unittest.TestCase):
    def test_parse_env_file_keeps_only_badc_keys(self):
        with tempfile.TemporaryDirectory() as td:
            env_path = Path(td) / "court.env"
            env_path.write_text(
                "\n".join(
                    [
                        "# comment",
                        "export BADC_ALPHA=1",
                        "BADC_BETA='on'",
                        "NOT_ALLOWED=abc",
                    ]
                ),
                encoding="utf-8",
            )
            out = parse_env_file(env_path)
        self.assertEqual(out.get("BADC_ALPHA"), "1")
        self.assertEqual(out.get("BADC_BETA"), "on")
        self.assertNotIn("NOT_ALLOWED", out)

    def test_resolve_and_apply_merges_profile_and_inline(self):
        old = dict(os.environ)
        try:
            with tempfile.TemporaryDirectory() as td:
                root = Path(td)
                env_path = root / "court.env"
                env_path.write_text("BADC_A=1\nBADC_B=2\n", encoding="utf-8")
                meta = resolve_and_apply_court_env_overrides(
                    {
                        "court_env_file": "court.env",
                        "court_env": {"BADC_B": 3, "BADC_C": "x", "BAD_KEY": "drop"},
                    },
                    base_dir=root,
                )

            self.assertEqual(meta.get("num_vars"), 3)
            self.assertEqual(os.environ.get("BADC_A"), "1")
            self.assertEqual(os.environ.get("BADC_B"), "3")
            self.assertEqual(os.environ.get("BADC_C"), "x")
            self.assertIsNone(os.environ.get("BAD_KEY"))
        finally:
            os.environ.clear()
            os.environ.update(old)

    def test_apply_filters_non_badc_keys(self):
        old = dict(os.environ)
        try:
            out = apply_court_env_overrides({"BADC_KEEP": "1", "OTHER_DROP": "2"})
            self.assertIn("BADC_KEEP", out)
            self.assertNotIn("OTHER_DROP", out)
            self.assertEqual(os.environ.get("BADC_KEEP"), "1")
            self.assertIsNone(os.environ.get("OTHER_DROP"))
        finally:
            os.environ.clear()
            os.environ.update(old)


if __name__ == "__main__":
    unittest.main()
