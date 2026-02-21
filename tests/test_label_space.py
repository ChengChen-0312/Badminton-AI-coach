import unittest

from src.pipeline.label_space import (
    DEFAULT_6CLASS_LABELS,
    DEFAULT_LABEL_SPACE_VERSION,
    build_label_space_metadata,
    infer_label_space_version,
)


class LabelSpaceTests(unittest.TestCase):
    def test_infer_default_version_for_builtin_6class(self):
        version = infer_label_space_version(DEFAULT_6CLASS_LABELS)
        self.assertEqual(version, DEFAULT_LABEL_SPACE_VERSION)

    def test_build_metadata_flags_mismatch(self):
        meta = build_label_space_metadata(
            ["a", "b", "c"],
            version_hint="stroke_6class_v1",
            expected_num_classes=6,
            expected_class_names=DEFAULT_6CLASS_LABELS,
        )
        self.assertTrue(meta["mismatch"])
        self.assertGreaterEqual(len(meta["mismatch_reasons"]), 1)

    def test_build_metadata_without_labels(self):
        meta = build_label_space_metadata(None, version_hint="stroke_6class_v1")
        self.assertEqual(meta["version"], "stroke_6class_v1")
        self.assertEqual(meta["num_classes"], 0)
        self.assertFalse(meta["mismatch"])


if __name__ == "__main__":
    unittest.main()
