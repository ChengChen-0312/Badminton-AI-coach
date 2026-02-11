import unittest
from pathlib import Path

from src.server.analysis_api import _resolve_video_path, _simple_summary


class AnalysisAPIUtilsTests(unittest.TestCase):
    def test_resolve_video_path_prefers_video_path_field(self):
        repo_root = Path("/tmp/repo")
        payload = {"video_path": "archive/demo.mp4"}
        out = _resolve_video_path(payload, repo_root)
        self.assertEqual(out, repo_root / "archive/demo.mp4")

    def test_resolve_video_path_accepts_file_url(self):
        repo_root = Path("/tmp/repo")
        payload = {"video_url": "file:///tmp/demo.mp4"}
        out = _resolve_video_path(payload, repo_root)
        self.assertEqual(out, Path("/tmp/demo.mp4"))

    def test_resolve_video_path_accepts_plain_local_video_url(self):
        repo_root = Path("/tmp/repo")
        payload = {"video_url": "archive/demo.mp4"}
        out = _resolve_video_path(payload, repo_root)
        self.assertEqual(out, repo_root / "archive/demo.mp4")

    def test_resolve_video_path_rejects_http_url(self):
        repo_root = Path("/tmp/repo")
        payload = {"video_url": "https://example.com/demo.mp4"}
        out = _resolve_video_path(payload, repo_root)
        self.assertIsNone(out)

    def test_simple_summary_from_confidence(self):
        strokes = [{"confidence": 0.7}, {"confidence": 0.9}]
        out = _simple_summary(strokes)
        self.assertEqual(out["overall_score"], 80.0)
        self.assertAlmostEqual(out["confidence"], 0.8, places=3)


if __name__ == "__main__":
    unittest.main()
