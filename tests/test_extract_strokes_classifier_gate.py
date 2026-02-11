import unittest

from src.pipeline.extract_strokes import summarise_strokes_from_analysis
from src.tracking.player_track import PlayerState


class _FakeBall:
    def __init__(self, cx: float, cy: float):
        self.cx = float(cx)
        self.cy = float(cy)
        self.predicted = False


class _FakeFrame:
    def __init__(self, frame_idx: int, cx: float, cy: float):
        self.frame_idx = int(frame_idx)
        self.ball = _FakeBall(cx, cy)
        self.player_states = [
            PlayerState(
                track_id=1,
                role="near",
                bboxes=[(40.0, 40.0, 120.0, 180.0)],
                frames=[int(frame_idx)],
            )
        ]


class ExtractStrokesClassifierGateTests(unittest.TestCase):
    def _build_analysis_stub(self):
        frames = []
        # Ball remains near player bbox so contact candidate exists.
        for idx in range(12):
            frames.append(_FakeFrame(idx, cx=80.0 + idx * 1.0, cy=90.0 + idx * 2.0))
        return {"frame_results": frames, "court_corners": None, "pose_results": None}

    def test_rejects_classifier_label_below_min_confidence(self):
        analysis = self._build_analysis_stub()
        summaries = summarise_strokes_from_analysis(
            analysis,
            classifier_outputs=[
                {
                    "label": "forehand_drive",
                    "confidence": 0.20,
                    "topk": [{"label": "forehand_drive", "confidence": 0.20}],
                }
            ],
            classifier_min_confidence=0.45,
            enable_hitter_inference=True,
        )

        self.assertGreaterEqual(len(summaries), 1)
        s0 = summaries[0]
        self.assertTrue(s0.classifier_rejected_low_conf)
        self.assertAlmostEqual(float(s0.classifier_confidence), 0.20, places=2)
        self.assertIsNone(s0.classifier_label)


if __name__ == "__main__":
    unittest.main()
