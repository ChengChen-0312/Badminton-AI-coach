import unittest

from src.ai_score.scoring_engine import normalize_llm_score_output


class ScoringEngineTests(unittest.TestCase):
    def test_prefers_rubric_score_when_dimensions_present(self):
        text = (
            '{"scores":{"technique":80,"footwork":70,"timing":90,'
            '"decision":60,"outcome":75},"mistake":"late contact","advice":"start split step earlier"}'
        )
        summary = {"final_type": "forehand_drive", "landing_region": "mid"}
        out = normalize_llm_score_output(text, summary)

        self.assertEqual(out["score_source"], "rubric")
        self.assertAlmostEqual(out["score"], 76.2, places=1)
        self.assertIn("mistake", out)
        self.assertIn("advice", out)

    def test_uses_declared_score_when_rubric_is_missing(self):
        text = '{"score": 88, "mistake":"none", "advice":"keep consistency"}'
        out = normalize_llm_score_output(text, {"final_type": "forehand_clear"})

        self.assertEqual(out["score_source"], "declared")
        self.assertEqual(out["score"], 88.0)

    def test_falls_back_when_json_is_invalid(self):
        text = "not-json-output"
        out = normalize_llm_score_output(text, {"final_type": "unknown", "landing_region": "out"})

        self.assertEqual(out["score_source"], "fallback")
        self.assertTrue(0.0 <= out["score"] <= 100.0)
        self.assertEqual(set(out["scores"].keys()), {"technique", "footwork", "timing", "decision", "outcome"})


if __name__ == "__main__":
    unittest.main()
