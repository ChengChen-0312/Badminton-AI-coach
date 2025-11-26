from __future__ import annotations

from typing import Any, Dict, List
import sys
from pathlib import Path

# Ensure repo root on path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ai_score.action_feedback import ActionFeedback
from src.pipeline.analyse_video import analyse_video
from src.pipeline.extract_strokes import summarise_strokes_from_analysis, stroke_summaries_to_dicts
import yaml


def run_action_feedback(
    video_path: str,
    cfg_path: str = "src/config/v3_ai_score.yaml",
    mode: str = "student",
) -> Dict[str, Any]:
    cfg = yaml.safe_load(open(cfg_path))
    analysis = analyse_video(video_path, config=cfg)
    summaries = summarise_strokes_from_analysis(
        analysis,
        classifier_labels=None,
        enable_hitter_inference=True,
        hitter_distance_max=cfg.get("spatial_logic", {}).get("hitter_distance_max", 200.0),
    )
    summaries_dict = stroke_summaries_to_dicts(summaries)

    feedback_engine = ActionFeedback(
        mode=mode,
        teacher_api=cfg.get("ai_score", {}).get("teacher_api"),
    )

    outputs: List[Any] = []
    for s in summaries_dict:
        desc = (
            f"stroke: {s.get('final_type')}, hitter: {s.get('hitter_role')}, "
            f"landing_region: {s.get('landing_region')}, contact_region: {s.get('contact_region')}"
        )
        outputs.append(feedback_engine.score_motion(desc))

    return {"analysis": analysis, "summaries": summaries_dict, "feedback": outputs}
