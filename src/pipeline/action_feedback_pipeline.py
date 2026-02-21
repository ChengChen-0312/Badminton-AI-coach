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
from src.pipeline.extract_strokes import (
    infer_segment_frame_ranges_from_analysis,
    summarise_strokes_from_analysis,
    stroke_summaries_to_dicts,
)
from src.pipeline.label_space import build_label_space_metadata
import yaml


def run_action_feedback(
    video_path: str,
    cfg_path: str = "src/config/v3_ai_score.yaml",
    mode: str = "student",
) -> Dict[str, Any]:
    cfg_path_obj = Path(cfg_path).expanduser()
    if not cfg_path_obj.is_absolute():
        cfg_path_obj = REPO_ROOT / cfg_path_obj
    with open(cfg_path_obj, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg["__config_dir__"] = str(cfg_path_obj.parent)

    analysis = analyse_video(video_path, config=cfg)
    classifier_labels = None
    classifier_outputs = None
    classifier_cfg = cfg.get("classifier", {}) if isinstance(cfg.get("classifier", {}), dict) else {}
    classifier_min_confidence = float(classifier_cfg.get("min_confidence", 0.0) or 0.0)
    expected_num_classes = classifier_cfg.get("expected_num_classes")
    try:
        expected_num_classes_i = int(expected_num_classes) if expected_num_classes is not None else None
    except Exception:
        expected_num_classes_i = None
    expected_class_names = classifier_cfg.get("class_names")
    label_space_meta = build_label_space_metadata(
        None,
        version_hint=classifier_cfg.get("label_space_version"),
        expected_num_classes=expected_num_classes_i,
        expected_class_names=expected_class_names,
    )
    classifier_ckpt = classifier_cfg.get("checkpoint")
    if classifier_ckpt:
        ckpt_path = Path(str(classifier_ckpt))
        if not ckpt_path.is_absolute():
            ckpt_path = REPO_ROOT / ckpt_path
        if ckpt_path.exists():
            try:
                from src.pipeline.stroke_classifier_runtime import StrokeClassifierRuntime

                runtime = StrokeClassifierRuntime.from_checkpoint(
                    checkpoint_path=ckpt_path,
                    device=str(classifier_cfg.get("device", "auto")),
                    frame_size=classifier_cfg.get("frame_size"),
                    num_frames=classifier_cfg.get("num_frames"),
                    topk=int(max(1, classifier_cfg.get("topk", 3))),
                )
                segs = infer_segment_frame_ranges_from_analysis(analysis)
                classifier_outputs = runtime.predict_labels_for_segments_from_video(video_path, segs)
                classifier_labels = [
                    str(item.get("label")) if isinstance(item, dict) and item.get("label") is not None else None
                    for item in classifier_outputs
                ]
                label_space_meta = build_label_space_metadata(
                    runtime.classes,
                    version_hint=classifier_cfg.get("label_space_version"),
                    expected_num_classes=expected_num_classes_i,
                    expected_class_names=expected_class_names,
                )
            except Exception:
                classifier_labels = None
                classifier_outputs = None

    summaries = summarise_strokes_from_analysis(
        analysis,
        classifier_labels=classifier_labels,
        classifier_outputs=classifier_outputs,
        classifier_min_confidence=float(max(0.0, classifier_min_confidence)),
        enable_hitter_inference=True,
        hitter_distance_max=cfg.get("spatial_logic", {}).get("hitter_distance_max", 200.0),
    )
    summaries_dict = stroke_summaries_to_dicts(summaries)

    runtime_mode = str(mode or "student").strip().lower()
    if runtime_mode in ("teacher", "teacher_mlx"):
        runtime_mode = "teacher_mlx"
    else:
        runtime_mode = "student_mlx"
    ai_cfg = cfg.get("ai_score", {}) if isinstance(cfg.get("ai_score", {}), dict) else {}
    feedback_engine = ActionFeedback(
        mode=runtime_mode,
        teacher_mlx_path=ai_cfg.get("teacher_mlx_path"),
        student_mlx_path=ai_cfg.get("student_mlx_path"),
        adapter_path=ai_cfg.get("adapter_path"),
    )

    outputs: List[Any] = []
    for s in summaries_dict:
        desc = (
            f"stroke: {s.get('final_type')}, hitter: {s.get('hitter_role')}, "
            f"landing_region: {s.get('landing_region')}, contact_region: {s.get('contact_region')}"
        )
        outputs.append(feedback_engine.score_motion(desc))

    return {
        "analysis": analysis,
        "summaries": summaries_dict,
        "feedback": outputs,
        "court_env": getattr(analysis, "court_env", None),
        "stroke_classifier": {
            "checkpoint": str(classifier_ckpt) if classifier_ckpt else None,
            "min_confidence": float(max(0.0, classifier_min_confidence)),
            "label_space": label_space_meta,
        },
        "label_space_version": label_space_meta.get("version"),
    }
