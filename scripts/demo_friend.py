#!/usr/bin/env python3
"""
One-command demo for friends:
- Run the v3 realtime pipeline on a sample video
- Summarize strokes
- Generate a Markdown/JSON/CSV report + heatmap/timeline images
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import yaml

# Ensure repo root on path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if str(os.environ.get("BADMINTON_DEBUG_IMPORTS", "0")).strip() in {"1", "true", "TRUE"}:
    # Optional import-path debug for environment diagnostics.
    import importlib
    import inspect

    print("[PY] CWD:", os.getcwd())
    print("[PY] sys.path[0:5]:", sys.path[:5])
    try:
        _cfh = importlib.import_module("src.vision.court_fit_homography")
        print("[PY] court_fit_homography loaded from:", inspect.getfile(_cfh))
    except Exception as _e:
        print("[PY] failed to import src.vision.court_fit_homography:", _e)

DEFAULT_DEMO_VIDEO = REPO_ROOT / "archive" / "demo.mp4"

from src.pipeline.analyse_video import AnalyseResult, analyse_video
from src.pipeline.extract_strokes import (
    infer_segment_frame_ranges_from_analysis,
    summarise_strokes_from_analysis,
    stroke_summaries_to_dicts,
)
from src.pipeline.label_space import build_label_space_metadata
from src.pipeline.report_generator import generate_match_report


def _adapter_file_in_dir(adapter_dir: Path) -> Optional[Path]:
    for fname in ("adapters.safetensors", "adapters.npz"):
        p = adapter_dir / fname
        if p.exists():
            return p
    return None


def _resolve_existing_path(path_str: str) -> Path:
    p = Path(path_str)
    if p.exists():
        return p
    p2 = REPO_ROOT / path_str
    if p2.exists():
        return p2
    return p


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def _looks_like_mediapipe_installed() -> bool:
    try:
        import mediapipe  # noqa: F401

        return True
    except Exception:
        return False


def _summarize_for_console(summary: Dict[str, Any]) -> Dict[str, Any]:
    keys = [
        "final_type",
        "confidence",
        "event_type",
        "classifier_label",
        "hitter_role",
        "hitter_track_id",
        "hitter_distance",
        "landing_region",
        "contact_region",
        "contact_frame",
        "landing_frame",
        "landing_predicted",
        "pose_features",
    ]
    return {k: summary.get(k) for k in keys if k in summary}


def _llm_description_from_summary(summary: Dict[str, Any]) -> str:
    dist_raw = summary.get("hitter_distance")
    dist_val: Optional[float] = None
    dist_valid = False
    if isinstance(dist_raw, (int, float)):
        d = float(dist_raw)
        # In current pipeline, 0.0 appears frequently as missing/invalid signal.
        if np.isfinite(d) and d > 1.0:
            dist_val = d
            dist_valid = True

    landing_region = summary.get("landing_region")
    landing_predicted = bool(summary.get("landing_predicted"))
    landing_is_out = str(landing_region).lower() == "out"

    def _fmt(key: str) -> str:
        v = summary.get(key)
        return f"{key}: {v}"

    lines = [
        _fmt("final_type"),
        _fmt("event_type"),
        _fmt("hitter_role"),
        _fmt("hitter_track_id"),
        f"hitter_distance_px: {dist_val if dist_valid else 'unknown'}",
        f"hitter_distance_valid: {str(dist_valid).lower()}",
        _fmt("landing_region"),
        _fmt("contact_region"),
        _fmt("contact_frame"),
        _fmt("landing_frame"),
        f"landing_predicted: {str(landing_predicted).lower()}",
        f"landing_is_out: {str(landing_is_out).lower()}",
    ]
    pose_features = summary.get("pose_features")
    if pose_features:
        lines.append(f"pose_features: {json.dumps(pose_features, ensure_ascii=False)}")
    return "\n".join(lines)


def _to_text(output: Any) -> str:
    if output is None:
        return ""
    if isinstance(output, str):
        return output
    if hasattr(output, "text"):
        try:
            return str(getattr(output, "text"))
        except Exception:
            pass
    return str(output)


def _extract_json_dict(text: str) -> Tuple[Optional[Dict[str, Any]], str]:
    t = (text or "").strip()
    if not t:
        return None, ""
    try:
        obj = json.loads(t)
        return obj if isinstance(obj, dict) else None, t
    except Exception:
        pass
    # best-effort: find first {...} block
    start = t.find("{")
    end = t.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = t[start : end + 1]
        try:
            obj = json.loads(candidate)
            return obj if isinstance(obj, dict) else None, t
        except Exception:
            return None, t
    return None, t


def _to_score_value(v: Any) -> Optional[float]:
    try:
        x = float(v)
    except Exception:
        return None
    if not np.isfinite(x):
        return None
    return float(max(0.0, min(100.0, x)))


def _extract_declared_score(parsed: Optional[Dict[str, Any]]) -> Optional[float]:
    if not isinstance(parsed, dict):
        return None
    for k in ("score", "final_score", "overall_score"):
        if k in parsed:
            sv = _to_score_value(parsed.get(k))
            if sv is not None:
                return sv
    return None


def _extract_dim_scores(parsed: Optional[Dict[str, Any]]) -> Dict[str, float]:
    if not isinstance(parsed, dict):
        return {}

    score_keys = ("technique", "footwork", "timing", "decision", "outcome")
    node = parsed.get("scores")
    src: Dict[str, Any] = node if isinstance(node, dict) else parsed

    out: Dict[str, float] = {}
    for k in score_keys:
        sv = _to_score_value(src.get(k))
        if sv is not None:
            out[k] = sv
    return out


def _compute_rubric_score(parsed: Optional[Dict[str, Any]]) -> Optional[float]:
    dim_scores = _extract_dim_scores(parsed)
    if len(dim_scores) < 3:
        return None

    weights = {
        "technique": 0.30,
        "footwork": 0.20,
        "timing": 0.20,
        "decision": 0.15,
        "outcome": 0.15,
    }
    used = [k for k in weights.keys() if k in dim_scores]
    if not used:
        return None
    w_sum = float(sum(weights[k] for k in used))
    if w_sum <= 0.0:
        return None
    score = sum(dim_scores[k] * weights[k] for k in used) / w_sum
    return float(max(0.0, min(100.0, round(score, 1))))


def _read_video_frame_bgr(video_path: Path, frame_idx: int = 0) -> Optional[Any]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame_bgr = cap.read()
    cap.release()
    if not ok or frame_bgr is None:
        return None
    return frame_bgr


def _looks_like_default_full_frame_corners(
    corners: list[list[float]], width: int, height: int, tol_px: float = 2.0
) -> bool:
    if len(corners) != 4:
        return False
    target = [
        [0.0, float(height - 1)],
        [float(width - 1), float(height - 1)],
        [float(width - 1), 0.0],
        [0.0, 0.0],
    ]
    for p, t in zip(corners, target):
        if len(p) != 2:
            return False
        if abs(float(p[0]) - float(t[0])) > tol_px:
            return False
        if abs(float(p[1]) - float(t[1])) > tol_px:
            return False
    return True


def _compute_court_roi(
    corners: list[list[float]],
    width: int,
    height: int,
    margin: float,
) -> Optional[tuple[int, int, int, int]]:
    if len(corners) != 4:
        return None
    try:
        xs = [float(p[0]) for p in corners]
        ys = [float(p[1]) for p in corners]
    except Exception:
        return None
    pad_x = float(width) * float(margin)
    pad_y = float(height) * float(margin)
    x1 = max(0, int(round(min(xs) - pad_x)))
    y1 = max(0, int(round(min(ys) - pad_y)))
    x2 = min(width, int(round(max(xs) + pad_x)))
    y2 = min(height, int(round(max(ys) + pad_y)))
    if x2 > x1 and y2 > y1:
        return x1, y1, x2, y2
    return None


def _draw_court_debug(
    frame_bgr: Any,
    corners: list[list[float]],
    corner_source: str,
    roi: Optional[tuple[int, int, int, int]] = None,
    roi_margin: Optional[float] = None,
) -> Any:
    out = frame_bgr.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX

    header = f"Court corners: {corner_source} (order: LB, RB, RT, LT)"
    cv2.putText(out, header, (20, 40), font, 0.9, (0, 0, 0), 5, lineType=cv2.LINE_AA)
    cv2.putText(out, header, (20, 40), font, 0.9, (255, 255, 255), 2, lineType=cv2.LINE_AA)

    if len(corners) == 4:
        pts = [(int(round(float(x))), int(round(float(y)))) for x, y in corners]
        labels = ["LB", "RB", "RT", "LT"]

        cv2.polylines(out, [np.array(pts, dtype=np.int32).reshape(-1, 1, 2)], True, (0, 255, 0), 3)
        for (x, y), label in zip(pts, labels):
            cv2.circle(out, (x, y), 7, (0, 0, 255), -1)
            cv2.putText(out, label, (x + 10, y - 10), font, 0.85, (0, 0, 0), 4, lineType=cv2.LINE_AA)
            cv2.putText(out, label, (x + 10, y - 10), font, 0.85, (0, 0, 255), 2, lineType=cv2.LINE_AA)

    if roi is not None:
        x1, y1, x2, y2 = roi
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 0, 0), 2)
        margin_txt = f"ROI (margin={roi_margin:.3f})" if roi_margin is not None else "ROI"
        cv2.putText(
            out,
            margin_txt,
            (x1 + 5, max(20, y1 - 10)),
            font,
            0.75,
            (0, 0, 0),
            4,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            out,
            margin_txt,
            (x1 + 5, max(20, y1 - 10)),
            font,
            0.75,
            (255, 0, 0),
            2,
            lineType=cv2.LINE_AA,
        )
    return out


def _llm_compare(
    summaries: list[Dict[str, Any]],
    variants: list[str],
    teacher_model_path: Path,
    student_model_path: Path,
    adapter_early: Path,
    adapter_highcap: Path,
    adapter_pose: Path,
    max_strokes: int,
) -> Dict[str, Any]:
    try:
        from src.ai_score.action_feedback import ActionFeedback
        from src.ai_score.scoring_engine import normalize_llm_score_output
    except Exception as exc:
        raise RuntimeError(
            "LLM dependencies not available. Install MLX stack (mlx-vlm/mlx) to enable --llm.\n"
            f"Import error: {exc}"
        )

    alias = {
        "teacher": "teacher_30b",
        "teacher_30b": "teacher_30b",
        "student": "student_4b_base",
        "student_base": "student_4b_base",
        "base": "student_4b_base",
        "student_4b_base": "student_4b_base",
        "early": "student_4b_lora_early",
        "student_lora_early": "student_4b_lora_early",
        "student_4b_lora_early": "student_4b_lora_early",
        "highcap": "student_4b_lora_highcap",
        "student_lora_highcap": "student_4b_lora_highcap",
        "student_4b_lora_highcap": "student_4b_lora_highcap",
        "pose": "student_4b_lora_pose",
        "student_lora_pose": "student_4b_lora_pose",
        "student_4b_lora_pose": "student_4b_lora_pose",
    }
    normalized = [alias.get(v.strip().lower(), v.strip()) for v in variants]
    if any(v.lower() == "all" for v in normalized):
        normalized = [
            "teacher_30b",
            "student_4b_base",
            "student_4b_lora_early",
            "student_4b_lora_highcap",
            "student_4b_lora_pose",
        ]

    max_strokes_i = int(max_strokes)
    # 0 or negative means: score all strokes.
    if max_strokes_i <= 0:
        strokes = list(summaries)
    else:
        strokes = summaries[:max_strokes_i]
    llm_inputs = [{"stroke_index": i, "summary": s, "description": _llm_description_from_summary(s)} for i, s in enumerate(strokes)]

    def _variant_engine(v: str) -> Tuple[str, Optional[ActionFeedback]]:
        if v == "teacher_30b":
            if not teacher_model_path.exists():
                print(f"[WARN] teacher model not found: {teacher_model_path} -> skipping {v}")
                return v, None
            return v, ActionFeedback(mode="teacher_mlx", teacher_mlx_path=str(teacher_model_path))
        if v == "student_4b_base":
            if not student_model_path.exists():
                print(f"[WARN] student model not found: {student_model_path} -> skipping {v}")
                return v, None
            return v, ActionFeedback(mode="student_mlx", student_mlx_path=str(student_model_path))
        if v == "student_4b_lora_early":
            if not student_model_path.exists():
                print(f"[WARN] student model not found: {student_model_path} -> skipping {v}")
                return v, None
            if not _adapter_file_in_dir(adapter_early):
                print(f"[WARN] adapter not found in: {adapter_early} -> skipping {v}")
                return v, None
            return v, ActionFeedback(
                mode="student_mlx",
                student_mlx_path=str(student_model_path),
                adapter_path=str(adapter_early),
            )
        if v == "student_4b_lora_highcap":
            if not student_model_path.exists():
                print(f"[WARN] student model not found: {student_model_path} -> skipping {v}")
                return v, None
            if not _adapter_file_in_dir(adapter_highcap):
                print(f"[WARN] adapter not found in: {adapter_highcap} -> skipping {v}")
                return v, None
            return v, ActionFeedback(
                mode="student_mlx",
                student_mlx_path=str(student_model_path),
                adapter_path=str(adapter_highcap),
            )
        if v == "student_4b_lora_pose":
            if not student_model_path.exists():
                print(f"[WARN] student model not found: {student_model_path} -> skipping {v}")
                return v, None
            if not _adapter_file_in_dir(adapter_pose):
                print(f"[WARN] adapter not found in: {adapter_pose} -> skipping {v}")
                return v, None
            return v, ActionFeedback(
                mode="student_mlx",
                student_mlx_path=str(student_model_path),
                adapter_path=str(adapter_pose),
            )
        print(f"[WARN] unknown llm variant: {v} -> skipping")
        return v, None

    results: Dict[str, Any] = {
        "variants": normalized,
        "strokes": [],
    }

    for inp in llm_inputs:
        results["strokes"].append(
            {
                "stroke_index": inp["stroke_index"],
                "description": inp["description"],
                "outputs": {},
            }
        )

    for v in normalized:
        print(f"\n[LLM] Running: {v}")
        variant_name, engine = _variant_engine(v)
        if engine is None:
            continue
        try:
            for i, inp in enumerate(llm_inputs):
                raw_out = engine.score_motion(inp["description"])
                text_out = _to_text(raw_out)
                normalized = normalize_llm_score_output(text_out, inp["summary"])
                score = normalized.get("score")
                results["strokes"][i]["outputs"][variant_name] = normalized
                if score is not None:
                    print(f"- stroke[{inp['stroke_index']}] score={score}")
        finally:
            # Best-effort cleanup (MLX models can be large)
            try:
                del engine
            except Exception:
                pass

    return results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Badminton AI Coach: one-command demo (report + visualizations).")
    p.add_argument(
        "--video",
        type=str,
        default=None,
        help='Video path. Default: "archive/demo.mp4" (if exists), otherwise first *.mp4 under archive/.',
    )
    p.add_argument("--config", type=str, default="src/config/v3_realtime.yaml", help="YAML config path.")
    p.add_argument("--match-name", type=str, default="demo_friend", help="Report file prefix.")
    p.add_argument("--out-dir", type=str, default="reports/demo_friend", help="Output directory.")
    p.add_argument(
        "--court-detect-once",
        choices=["auto", "on", "off"],
        default="auto",
        help='Override single-shot court detection mode. "auto" uses config/default behavior.',
    )
    p.add_argument(
        "--court-detect-frame-idx",
        type=int,
        default=None,
        help="Frame index used for single-shot court detection lock.",
    )
    p.add_argument(
        "--stroke-classifier",
        choices=["auto", "on", "off"],
        default="auto",
        help='Stroke classifier mode: "on" force enable, "off" disable, "auto" enable when checkpoint exists.',
    )
    p.add_argument(
        "--stroke-classifier-checkpoint",
        type=str,
        default=os.environ.get("BADMINTON_STROKE_CKPT", "runs/v2_highcap_stable/best_model.pt"),
        help="Path to stroke classifier checkpoint (best_model.pt).",
    )
    p.add_argument(
        "--stroke-classifier-device",
        type=str,
        default=os.environ.get("BADMINTON_STROKE_DEVICE", "auto"),
        help='Stroke classifier device: "auto" | "mps" | "cuda" | "cpu".',
    )
    p.add_argument(
        "--stroke-classifier-topk",
        type=int,
        default=3,
        help="Top-K class predictions stored into each stroke summary.",
    )
    p.add_argument(
        "--stroke-classifier-frame-size",
        type=int,
        default=None,
        help="Optional override for classifier input size.",
    )
    p.add_argument(
        "--stroke-classifier-num-frames",
        type=int,
        default=None,
        help="Optional override for classifier clip length.",
    )
    p.add_argument(
        "--stroke-classifier-min-confidence",
        type=float,
        default=None,
        help="If set, reject classifier labels below this confidence and fallback to spatial logic.",
    )

    p.add_argument("--no-heatmap", action="store_true", help="Disable heatmap image.")
    p.add_argument("--no-timeline", action="store_true", help="Disable timeline image.")
    p.add_argument("--no-court-debug", action="store_true", help="Disable saving a court detection debug image.")
    p.add_argument(
        "--court-debug-frame",
        type=int,
        default=None,
        help="Frame index used for court debug image (default: use the detection frame if available).",
    )

    p.add_argument(
        "--pose",
        choices=["auto", "on", "off"],
        default="auto",
        help='Pose mode: "auto" keeps config but disables pose if mediapipe missing; "on"/"off" override.',
    )
    p.add_argument("--open", action="store_true", help="Open report/figures after generation (macOS: open).")

    # LLM comparison (optional)
    p.add_argument(
        "--llm",
        dest="llm",
        action="store_true",
        default=True,
        help="Run LLM coach feedback comparison (default: on).",
    )
    p.add_argument(
        "--no-llm",
        dest="llm",
        action="store_false",
        help="Disable LLM coach feedback comparison.",
    )
    p.add_argument(
        "--llm-variants",
        nargs="+",
        default=None,
        help=(
            "Variants to run. Default: teacher_30b student_4b_base student_4b_lora_pose. "
            'Use "all" for all built-ins. '
            "Built-ins: teacher_30b, student_4b_base, student_4b_lora_early, student_4b_lora_highcap, student_4b_lora_pose."
        ),
    )
    p.add_argument(
        "--llm-max-strokes",
        type=int,
        default=0,
        help="Limit number of strokes sent to LLM (speed). Use 0 to score all strokes.",
    )
    p.add_argument(
        "--teacher-model-path",
        type=str,
        default=os.environ.get("BADMINTON_COACH_TEACHER_MODEL", "/Users/chencheng/llm/qwen3-30b"),
        help="Local path to teacher model (optional). Can also set env BADMINTON_COACH_TEACHER_MODEL.",
    )
    p.add_argument(
        "--student-model-path",
        type=str,
        default=os.environ.get("BADMINTON_COACH_STUDENT_MODEL", "/Users/chencheng/llm/qwen3-4b"),
        help="Local path to student model (optional). Can also set env BADMINTON_COACH_STUDENT_MODEL.",
    )
    p.add_argument("--adapter-early", type=str, default="outputs/lora_adapters")
    p.add_argument("--adapter-highcap", type=str, default="outputs/lora_adapters_highcap")
    p.add_argument("--adapter-pose", type=str, default="outputs/lora_adapters_pose")
    return p.parse_args()


def _maybe_open(path: Optional[str]) -> None:
    if not path:
        return
    try:
        import subprocess

        subprocess.run(["open", path], check=False)
    except Exception:
        return


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir

    if args.video:
        video_path = _resolve_existing_path(args.video)
    else:
        video_path = DEFAULT_DEMO_VIDEO
        if not video_path.exists():
            archive_dir = REPO_ROOT / "archive"
            candidates = sorted(archive_dir.glob("**/*.mp4")) if archive_dir.exists() else []
            demo_like = [p for p in candidates if p.name.lower().startswith("demo")]
            if demo_like:
                video_path = demo_like[0]
            elif candidates:
                video_path = candidates[0]

    if not video_path.exists():
        raise SystemExit(
            "[ERROR] demo video not found.\n"
            f"- Checked: {DEFAULT_DEMO_VIDEO}\n"
            "Provide a video path, e.g.:\n"
            "  python scripts/demo_friend.py --video path/to/video.mp4"
        )

    cfg_path = _resolve_existing_path(args.config)
    if not cfg_path.exists():
        raise SystemExit(f"[ERROR] config not found: {args.config}")
    cfg = _load_yaml(str(cfg_path))
    cfg["__config_dir__"] = str(cfg_path.parent)

    pose_cfg = cfg.get("pose", {}) if isinstance(cfg.get("pose", {}), dict) else {}
    if args.pose == "off":
        pose_cfg["enable"] = False
    elif args.pose == "on":
        pose_cfg["enable"] = True
    else:
        if pose_cfg.get("enable") and not _looks_like_mediapipe_installed():
            pose_cfg["enable"] = False
            print('[INFO] mediapipe not found -> pose disabled (use "--pose on" after installing mediapipe).')
    cfg["pose"] = pose_cfg

    vision_cfg = cfg.get("vision", {}) if isinstance(cfg.get("vision", {}), dict) else {}
    if args.court_detect_once == "on":
        vision_cfg["court_detect_once"] = True
    elif args.court_detect_once == "off":
        vision_cfg["court_detect_once"] = False
    if args.court_detect_frame_idx is not None:
        vision_cfg["court_detect_frame_idx"] = int(max(0, args.court_detect_frame_idx))
    # Keep all court debug artifacts in this run's output folder to avoid cross-run overwrite.
    vision_cfg["model_fit_debug_dir"] = str(out_dir)
    vision_cfg.pop("model_fit_debug_path", None)
    cfg["vision"] = vision_cfg
    classifier_cfg = cfg.get("classifier", {}) if isinstance(cfg.get("classifier", {}), dict) else {}
    if args.stroke_classifier_min_confidence is not None:
        classifier_min_confidence = float(max(0.0, args.stroke_classifier_min_confidence))
    else:
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
    player_model = vision_cfg.get("yolo_model", "yolov8n.pt")
    ball_model = vision_cfg.get("ball_model", "yolov8n.pt")
    player_model_path = _resolve_existing_path(str(player_model))
    ball_model_path = _resolve_existing_path(str(ball_model))

    print("[DEMO] Badminton AI Coach")
    print(f"- Video:  {video_path}")
    print(f"- Config: {cfg_path}")
    if not player_model_path.exists():
        print(f"[WARN] player model not found locally: {player_model_path} (ultralytics may try to download)")
    if not ball_model_path.exists():
        print(f"[WARN] ball model not found locally: {ball_model_path} (ultralytics may try to download)")

    t0 = time.time()
    analysis: AnalyseResult = analyse_video(
        str(video_path),
        config=cfg,
        player_model_path=str(player_model_path),
        ball_model_path=str(ball_model_path),
    )
    t1 = time.time()

    spatial_cfg = cfg.get("spatial_logic", {}) if isinstance(cfg.get("spatial_logic", {}), dict) else {}
    classifier_labels = None
    classifier_outputs = None
    classifier_enabled = False
    classifier_ckpt = _resolve_existing_path(args.stroke_classifier_checkpoint) if args.stroke_classifier_checkpoint else None
    want_classifier = args.stroke_classifier == "on" or (
        args.stroke_classifier == "auto" and classifier_ckpt is not None and classifier_ckpt.exists()
    )
    if args.stroke_classifier == "on" and (classifier_ckpt is None or not classifier_ckpt.exists()):
        print(f"[WARN] stroke classifier forced on but checkpoint not found: {classifier_ckpt}")
    if want_classifier and classifier_ckpt is not None and classifier_ckpt.exists():
        try:
            from src.pipeline.stroke_classifier_runtime import StrokeClassifierRuntime

            classifier_runtime = StrokeClassifierRuntime.from_checkpoint(
                checkpoint_path=classifier_ckpt,
                device=args.stroke_classifier_device,
                frame_size=args.stroke_classifier_frame_size,
                num_frames=args.stroke_classifier_num_frames,
                topk=max(1, int(args.stroke_classifier_topk)),
            )
            segment_ranges = infer_segment_frame_ranges_from_analysis(analysis)
            classifier_outputs = classifier_runtime.predict_labels_for_segments_from_video(video_path, segment_ranges)
            classifier_labels = [
                str(item.get("label")) if isinstance(item, dict) and item.get("label") is not None else None
                for item in classifier_outputs
            ]
            label_space_meta = build_label_space_metadata(
                classifier_runtime.classes,
                version_hint=classifier_cfg.get("label_space_version"),
                expected_num_classes=expected_num_classes_i,
                expected_class_names=expected_class_names,
            )
            if label_space_meta.get("mismatch"):
                print(
                    "[WARN] classifier label-space mismatch:",
                    ",".join(label_space_meta.get("mismatch_reasons", [])),
                )
            classifier_enabled = True
            print(
                f"[CLS] enabled checkpoint={classifier_ckpt} segments={len(segment_ranges)} "
                f"predictions={sum(1 for x in classifier_labels if x)}"
            )
        except Exception as exc:
            print(f"[WARN] stroke classifier runtime failed, fallback to spatial-only labels: {exc}")
            classifier_labels = None
            classifier_outputs = None

    summaries = summarise_strokes_from_analysis(
        analysis,
        classifier_labels=classifier_labels,
        classifier_outputs=classifier_outputs,
        classifier_min_confidence=float(max(0.0, classifier_min_confidence)),
        enable_hitter_inference=spatial_cfg.get("enable_hitter_inference", True),
        hitter_distance_max=float(spatial_cfg.get("hitter_distance_max", 200.0)),
        pose_window=int(pose_cfg.get("window", 3)) if isinstance(pose_cfg.get("window", 3), int) else 3,
    )
    summary_dicts = stroke_summaries_to_dicts(summaries)

    viz_cfg = cfg.get("visualization", {}) if isinstance(cfg.get("visualization", {}), dict) else {}
    out = generate_match_report(
        args.match_name,
        summary_dicts,
        str(out_dir),
        enable_heatmap=bool(viz_cfg.get("enable_heatmap", True)) and (not args.no_heatmap),
        enable_timeline=bool(viz_cfg.get("enable_timeline", True)) and (not args.no_timeline),
        heatmap_bins=int(viz_cfg.get("heatmap_bins", 32)),
    )
    report_json = out.get("json")
    if report_json:
        try:
            report_path = Path(report_json)
            if not report_path.is_absolute():
                report_path = out_dir / report_path
            report_data: Any = {}
            if report_path.exists():
                report_data = json.loads(report_path.read_text(encoding="utf-8"))
            if isinstance(report_data, list):
                report_data = {"strokes": report_data}
            det = getattr(analysis, "court_detection", None)
            corners = None
            if isinstance(getattr(analysis, "court_corners", None), list) and len(analysis.court_corners) == 4:
                try:
                    corners = [[float(x), float(y)] for x, y in analysis.court_corners]
                except Exception:
                    corners = analysis.court_corners
            source = None
            confidence = None
            reason = None
            last_metrics = None
            frame_idx = None
            detect_mode = None
            lock_enabled = None
            sample_frame_indices = None
            sample_frame_count = None
            fallback_used = None
            if isinstance(det, dict):
                source = det.get("source")
                confidence = det.get("confidence")
                reason = det.get("reason")
                last_metrics = det.get("metrics")
                frame_idx = det.get("frame_idx")
                detect_mode = det.get("detect_mode")
                lock_enabled = det.get("lock_enabled")
                sample_frame_indices = det.get("sample_frame_indices")
                sample_frame_count = det.get("sample_frame_count")
                fallback_used = det.get("fallback_used")
            if source not in ("manual", "auto", "auto_failed"):
                source = "auto" if corners is not None else "auto_failed"
            report_data["court_detection"] = {
                "source": source,
                "corners": corners,
                "confidence": confidence,
                "reason": reason,
                "frame_idx": frame_idx,
                "detect_mode": detect_mode,
                "lock_enabled": lock_enabled,
                "sample_frame_indices": sample_frame_indices,
                "sample_frame_count": sample_frame_count,
                "fallback_used": fallback_used,
                "last_metrics": last_metrics,
            }
            report_data["stroke_classifier"] = {
                "enabled": bool(classifier_enabled),
                "checkpoint": str(classifier_ckpt) if classifier_ckpt is not None else None,
                "mode": args.stroke_classifier,
                "device": args.stroke_classifier_device,
                "topk": int(max(1, args.stroke_classifier_topk)),
                "min_confidence": float(max(0.0, classifier_min_confidence)),
                "label_space": label_space_meta,
            }
            report_data["label_space_version"] = label_space_meta.get("version")
            if isinstance(getattr(analysis, "court_env", None), dict):
                report_data["court_env"] = analysis.court_env
            report_path.write_text(json.dumps(report_data, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception as exc:
            print(f"[WARN] Failed to write court_detection into report JSON: {exc}")

    court_debug_path: Optional[Path] = None
    if not args.no_court_debug:
        try:
            det = getattr(analysis, "court_detection", None)
            det_frame_idx = det.get("frame_idx") if isinstance(det, dict) else None
            debug_frame_idx = (
                int(args.court_debug_frame)
                if args.court_debug_frame is not None
                else int(det_frame_idx) if det_frame_idx is not None else 0
            )
            frame_bgr = _read_video_frame_bgr(video_path, frame_idx=debug_frame_idx)
            if frame_bgr is None:
                print("[WARN] Court debug image skipped (failed to read frame).")
            else:
                h0, w0 = frame_bgr.shape[:2]
                corners = (
                    analysis.court_corners
                    if isinstance(getattr(analysis, "court_corners", None), list) and len(analysis.court_corners) == 4
                    else []
                )
                if isinstance(det, dict):
                    src = det.get("source", "unknown")
                    conf = det.get("confidence", None)
                    reason = det.get("reason", None)
                    edge_support = det.get("edge_support", None)
                    extra = []
                    if conf is not None:
                        try:
                            extra.append(f"conf={float(conf):.2f}")
                        except Exception:
                            extra.append(f"conf={conf}")
                    if reason:
                        extra.append(f"reason={reason}")
                    if isinstance(edge_support, (list, tuple)) and len(edge_support) == 4:
                        try:
                            extra.append(f"top_edge={float(edge_support[2]):.2f}")
                        except Exception:
                            pass
                    corner_source = f"{src} ({', '.join(extra)})" if extra else str(src)
                else:
                    corner_source = "unknown"

                use_court_roi = bool(vision_cfg.get("use_court_roi", False))
                court_margin = float(vision_cfg.get("court_roi_margin", 0.0))
                roi = _compute_court_roi(corners, w0, h0, margin=court_margin) if (use_court_roi and len(corners) == 4) else None
                debug_img = _draw_court_debug(
                    frame_bgr,
                    corners=corners,
                    corner_source=corner_source,
                    roi=roi,
                    roi_margin=court_margin if use_court_roi else None,
                )
                court_debug_path = out_dir / f"{args.match_name}_court_detect_debug.jpg"
                cv2.imwrite(str(court_debug_path), debug_img)
        except Exception as exc:
            print(f"[WARN] Court debug image failed: {exc}")

    frames = len(analysis.frame_results)
    fps = frames / max(t1 - t0, 1e-6)

    print(f"- Processed: {frames} frames in {t1 - t0:.2f}s -> FPS={fps:.2f}")
    if isinstance(getattr(analysis, "court_env", None), dict):
        ce = analysis.court_env
        print(
            f"- Court Env: profile={ce.get('profile_path')} num_vars={ce.get('num_vars')}"
        )
    if summary_dicts:
        print(f"- Stroke Summaries: {len(summary_dicts)}")
        print("- First Stroke Summary:")
        print(json.dumps(_summarize_for_console(summary_dicts[0]), indent=2, ensure_ascii=False))
    else:
        print("- Stroke Summary: (none)")

    print("- Report Outputs:")
    print(f"  - Markdown: {out.get('markdown')}")
    print(f"  - JSON:     {out.get('json')}")
    print(f"  - CSV:      {out.get('csv')}")
    if out.get("heatmap"):
        print(f"  - Heatmap:  {out.get('heatmap')}")
    if out.get("hitter_heatmap"):
        print(f"  - Hitter:   {out.get('hitter_heatmap')}")
    if out.get("timeline"):
        print(f"  - Timeline: {out.get('timeline')}")
    if court_debug_path is not None:
        print(f"  - Court:    {court_debug_path}")

    if args.llm:
        if not summary_dicts:
            print("[INFO] LLM compare skipped (no stroke summaries).")
        else:
            teacher_model_path = (
                _resolve_existing_path(args.teacher_model_path) if args.teacher_model_path else None
            )
            student_model_path = (
                _resolve_existing_path(args.student_model_path) if args.student_model_path else None
            )
            if not (
                (teacher_model_path is not None and teacher_model_path.exists())
                or (student_model_path is not None and student_model_path.exists())
            ):
                print("[INFO] LLM compare skipped (teacher/student model paths not provided or not found).")
                teacher_model_path = REPO_ROOT / "__missing_teacher_model__"
                student_model_path = REPO_ROOT / "__missing_student_model__"
            variants = (
                args.llm_variants
                if args.llm_variants is not None
                else ["teacher_30b", "student_4b_base", "student_4b_lora_pose"]
            )
            try:
                llm_results = _llm_compare(
                    summaries=summary_dicts,
                    variants=variants,
                    teacher_model_path=teacher_model_path,
                    student_model_path=student_model_path,
                    adapter_early=_resolve_existing_path(args.adapter_early),
                    adapter_highcap=_resolve_existing_path(args.adapter_highcap),
                    adapter_pose=_resolve_existing_path(args.adapter_pose),
                    max_strokes=args.llm_max_strokes,
                )
                llm_json = out_dir / f"{args.match_name}_llm_compare.json"
                llm_json.write_text(json.dumps(llm_results, indent=2, ensure_ascii=False), encoding="utf-8")
                print(f"  - LLM Compare (json): {llm_json}")
            except Exception as exc:
                print(f"[WARN] LLM compare failed: {exc}")

    if args.open:
        _maybe_open(out.get("markdown"))
        _maybe_open(out.get("heatmap"))
        _maybe_open(out.get("hitter_heatmap"))
        _maybe_open(out.get("timeline"))
        if court_debug_path is not None:
            _maybe_open(str(court_debug_path))

    print("\nTip: open the Markdown in VS Code preview, or share the PNGs with your friends.")


if __name__ == "__main__":
    main()
