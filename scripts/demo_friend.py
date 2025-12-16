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

DEFAULT_DEMO_VIDEO = REPO_ROOT / "archive" / "demo.mp4"

from src.pipeline.analyse_video import AnalyseResult, analyse_video
from src.pipeline.extract_strokes import summarise_strokes_from_analysis, stroke_summaries_to_dicts
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
    def _fmt(key: str) -> str:
        v = summary.get(key)
        return f"{key}: {v}"

    lines = [
        _fmt("final_type"),
        _fmt("event_type"),
        _fmt("hitter_role"),
        _fmt("hitter_track_id"),
        _fmt("hitter_distance"),
        _fmt("landing_region"),
        _fmt("contact_region"),
        _fmt("contact_frame"),
        _fmt("landing_frame"),
        _fmt("landing_predicted"),
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

    strokes = summaries[: max(0, int(max_strokes))]
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
                parsed, raw_text = _extract_json_dict(text_out)
                score = None
                if isinstance(parsed, dict) and "score" in parsed:
                    try:
                        score = float(parsed["score"])
                    except Exception:
                        score = parsed.get("score")
                results["strokes"][i]["outputs"][variant_name] = {
                    "raw": raw_text,
                    "parsed": parsed,
                    "score": score,
                }
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

    p.add_argument("--no-heatmap", action="store_true", help="Disable heatmap image.")
    p.add_argument("--no-timeline", action="store_true", help="Disable timeline image.")
    p.add_argument("--no-court-debug", action="store_true", help="Disable saving a court detection debug image.")
    p.add_argument("--court-debug-frame", type=int, default=0, help="Frame index used for court debug image.")

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
    p.add_argument("--llm-max-strokes", type=int, default=1, help="Limit number of strokes sent to LLM (speed).")
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
    summaries = summarise_strokes_from_analysis(
        analysis,
        classifier_labels=None,
        enable_hitter_inference=spatial_cfg.get("enable_hitter_inference", True),
        hitter_distance_max=float(spatial_cfg.get("hitter_distance_max", 200.0)),
        pose_window=int(pose_cfg.get("window", 3)) if isinstance(pose_cfg.get("window", 3), int) else 3,
    )
    summary_dicts = stroke_summaries_to_dicts(summaries)

    viz_cfg = cfg.get("visualization", {}) if isinstance(cfg.get("visualization", {}), dict) else {}
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out = generate_match_report(
        args.match_name,
        summary_dicts,
        str(out_dir),
        enable_heatmap=bool(viz_cfg.get("enable_heatmap", True)) and (not args.no_heatmap),
        enable_timeline=bool(viz_cfg.get("enable_timeline", True)) and (not args.no_timeline),
        heatmap_bins=int(viz_cfg.get("heatmap_bins", 32)),
    )

    court_debug_path: Optional[Path] = None
    if not args.no_court_debug:
        try:
            frame_bgr = _read_video_frame_bgr(video_path, frame_idx=int(args.court_debug_frame))
            corners = (
                analysis.court_corners
                if isinstance(analysis.court_corners, list) and len(analysis.court_corners) == 4
                else None
            )
            if frame_bgr is None or corners is None:
                print("[WARN] Court debug image skipped (failed to read frame or missing corners).")
            else:
                h0, w0 = frame_bgr.shape[:2]
                manual_corners = vision_cfg.get("court_corners")
                if isinstance(manual_corners, (list, tuple)) and len(manual_corners) == 4:
                    corner_source = "manual_config"
                elif _looks_like_default_full_frame_corners(corners, w0, h0):
                    corner_source = "fallback_full_frame (detector failed)"
                else:
                    corner_source = "auto_detector"

                use_court_roi = bool(vision_cfg.get("use_court_roi", False))
                court_margin = float(vision_cfg.get("court_roi_margin", 0.0))
                roi = _compute_court_roi(corners, w0, h0, margin=court_margin) if use_court_roi else None
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
    if summary_dicts:
        print("- Stroke Summary:")
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
