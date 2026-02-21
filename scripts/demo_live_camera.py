#!/usr/bin/env python3
"""Live camera demo: lock court once, then run realtime player/ball/stroke inference."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import deque
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.pipeline.analyse_video import FrameResult
from src.pipeline.extract_strokes import (
    StrokeSummary,
    infer_segment_frame_ranges_from_analysis,
    summarise_strokes_from_analysis,
)
from src.pipeline.label_space import build_label_space_metadata
from src.tracking.ball_track import BallTrackState, SingleBallTracker
from src.tracking.bytetrack import SimpleByteTrack, Track
from src.tracking.player_track import PlayerState
from src.vision.ball_detector import BallDetector
from src.vision.court_detector import CourtDetector
from src.vision.court_fit_homography import fit_court_homography
from src.vision.court_env_profile import resolve_and_apply_court_env_overrides
from src.vision.detectors import PlayerDetector
from src.vision.pose_estimator import PoseEstimator, PoseKeypoints


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    return obj if isinstance(obj, dict) else {}


def _normalize_corners(value: Any) -> Optional[List[List[float]]]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    out: List[List[float]] = []
    for p in value:
        if not isinstance(p, (list, tuple)) or len(p) != 2:
            return None
        try:
            out.append([float(p[0]), float(p[1])])
        except Exception:
            return None
    return out


def _load_corners_from_report(report_path: Path) -> tuple[Optional[List[List[float]]], Dict[str, Any]]:
    meta: Dict[str, Any] = {"report_path": str(report_path), "source": "report", "reason": "report_invalid"}
    if not report_path.exists():
        meta["reason"] = "report_not_found"
        return None, meta
    try:
        data = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception as exc:
        meta["reason"] = f"report_json_error:{exc}"
        return None, meta

    corners = None
    frame_size = None
    if isinstance(data, dict):
        cd = data.get("court_detection")
        if isinstance(cd, dict):
            corners = _normalize_corners(cd.get("corners"))
            fs = cd.get("frame_size")
            if isinstance(fs, dict):
                frame_size = {
                    "width": int(fs.get("width", 0) or 0),
                    "height": int(fs.get("height", 0) or 0),
                }
        if corners is None:
            corners = _normalize_corners(data.get("corners"))

    if corners is None:
        return None, meta
    meta["reason"] = "OK"
    meta["confidence"] = 1.0
    meta["frame_size"] = frame_size
    return corners, meta


def _candidate_score(conf: float, reason: str, metrics: Optional[Dict[str, Any]]) -> float:
    m = metrics if isinstance(metrics, dict) else {}
    score = 100.0 * float(conf)
    r = str(reason or "")
    if r == "OK":
        score += 100.0
    elif r.startswith("R_fit_poor"):
        score -= 35.0
    elif r.startswith("R_hough"):
        score -= 25.0
    elif r.startswith("R_"):
        score -= 20.0

    na = float(m.get("num_hough_lines_a", 0) or 0)
    nb = float(m.get("num_hough_lines_b", 0) or 0)
    npairs_a = float(m.get("num_hough_pairs_a", 0) or 0)
    npairs_b = float(m.get("num_hough_pairs_b", 0) or 0)
    if na > 0 or nb > 0:
        score += min(35.0, 1.4 * (na + nb))
    if na > 0 and nb > 0:
        ratio = min(na, nb) / max(na, nb)
        score += 40.0 * float(ratio)
    if npairs_a > 0 or npairs_b > 0:
        score += min(24.0, 3.0 * (npairs_a + npairs_b))

    score -= 2.0 * float(m.get("reject_aspect_ratio", 0) or 0)
    score -= 0.8 * float(m.get("reject_area_ratio_low", 0) or 0)
    score -= 0.6 * float(m.get("reject_degenerate_min_edge", 0) or 0)
    return float(score)


def _compute_court_roi(
    corners: Sequence[Sequence[float]],
    width: int,
    height: int,
    margin: float,
) -> Optional[Tuple[int, int, int, int]]:
    if len(corners) != 4:
        return None
    xs = [float(p[0]) for p in corners]
    ys = [float(p[1]) for p in corners]
    pad_x = float(width) * float(margin)
    pad_y = float(height) * float(margin)
    x1 = max(0, int(round(min(xs) - pad_x)))
    y1 = max(0, int(round(min(ys) - pad_y)))
    x2 = min(width, int(round(max(xs) + pad_x)))
    y2 = min(height, int(round(max(ys) + pad_y)))
    if x2 > x1 and y2 > y1:
        return x1, y1, x2, y2
    return None


def _draw_corners(frame_bgr: np.ndarray, corners: Optional[List[List[float]]]) -> None:
    if corners is None or len(corners) != 4:
        return
    pts = np.array([(int(round(p[0])), int(round(p[1]))) for p in corners], dtype=np.int32)
    cv2.polylines(frame_bgr, [pts.reshape(-1, 1, 2)], True, (0, 255, 0), 2)
    for idx, (x, y) in enumerate(pts):
        cv2.circle(frame_bgr, (int(x), int(y)), 5, (0, 0, 255), -1)
        cv2.putText(
            frame_bgr,
            ["LB", "RB", "RT", "LT"][idx],
            (int(x) + 6, int(y) - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
            lineType=cv2.LINE_AA,
        )


def _overlay_text(img: np.ndarray, lines: List[str], x: int = 12, y0: int = 24, step: int = 24) -> None:
    y = y0
    for txt in lines:
        cv2.putText(img, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3, lineType=cv2.LINE_AA)
        cv2.putText(img, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, lineType=cv2.LINE_AA)
        y += step


def _filter_player_detections_to_court(court_poly: Optional[np.ndarray], dets: list, min_keep: int = 1):
    if court_poly is None or not dets:
        return dets
    kept = []
    for det in dets:
        x1, y1, x2, y2 = det.bbox
        px = float((x1 + x2) / 2.0)
        py = float(y2)
        if cv2.pointPolygonTest(court_poly, (px, py), False) >= 0:
            kept.append(det)
    return kept if len(kept) >= min_keep else dets


def _roles_for_tracks(
    tracks: List[Track],
    court_h: Optional[Any],
    court_half_len: Optional[float],
) -> Dict[int, str]:
    if not tracks:
        return {}
    if court_h is not None and court_half_len is not None:
        roles: Dict[int, str] = {}
        for tr in tracks:
            x1, y1, x2, y2 = [float(v) for v in tr.bbox]
            foot = ((x1 + x2) / 2.0, y2)
            _, cy = court_h.to_court(foot)
            roles[tr.track_id] = "near" if float(cy) < court_half_len else "far"
        return roles
    centers = [(tr.track_id, float((tr.bbox[1] + tr.bbox[3]) / 2.0)) for tr in tracks]
    thresh = float(np.median([c for _, c in centers]))
    return {tid: ("near" if cy >= thresh else "far") for tid, cy in centers}


def _try_lock_court_once(
    frame_bgr: np.ndarray,
    use_model_fit: bool,
    model_fit_method: str,
    allow_fallback_lsd: bool,
    court_detector: Optional[CourtDetector],
) -> tuple[Optional[List[List[float]]], Dict[str, Any], Optional[np.ndarray]]:
    if use_model_fit:
        fit = fit_court_homography(
            frame_bgr,
            method=model_fit_method,
            allow_fallback_lsd=allow_fallback_lsd,
        )
        metrics = fit.metrics if isinstance(fit.metrics, dict) else {}
        meta = {
            "source": "auto",
            "confidence": float(fit.confidence),
            "reason": str(fit.reason),
            "metrics": metrics,
            "candidate_score": float(_candidate_score(float(fit.confidence), str(fit.reason), metrics)),
        }
        corners = fit.corners.tolist() if fit.corners is not None else None
        return corners, meta, fit.debug_image if isinstance(fit.debug_image, np.ndarray) else None

    if court_detector is None:
        return None, {"source": "auto_failed", "confidence": None, "reason": "detector_missing"}, None

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    lines = court_detector.detect_court(rgb)
    conf = float(court_detector.last_confidence or 0.0)
    reason = str(court_detector.last_reason or "unknown")
    metrics = court_detector.last_metrics if isinstance(court_detector.last_metrics, dict) else {}
    meta = {
        "source": "auto",
        "confidence": conf,
        "reason": reason,
        "metrics": metrics,
        "candidate_score": float(_candidate_score(conf, reason, metrics)),
    }
    corners = lines.corners.tolist() if (lines is not None and lines.corners is not None) else None
    return corners, meta, None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Badminton live camera demo (lock court once + realtime stroke view).")
    p.add_argument("--camera-index", type=int, default=0, help="Camera index for cv2.VideoCapture.")
    p.add_argument("--config", type=str, default="src/config/v3_realtime.yaml", help="YAML config path.")
    p.add_argument("--out-dir", type=str, default="reports/live_camera", help="Output directory.")
    p.add_argument("--session-name", type=str, default="live_camera", help="Output report prefix.")
    p.add_argument(
        "--court-corners-report",
        type=str,
        default="",
        help="Reuse court corners from an existing report JSON and skip auto lock.",
    )
    p.add_argument("--court-lock-attempts", type=int, default=24, help="Max lock attempts before fallback.")
    p.add_argument("--court-lock-step", type=int, default=5, help="Frames skipped between lock attempts.")
    p.add_argument(
        "--court-lock-seconds",
        type=float,
        default=0.0,
        help="If >0, record this many seconds for lock phase before choosing best court.",
    )
    p.add_argument(
        "--lock-only",
        action="store_true",
        help="Only lock court and write report, do not run realtime stroke inference.",
    )
    p.add_argument("--stroke-buffer-frames", type=int, default=360, help="Rolling buffer length for stroke inference.")
    p.add_argument("--stroke-eval-stride", type=int, default=6, help="Evaluate stroke summaries every N frames.")
    p.add_argument("--stroke-min-frames", type=int, default=60, help="Minimum buffered frames before stroke eval.")
    p.add_argument(
        "--stroke-classifier",
        choices=["auto", "on", "off"],
        default="auto",
        help='Stroke classifier mode: "on" force enable, "off" disable, "auto" enable when checkpoint exists.',
    )
    p.add_argument(
        "--stroke-classifier-checkpoint",
        type=str,
        default="",
        help="Path to stroke classifier checkpoint (best_model.pt).",
    )
    p.add_argument(
        "--stroke-classifier-device",
        type=str,
        default="auto",
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
    p.add_argument("--display-width", type=int, default=1280, help="Display width; <=0 keeps original size.")
    p.add_argument("--max-frames", type=int, default=0, help="Max processed frames. 0 means unlimited.")
    p.add_argument("--no-window", action="store_true", help="Do not open visualization window.")
    p.add_argument("--save-court-debug", action="store_true", help="Save locked-court debug image.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = REPO_ROOT / cfg_path
    if not cfg_path.exists():
        raise SystemExit(f"[ERROR] config not found: {cfg_path}")
    cfg = _load_yaml(cfg_path)

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    vision_cfg = cfg.get("vision", {}) if isinstance(cfg.get("vision", {}), dict) else {}
    tracking_cfg = cfg.get("tracking", {}) if isinstance(cfg.get("tracking", {}), dict) else {}
    pose_cfg = cfg.get("pose", {}) if isinstance(cfg.get("pose", {}), dict) else {}
    device_cfg = cfg.get("device", {}) if isinstance(cfg.get("device", {}), dict) else {}
    spatial_cfg = cfg.get("spatial_logic", {}) if isinstance(cfg.get("spatial_logic", {}), dict) else {}
    classifier_cfg = cfg.get("classifier", {}) if isinstance(cfg.get("classifier", {}), dict) else {}
    court_env = resolve_and_apply_court_env_overrides(vision_cfg, base_dir=cfg_path.parent)
    if int(court_env.get("num_vars", 0)) > 0:
        print(
            f"[COURT_ENV] applied {court_env.get('num_vars')} vars "
            f"(profile={court_env.get('profile_path')})"
        )
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

    classifier_runtime = None
    classifier_enabled = False
    classifier_ckpt = None
    classifier_mode = str(args.stroke_classifier or "auto").strip().lower()
    classifier_ckpt_raw = str(args.stroke_classifier_checkpoint or "").strip()
    if not classifier_ckpt_raw:
        classifier_ckpt_raw = str(classifier_cfg.get("checkpoint", "") or "").strip()
    if not classifier_ckpt_raw:
        classifier_ckpt_raw = str(os.environ.get("BADMINTON_STROKE_CKPT", "") or "").strip()
    if classifier_ckpt_raw:
        classifier_ckpt = Path(classifier_ckpt_raw).expanduser()
        if not classifier_ckpt.is_absolute():
            classifier_ckpt = REPO_ROOT / classifier_ckpt

    want_classifier = classifier_mode == "on" or (
        classifier_mode == "auto" and classifier_ckpt is not None and classifier_ckpt.exists()
    )
    if classifier_mode == "on" and (classifier_ckpt is None or not classifier_ckpt.exists()):
        print(f"[WARN] stroke classifier forced on but checkpoint not found: {classifier_ckpt}")

    if want_classifier and classifier_ckpt is not None and classifier_ckpt.exists():
        try:
            from src.pipeline.stroke_classifier_runtime import StrokeClassifierRuntime

            classifier_runtime = StrokeClassifierRuntime.from_checkpoint(
                checkpoint_path=classifier_ckpt,
                device=str(args.stroke_classifier_device or classifier_cfg.get("device", "auto")),
                frame_size=args.stroke_classifier_frame_size,
                num_frames=args.stroke_classifier_num_frames,
                topk=max(1, int(args.stroke_classifier_topk)),
            )
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
            print(f"[CLS] enabled checkpoint={classifier_ckpt}")
        except Exception as exc:
            classifier_runtime = None
            classifier_enabled = False
            print(f"[WARN] failed to init stroke classifier runtime: {exc}")

    yolo_device = device_cfg.get("yolo_device", "cpu")
    yolo_imgsz = int(vision_cfg.get("yolo_imgsz", 640))
    yolo_conf = float(vision_cfg.get("yolo_conf", 0.25))

    player_model = str(vision_cfg.get("yolo_model", "yolov8n.pt"))
    ball_model = str(vision_cfg.get("ball_model", "yolov8n.pt"))
    player_model_path = Path(player_model)
    if not player_model_path.is_absolute():
        player_model_path = REPO_ROOT / player_model_path
    ball_model_path = Path(ball_model)
    if not ball_model_path.is_absolute():
        ball_model_path = REPO_ROOT / ball_model_path

    if not player_model_path.exists():
        print(f"[WARN] player model not found locally: {player_model_path}")
    if not ball_model_path.exists():
        print(f"[WARN] ball model not found locally: {ball_model_path}")

    detect_court_corners = bool(vision_cfg.get("detect_court_corners", True))
    court_method = str(vision_cfg.get("court_detector_method", "heuristic")).lower()
    use_model_fit = court_method in ("model_fit", "model_fit_bwf")
    model_fit_method = str(vision_cfg.get("model_fit_method", "lsd"))
    allow_fallback_lsd = bool(vision_cfg.get("model_fit_allow_fallback_lsd", True))

    court_detector = (
        CourtDetector(
            edge_top_min=float(vision_cfg.get("edge_top_min", 0.20)),
            edge_bottom_min=float(vision_cfg.get("edge_bottom_min", 0.18)),
            edge_min_floor=float(vision_cfg.get("edge_min_floor", 0.10)),
            net_suppress_y_min=float(vision_cfg.get("net_suppress_y_min", 0.35)),
            net_suppress_y_max=float(vision_cfg.get("net_suppress_y_max", 0.55)),
            tpl_net_reject_max=float(vision_cfg.get("tpl_net_reject_max", 0.14)),
            top_edge_net_reject_max=float(vision_cfg.get("top_edge_net_reject_max", 0.18)),
        )
        if detect_court_corners and (not use_model_fit)
        else None
    )

    cap = cv2.VideoCapture(int(args.camera_index))
    if not cap.isOpened():
        raise SystemExit(f"[ERROR] failed to open camera index {args.camera_index}")

    ok, first_bgr = cap.read()
    if not ok or first_bgr is None:
        cap.release()
        raise SystemExit("[ERROR] failed to read first camera frame")

    lock_best = None
    lock_best_meta: Dict[str, Any] = {
        "source": "auto_failed",
        "confidence": None,
        "reason": "no_valid_candidate",
        "candidate_score": -1e9,
    }
    lock_best_debug: Optional[np.ndarray] = None
    lock_frame = first_bgr
    checked = 0
    lock_buffer_frames = 0
    lock_buffer_seconds = float(max(0.0, args.court_lock_seconds))

    report_path_raw = str(args.court_corners_report or "").strip()
    report_path = None
    if report_path_raw:
        rp = Path(report_path_raw).expanduser()
        if not rp.is_absolute():
            rp = REPO_ROOT / rp
        report_path = rp
        rep_corners, rep_meta = _load_corners_from_report(report_path)
        if rep_corners is not None:
            lock_best = rep_corners
            lock_best_meta = {
                "source": "report",
                "confidence": 1.0,
                "reason": "OK",
                "candidate_score": 1e9,
                "report_path": str(report_path),
                "report_meta": rep_meta,
            }
        else:
            print(f"[WARN] failed to use report corners: {report_path} reason={rep_meta.get('reason')}")

    if lock_best is None:
        eval_frames: List[np.ndarray] = []
        if lock_buffer_seconds > 0.0:
            t_start = time.monotonic()
            eval_frames.append(first_bgr)
            while True:
                if time.monotonic() - t_start >= lock_buffer_seconds:
                    break
                ok, buf_frame = cap.read()
                if not ok or buf_frame is None:
                    break
                eval_frames.append(buf_frame)
            lock_buffer_frames = int(len(eval_frames))

        if eval_frames:
            step = max(1, int(args.court_lock_step))
            max_eval = max(1, int(args.court_lock_attempts))
            for idx in range(0, len(eval_frames), step):
                if checked >= max_eval:
                    break
                lock_frame = eval_frames[idx]
                checked += 1
                corners_i, meta_i, dbg_i = _try_lock_court_once(
                    lock_frame,
                    use_model_fit=use_model_fit,
                    model_fit_method=model_fit_method,
                    allow_fallback_lsd=allow_fallback_lsd,
                    court_detector=court_detector,
                )
                score_i = float(meta_i.get("candidate_score", -1e9))
                if corners_i is not None and (
                    lock_best is None
                    or score_i > float(lock_best_meta.get("candidate_score", -1e9))
                    or (
                        score_i == float(lock_best_meta.get("candidate_score", -1e9))
                        and float(meta_i.get("confidence", 0.0) or 0.0)
                        > float(lock_best_meta.get("confidence", 0.0) or 0.0)
                    )
                ):
                    lock_best = corners_i
                    lock_best_meta = meta_i
                    lock_best_debug = dbg_i
                if str(meta_i.get("reason", "")) == "OK" and corners_i is not None:
                    break
        else:
            for i in range(max(1, int(args.court_lock_attempts))):
                if i > 0:
                    for _ in range(max(1, int(args.court_lock_step))):
                        ok, lock_frame = cap.read()
                        if not ok or lock_frame is None:
                            break
                    if not ok or lock_frame is None:
                        break
                checked += 1
                corners_i, meta_i, dbg_i = _try_lock_court_once(
                    lock_frame,
                    use_model_fit=use_model_fit,
                    model_fit_method=model_fit_method,
                    allow_fallback_lsd=allow_fallback_lsd,
                    court_detector=court_detector,
                )
                score_i = float(meta_i.get("candidate_score", -1e9))
                if corners_i is not None and (
                    lock_best is None
                    or score_i > float(lock_best_meta.get("candidate_score", -1e9))
                    or (
                        score_i == float(lock_best_meta.get("candidate_score", -1e9))
                        and float(meta_i.get("confidence", 0.0) or 0.0)
                        > float(lock_best_meta.get("confidence", 0.0) or 0.0)
                    )
                ):
                    lock_best = corners_i
                    lock_best_meta = meta_i
                    lock_best_debug = dbg_i
                if str(meta_i.get("reason", "")) == "OK" and corners_i is not None:
                    break

    h0, w0 = first_bgr.shape[:2]
    default_corners = [
        [0.0, float(h0 - 1)],
        [float(w0 - 1), float(h0 - 1)],
        [float(w0 - 1), 0.0],
        [0.0, 0.0],
    ]
    court_corners = lock_best if lock_best is not None else None
    court_detection = dict(lock_best_meta)
    court_detection["checked_frames"] = int(checked)
    court_detection["source"] = "auto" if court_corners is not None else "auto_failed"
    court_detection["lock_buffer_frames"] = int(lock_buffer_frames)
    court_detection["lock_buffer_seconds"] = float(lock_buffer_seconds)
    if report_path is not None:
        court_detection["report_path"] = str(report_path)
    if str(lock_best_meta.get("source", "")) == "report" and court_corners is not None:
        court_detection["source"] = "report"

    if args.save_court_debug:
        if lock_best_debug is not None:
            cv2.imwrite(str(out_dir / f"{args.session_name}_court_fit_debug.jpg"), lock_best_debug)
        dbg = lock_frame.copy()
        _draw_corners(dbg, court_corners if court_corners is not None else default_corners)
        _overlay_text(
            dbg,
            [
                f"Court source: {court_detection.get('source')}",
                f"reason: {court_detection.get('reason')}",
                f"conf: {float(court_detection.get('confidence') or 0.0):.3f}",
            ],
        )
        cv2.imwrite(str(out_dir / f"{args.session_name}_court_detect_debug.jpg"), dbg)

    use_court_roi = bool(vision_cfg.get("use_court_roi", False))
    court_margin = float(vision_cfg.get("court_roi_margin", 0.0))
    court_roi: Optional[Tuple[int, int, int, int]] = None
    if use_court_roi and court_corners is not None:
        court_roi = _compute_court_roi(court_corners, w0, h0, margin=court_margin)

    roi_dx = int(court_roi[0]) if court_roi is not None else 0
    roi_dy = int(court_roi[1]) if court_roi is not None else 0

    court_h = None
    court_half_len = None
    if court_corners is not None and len(court_corners) == 4:
        try:
            from src.geometry.homography import COURT_LENGTH_M, CourtHomography

            court_h = CourtHomography.from_corners(court_corners)
            court_half_len = float(COURT_LENGTH_M) / 2.0
        except Exception:
            court_h = None
            court_half_len = None

    poly_src = court_corners if (court_corners is not None and len(court_corners) == 4) else default_corners
    court_poly = np.array(poly_src, dtype=np.float32).reshape(-1, 1, 2) if len(poly_src) == 4 else None

    player_det = PlayerDetector(
        model_path=str(player_model_path),
        device=str(yolo_device),
        conf=float(yolo_conf),
        imgsz=int(yolo_imgsz),
    )
    ball_det = BallDetector(
        model_path=str(ball_model_path),
        device=str(yolo_device),
        conf=float(yolo_conf),
        imgsz=int(yolo_imgsz),
        allow_all_when_empty=bool(vision_cfg.get("ball_allow_all_if_empty", False)),
    )
    player_tracker = SimpleByteTrack(
        iou_thresh=float(tracking_cfg.get("iou_thresh", 0.3)),
        max_time_since_update=int(tracking_cfg.get("max_age", 30)),
    )
    ball_tracker = SingleBallTracker(
        iou_thresh=float(tracking_cfg.get("iou_thresh", 0.3)),
        ema_alpha=0.6,
        max_age=int(tracking_cfg.get("ball_max_age", 5)),
    )

    enable_pose = bool(pose_cfg.get("enable", False))
    pose_stride = int(pose_cfg.get("stride", 2))
    pose_window = int(pose_cfg.get("window", 3))
    pose_estimator: Optional[PoseEstimator] = None
    if enable_pose:
        try:
            pose_estimator = PoseEstimator()
        except Exception as exc:
            print(f"[WARN] PoseEstimator init failed: {exc}")
            pose_estimator = None
            enable_pose = False

    detect_stride = int(tracking_cfg.get("detect_stride", 1))
    detect_stride = max(1, detect_stride)
    last_ball_state: Optional[BallTrackState] = None

    frame_history: deque[FrameResult] = deque(maxlen=max(30, int(args.stroke_buffer_frames)))
    frame_rgb_history: Dict[int, np.ndarray] = {}
    pose_history: Dict[int, Dict] = {}
    classifier_cache: Dict[Tuple[int, int], Dict[str, Any]] = {}
    emitted_keys: set[tuple[int, int, str]] = set()
    emitted_summaries: List[Dict[str, Any]] = []
    latest_summary: Optional[StrokeSummary] = None

    frame_idx = 0
    t_last = time.time()
    fps_ema = 0.0

    print(
        "[LIVE] started. keys: q/ESC quit, r relock court.\n"
        f"[LIVE] camera={args.camera_index} court_source={court_detection.get('source')} "
        f"reason={court_detection.get('reason')} conf={court_detection.get('confidence')} "
        f"classifier={'on' if classifier_enabled else 'off'}"
    )

    if bool(args.lock_only):
        cap.release()
        cv2.destroyAllWindows()
        report = {
            "mode": "live_camera",
            "camera_index": int(args.camera_index),
            "frames_processed": 0,
            "court_detection": {
                "source": court_detection.get("source"),
                "confidence": court_detection.get("confidence"),
                "reason": court_detection.get("reason"),
                "corners": court_corners,
                "checked_frames": court_detection.get("checked_frames"),
                "candidate_score": court_detection.get("candidate_score"),
                "lock_buffer_frames": court_detection.get("lock_buffer_frames"),
                "lock_buffer_seconds": court_detection.get("lock_buffer_seconds"),
                "report_path": court_detection.get("report_path"),
                "frame_size": {"width": int(w0), "height": int(h0)},
            },
            "stroke_classifier": {
                "enabled": bool(classifier_enabled),
                "mode": classifier_mode,
                "checkpoint": str(classifier_ckpt) if classifier_ckpt is not None else None,
                "device": str(args.stroke_classifier_device),
                "topk": int(max(1, args.stroke_classifier_topk)),
                "min_confidence": float(max(0.0, classifier_min_confidence)),
                "label_space": label_space_meta,
            },
            "court_env": court_env,
            "label_space_version": label_space_meta.get("version"),
            "stroke_count": 0,
            "latest_stroke": None,
            "strokes": [],
        }
        out_json = out_dir / f"{args.session_name}_live_report.json"
        report["court_corners_only"] = True
        out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[LIVE] lock-only report saved: {out_json}")
        return

    while True:
        if int(args.max_frames) > 0 and frame_idx >= int(args.max_frames):
            break
        ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if court_roi is not None:
            x1, y1, x2, y2 = court_roi
            frame_proc = frame_rgb[y1:y2, x1:x2]
        else:
            frame_proc = frame_rgb
        frame_rgb_history[int(frame_idx)] = frame_rgb.copy()

        if frame_idx % detect_stride == 0:
            pdets = player_det.detect([frame_proc])[0]
            bdets = ball_det.detect([frame_proc], exclude_bboxes_batch=[[d.bbox for d in pdets]])[0]
            if roi_dx != 0 or roi_dy != 0:
                for det in pdets:
                    det.bbox = det.bbox + np.array([roi_dx, roi_dy, roi_dx, roi_dy], dtype=float)
                for det in bdets:
                    det.bbox = det.bbox + np.array([roi_dx, roi_dy, roi_dx, roi_dy], dtype=float)
            pdets = _filter_player_detections_to_court(court_poly, pdets)
            player_tracks = player_tracker.update(pdets, frame_idx=frame_idx)
            ball_state = ball_tracker.update(frame_idx, bdets)
            if ball_state is None:
                pred = ball_tracker.predict_only()
                if pred is not None:
                    ball_state = BallTrackState(
                        frame_idx=frame_idx,
                        bbox=pred.bbox,
                        score=pred.score,
                        cx=pred.cx,
                        cy=pred.cy,
                        predicted=True,
                    )
            if ball_state is None and last_ball_state is not None:
                ball_state = BallTrackState(
                    frame_idx=frame_idx,
                    bbox=last_ball_state.bbox,
                    score=last_ball_state.score,
                    cx=last_ball_state.cx,
                    cy=last_ball_state.cy,
                    predicted=True,
                )
        else:
            player_tracks = player_tracker.predict_only()
            pred = ball_tracker.predict_only()
            ball_state = None
            if pred is not None:
                ball_state = BallTrackState(
                    frame_idx=frame_idx,
                    bbox=pred.bbox,
                    score=pred.score,
                    cx=pred.cx,
                    cy=pred.cy,
                    predicted=True,
                )
            elif last_ball_state is not None:
                ball_state = BallTrackState(
                    frame_idx=frame_idx,
                    bbox=last_ball_state.bbox,
                    score=last_ball_state.score,
                    cx=last_ball_state.cx,
                    cy=last_ball_state.cy,
                    predicted=True,
                )

        if ball_state is not None:
            last_ball_state = ball_state

        roles = _roles_for_tracks(player_tracks, court_h, court_half_len)
        player_states = [
            PlayerState(
                track_id=tr.track_id,
                role=roles.get(tr.track_id, "unknown"),
                bboxes=[tuple(float(v) for v in tr.bbox)],
                frames=[int(frame_idx)],
            )
            for tr in player_tracks
        ]

        if enable_pose and pose_estimator is not None and frame_idx % max(1, pose_stride) == 0:
            try:
                poses: List[PoseKeypoints] = pose_estimator.estimate(frame_proc)
                if poses:
                    pts = poses[0].points.copy()
                    if roi_dx != 0 or roi_dy != 0:
                        pts[:, 0] += float(roi_dx)
                        pts[:, 1] += float(roi_dy)
                    pose_history[int(frame_idx)] = {"points": pts.tolist()}
            except Exception:
                pass

        fr = FrameResult(
            frame_idx=frame_idx,
            players=player_tracks,
            player_states=player_states,
            player_roles=roles,
            ball=ball_state,
        )
        frame_history.append(fr)

        old_limit = int(frame_idx) - int(frame_history.maxlen) - 5
        if old_limit > 0:
            stale = [k for k in pose_history.keys() if int(k) < old_limit]
            for k in stale:
                pose_history.pop(k, None)
            stale_rgb = [k for k in frame_rgb_history.keys() if int(k) < old_limit]
            for k in stale_rgb:
                frame_rgb_history.pop(k, None)
            stale_cache = [k for k in classifier_cache.keys() if int(k[1]) < old_limit]
            for k in stale_cache:
                classifier_cache.pop(k, None)

        if (
            len(frame_history) >= max(10, int(args.stroke_min_frames))
            and frame_idx % max(1, int(args.stroke_eval_stride)) == 0
        ):
            analysis_stub = SimpleNamespace(
                frame_results=list(frame_history),
                court_corners=court_corners,
                pose_results=pose_history if enable_pose else None,
            )
            try:
                classifier_labels = None
                classifier_outputs = None
                if classifier_enabled and classifier_runtime is not None:
                    seg_ranges = infer_segment_frame_ranges_from_analysis(analysis_stub)
                    classifier_outputs = []
                    for sr in seg_ranges:
                        if len(sr) != 2:
                            classifier_outputs.append({"label": None, "confidence": None, "topk": []})
                            continue
                        s_f, e_f = int(sr[0]), int(sr[1])
                        ckey = (s_f, e_f)
                        cached = classifier_cache.get(ckey)
                        if cached is None:
                            pred = classifier_runtime.predict_from_frame_store(frame_rgb_history, s_f, e_f)
                            if pred is None:
                                cached = {"label": None, "confidence": None, "topk": []}
                            else:
                                cached = {
                                    "label": pred.label,
                                    "confidence": float(pred.confidence),
                                    "topk": pred.topk,
                                }
                            classifier_cache[ckey] = cached
                        classifier_outputs.append(cached)
                    classifier_labels = [
                        str(item.get("label")) if isinstance(item, dict) and item.get("label") is not None else None
                        for item in classifier_outputs
                    ]

                summaries = summarise_strokes_from_analysis(
                    analysis_stub,
                    classifier_labels=classifier_labels,
                    classifier_outputs=classifier_outputs,
                    classifier_min_confidence=float(max(0.0, classifier_min_confidence)),
                    enable_hitter_inference=bool(spatial_cfg.get("enable_hitter_inference", True)),
                    hitter_distance_max=float(spatial_cfg.get("hitter_distance_max", 200.0)),
                    pose_window=pose_window,
                )
                for s in summaries:
                    key = (int(s.frame_range[0]), int(s.frame_range[1]), str(s.final_type))
                    if key in emitted_keys:
                        continue
                    emitted_keys.add(key)
                    emitted_summaries.append(asdict(s))
                    latest_summary = s
            except Exception:
                pass

        draw = frame_bgr.copy()
        _draw_corners(draw, court_corners if court_corners is not None else default_corners)
        if court_roi is not None:
            x1, y1, x2, y2 = court_roi
            cv2.rectangle(draw, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

        for tr in player_tracks:
            x1, y1, x2, y2 = [int(round(float(v))) for v in tr.bbox]
            role = roles.get(tr.track_id, "unknown")
            color = (0, 220, 0) if role == "near" else (0, 180, 255)
            cv2.rectangle(draw, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                draw,
                f"id={tr.track_id} {role}",
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                lineType=cv2.LINE_AA,
            )

        if ball_state is not None:
            bx = int(round(float(ball_state.cx)))
            by = int(round(float(ball_state.cy)))
            bcolor = (0, 180, 255) if bool(ball_state.predicted) else (0, 0, 255)
            cv2.circle(draw, (bx, by), 5, bcolor, -1)

        if latest_summary is not None:
            if latest_summary.contact_x is not None and latest_summary.contact_y is not None:
                cv2.circle(
                    draw,
                    (int(round(latest_summary.contact_x)), int(round(latest_summary.contact_y))),
                    6,
                    (255, 255, 0),
                    -1,
                )
            if latest_summary.landing_x is not None and latest_summary.landing_y is not None:
                cv2.circle(
                    draw,
                    (int(round(latest_summary.landing_x)), int(round(latest_summary.landing_y))),
                    6,
                    (255, 0, 255),
                    -1,
                )

        t_now = time.time()
        dt = max(1e-6, t_now - t_last)
        inst = 1.0 / dt
        fps_ema = inst if fps_ema <= 0.0 else (0.9 * fps_ema + 0.1 * inst)
        t_last = t_now

        txt = [
            f"LIVE camera={args.camera_index} fps={fps_ema:.1f}",
            f"Court: {court_detection.get('source')} reason={court_detection.get('reason')} conf={float(court_detection.get('confidence') or 0.0):.2f}",
            f"Classifier: {'on' if classifier_enabled else 'off'}",
            f"Strokes detected: {len(emitted_summaries)}",
            "Keys: q/ESC quit, r relock",
        ]
        if latest_summary is not None:
            txt.append(
                f"Last: {latest_summary.final_type} | hitter={latest_summary.hitter_role} | landing={latest_summary.landing_region}"
            )
        _overlay_text(draw, txt)

        if int(args.display_width) > 0:
            h, w = draw.shape[:2]
            if w > int(args.display_width):
                nh = int(round(h * float(args.display_width) / float(w)))
                draw = cv2.resize(draw, (int(args.display_width), max(1, nh)))

        if not args.no_window:
            cv2.imshow("Badminton Live Camera", draw)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("r"):
                corners_i, meta_i, dbg_i = _try_lock_court_once(
                    frame_bgr,
                    use_model_fit=use_model_fit,
                    model_fit_method=model_fit_method,
                    allow_fallback_lsd=allow_fallback_lsd,
                    court_detector=court_detector,
                )
                if corners_i is not None:
                    court_corners = corners_i
                    court_detection = dict(meta_i)
                    court_detection["source"] = "auto"
                    if args.save_court_debug and dbg_i is not None:
                        cv2.imwrite(str(out_dir / f"{args.session_name}_court_fit_debug_relock.jpg"), dbg_i)
                    if use_court_roi:
                        court_roi = _compute_court_roi(court_corners, w0, h0, margin=court_margin)
                        roi_dx = int(court_roi[0]) if court_roi is not None else 0
                        roi_dy = int(court_roi[1]) if court_roi is not None else 0
                    if court_corners is not None and len(court_corners) == 4:
                        try:
                            from src.geometry.homography import COURT_LENGTH_M, CourtHomography

                            court_h = CourtHomography.from_corners(court_corners)
                            court_half_len = float(COURT_LENGTH_M) / 2.0
                            court_poly = np.array(court_corners, dtype=np.float32).reshape(-1, 1, 2)
                        except Exception:
                            court_h = None
                            court_half_len = None
                            court_poly = np.array(default_corners, dtype=np.float32).reshape(-1, 1, 2)

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

    report = {
        "mode": "live_camera",
        "camera_index": int(args.camera_index),
        "frames_processed": int(frame_idx),
        "court_detection": {
            "source": court_detection.get("source"),
            "confidence": court_detection.get("confidence"),
            "reason": court_detection.get("reason"),
            "corners": court_corners,
            "checked_frames": court_detection.get("checked_frames"),
            "candidate_score": court_detection.get("candidate_score"),
            "lock_buffer_frames": court_detection.get("lock_buffer_frames"),
            "lock_buffer_seconds": court_detection.get("lock_buffer_seconds"),
            "report_path": court_detection.get("report_path"),
            "frame_size": {"width": int(w0), "height": int(h0)},
        },
        "stroke_classifier": {
            "enabled": bool(classifier_enabled),
            "mode": classifier_mode,
            "checkpoint": str(classifier_ckpt) if classifier_ckpt is not None else None,
            "device": str(args.stroke_classifier_device),
            "topk": int(max(1, args.stroke_classifier_topk)),
            "min_confidence": float(max(0.0, classifier_min_confidence)),
            "label_space": label_space_meta,
        },
        "court_env": court_env,
        "label_space_version": label_space_meta.get("version"),
        "stroke_count": int(len(emitted_summaries)),
        "latest_stroke": asdict(latest_summary) if latest_summary is not None else None,
        "strokes": emitted_summaries,
    }
    out_json = out_dir / f"{args.session_name}_live_report.json"
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[LIVE] report saved: {out_json}")


if __name__ == "__main__":
    main()
