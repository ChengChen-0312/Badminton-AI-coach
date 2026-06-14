"""Unified video analysis pipeline placeholder."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from src.tracking.ball_track import BallTrackState, SingleBallTracker
from src.tracking.bytetrack import SimpleByteTrack, Track
from src.tracking.player_track import PlayerState
from src.vision.ball_detector import BallDetector
from src.vision.court_detector import CourtDetector
from src.vision.court_fit_homography import fit_court_homography
from src.vision.court_env_profile import resolve_and_apply_court_env_overrides, resolve_court_env_overrides
from src.vision.detectors import PlayerDetector
from src.vision.pose_estimator import PoseEstimator, PoseKeypoints

try:
    from src.vision.court_fit_homography_modern import fit_court_homography as fit_court_homography_modern
except Exception:  # pragma: no cover - optional hybrid backend
    fit_court_homography_modern = None


@dataclass
class FrameResult:
    frame_idx: int
    players: List[Track]
    player_states: List[PlayerState]
    player_roles: Dict[int, str]
    ball: BallTrackState | None


@dataclass
class AnalyseResult:
    video_path: str
    fps: float
    frame_results: List[FrameResult]
    court_corners: Optional[List[List[float]]] = None
    court_detection: Optional[Dict[str, Any]] = None
    player_tracks: Optional[List[PlayerState]] = None
    pose_results: Optional[Dict[int, Dict]] = None
    court_env: Optional[Dict[str, Any]] = None


def analyse_video(
    video_path: str | Path,
    device: str = "mps",
    player_model_path: str = "yolov8n.pt",
    ball_model_path: str = "yolov8n.pt",
    config: dict | None = None,
) -> AnalyseResult:
    cfg = config or {}
    vision_cfg = cfg.get("vision", {})
    tracking_cfg = cfg.get("tracking", {})
    realtime_cfg = cfg.get("realtime", {})
    pose_cfg = cfg.get("pose", {})

    # Optional reproducible env profile for court-fitting knobs (BADC_*).
    cfg_base_raw = cfg.get("__config_dir__")
    cfg_base_dir = Path(str(cfg_base_raw)).expanduser() if isinstance(cfg_base_raw, str) and cfg_base_raw else None
    court_env = resolve_and_apply_court_env_overrides(vision_cfg, base_dir=cfg_base_dir)
    profile_retry_enabled = bool(vision_cfg.get("court_detect_profile_retry_on_fail", False))
    profile_retry_overrides: Dict[str, str] = {}
    profile_retry_path: Optional[str] = None
    profile_retry_raw = vision_cfg.get("court_detect_profile_retry_file")
    if profile_retry_enabled and isinstance(profile_retry_raw, str) and profile_retry_raw.strip():
        profile_retry_overrides, profile_retry_path = resolve_court_env_overrides(
            {"court_env_file": profile_retry_raw.strip()},
            base_dir=cfg_base_dir,
        )
        court_env["retry_profile_path"] = profile_retry_path
        court_env["retry_num_vars"] = int(len(profile_retry_overrides))

    detect_stride = tracking_cfg.get("detect_stride", 1)
    batch_size = realtime_cfg.get("batch_size", 1)

    yolo_device = cfg.get("device", {}).get("yolo_device", device)
    yolo_imgsz = vision_cfg.get("yolo_imgsz", 640)
    yolo_conf = vision_cfg.get("yolo_conf", 0.25)
    use_court_roi = vision_cfg.get("use_court_roi", False)
    court_margin = vision_cfg.get("court_roi_margin", 0.0)
    edge_top_min = float(vision_cfg.get("edge_top_min", 0.20))
    edge_bottom_min = float(vision_cfg.get("edge_bottom_min", 0.18))
    edge_min_floor = float(vision_cfg.get("edge_min_floor", 0.10))
    net_suppress_y_min = float(vision_cfg.get("net_suppress_y_min", 0.35))
    net_suppress_y_max = float(vision_cfg.get("net_suppress_y_max", 0.55))
    tpl_net_reject_max = float(vision_cfg.get("tpl_net_reject_max", 0.14))
    top_edge_net_reject_max = float(vision_cfg.get("top_edge_net_reject_max", 0.18))
    court_corners: Optional[List[List[float]]] = None
    court_detection: Dict[str, Any] = {"source": "none", "confidence": None, "reason": None}
    manual_corners = vision_cfg.get("court_corners")
    if isinstance(manual_corners, (list, tuple)) and len(manual_corners) == 4:
        try:
            court_corners = [[float(x), float(y)] for x, y in manual_corners]  # type: ignore[misc]
            court_detection = {"source": "manual", "confidence": 1.0, "reason": "OK"}
        except Exception:
            court_corners = None
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)

    player_det = PlayerDetector(
        model_path=player_model_path,
        device=yolo_device,
        conf=yolo_conf,
        imgsz=yolo_imgsz,
    )
    ball_det = BallDetector(
        model_path=ball_model_path,
        device=yolo_device,
        conf=yolo_conf,
        imgsz=yolo_imgsz,
        allow_all_when_empty=vision_cfg.get("ball_allow_all_if_empty", False),
    )

    player_tracker = SimpleByteTrack(
        iou_thresh=tracking_cfg.get("iou_thresh", 0.3),
        max_time_since_update=tracking_cfg.get("max_age", 30),
    )
    ball_tracker = SingleBallTracker(
        iou_thresh=tracking_cfg.get("iou_thresh", 0.3),
        ema_alpha=0.6,
        max_age=tracking_cfg.get("ball_max_age", 5),
    )
    detect_court_corners = bool(vision_cfg.get("detect_court_corners", True))
    court_method = str(vision_cfg.get("court_detector_method", "heuristic")).lower()
    use_model_fit = court_method in ("model_fit", "model_fit_bwf")
    model_fit_method = vision_cfg.get("model_fit_method", "lsd")
    allow_fallback_lsd = vision_cfg.get("model_fit_allow_fallback_lsd", True)
    court_detect_once = bool(vision_cfg.get("court_detect_once", use_model_fit))
    court_detect_frame_idx = int(vision_cfg.get("court_detect_frame_idx", 0))
    print("[CFG] court_detector_method =", vision_cfg.get("court_detector_method"))
    print("[CFG] model_fit_method      =", vision_cfg.get("model_fit_method"))
    print("[CFG] allow_fallback_lsd    =", vision_cfg.get("model_fit_allow_fallback_lsd", True))
    print("[CFG] court_detect_once     =", court_detect_once)
    print("[CFG] court_detect_frame_idx=", court_detect_frame_idx)
    court_detector = (
        CourtDetector(
            edge_top_min=edge_top_min,
            edge_bottom_min=edge_bottom_min,
            edge_min_floor=edge_min_floor,
            net_suppress_y_min=net_suppress_y_min,
            net_suppress_y_max=net_suppress_y_max,
            tpl_net_reject_max=tpl_net_reject_max,
            top_edge_net_reject_max=top_edge_net_reject_max,
        )
        if (detect_court_corners or use_court_roi) and not use_model_fit
        else None
    )

    # Pose settings
    enable_pose = pose_cfg.get("enable", False)
    pose_stride = pose_cfg.get("stride", 2)
    pose_backend = pose_cfg.get("backend", "mediapipe")
    pose_estimator: PoseEstimator | None = None
    if enable_pose:
        try:
            pose_estimator = PoseEstimator()  # currently only mediapipe backend
        except Exception as exc:  # pragma: no cover
            print(f"[WARN] PoseEstimator init failed: {exc}")
            pose_estimator = None
            enable_pose = False

    frame_idx = 0
    frame_results: List[FrameResult] = []
    pose_results: Dict[int, Dict] = {}

    # Court corners / ROI from the first frame.
    court_roi = None
    default_corners = None
    ok, first_bgr = cap.read()
    if ok:
        h0, w0 = first_bgr.shape[:2]
        first_rgb = cv2.cvtColor(first_bgr, cv2.COLOR_BGR2RGB)
        # Default full-frame corners: LB, RB, RT, LT
        default_corners = [
            [0.0, float(h0 - 1)],
            [float(w0 - 1), float(h0 - 1)],
            [float(w0 - 1), 0.0],
            [0.0, 0.0],
        ]

        if court_corners is None and (detect_court_corners or use_court_roi):
            # Try multiple early frames and keep the best-scoring detection.
            # When single-frame lock is enabled, optionally fallback-scan if the chosen frame is weak.
            samples = int(vision_cfg.get("court_detect_samples", 10))
            stride = int(vision_cfg.get("court_detect_stride", 5))
            samples = max(1, samples)
            stride = max(1, stride)
            if court_detect_once:
                samples = 1

            detect_mode = "single_frame" if court_detect_once else "best_of_samples"
            detect_fallback_on_fail = bool(vision_cfg.get("court_detect_fallback_on_fail", True))
            detect_fallback_samples = int(vision_cfg.get("court_detect_fallback_samples", 24))
            detect_fallback_stride = int(vision_cfg.get("court_detect_fallback_stride", 10))
            detect_fallback_min_conf = float(vision_cfg.get("court_detect_fallback_min_conf", 0.45))
            detect_fallback_min_score = float(vision_cfg.get("court_detect_fallback_min_score", 120.0))
            detect_precision_max_p90 = float(vision_cfg.get("court_detect_precision_max_p90", 18.0))
            detect_precision_max_mean = float(vision_cfg.get("court_detect_precision_max_mean", 4.5))
            detect_precision_joint_mean_min = float(
                vision_cfg.get("court_detect_precision_joint_mean_min", 0.0)
            )
            detect_precision_joint_p90_min = float(vision_cfg.get("court_detect_precision_joint_p90_min", 0.0))
            detect_precision_max_far_gap_err = float(
                vision_cfg.get("court_detect_precision_max_far_gap_err_m", 0.25)
            )
            detect_fallback_samples = max(2, detect_fallback_samples)
            detect_fallback_stride = max(1, detect_fallback_stride)
            detect_temporal_refine = bool(vision_cfg.get("court_detect_temporal_refine", True))
            detect_temporal_refine_topk = max(2, int(vision_cfg.get("court_detect_temporal_refine_topk", 5)))
            detect_temporal_refine_drift_ref_px = float(
                vision_cfg.get("court_detect_temporal_refine_drift_ref_px", 18.0)
            )
            detect_temporal_refine_weight = float(vision_cfg.get("court_detect_temporal_refine_weight", 30.0))
            detect_hybrid_modern = bool(vision_cfg.get("court_detect_hybrid_modern", False))
            detect_hybrid_modern = bool(detect_hybrid_modern and fit_court_homography_modern is not None)
            detect_hybrid_always_scan = bool(vision_cfg.get("court_detect_hybrid_always_scan", detect_hybrid_modern))
            sample_frame_indices = (
                [max(0, int(court_detect_frame_idx))]
                if court_detect_once
                else [int(i * stride) for i in range(samples)]
            )
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            max_frame_idx = max(0, total_frames - 1) if total_frames > 0 else None

            def _clip_unique_indices(indices: List[int]) -> List[int]:
                out: List[int] = []
                seen = set()
                for fi_raw in indices:
                    fi = int(fi_raw)
                    if max_frame_idx is not None:
                        fi = max(0, min(fi, max_frame_idx))
                    else:
                        fi = max(0, fi)
                    if fi in seen:
                        continue
                    seen.add(fi)
                    out.append(fi)
                return out

            sample_frame_indices = _clip_unique_indices(sample_frame_indices)

            def _candidate_score(conf: float, reason: str, metrics: Optional[Dict[str, Any]]) -> float:
                m = metrics if isinstance(metrics, dict) else {}

                def _num(value: Any) -> Optional[float]:
                    try:
                        if value is None:
                            return None
                        out = float(value)
                    except (TypeError, ValueError):
                        return None
                    if not np.isfinite(out):
                        return None
                    return out

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

                # Penalize structurally unstable hypotheses.
                score -= 2.0 * float(m.get("reject_aspect_ratio", 0) or 0)
                score -= 0.8 * float(m.get("reject_area_ratio_low", 0) or 0)
                score -= 0.6 * float(m.get("reject_degenerate_min_edge", 0) or 0)

                p90_px = _num(m.get("p90_dist_px") or m.get("sample_dist_p90"))
                mean_px = _num(m.get("mean_dist_px") or m.get("sample_dist_mean"))
                cost_total = _num(m.get("cost_total"))
                area_ratio = _num(
                    m.get("selected_area_ratio_guard")
                    or m.get("area_ratio")
                    or m.get("raw_floor_selected_area_ratio")
                )
                if p90_px is not None:
                    score -= min(70.0, 2.0 * max(0.0, p90_px - 12.0))
                if mean_px is not None:
                    score -= min(35.0, 4.0 * max(0.0, mean_px - 4.0))
                if cost_total is not None:
                    score -= min(45.0, 0.45 * max(0.0, cost_total - 18.0))
                if area_ratio is not None and area_ratio < 0.18:
                    score -= min(80.0, 160.0 * (0.18 - area_ratio))
                if area_ratio is not None and area_ratio > 0.0:
                    score += min(28.0, 36.0 * area_ratio)
                return float(score)

            def _candidate_good(meta: Dict[str, Any]) -> bool:
                conf_i = float(meta.get("confidence", 0.0) or 0.0)
                score_i = float(meta.get("candidate_score", -1e9) or -1e9)
                reason_i = str(meta.get("reason", ""))
                metrics_i = meta.get("metrics") if isinstance(meta.get("metrics"), dict) else {}

                def _num_good(value: Any) -> Optional[float]:
                    try:
                        if value is None:
                            return None
                        out = float(value)
                    except (TypeError, ValueError):
                        return None
                    if not np.isfinite(out):
                        return None
                    return out

                p90_px = _num_good(metrics_i.get("p90_dist_px") or metrics_i.get("sample_dist_p90"))
                mean_px = _num_good(metrics_i.get("mean_dist_px") or metrics_i.get("sample_dist_mean"))
                if detect_precision_max_p90 > 0.0 and p90_px is not None and p90_px > detect_precision_max_p90:
                    return False
                if detect_precision_max_mean > 0.0 and mean_px is not None and mean_px > detect_precision_max_mean:
                    return False
                if (
                    detect_precision_joint_mean_min > 0.0
                    and detect_precision_joint_p90_min > 0.0
                    and mean_px is not None
                    and p90_px is not None
                    and mean_px > detect_precision_joint_mean_min
                    and p90_px > detect_precision_joint_p90_min
                ):
                    return False

                semantic_meta = None
                legacy_refine_meta = metrics_i.get("legacy_semantic_refine_meta")
                if isinstance(legacy_refine_meta, dict) and isinstance(legacy_refine_meta.get("semantic_meta"), dict):
                    semantic_meta = legacy_refine_meta.get("semantic_meta")
                elif isinstance(metrics_i.get("semantic_refine_meta"), dict):
                    semantic_meta = metrics_i.get("semantic_refine_meta")
                if isinstance(semantic_meta, dict) and detect_precision_max_far_gap_err > 0.0:
                    far_gap_err = _num_good(semantic_meta.get("far_pair_gap_err_m"))
                    if (
                        far_gap_err is not None
                        and bool(semantic_meta.get("far_observable", False))
                        and bool(semantic_meta.get("far_pair_found", False))
                        and far_gap_err > detect_precision_max_far_gap_err
                    ):
                        return False

                return bool(reason_i == "OK" and conf_i >= detect_fallback_min_conf and score_i >= detect_fallback_min_score)

            best = None  # (score, confidence, corners, meta, fit)
            best_fail = None  # (score, confidence, meta, fit)
            checked_indices: List[int] = []
            fallback_used = False
            candidate_by_frame: Dict[int, Dict[str, Any]] = {}

            def _maybe_update_best(
                score_i: float,
                conf_i: float,
                corners_i: Optional[List[List[float]]],
                meta_i: Dict[str, Any],
                fit_i: Any,
            ) -> None:
                nonlocal best, best_fail
                if corners_i is None:
                    if best_fail is None or score_i > float(best_fail[0]) or (
                        score_i == float(best_fail[0]) and conf_i > float(best_fail[1])
                    ):
                        best_fail = (score_i, conf_i, meta_i, fit_i)
                    return
                if not _candidate_good(meta_i):
                    if best_fail is None or score_i > float(best_fail[0]) or (
                        score_i == float(best_fail[0]) and conf_i > float(best_fail[1])
                    ):
                        best_fail = (score_i, conf_i, meta_i, fit_i)
                    return
                if best is None or score_i > float(best[0]) or (score_i == float(best[0]) and conf_i > float(best[1])):
                    best = (score_i, conf_i, corners_i, meta_i, fit_i)
                fi = int(meta_i.get("frame_idx", -1))
                if fi >= 0:
                    prev = candidate_by_frame.get(fi)
                    if prev is None or score_i > float(prev.get("score", -1e9)):
                        candidate_by_frame[fi] = {
                            "frame_idx": fi,
                            "score": float(score_i),
                            "confidence": float(conf_i),
                            "corners": corners_i,
                            "meta": dict(meta_i),
                            "fit": fit_i,
                        }

            def _mean_corner_dist(c0: Any, c1: Any) -> Optional[float]:
                try:
                    a = np.asarray(c0, dtype=np.float32).reshape(4, 2)
                    b = np.asarray(c1, dtype=np.float32).reshape(4, 2)
                    d = np.linalg.norm(a - b, axis=1)
                    if d.size == 0:
                        return None
                    return float(np.mean(d))
                except Exception:
                    return None

            def _run_model_fit(
                bgr_i: np.ndarray,
                *,
                use_profile_retry: bool = False,
                backend: str = "legacy",
            ) -> Any:
                fit_fn = fit_court_homography
                if backend == "modern":
                    if fit_court_homography_modern is None:
                        raise RuntimeError("modern court-fit backend is not available")
                    fit_fn = fit_court_homography_modern
                if not use_profile_retry:
                    return fit_fn(
                        bgr_i,
                        method=model_fit_method,
                        allow_fallback_lsd=allow_fallback_lsd,
                    )
                badc_snapshot = {k: v for k, v in os.environ.items() if k.startswith("BADC_")}
                try:
                    for key in list(os.environ):
                        if key.startswith("BADC_"):
                            os.environ.pop(key, None)
                    os.environ.update(profile_retry_overrides)
                    return fit_fn(
                        bgr_i,
                        method=model_fit_method,
                        allow_fallback_lsd=allow_fallback_lsd,
                    )
                finally:
                    for key in list(os.environ):
                        if key.startswith("BADC_"):
                            os.environ.pop(key, None)
                    os.environ.update(badc_snapshot)

            def _submit_model_fit(
                fi: int,
                fit_i: Any,
                *,
                profile_retry_used: bool = False,
                backend: str = "legacy",
            ) -> Optional[List[List[float]]]:
                label_bits = ["FIT"]
                if backend != "legacy":
                    label_bits.append(backend)
                if profile_retry_used:
                    label_bits.append("profile_retry")
                label = "[" + ":".join(label_bits) + "]"
                print(label, "method_used =", fit_i.method_used, "reason =", fit_i.reason)
                metrics_i = fit_i.metrics if isinstance(fit_i.metrics, dict) else {}
                metrics_i = dict(metrics_i)
                metrics_i["fit_backend"] = str(backend)
                if profile_retry_used:
                    metrics_i["profile_retry_used"] = True
                    metrics_i["profile_retry_path"] = profile_retry_path
                fit_i.metrics = metrics_i
                print(
                    label,
                    "raw_full_ratio=",
                    metrics_i.get("white_mask_raw_full_ratio"),
                    "raw_floor_ratio=",
                    metrics_i.get("white_mask_raw_floor_ratio"),
                    "clean_ratio=",
                    metrics_i.get("white_mask_clean_ratio"),
                    "floor_roi_ratio=",
                    metrics_i.get("floor_roi_ratio"),
                    "floor_bbox=",
                    metrics_i.get("floor_bbox"),
                    "fallback_floor_roi=",
                    metrics_i.get("fallback_floor_roi"),
                    "fallback_mode=",
                    metrics_i.get("fallback_mode"),
                    "floor_y_cut=",
                    metrics_i.get("floor_y_cut"),
                )
                conf_i = float(fit_i.confidence)
                reason_i = str(fit_i.reason)
                score_i = _candidate_score(conf_i, reason_i, metrics_i)
                meta_i = {
                    "source": "auto",
                    "confidence": conf_i,
                    "reason": reason_i,
                    "frame_idx": int(fi),
                    "detect_mode": detect_mode,
                    "lock_enabled": bool(court_detect_once),
                    "metrics": fit_i.metrics,
                    "candidate_score": float(score_i),
                    "fit_backend": str(backend),
                }
                if profile_retry_used:
                    meta_i["profile_retry_used"] = True
                    meta_i["profile_retry_path"] = profile_retry_path
                corners_i = fit_i.corners.tolist() if fit_i.corners is not None else None
                _maybe_update_best(score_i, conf_i, corners_i, meta_i, fit_i)
                return corners_i

            def _eval_candidates(indices: List[int]) -> None:
                for fi in indices:
                    if fi in checked_indices:
                        continue
                    checked_indices.append(int(fi))
                    if fi == 0:
                        bgr_i = first_bgr
                    else:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
                        ok_i, bgr_i = cap.read()
                        if not ok_i or bgr_i is None:
                            continue
                    if use_model_fit:
                        fit_i = _run_model_fit(bgr_i, backend="legacy")
                        corners_i = _submit_model_fit(int(fi), fit_i, backend="legacy")
                        if detect_hybrid_modern:
                            modern_fit = _run_model_fit(bgr_i, backend="modern")
                            modern_corners_i = _submit_model_fit(int(fi), modern_fit, backend="modern")
                            if modern_corners_i is not None:
                                corners_i = modern_corners_i
                        if (
                            corners_i is None
                            and profile_retry_enabled
                            and profile_retry_overrides
                            and profile_retry_path
                        ):
                            retry_fit = _run_model_fit(bgr_i, use_profile_retry=True, backend="legacy")
                            retry_corners_i = _submit_model_fit(
                                int(fi),
                                retry_fit,
                                profile_retry_used=True,
                                backend="legacy",
                            )
                            if detect_hybrid_modern:
                                modern_retry_fit = _run_model_fit(bgr_i, use_profile_retry=True, backend="modern")
                                modern_retry_corners_i = _submit_model_fit(
                                    int(fi),
                                    modern_retry_fit,
                                    profile_retry_used=True,
                                    backend="modern",
                                )
                                if modern_retry_corners_i is not None:
                                    retry_corners_i = modern_retry_corners_i
                    else:
                        rgb = cv2.cvtColor(bgr_i, cv2.COLOR_BGR2RGB)
                        lines = court_detector.detect_court(rgb) if court_detector is not None else None
                        conf_i = float(court_detector.last_confidence or 0.0) if court_detector is not None else 0.0
                        reason_i = str(court_detector.last_reason or "unknown") if court_detector is not None else "unknown"
                        edge_support_i = (
                            [float(v) for v in court_detector.last_edge_support]
                            if court_detector is not None and isinstance(court_detector.last_edge_support, list)
                            else None
                        )
                        metrics_i = (
                            court_detector.last_metrics
                            if court_detector is not None and isinstance(court_detector.last_metrics, dict)
                            else None
                        )
                        score_i = _candidate_score(conf_i, reason_i, metrics_i if isinstance(metrics_i, dict) else None)
                        meta_i = {
                            "source": "auto",
                            "confidence": conf_i,
                            "reason": reason_i,
                            "frame_idx": int(fi),
                            "detect_mode": detect_mode,
                            "lock_enabled": bool(court_detect_once),
                            "edge_support": edge_support_i,
                            "metrics": metrics_i,
                            "candidate_score": float(score_i),
                        }
                        corners_i = lines.corners.tolist() if (lines is not None and lines.corners is not None) else None
                        _maybe_update_best(score_i, conf_i, corners_i, meta_i, None)

            _eval_candidates(sample_frame_indices)

            # Single-frame mode fallback: if target frame is weak, scan more frames and pick the best.
            if court_detect_once and detect_fallback_on_fail:
                best_meta = best[3] if best is not None else None
                if detect_hybrid_always_scan or best_meta is None or not _candidate_good(best_meta):
                    fallback_indices: List[int] = [0]
                    fallback_indices.extend([int(i * detect_fallback_stride) for i in range(detect_fallback_samples)])
                    swing = max(2, detect_fallback_samples // 4)
                    for d in range(1, swing + 1):
                        fallback_indices.append(int(court_detect_frame_idx - d * detect_fallback_stride))
                        fallback_indices.append(int(court_detect_frame_idx + d * detect_fallback_stride))
                    fallback_indices = _clip_unique_indices(fallback_indices)
                    fallback_indices = [fi for fi in fallback_indices if fi not in checked_indices]
                    if fallback_indices:
                        fallback_used = True
                        detect_mode = "single_frame_with_fallback"
                        _eval_candidates(fallback_indices)

            if best is not None and detect_temporal_refine and len(candidate_by_frame) > 1:
                candidates = list(candidate_by_frame.values())
                candidates = [c for c in candidates if c.get("corners") is not None]
                if len(candidates) > 1:
                    candidates.sort(
                        key=lambda c: (float(c.get("score", -1e9)), float(c.get("confidence", 0.0))),
                        reverse=True,
                    )
                    top = candidates[: int(max(2, detect_temporal_refine_topk))]
                    best_refined = None
                    for cand in top:
                        dists: List[float] = []
                        for peer in top:
                            if int(peer.get("frame_idx", -1)) == int(cand.get("frame_idx", -2)):
                                continue
                            md = _mean_corner_dist(cand.get("corners"), peer.get("corners"))
                            if md is not None:
                                dists.append(float(md))
                        mean_dist = float(np.mean(dists)) if dists else None
                        consensus = (
                            float(1.0 / (1.0 + mean_dist / max(1.0, detect_temporal_refine_drift_ref_px)))
                            if mean_dist is not None
                            else 0.0
                        )
                        refined_score = float(cand.get("score", -1e9)) + float(detect_temporal_refine_weight) * float(
                            consensus
                        )
                        meta_c = dict(cand.get("meta", {}))
                        meta_c["temporal_refine_enabled"] = True
                        meta_c["temporal_refine_topk"] = int(len(top))
                        meta_c["temporal_refine_consensus"] = float(consensus)
                        meta_c["temporal_refine_mean_corner_dist_px"] = mean_dist
                        meta_c["temporal_refine_refined_score"] = float(refined_score)
                        cand["meta"] = meta_c
                        cand["refined_score"] = refined_score
                        if best_refined is None:
                            best_refined = cand
                        else:
                            if refined_score > float(best_refined.get("refined_score", -1e9)):
                                best_refined = cand
                            elif refined_score == float(best_refined.get("refined_score", -1e9)) and float(
                                cand.get("confidence", 0.0)
                            ) > float(best_refined.get("confidence", 0.0)):
                                best_refined = cand
                    if best_refined is not None:
                        best = (
                            float(best_refined.get("score", -1e9)),
                            float(best_refined.get("confidence", 0.0)),
                            best_refined.get("corners"),
                            dict(best_refined.get("meta", {})),
                            best_refined.get("fit"),
                        )

            if best is not None and not _candidate_good(best[3]):
                score_i, conf_i, _corners_i, meta_i, fit_i = best
                meta_i = dict(meta_i)
                original_reason = str(meta_i.get("reason", ""))
                meta_i["fit_reason"] = original_reason
                meta_i["reason"] = "R_candidate_below_threshold"
                meta_i["final_reject_min_conf"] = float(detect_fallback_min_conf)
                meta_i["final_reject_min_score"] = float(detect_fallback_min_score)
                metrics_i = meta_i.get("metrics")
                if isinstance(metrics_i, dict):
                    metrics_i["final_reject_reason"] = "R_candidate_below_threshold"
                    metrics_i["final_reject_fit_reason"] = original_reason
                    metrics_i["final_reject_min_conf"] = float(detect_fallback_min_conf)
                    metrics_i["final_reject_min_score"] = float(detect_fallback_min_score)
                if best_fail is None or score_i > float(best_fail[0]) or (
                    score_i == float(best_fail[0]) and conf_i > float(best_fail[1])
                ):
                    best_fail = (score_i, conf_i, meta_i, fit_i)
                best = None

            if best is not None:
                _score, conf, corners_i, meta_i, fit = best
                meta_i = dict(meta_i)
                meta_i["detect_mode"] = detect_mode
                meta_i["lock_enabled"] = bool(court_detect_once)
                meta_i["sample_frame_indices"] = [int(v) for v in checked_indices]
                meta_i["sample_frame_count"] = int(len(checked_indices))
                meta_i["fallback_used"] = bool(fallback_used)
                meta_i.setdefault("temporal_refine_enabled", bool(detect_temporal_refine))
                if use_model_fit:
                    debug_dir = vision_cfg.get("model_fit_debug_dir", None)
                    debug_path = vision_cfg.get("model_fit_debug_path", None)
                    if debug_path:
                        path = Path(debug_path)
                    else:
                        if debug_dir is None:
                            debug_dir = Path("reports") / "demo_friend"
                        debug_dir = Path(debug_dir)
                        debug_dir.mkdir(parents=True, exist_ok=True)
                        path = debug_dir / f"{Path(video_path).stem}_court_fit_debug.jpg"
                    metrics = meta_i.get("metrics") if isinstance(meta_i, dict) else None
                    if isinstance(metrics, dict) and fit is not None:
                        if isinstance(fit.debug_image, np.ndarray):
                            cv2.imwrite(str(path), fit.debug_image)
                            metrics["debug_image_path"] = str(path)
                        if isinstance(fit.debug_image_init, np.ndarray):
                            init_path = path.with_name(f"{Path(video_path).stem}_court_fit_init_overlay.jpg")
                            cv2.imwrite(str(init_path), fit.debug_image_init)
                            metrics["debug_init_path"] = str(init_path)
                        raw_full = fit.white_mask_raw_full if isinstance(fit.white_mask_raw_full, np.ndarray) else fit.white_mask_raw
                        if isinstance(raw_full, np.ndarray):
                            raw_full_path = path.with_name(f"{Path(video_path).stem}_court_fit_white_mask_raw_full.png")
                            cv2.imwrite(str(raw_full_path), raw_full)
                            metrics["white_mask_raw_full_path"] = str(raw_full_path)
                            metrics["white_mask_raw_path"] = str(raw_full_path)
                        if isinstance(fit.white_mask_raw_floor, np.ndarray):
                            raw_floor_path = path.with_name(f"{Path(video_path).stem}_court_fit_white_mask_raw_floor.png")
                            cv2.imwrite(str(raw_floor_path), fit.white_mask_raw_floor)
                            metrics["white_mask_raw_floor_path"] = str(raw_floor_path)
                        if isinstance(fit.white_mask_raw_floor_preblob, np.ndarray):
                            preblob_path = path.with_name(
                                f"{Path(video_path).stem}_court_fit_white_mask_raw_floor_preblob.png"
                            )
                            cv2.imwrite(str(preblob_path), fit.white_mask_raw_floor_preblob)
                            metrics["white_mask_raw_floor_preblob_path"] = str(preblob_path)
                        if isinstance(fit.white_mask_raw_floor_postblob, np.ndarray):
                            postblob_path = path.with_name(
                                f"{Path(video_path).stem}_court_fit_white_mask_raw_floor_postblob.png"
                            )
                            cv2.imwrite(str(postblob_path), fit.white_mask_raw_floor_postblob)
                            metrics["white_mask_raw_floor_postblob_path"] = str(postblob_path)
                        if isinstance(fit.white_mask_raw_floor_noblob, np.ndarray):
                            noblob_path = path.with_name(
                                f"{Path(video_path).stem}_court_fit_white_mask_raw_floor_noblob.png"
                            )
                            cv2.imwrite(str(noblob_path), fit.white_mask_raw_floor_noblob)
                            metrics["white_mask_raw_floor_noblob_path"] = str(noblob_path)
                        if isinstance(fit.floor_roi_mask, np.ndarray):
                            floor_roi_path = path.with_name(f"{Path(video_path).stem}_court_fit_floor_roi.png")
                            cv2.imwrite(str(floor_roi_path), fit.floor_roi_mask)
                            metrics["floor_roi_path"] = str(floor_roi_path)
                        if isinstance(fit.floor_roi_overlay, np.ndarray):
                            floor_roi_overlay_path = path.with_name(
                                f"{Path(video_path).stem}_court_fit_floor_roi_overlay.jpg"
                            )
                            cv2.imwrite(str(floor_roi_overlay_path), fit.floor_roi_overlay)
                            metrics["floor_roi_overlay_path"] = str(floor_roi_overlay_path)
                        if isinstance(fit.seed_bottom_mask, np.ndarray):
                            seed_path = path.with_name(f"{Path(video_path).stem}_court_fit_seed_bottom_mask.png")
                            cv2.imwrite(str(seed_path), fit.seed_bottom_mask)
                            metrics["seed_bottom_path"] = str(seed_path)
                        if isinstance(fit.green_mask, np.ndarray):
                            green_path = path.with_name(f"{Path(video_path).stem}_court_fit_green_mask.png")
                            cv2.imwrite(str(green_path), fit.green_mask)
                            metrics["green_mask_path"] = str(green_path)
                        if isinstance(fit.largest_cc_mask, np.ndarray):
                            cc_path = path.with_name(f"{Path(video_path).stem}_court_fit_largest_cc_mask.png")
                            cv2.imwrite(str(cc_path), fit.largest_cc_mask)
                            metrics["largest_cc_path"] = str(cc_path)
                        if isinstance(fit.exg_row_plot, np.ndarray):
                            exg_path = path.with_name(f"{Path(video_path).stem}_court_fit_exg_row.png")
                            cv2.imwrite(str(exg_path), fit.exg_row_plot)
                            metrics["exg_row_path"] = str(exg_path)
                        if isinstance(fit.white_mask_clean, np.ndarray):
                            clean_path = path.with_name(f"{Path(video_path).stem}_court_fit_white_mask_clean.png")
                            cv2.imwrite(str(clean_path), fit.white_mask_clean)
                            metrics["white_mask_clean_path"] = str(clean_path)
                            metrics["white_mask_path"] = str(clean_path)
                        if isinstance(fit.lsd_lines_a, np.ndarray):
                            lsd_path = path.with_name(f"{Path(video_path).stem}_court_fit_lsd_dirA.png")
                            cv2.imwrite(str(lsd_path), fit.lsd_lines_a)
                            metrics["lsd_dirA_path"] = str(lsd_path)
                        if isinstance(fit.lsd_lines_b, np.ndarray):
                            lsd_path = path.with_name(f"{Path(video_path).stem}_court_fit_lsd_dirB.png")
                            cv2.imwrite(str(lsd_path), fit.lsd_lines_b)
                            metrics["lsd_dirB_path"] = str(lsd_path)
                        if isinstance(fit.dt_debug, np.ndarray):
                            dt_path = path.with_name(f"{Path(video_path).stem}_court_fit_dt.png")
                            cv2.imwrite(str(dt_path), fit.dt_debug)
                            metrics["dt_debug_path"] = str(dt_path)
                        if isinstance(fit.ori_mask_a, np.ndarray):
                            ori_path = path.with_name(f"{Path(video_path).stem}_court_fit_ori_maskA.png")
                            cv2.imwrite(str(ori_path), fit.ori_mask_a)
                            metrics["ori_maskA_path"] = str(ori_path)
                        if isinstance(fit.ori_mask_b, np.ndarray):
                            ori_path = path.with_name(f"{Path(video_path).stem}_court_fit_ori_maskB.png")
                            cv2.imwrite(str(ori_path), fit.ori_mask_b)
                            metrics["ori_maskB_path"] = str(ori_path)
                        if isinstance(fit.hough_lines_img, np.ndarray):
                            hough_path = path.with_name(f"{Path(video_path).stem}_court_fit_hough_lines.png")
                            cv2.imwrite(str(hough_path), fit.hough_lines_img)
                            metrics["hough_lines_path"] = str(hough_path)
                        if isinstance(fit.raw_floor_hough_lines_img, np.ndarray):
                            hough_path = path.with_name(f"{Path(video_path).stem}_court_fit_raw_floor_hough_lines.png")
                            cv2.imwrite(str(hough_path), fit.raw_floor_hough_lines_img)
                            metrics["raw_floor_hough_lines_path"] = str(hough_path)
                        if isinstance(fit.raw_floor_hough_lines_a, np.ndarray):
                            hough_path = path.with_name(f"{Path(video_path).stem}_court_fit_raw_floor_hough_lines_A.png")
                            cv2.imwrite(str(hough_path), fit.raw_floor_hough_lines_a)
                            metrics["raw_floor_hough_lines_a_path"] = str(hough_path)
                        if isinstance(fit.raw_floor_hough_lines_b, np.ndarray):
                            hough_path = path.with_name(f"{Path(video_path).stem}_court_fit_raw_floor_hough_lines_B.png")
                            cv2.imwrite(str(hough_path), fit.raw_floor_hough_lines_b)
                            metrics["raw_floor_hough_lines_b_path"] = str(hough_path)
                        if isinstance(fit.raw_floor_dt_debug, np.ndarray):
                            dt_path = path.with_name(f"{Path(video_path).stem}_court_fit_raw_floor_dt.png")
                            cv2.imwrite(str(dt_path), fit.raw_floor_dt_debug)
                            metrics["raw_floor_dt_path"] = str(dt_path)
                        if isinstance(fit.raw_floor_preprocessed, np.ndarray):
                            prep_path = path.with_name(f"{Path(video_path).stem}_court_fit_raw_floor_preprocessed.png")
                            cv2.imwrite(str(prep_path), fit.raw_floor_preprocessed)
                            metrics["raw_floor_preprocessed_path"] = str(prep_path)
                        if isinstance(fit.raw_floor_edges, np.ndarray):
                            edge_path = path.with_name(f"{Path(video_path).stem}_court_fit_raw_floor_edges.png")
                            cv2.imwrite(str(edge_path), fit.raw_floor_edges)
                            metrics["raw_floor_edges_path"] = str(edge_path)
                        if isinstance(fit.linepix_mask, np.ndarray):
                            lp_path = path.with_name(f"{Path(video_path).stem}_court_fit_linepix.png")
                            cv2.imwrite(str(lp_path), fit.linepix_mask)
                            metrics["linepix_path"] = str(lp_path)
                        if isinstance(fit.linepix_overlay, np.ndarray):
                            lp_path = path.with_name(f"{Path(video_path).stem}_court_fit_linepix_overlay.jpg")
                            cv2.imwrite(str(lp_path), fit.linepix_overlay)
                            metrics["linepix_overlay_path"] = str(lp_path)
                        if isinstance(fit.linepix_mask, np.ndarray):
                            lp_mask_path = path.with_name(f"{Path(video_path).stem}_court_fit_linepix.png")
                            cv2.imwrite(str(lp_mask_path), fit.linepix_mask)
                            metrics["linepix_mask_path"] = str(lp_mask_path)
                        if isinstance(fit.ransac_lines_img, np.ndarray):
                            lp_path = path.with_name(f"{Path(video_path).stem}_court_fit_ransac_lines.jpg")
                            cv2.imwrite(str(lp_path), fit.ransac_lines_img)
                            metrics["ransac_lines_path"] = str(lp_path)
                        if isinstance(fit.raw_floor_top5_overlay, np.ndarray):
                            ov_path = path.with_name(f"{Path(video_path).stem}_court_fit_top5_candidate_overlays.png")
                            cv2.imwrite(str(ov_path), fit.raw_floor_top5_overlay)
                            metrics["raw_floor_top5_overlay_path"] = str(ov_path)
                        if isinstance(fit.raw_floor_model_overlay, np.ndarray):
                            ov_path = path.with_name(f"{Path(video_path).stem}_court_fit_raw_floor_overlay.png")
                            cv2.imwrite(str(ov_path), fit.raw_floor_model_overlay)
                            metrics["raw_floor_overlay_path"] = str(ov_path)
                        if isinstance(fit.frame_model_overlay, np.ndarray):
                            ov_path = path.with_name(f"{Path(video_path).stem}_court_fit_frame_overlay.jpg")
                            cv2.imwrite(str(ov_path), fit.frame_model_overlay)
                            metrics["raw_floor_frame_overlay_path"] = str(ov_path)
                court_corners = corners_i
                court_detection = meta_i
            else:
                # Auto detector rejected all candidates.
                if best_fail is not None:
                    _score, _conf, _meta, fit = best_fail
                    _meta = dict(_meta) if isinstance(_meta, dict) else {}
                    _meta["detect_mode"] = detect_mode
                    _meta["lock_enabled"] = bool(court_detect_once)
                    _meta["sample_frame_indices"] = [int(v) for v in checked_indices]
                    _meta["sample_frame_count"] = int(len(checked_indices))
                    _meta["fallback_used"] = bool(fallback_used)
                    if use_model_fit:
                        debug_dir = vision_cfg.get("model_fit_debug_dir", None)
                        debug_path = vision_cfg.get("model_fit_debug_path", None)
                        if debug_path:
                            path = Path(debug_path)
                        else:
                            if debug_dir is None:
                                debug_dir = Path("reports") / "demo_friend"
                            debug_dir = Path(debug_dir)
                            debug_dir.mkdir(parents=True, exist_ok=True)
                            path = debug_dir / f"{Path(video_path).stem}_court_fit_debug.jpg"
                        metrics = _meta.get("metrics") if isinstance(_meta, dict) else None
                        if isinstance(metrics, dict) and fit is not None:
                            if isinstance(fit.debug_image, np.ndarray):
                                cv2.imwrite(str(path), fit.debug_image)
                                metrics["debug_image_path"] = str(path)
                            if isinstance(fit.debug_image_init, np.ndarray):
                                init_path = path.with_name(f"{Path(video_path).stem}_court_fit_init_overlay.jpg")
                                cv2.imwrite(str(init_path), fit.debug_image_init)
                                metrics["debug_init_path"] = str(init_path)
                            raw_full = fit.white_mask_raw_full if isinstance(fit.white_mask_raw_full, np.ndarray) else fit.white_mask_raw
                            if isinstance(raw_full, np.ndarray):
                                raw_full_path = path.with_name(f"{Path(video_path).stem}_court_fit_white_mask_raw_full.png")
                                cv2.imwrite(str(raw_full_path), raw_full)
                                metrics["white_mask_raw_full_path"] = str(raw_full_path)
                                metrics["white_mask_raw_path"] = str(raw_full_path)
                            if isinstance(fit.white_mask_raw_floor, np.ndarray):
                                raw_floor_path = path.with_name(f"{Path(video_path).stem}_court_fit_white_mask_raw_floor.png")
                                cv2.imwrite(str(raw_floor_path), fit.white_mask_raw_floor)
                                metrics["white_mask_raw_floor_path"] = str(raw_floor_path)
                            if isinstance(fit.white_mask_raw_floor_preblob, np.ndarray):
                                preblob_path = path.with_name(
                                    f"{Path(video_path).stem}_court_fit_white_mask_raw_floor_preblob.png"
                                )
                                cv2.imwrite(str(preblob_path), fit.white_mask_raw_floor_preblob)
                                metrics["white_mask_raw_floor_preblob_path"] = str(preblob_path)
                            if isinstance(fit.white_mask_raw_floor_postblob, np.ndarray):
                                postblob_path = path.with_name(
                                    f"{Path(video_path).stem}_court_fit_white_mask_raw_floor_postblob.png"
                                )
                                cv2.imwrite(str(postblob_path), fit.white_mask_raw_floor_postblob)
                                metrics["white_mask_raw_floor_postblob_path"] = str(postblob_path)
                            if isinstance(fit.white_mask_raw_floor_noblob, np.ndarray):
                                noblob_path = path.with_name(
                                    f"{Path(video_path).stem}_court_fit_white_mask_raw_floor_noblob.png"
                                )
                                cv2.imwrite(str(noblob_path), fit.white_mask_raw_floor_noblob)
                                metrics["white_mask_raw_floor_noblob_path"] = str(noblob_path)
                            if isinstance(fit.floor_roi_mask, np.ndarray):
                                floor_roi_path = path.with_name(f"{Path(video_path).stem}_court_fit_floor_roi.png")
                                cv2.imwrite(str(floor_roi_path), fit.floor_roi_mask)
                                metrics["floor_roi_path"] = str(floor_roi_path)
                            if isinstance(fit.floor_roi_overlay, np.ndarray):
                                floor_roi_overlay_path = path.with_name(
                                    f"{Path(video_path).stem}_court_fit_floor_roi_overlay.jpg"
                                )
                                cv2.imwrite(str(floor_roi_overlay_path), fit.floor_roi_overlay)
                                metrics["floor_roi_overlay_path"] = str(floor_roi_overlay_path)
                            if isinstance(fit.seed_bottom_mask, np.ndarray):
                                seed_path = path.with_name(f"{Path(video_path).stem}_court_fit_seed_bottom_mask.png")
                                cv2.imwrite(str(seed_path), fit.seed_bottom_mask)
                                metrics["seed_bottom_path"] = str(seed_path)
                            if isinstance(fit.green_mask, np.ndarray):
                                green_path = path.with_name(f"{Path(video_path).stem}_court_fit_green_mask.png")
                                cv2.imwrite(str(green_path), fit.green_mask)
                                metrics["green_mask_path"] = str(green_path)
                            if isinstance(fit.largest_cc_mask, np.ndarray):
                                cc_path = path.with_name(f"{Path(video_path).stem}_court_fit_largest_cc_mask.png")
                                cv2.imwrite(str(cc_path), fit.largest_cc_mask)
                                metrics["largest_cc_path"] = str(cc_path)
                            if isinstance(fit.exg_row_plot, np.ndarray):
                                exg_path = path.with_name(f"{Path(video_path).stem}_court_fit_exg_row.png")
                                cv2.imwrite(str(exg_path), fit.exg_row_plot)
                                metrics["exg_row_path"] = str(exg_path)
                            if isinstance(fit.white_mask_clean, np.ndarray):
                                clean_path = path.with_name(f"{Path(video_path).stem}_court_fit_white_mask_clean.png")
                                cv2.imwrite(str(clean_path), fit.white_mask_clean)
                                metrics["white_mask_clean_path"] = str(clean_path)
                                metrics["white_mask_path"] = str(clean_path)
                            if isinstance(fit.lsd_lines_a, np.ndarray):
                                lsd_path = path.with_name(f"{Path(video_path).stem}_court_fit_lsd_dirA.png")
                                cv2.imwrite(str(lsd_path), fit.lsd_lines_a)
                                metrics["lsd_dirA_path"] = str(lsd_path)
                            if isinstance(fit.lsd_lines_b, np.ndarray):
                                lsd_path = path.with_name(f"{Path(video_path).stem}_court_fit_lsd_dirB.png")
                                cv2.imwrite(str(lsd_path), fit.lsd_lines_b)
                                metrics["lsd_dirB_path"] = str(lsd_path)
                            if isinstance(fit.dt_debug, np.ndarray):
                                dt_path = path.with_name(f"{Path(video_path).stem}_court_fit_dt.png")
                                cv2.imwrite(str(dt_path), fit.dt_debug)
                                metrics["dt_debug_path"] = str(dt_path)
                            if isinstance(fit.ori_mask_a, np.ndarray):
                                ori_path = path.with_name(f"{Path(video_path).stem}_court_fit_ori_maskA.png")
                                cv2.imwrite(str(ori_path), fit.ori_mask_a)
                                metrics["ori_maskA_path"] = str(ori_path)
                            if isinstance(fit.ori_mask_b, np.ndarray):
                                ori_path = path.with_name(f"{Path(video_path).stem}_court_fit_ori_maskB.png")
                                cv2.imwrite(str(ori_path), fit.ori_mask_b)
                                metrics["ori_maskB_path"] = str(ori_path)
                            if isinstance(fit.hough_lines_img, np.ndarray):
                                hough_path = path.with_name(f"{Path(video_path).stem}_court_fit_hough_lines.png")
                                cv2.imwrite(str(hough_path), fit.hough_lines_img)
                                metrics["hough_lines_path"] = str(hough_path)
                            if isinstance(fit.raw_floor_hough_lines_img, np.ndarray):
                                hough_path = path.with_name(
                                    f"{Path(video_path).stem}_court_fit_raw_floor_hough_lines.png"
                                )
                                cv2.imwrite(str(hough_path), fit.raw_floor_hough_lines_img)
                                metrics["raw_floor_hough_lines_path"] = str(hough_path)
                            if isinstance(fit.raw_floor_hough_lines_a, np.ndarray):
                                hough_path = path.with_name(
                                    f"{Path(video_path).stem}_court_fit_raw_floor_hough_lines_A.png"
                                )
                                cv2.imwrite(str(hough_path), fit.raw_floor_hough_lines_a)
                                metrics["raw_floor_hough_lines_a_path"] = str(hough_path)
                            if isinstance(fit.raw_floor_hough_lines_b, np.ndarray):
                                hough_path = path.with_name(
                                    f"{Path(video_path).stem}_court_fit_raw_floor_hough_lines_B.png"
                                )
                                cv2.imwrite(str(hough_path), fit.raw_floor_hough_lines_b)
                                metrics["raw_floor_hough_lines_b_path"] = str(hough_path)
                            if isinstance(fit.raw_floor_dt_debug, np.ndarray):
                                dt_path = path.with_name(f"{Path(video_path).stem}_court_fit_raw_floor_dt.png")
                                cv2.imwrite(str(dt_path), fit.raw_floor_dt_debug)
                                metrics["raw_floor_dt_path"] = str(dt_path)
                            if isinstance(fit.raw_floor_preprocessed, np.ndarray):
                                prep_path = path.with_name(f"{Path(video_path).stem}_court_fit_raw_floor_preprocessed.png")
                                cv2.imwrite(str(prep_path), fit.raw_floor_preprocessed)
                                metrics["raw_floor_preprocessed_path"] = str(prep_path)
                            if isinstance(fit.raw_floor_edges, np.ndarray):
                                edge_path = path.with_name(f"{Path(video_path).stem}_court_fit_raw_floor_edges.png")
                                cv2.imwrite(str(edge_path), fit.raw_floor_edges)
                                metrics["raw_floor_edges_path"] = str(edge_path)
                            if isinstance(fit.linepix_mask, np.ndarray):
                                lp_path = path.with_name(f"{Path(video_path).stem}_court_fit_linepix.png")
                                cv2.imwrite(str(lp_path), fit.linepix_mask)
                                metrics["linepix_path"] = str(lp_path)
                            if isinstance(fit.linepix_overlay, np.ndarray):
                                lp_path = path.with_name(f"{Path(video_path).stem}_court_fit_linepix_overlay.jpg")
                                cv2.imwrite(str(lp_path), fit.linepix_overlay)
                                metrics["linepix_overlay_path"] = str(lp_path)
                            if isinstance(fit.linepix_mask, np.ndarray):
                                lp_mask_path = path.with_name(f"{Path(video_path).stem}_court_fit_linepix.png")
                                cv2.imwrite(str(lp_mask_path), fit.linepix_mask)
                                metrics["linepix_mask_path"] = str(lp_mask_path)
                            if isinstance(fit.ransac_lines_img, np.ndarray):
                                lp_path = path.with_name(f"{Path(video_path).stem}_court_fit_ransac_lines.jpg")
                                cv2.imwrite(str(lp_path), fit.ransac_lines_img)
                                metrics["ransac_lines_path"] = str(lp_path)
                            if isinstance(fit.raw_floor_top5_overlay, np.ndarray):
                                ov_path = path.with_name(
                                    f"{Path(video_path).stem}_court_fit_top5_candidate_overlays.png"
                                )
                                cv2.imwrite(str(ov_path), fit.raw_floor_top5_overlay)
                                metrics["raw_floor_top5_overlay_path"] = str(ov_path)
                            if isinstance(fit.raw_floor_model_overlay, np.ndarray):
                                ov_path = path.with_name(f"{Path(video_path).stem}_court_fit_raw_floor_overlay.png")
                                cv2.imwrite(str(ov_path), fit.raw_floor_model_overlay)
                                metrics["raw_floor_overlay_path"] = str(ov_path)
                            if isinstance(fit.frame_model_overlay, np.ndarray):
                                ov_path = path.with_name(f"{Path(video_path).stem}_court_fit_frame_overlay.jpg")
                                cv2.imwrite(str(ov_path), fit.frame_model_overlay)
                                metrics["raw_floor_frame_overlay_path"] = str(ov_path)
                    court_detection = {**_meta, "source": "auto_failed"}
                else:
                    court_detection = {"source": "auto_failed", "confidence": None, "reason": "no_valid_candidate"}

        if court_corners is not None and use_court_roi:
            xs = np.array([p[0] for p in court_corners], dtype=np.float32)
            ys = np.array([p[1] for p in court_corners], dtype=np.float32)
            pad_x = float(w0) * float(court_margin)
            pad_y = float(h0) * float(court_margin)
            x1 = max(0, int(round(float(xs.min() - pad_x))))
            y1 = max(0, int(round(float(ys.min() - pad_y))))
            x2 = min(w0, int(round(float(xs.max() + pad_x))))
            y2 = min(h0, int(round(float(ys.max() + pad_y))))
            if x2 > x1 and y2 > y1:
                court_roi = (x1, y1, x2, y2)

        if use_court_roi and court_roi is None:
            x1 = int(round(w0 * float(court_margin)))
            y1 = int(round(h0 * float(court_margin)))
            x2 = int(round(w0 * (1.0 - float(court_margin))))
            y2 = int(round(h0 * (1.0 - float(court_margin))))
            if x2 > x1 and y2 > y1:
                court_roi = (x1, y1, x2, y2)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    batch_frames: list[np.ndarray] = []
    batch_indices: list[int] = []
    last_ball_state: BallTrackState | None = None
    roi_dx = int(court_roi[0]) if court_roi is not None else 0
    roi_dy = int(court_roi[1]) if court_roi is not None else 0
    court_poly = None
    poly_src = court_corners if (court_corners is not None and len(court_corners) == 4) else default_corners
    if poly_src is not None and len(poly_src) == 4:
        court_poly = np.array(poly_src, dtype=np.float32).reshape(-1, 1, 2)

    court_h = None
    court_half_len = None
    # Only build homography if we have real court corners (manual or validated auto detection).
    if court_corners is not None and len(court_corners) == 4:
        try:
            from src.geometry.homography import COURT_LENGTH_M, CourtHomography

            court_h = CourtHomography.from_corners(court_corners)
            court_half_len = float(COURT_LENGTH_M) / 2.0
        except Exception:
            court_h = None
            court_half_len = None

    def _filter_player_detections_to_court(dets: list, min_keep: int = 1):
        if court_poly is None or not dets:
            return dets
        kept = []
        for det in dets:
            x1, y1, x2, y2 = det.bbox
            px = float((x1 + x2) / 2.0)
            py = float(y2)  # bottom-center is a better proxy for "on court"
            if cv2.pointPolygonTest(court_poly, (px, py), False) >= 0:
                kept.append(det)
        # If filtering removes everything (e.g., failed court detection), keep original.
        return kept if len(kept) >= min_keep else dets

    def _roles_for_tracks(tracks: List[Track]) -> Dict[int, str]:
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

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, _ = frame_rgb.shape

        if court_roi is not None:
            x1, y1, x2, y2 = court_roi
            frame_proc = frame_rgb[y1:y2, x1:x2]
        else:
            frame_proc = frame_rgb

        if frame_idx % detect_stride == 0:
            batch_frames.append(frame_proc)
            batch_indices.append(frame_idx)

            if len(batch_frames) >= batch_size:
                player_dets_batch = player_det.detect(batch_frames)
                exclude_bboxes_batch = [[d.bbox for d in pdets] for pdets in player_dets_batch]
                ball_dets_batch = ball_det.detect(batch_frames, exclude_bboxes_batch=exclude_bboxes_batch)
                for fi, pdets, bdets, fproc in zip(batch_indices, player_dets_batch, ball_dets_batch, batch_frames):
                    if roi_dx != 0 or roi_dy != 0:
                        for det in pdets:
                            det.bbox = det.bbox + np.array([roi_dx, roi_dy, roi_dx, roi_dy], dtype=float)
                        for det in bdets:
                            det.bbox = det.bbox + np.array([roi_dx, roi_dy, roi_dx, roi_dy], dtype=float)
                    pdets = _filter_player_detections_to_court(pdets)
                    player_tracks = player_tracker.update(pdets, frame_idx=fi)
                    ball_state = ball_tracker.update(fi, bdets)
                    if ball_state is None:
                        pred = ball_tracker.predict_only()
                        if pred is not None:
                            ball_state = BallTrackState(
                                frame_idx=fi,
                                bbox=pred.bbox,
                                score=pred.score,
                                cx=pred.cx,
                                cy=pred.cy,
                                predicted=True,
                            )
                    if ball_state is None and last_ball_state is not None:
                        ball_state = BallTrackState(
                            frame_idx=fi,
                            bbox=last_ball_state.bbox,
                            score=last_ball_state.score,
                            cx=last_ball_state.cx,
                            cy=last_ball_state.cy,
                            predicted=True,
                        )
                    if ball_state is not None:
                        last_ball_state = ball_state
                    roles = _roles_for_tracks(player_tracks)
                    player_states = [
                        PlayerState(
                            track_id=tr.track_id,
                            role=roles.get(tr.track_id, "unknown"),
                            bboxes=[tuple(float(v) for v in tr.bbox)],
                            frames=[int(fi)],
                        )
                        for tr in player_tracks
                    ]
                    # Pose inference (sparse by stride)
                    if enable_pose and pose_estimator is not None and (fi % pose_stride == 0):
                        try:
                            poses: List[PoseKeypoints] = pose_estimator.estimate(fproc)
                            if poses:
                                # store first person for now
                                pts = poses[0].points.copy()
                                if roi_dx != 0 or roi_dy != 0:
                                    pts[:, 0] += float(roi_dx)
                                    pts[:, 1] += float(roi_dy)
                                pose_results[fi] = {"points": pts.tolist()}
                        except Exception as exc:  # pragma: no cover
                            print(f"[WARN] Pose inference failed at frame {fi}: {exc}")
                    frame_results.append(
                        FrameResult(
                            frame_idx=fi,
                            players=player_tracks,
                            player_states=player_states,
                            player_roles=roles,
                            ball=ball_state,
                        )
                    )
                batch_frames.clear()
                batch_indices.clear()
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
            roles = _roles_for_tracks(player_tracks)
            player_states = [
                PlayerState(
                    track_id=tr.track_id,
                    role=roles.get(tr.track_id, "unknown"),
                    bboxes=[tuple(float(v) for v in tr.bbox)],
                    frames=[int(frame_idx)],
                )
                for tr in player_tracks
            ]
            if enable_pose and pose_estimator is not None and (frame_idx % pose_stride == 0):
                try:
                    poses: List[PoseKeypoints] = pose_estimator.estimate(frame_proc)
                    if poses:
                        pts = poses[0].points.copy()
                        if roi_dx != 0 or roi_dy != 0:
                            pts[:, 0] += float(roi_dx)
                            pts[:, 1] += float(roi_dy)
                        pose_results[frame_idx] = {"points": pts.tolist()}
                except Exception as exc:  # pragma: no cover
                    print(f"[WARN] Pose inference failed at frame {frame_idx}: {exc}")
            frame_results.append(
                FrameResult(
                    frame_idx=frame_idx,
                    players=player_tracks,
                    player_states=player_states,
                    player_roles=roles,
                    ball=ball_state,
                )
            )

        frame_idx += 1

    cap.release()
    # process leftover batch
    if batch_frames:
        player_dets_batch = player_det.detect(batch_frames)
        exclude_bboxes_batch = [[d.bbox for d in pdets] for pdets in player_dets_batch]
        ball_dets_batch = ball_det.detect(batch_frames, exclude_bboxes_batch=exclude_bboxes_batch)
        for fi, pdets, bdets in zip(batch_indices, player_dets_batch, ball_dets_batch):
            if roi_dx != 0 or roi_dy != 0:
                for det in pdets:
                    det.bbox = det.bbox + np.array([roi_dx, roi_dy, roi_dx, roi_dy], dtype=float)
                for det in bdets:
                    det.bbox = det.bbox + np.array([roi_dx, roi_dy, roi_dx, roi_dy], dtype=float)
            pdets = _filter_player_detections_to_court(pdets)
            player_tracks = player_tracker.update(pdets, frame_idx=fi)
            ball_state = ball_tracker.update(fi, bdets)
            if ball_state is None:
                pred = ball_tracker.predict_only()
                if pred is not None:
                    ball_state = BallTrackState(
                        frame_idx=fi,
                        bbox=pred.bbox,
                        score=pred.score,
                        cx=pred.cx,
                        cy=pred.cy,
                        predicted=True,
                    )
            if ball_state is None and last_ball_state is not None:
                ball_state = BallTrackState(
                    frame_idx=fi,
                    bbox=last_ball_state.bbox,
                    score=last_ball_state.score,
                    cx=last_ball_state.cx,
                    cy=last_ball_state.cy,
                    predicted=True,
                )
            if ball_state is not None:
                last_ball_state = ball_state
            roles = _roles_for_tracks(player_tracks)
            player_states = [
                PlayerState(
                    track_id=tr.track_id,
                    role=roles.get(tr.track_id, "unknown"),
                    bboxes=[tuple(float(v) for v in tr.bbox)],
                    frames=[int(fi)],
                )
                for tr in player_tracks
            ]
            frame_results.append(
                FrameResult(
                    frame_idx=fi,
                    players=player_tracks,
                    player_states=player_states,
                    player_roles=roles,
                    ball=ball_state,
                )
            )

    return AnalyseResult(
        video_path=str(video_path),
        fps=fps,
        frame_results=frame_results,
        court_corners=court_corners,
        court_detection=court_detection,
        player_tracks=None,
        pose_results=pose_results if enable_pose else None,
        court_env=court_env,
    )
