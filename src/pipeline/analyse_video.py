"""Unified video analysis pipeline placeholder."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from src.tracking.ball_track import BallTrackState, SingleBallTracker
from src.tracking.bytetrack import SimpleByteTrack, Track
from src.tracking.player_track import PlayerState
from src.vision.ball_detector import BallDetector
from src.vision.court_detector import CourtDetector
from src.vision.detectors import PlayerDetector
from src.vision.pose_estimator import PoseEstimator, PoseKeypoints


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

    detect_stride = tracking_cfg.get("detect_stride", 1)
    batch_size = realtime_cfg.get("batch_size", 1)

    yolo_device = cfg.get("device", {}).get("yolo_device", device)
    yolo_imgsz = vision_cfg.get("yolo_imgsz", 640)
    yolo_conf = vision_cfg.get("yolo_conf", 0.25)
    use_court_roi = vision_cfg.get("use_court_roi", False)
    court_margin = vision_cfg.get("court_roi_margin", 0.0)
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
    court_detector = CourtDetector() if (detect_court_corners or use_court_roi) else None

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

        if court_corners is None and court_detector is not None:
            # Try multiple early frames and keep the best-scoring detection.
            samples = int(vision_cfg.get("court_detect_samples", 10))
            stride = int(vision_cfg.get("court_detect_stride", 5))
            samples = max(1, samples)
            stride = max(1, stride)

            best = None  # (confidence, corners, meta)
            for i in range(samples):
                fi = int(i * stride)
                if fi == 0:
                    rgb = first_rgb
                else:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
                    ok_i, bgr_i = cap.read()
                    if not ok_i or bgr_i is None:
                        break
                    rgb = cv2.cvtColor(bgr_i, cv2.COLOR_BGR2RGB)
                lines = court_detector.detect_court(rgb)
                conf = float(court_detector.last_confidence or 0.0)
                reason = str(court_detector.last_reason or "unknown")
                if lines is None or lines.corners is None:
                    continue
                corners_i = lines.corners.tolist()
                meta_i = {"source": "auto", "confidence": conf, "reason": reason, "frame_idx": fi}
                if best is None or conf > float(best[0]):
                    best = (conf, corners_i, meta_i)

            if best is not None:
                court_corners = best[1]
                court_detection = best[2]
            else:
                # Auto detector rejected all candidates.
                court_detection = {
                    "source": "auto_failed",
                    "confidence": float(court_detector.last_confidence) if court_detector.last_confidence is not None else None,
                    "reason": str(court_detector.last_reason) if court_detector.last_reason is not None else None,
                }

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
    )
