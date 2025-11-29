"""Unified video analysis pipeline placeholder."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from src.tracking.ball_track import BallTrackState, SingleBallTracker
from src.tracking.bytetrack import SimpleByteTrack, Track
from src.tracking.id_assign import assign_player_roles
from src.tracking.player_track import PlayerState, PlayerTracker
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
    identity_tracker = PlayerTracker(max_iou_mismatch=tracking_cfg.get("iou_thresh", 0.3))
    court_detector = CourtDetector() if use_court_roi else None

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

    # optional ROI from first frame; also capture default full-frame corners
    court_roi = None
    default_corners = None
    if use_court_roi:
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
            if court_detector is not None:
                lines = court_detector.detect_court(first_rgb)
                if lines is not None and lines.corners is not None:
                    court_corners = lines.corners.tolist()
                    xs = lines.corners[:, 0]
                    ys = lines.corners[:, 1]
                    x1 = max(0, int(xs.min() * (1 - court_margin)))
                    y1 = max(0, int(ys.min() * (1 - court_margin)))
                    x2 = min(w0, int(xs.max() * (1 + court_margin)))
                    y2 = min(h0, int(ys.max() * (1 + court_margin)))
                    court_roi = (x1, y1, x2, y2)
            if court_roi is None:
                x1 = int(w0 * court_margin)
                y1 = int(h0 * court_margin)
                x2 = int(w0 * (1 - court_margin))
                y2 = int(h0 * (1 - court_margin))
                court_roi = (x1, y1, x2, y2)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    else:
        # No ROI, use full-frame corners for downstream homography if needed
        cap_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
        ok, first_bgr = cap.read()
        if ok:
            h0, w0 = first_bgr.shape[:2]
            default_corners = [
                [0.0, float(h0 - 1)],
                [float(w0 - 1), float(h0 - 1)],
                [float(w0 - 1), 0.0],
                [0.0, 0.0],
            ]
        if cap_pos is not None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, cap_pos)

    batch_frames: list[np.ndarray] = []
    batch_indices: list[int] = []
    last_ball_state: BallTrackState | None = None

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
                ball_dets_batch = ball_det.detect(batch_frames)
                for fi, pdets, bdets, fproc in zip(batch_indices, player_dets_batch, ball_dets_batch, batch_frames):
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
                            )
                    if ball_state is None and last_ball_state is not None:
                        ball_state = BallTrackState(
                            frame_idx=fi,
                            bbox=last_ball_state.bbox,
                            score=last_ball_state.score,
                            cx=last_ball_state.cx,
                            cy=last_ball_state.cy,
                        )
                    if ball_state is not None:
                        last_ball_state = ball_state
                    roles = assign_player_roles(player_tracks, frame_height=h)
                    # update identity tracker with current player boxes
                    identity_tracker.update(fi, [tuple(t.bbox) for t in player_tracks])
                    player_states = identity_tracker.get_players_at(fi)
                    # Pose inference (sparse by stride)
                    if enable_pose and pose_estimator is not None and (fi % pose_stride == 0):
                        try:
                            poses: List[PoseKeypoints] = pose_estimator.estimate(fproc)
                            if poses:
                                # store first person for now
                                pose_results[fi] = {"points": poses[0].points.tolist()}
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
                )
            elif last_ball_state is not None:
                ball_state = BallTrackState(
                    frame_idx=frame_idx,
                    bbox=last_ball_state.bbox,
                    score=last_ball_state.score,
                    cx=last_ball_state.cx,
                    cy=last_ball_state.cy,
                )
            if ball_state is not None:
                last_ball_state = ball_state
            roles = assign_player_roles(player_tracks, frame_height=h)
            identity_tracker.update(frame_idx, [tuple(t.bbox) for t in player_tracks])
            player_states = identity_tracker.get_players_at(frame_idx)
            if enable_pose and pose_estimator is not None and (frame_idx % pose_stride == 0):
                try:
                    poses: List[PoseKeypoints] = pose_estimator.estimate(frame_proc)
                    if poses:
                        pose_results[frame_idx] = {"points": poses[0].points.tolist()}
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
        ball_dets_batch = ball_det.detect(batch_frames)
        for fi, pdets, bdets in zip(batch_indices, player_dets_batch, ball_dets_batch):
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
                    )
            if ball_state is None and last_ball_state is not None:
                ball_state = BallTrackState(
                    frame_idx=fi,
                    bbox=last_ball_state.bbox,
                    score=last_ball_state.score,
                    cx=last_ball_state.cx,
                    cy=last_ball_state.cy,
                )
            if ball_state is not None:
                last_ball_state = ball_state
            roles = assign_player_roles(player_tracks, frame_height=h)
            identity_tracker.update(fi, [tuple(t.bbox) for t in player_tracks])
            player_states = identity_tracker.get_players_at(fi)
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
        court_corners=court_corners if court_corners is not None else default_corners,
        player_tracks=identity_tracker.players,
        pose_results=pose_results if enable_pose else None,
    )
