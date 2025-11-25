"""Unified video analysis pipeline placeholder."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

from src.tracking.ball_track import BallTrackState, SingleBallTracker
from src.tracking.bytetrack import SimpleByteTrack, Track
from src.tracking.id_assign import assign_player_roles
from src.vision.ball_detector import BallDetector
from src.vision.detectors import PlayerDetector


@dataclass
class FrameResult:
    frame_idx: int
    players: List[Track]
    player_roles: Dict[int, str]
    ball: BallTrackState | None


@dataclass
class AnalyseResult:
    video_path: str
    fps: float
    frame_results: List[FrameResult]


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

    detect_stride = tracking_cfg.get("detect_stride", 1)
    batch_size = realtime_cfg.get("batch_size", 1)

    yolo_device = cfg.get("device", {}).get("yolo_device", device)
    yolo_imgsz = vision_cfg.get("yolo_imgsz", 640)
    yolo_conf = vision_cfg.get("yolo_conf", 0.25)
    use_court_roi = vision_cfg.get("use_court_roi", False)
    court_margin = vision_cfg.get("court_roi_margin", 0.0)
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

    frame_idx = 0
    frame_results: List[FrameResult] = []

    # optional ROI from first frame
    court_roi = None
    if use_court_roi:
        ok, first_bgr = cap.read()
        if ok:
            h0, w0 = first_bgr.shape[:2]
            # simple full-frame ROI with margin; replace with court detector if needed
            x1 = int(w0 * court_margin)
            y1 = int(h0 * court_margin)
            x2 = int(w0 * (1 - court_margin))
            y2 = int(h0 * (1 - court_margin))
            court_roi = (x1, y1, x2, y2)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    batch_frames: list[np.ndarray] = []
    batch_indices: list[int] = []

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
                for fi, pdets, bdets in zip(batch_indices, player_dets_batch, ball_dets_batch):
                    player_tracks = player_tracker.update(pdets, frame_idx=fi)
                    ball_state = ball_tracker.update(fi, bdets)
                    roles = assign_player_roles(player_tracks, frame_height=h)
                    frame_results.append(
                        FrameResult(
                            frame_idx=fi,
                            players=player_tracks,
                            player_roles=roles,
                            ball=ball_state,
                        )
                    )
                batch_frames.clear()
                batch_indices.clear()
        else:
            player_tracks = player_tracker.predict_only()
            ball_state = ball_tracker.predict_only()
            roles = assign_player_roles(player_tracks, frame_height=h)
            frame_results.append(
                FrameResult(
                    frame_idx=frame_idx,
                    players=player_tracks,
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
            roles = assign_player_roles(player_tracks, frame_height=h)
            frame_results.append(
                FrameResult(
                    frame_idx=fi,
                    players=player_tracks,
                    player_roles=roles,
                    ball=ball_state,
                )
            )

    return AnalyseResult(
        video_path=str(video_path),
        fps=fps,
        frame_results=frame_results,
    )
