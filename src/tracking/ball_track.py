from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from src.vision.detectors import Detection
from .utils import iou_xyxy


@dataclass
class BallTrackState:
    frame_idx: int
    bbox: np.ndarray  # [x1, y1, x2, y2]
    score: float
    cx: float
    cy: float


class SingleBallTracker:
    """
    Track a single ball/shuttle by selecting the best detection each frame and smoothing bbox.
    """

    def __init__(
        self,
        iou_thresh: float = 0.2,
        ema_alpha: float = 0.6,
        max_age: int = 5,
    ) -> None:
        self.iou_thresh = iou_thresh
        self.ema_alpha = ema_alpha
        self.prev_bbox: Optional[np.ndarray] = None
        self.prev_score: float = 0.0
        self.time_since_update: int = 0
        self.max_age = max_age

    def update(
        self,
        frame_idx: int,
        ball_dets: List[Detection],
    ) -> Optional[BallTrackState]:
        if not ball_dets:
            self.time_since_update += 1
            if self.time_since_update > self.max_age:
                self.prev_bbox = None
                self.prev_score = 0.0
            return None

        if self.prev_bbox is None:
            best_det = max(ball_dets, key=lambda d: d.score)
        else:
            best_det = None
            best_score = -1.0
            for det in ball_dets:
                iou = iou_xyxy(self.prev_bbox, det.bbox)
                combined = det.score + 0.5 * iou
                if combined > best_score:
                    best_score = combined
                    best_det = det

        if best_det is None:
            return None

        if self.prev_bbox is None:
            smoothed = best_det.bbox.copy()
        else:
            smoothed = (
                self.ema_alpha * best_det.bbox
                + (1.0 - self.ema_alpha) * self.prev_bbox
            )

        self.prev_bbox = smoothed
        self.prev_score = best_det.score
        self.time_since_update = 0
        cx = float((smoothed[0] + smoothed[2]) / 2.0)
        cy = float((smoothed[1] + smoothed[3]) / 2.0)

        return BallTrackState(
            frame_idx=frame_idx,
            bbox=smoothed,
            score=best_det.score,
            cx=cx,
            cy=cy,
        )

    def predict_only(self) -> Optional[BallTrackState]:
        """Return last known ball state if not aged out."""
        if self.prev_bbox is None:
            return None
        self.time_since_update += 1
        if self.time_since_update > self.max_age:
            self.prev_bbox = None
            self.prev_score = 0.0
            return None
        cx = float((self.prev_bbox[0] + self.prev_bbox[2]) / 2.0)
        cy = float((self.prev_bbox[1] + self.prev_bbox[3]) / 2.0)
        return BallTrackState(frame_idx=-1, bbox=self.prev_bbox, score=self.prev_score, cx=cx, cy=cy)
