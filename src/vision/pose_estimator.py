from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

try:
    import mediapipe as mp
except ImportError:
    mp = None


@dataclass
class PoseKeypoints:
    """Pose keypoints for a single person."""

    points: np.ndarray  # [num_kpts, 3] -> (x, y, visibility)


class PoseEstimator:
    """
    Optional pose estimator (MediaPipe).

    If mediapipe is not installed, constructing this class will raise an error.
    """

    def __init__(self) -> None:
        if mp is None:
            raise ImportError(
                "mediapipe is not installed. Please `pip install mediapipe` or avoid using PoseEstimator."
            )

        self.mp_pose = mp.solutions.pose
        self.model = self.mp_pose.Pose(static_image_mode=False)

    def estimate(self, frame: np.ndarray) -> List[PoseKeypoints]:
        """
        Args:
            frame: RGB np.ndarray [H, W, 3]

        Returns:
            List[PoseKeypoints]
        """
        results = self.model.process(frame)
        if not results.pose_landmarks:
            return []

        h, w, _ = frame.shape
        pts = []
        for lm in results.pose_landmarks.landmark:
            x = lm.x * w
            y = lm.y * h
            v = lm.visibility
            pts.append([x, y, v])

        return [PoseKeypoints(points=np.array(pts, dtype=np.float32))]
