"""Pose estimator placeholder."""

from __future__ import annotations

from typing import Any, List, Tuple

import numpy as np


class PoseEstimator:
    def __init__(self, model_name: str = "yolo-pose-s", device: str = "mps") -> None:
        self.model_name = model_name
        self.device = device
        self.model = None  # plug in your pose model here

    def estimate(self, frame: np.ndarray) -> List[Tuple[Any, float]]:
        """Return list of keypoints + confidence."""
        # TODO: integrate an actual pose model (e.g., YOLO-Pose or RTMDet-Pose)
        return []
