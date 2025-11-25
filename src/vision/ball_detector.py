"""High-speed shuttle detector placeholder."""

from __future__ import annotations

import numpy as np


class BallDetector:
    def __init__(self, model_path: str = "yolo-ball.pt", device: str = "mps") -> None:
        self.model_path = model_path
        self.device = device
        self.model = None

    def detect(self, frame: np.ndarray):
        """Return shuttle detections (bbox, score)."""
        # TODO: integrate small-object tuned model
        return []
