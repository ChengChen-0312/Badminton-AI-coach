"""Court line detector / segmenter placeholder."""

from __future__ import annotations

from typing import Any

import numpy as np


class CourtDetector:
    def __init__(self, model_path: str = "court_segmenter.pt", device: str = "mps") -> None:
        self.model_path = model_path
        self.device = device
        self.model: Any = None

    def detect_lines(self, frame: np.ndarray):
        """Return court line masks or keypoints."""
        # TODO: implement segmentation or line detection
        return None
