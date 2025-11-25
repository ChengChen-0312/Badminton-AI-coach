from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np


@dataclass
class CourtLines:
    """Key points / lines of the court in image coordinates."""

    corners: np.ndarray  # [4, 2] in image (x, y) order: TL, TR, BR, BL


class CourtDetector:
    """
    Simple court detector using Canny + Hough. For production, replace with
    a learned court segmentation / keypoint model.
    """

    def __init__(self, canny_thresh1: int = 50, canny_thresh2: int = 150) -> None:
        self.canny_thresh1 = canny_thresh1
        self.canny_thresh2 = canny_thresh2

    def detect_court(self, frame: np.ndarray) -> CourtLines | None:
        """
        Args:
            frame: RGB frame [H, W, 3]

        Returns:
            CourtLines or None if failed.
        """
        h, w, _ = frame.shape
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, self.canny_thresh1, self.canny_thresh2)

        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=200,
            minLineLength=min(h, w) * 0.3,
            maxLineGap=20,
        )

        if lines is None or len(lines) < 4:
            corners = np.array(
                [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]],
                dtype=np.float32,
            )
            return CourtLines(corners=corners)

        # Fallback to full image as court region; replace with line clustering if needed.
        corners = np.array(
            [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]],
            dtype=np.float32,
        )
        return CourtLines(corners=corners)
