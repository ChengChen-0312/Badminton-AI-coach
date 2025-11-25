from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import cv2
import numpy as np

# Standard badminton court dimensions (approx, singles)
COURT_LENGTH_M = 13.40
COURT_WIDTH_M = 6.10


@dataclass
class CourtHomography:
    H: np.ndarray  # 3x3

    @classmethod
    def from_corners(
        cls,
        image_corners: Sequence[Sequence[float]],
        court_length: float = COURT_LENGTH_M,
        court_width: float = COURT_WIDTH_M,
    ) -> "CourtHomography":
        """
        Build homography from four image-space corners to a standard court.
        image_corners order: LB, RB, RT, LT (or any consistent order).
        Court coords: origin at left baseline, x right, y up (meters).
        """
        if len(image_corners) != 4:
            raise ValueError("image_corners must have 4 points")

        src = np.array(image_corners, dtype=np.float32)
        dst = np.array(
            [
                [0.0, 0.0],
                [court_width, 0.0],
                [court_width, court_length],
                [0.0, court_length],
            ],
            dtype=np.float32,
        )

        H, _ = cv2.findHomography(src, dst, method=0)
        if H is None:
            raise RuntimeError("Failed to compute homography")
        return cls(H=H)

    def to_court(self, pt_xy: Sequence[float]) -> Tuple[float, float]:
        """Map image coords (x, y) to court coords (cx, cy) in meters."""
        x, y = pt_xy
        pts = np.array([[x, y]], dtype=np.float32).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, self.H).reshape(-1, 2)[0]
        return float(dst[0]), float(dst[1])
