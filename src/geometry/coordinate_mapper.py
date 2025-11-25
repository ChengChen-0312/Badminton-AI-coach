"""Coordinate mapping helpers."""

from __future__ import annotations

import numpy as np


class CoordinateMapper:
    def __init__(self, homography: np.ndarray | None = None) -> None:
        self.H = homography if homography is not None else np.eye(3)

    def image_to_court(self, point):
        pt = np.array([point[0], point[1], 1.0])
        mapped = self.H @ pt
        mapped = mapped / (mapped[2] + 1e-6)
        return mapped[:2]
