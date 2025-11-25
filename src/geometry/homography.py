"""Court homography utilities."""

from __future__ import annotations

import numpy as np


class CourtHomography:
    def compute(self, court_lines) -> np.ndarray:
        """Compute homography matrix H from detected court lines/points."""
        # TODO: derive source/target points then call cv2.findHomography
        return np.eye(3)

    def warp(self, point, H: np.ndarray | None = None):
        """Map image coords to real court coords."""
        if H is None:
            H = np.eye(3)
        pt = np.array([point[0], point[1], 1.0])
        mapped = H @ pt
        mapped = mapped / (mapped[2] + 1e-6)
        return mapped[:2]
