"""Geometry-based metrics (distances, angles)."""

from __future__ import annotations

import numpy as np


def distance(p1, p2) -> float:
    p1 = np.array(p1)
    p2 = np.array(p2)
    return float(np.linalg.norm(p1 - p2))


def angle(a, b, c) -> float:
    """Angle at point b between a-b and c-b."""
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cosang = ba.dot(bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0))))
