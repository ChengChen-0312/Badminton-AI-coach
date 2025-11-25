"""Rendering utilities to overlay detections/metrics on video frames."""

from __future__ import annotations

import cv2
import numpy as np


def render_players(frame: np.ndarray, players) -> np.ndarray:
    """Draw player boxes on frame."""
    out = frame.copy()
    for det in players:
        # TODO: unpack bbox, score
        cv2.rectangle(out, (10, 10), (100, 100), (0, 255, 0), 2)
    return out
