"""Rendering utilities to overlay detections/metrics on video frames."""

from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np


def render_players(frame: np.ndarray, players: Iterable) -> np.ndarray:
    """Draw player boxes on frame."""
    out = frame.copy()
    for det in players:
        if isinstance(det, (list, tuple)) and len(det) >= 4:
            x1, y1, x2, y2 = map(int, det[:4])
        else:
            x1, y1, x2, y2 = 10, 10, 100, 100
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return out


def render_ball(frame: np.ndarray, ball_pos) -> np.ndarray:
    """Draw ball position."""
    out = frame.copy()
    if ball_pos is not None and isinstance(ball_pos, (tuple, list)) and len(ball_pos) >= 2:
        x, y = map(int, ball_pos[:2])
        cv2.circle(out, (x, y), 5, (0, 0, 255), -1)
    return out
