"""Utility helpers for vision preprocessing/postprocessing."""

from __future__ import annotations

import numpy as np


def letterbox(image: np.ndarray, new_shape=(640, 640)):
    """Placeholder for letterboxing if needed."""
    return image


def nms(detections, iou_threshold: float = 0.5):
    """Placeholder NMS."""
    return detections
