from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np


# MediaPipe Pose landmark indices (right side)
RIGHT_SHOULDER = 12
RIGHT_ELBOW = 14
RIGHT_WRIST = 16
RIGHT_INDEX = 20
RIGHT_HIP = 24
LEFT_SHOULDER = 11
LEFT_HIP = 23


def _to_np(points: Sequence[Sequence[float]]) -> np.ndarray:
    return np.asarray(points, dtype=float)


def _get_pt(arr: np.ndarray, idx: int) -> Optional[np.ndarray]:
    if idx < 0 or idx >= arr.shape[0]:
        return None
    return arr[idx, :2]


def _angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> Optional[float]:
    """Compute angle (in degrees) at p2 formed by p1-p2-p3."""
    v1 = p1 - p2
    v2 = p3 - p2
    if np.linalg.norm(v1) < 1e-6 or np.linalg.norm(v2) < 1e-6:
        return None
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
    return float(np.degrees(np.arccos(dot)))


def _angle_with_vertical(p_top: np.ndarray, p_bottom: np.ndarray) -> Optional[float]:
    """Angle between the segment (bottom -> top) and the vertical axis."""
    v = p_top - p_bottom
    if np.linalg.norm(v) < 1e-6:
        return None
    v = v / np.linalg.norm(v)
    vertical = np.array([0.0, -1.0])  # up
    dot = np.clip(np.dot(v, vertical), -1.0, 1.0)
    return float(np.degrees(np.arccos(dot)))


def extract_pose_features(pose_landmarks: Dict) -> Optional[Dict[str, Optional[float]]]:
    """
    Extract key angles from pose landmarks (MediaPipe Pose 33 keypoints expected).
    Returns a dict with angles in degrees; missing values are None.
    """
    if pose_landmarks is None:
        return None

    points = pose_landmarks.get("points") if isinstance(pose_landmarks, dict) else None
    if points is None:
        return None

    pts = _to_np(points)
    features: Dict[str, Optional[float]] = {
        "right_elbow": None,
        "right_shoulder": None,
        "trunk_angle": None,
        "racket_angle": None,
    }

    sh = _get_pt(pts, RIGHT_SHOULDER)
    el = _get_pt(pts, RIGHT_ELBOW)
    wr = _get_pt(pts, RIGHT_WRIST)
    idx = _get_pt(pts, RIGHT_INDEX)
    hip_r = _get_pt(pts, RIGHT_HIP)
    sh_l = _get_pt(pts, LEFT_SHOULDER)
    hip_l = _get_pt(pts, LEFT_HIP)

    if sh is not None and el is not None and wr is not None:
        features["right_elbow"] = _angle(sh, el, wr)

    if hip_r is not None and sh is not None and el is not None:
        features["right_shoulder"] = _angle(hip_r, sh, el)

    # Trunk angle: line from mid-hip to mid-shoulder vs vertical
    if hip_r is not None and hip_l is not None and sh is not None and sh_l is not None:
        hip_mid = (hip_r + hip_l) / 2.0
        sh_mid = (sh + sh_l) / 2.0
        features["trunk_angle"] = _angle_with_vertical(sh_mid, hip_mid)

    # Racket angle: wrist -> index vs horizontal (x-axis)
    if wr is not None and idx is not None:
        v = idx - wr
        if np.linalg.norm(v) >= 1e-6:
            v = v / np.linalg.norm(v)
            horiz = np.array([1.0, 0.0])
            dot = np.clip(np.dot(v, horiz), -1.0, 1.0)
            features["racket_angle"] = float(np.degrees(np.arccos(dot)))

    return features
