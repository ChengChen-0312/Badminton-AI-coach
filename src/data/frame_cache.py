from __future__ import annotations

import os
from pathlib import Path
from typing import List

import cv2
import numpy as np


def extract_frames(video_path: str | Path, num_frames: int = 16) -> np.ndarray:
    """Decode a video and return uniformly sampled RGB frames (H, W, 3)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise RuntimeError(f"No frames in video: {video_path}")

    idxs = np.linspace(0, total - 1, num_frames).astype(int)
    frames: List[np.ndarray] = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()

    if len(frames) == 0:
        raise RuntimeError(f"Failed to read frames from {video_path}")
    return np.stack(frames)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
