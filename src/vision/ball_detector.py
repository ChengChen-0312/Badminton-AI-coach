from __future__ import annotations

from typing import List, Sequence

import numpy as np

from .detectors import Detection, YoloDetector, filter_by_class_name


class BallDetector:
    """
    Shuttlecock / ball detector.

    Usage:
      det = BallDetector(model_path="badminton_ball.pt")
      det.detect_balls(frame) -> List[Detection]
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        device: str = "cpu",
        conf: float = 0.25,
        allowed_class_names: Sequence[str] | None = None,
        imgsz: int = 640,
    ) -> None:
        self.det = YoloDetector(model_path=model_path, device=device, conf=conf, imgsz=imgsz)
        self.allowed_class_names = (
            [name.lower() for name in allowed_class_names]
            if allowed_class_names is not None
            else ["shuttlecock", "badminton", "sports ball", "ball"]
        )

    def detect_balls(self, frame: np.ndarray) -> List[Detection]:
        raw = self.det.detect(frame)
        return filter_by_class_name(raw, self.allowed_class_names)

    def detect(self, frames):
        """Alias to allow batch or single-frame detection."""
        raw = self.det.detect(frames)
        if isinstance(raw, list):
            return [filter_by_class_name(r, self.allowed_class_names) for r in raw]
        return filter_by_class_name(raw, self.allowed_class_names)
