"""YOLO-based detectors for players and shuttle (placeholders).

These wrappers are lightweight and avoid hard dependencies on ultralytics/YOLO.
Replace the TODO sections with actual model loading when integrating detection.
"""

from __future__ import annotations

from typing import Any, List, Tuple

import numpy as np


class YOLODetector:
    """Thin wrapper around a YOLO detector."""

    def __init__(self, model_path: str = "yolov8n.pt", device: str = "mps") -> None:
        self.model_path = model_path
        self.device = device
        self.model = None  # lazy load to avoid hard dependency

    def _ensure_model(self) -> None:
        if self.model is not None:
            return
        try:
            from ultralytics import YOLO
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "YOLO dependency missing. Install `ultralytics` or swap in your detector."
            ) from exc
        self.model = YOLO(self.model_path)
        self.model.to(self.device)

    def detect_players(self, frame: np.ndarray) -> List[Tuple[Any, float]]:
        """Return player bounding boxes + confidence."""
        self._ensure_model()
        results = self.model(frame)
        return self._parse_person_detections(results)

    def detect_ball(self, frame: np.ndarray) -> List[Tuple[Any, float]]:
        """Detect shuttle using a small-object-tuned YOLO."""
        self._ensure_model()
        results = self.model(frame)
        return self._parse_ball_detections(results)

    @staticmethod
    def _parse_person_detections(results) -> List[Tuple[Any, float]]:
        # TODO: filter by class==person and return (bbox, score)
        return []

    @staticmethod
    def _parse_ball_detections(results) -> List[Tuple[Any, float]]:
        # TODO: filter by shuttle/ball class and return (bbox, score)
        return []
