from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


@dataclass
class Detection:
    """Single object detection result."""

    bbox: np.ndarray  # [x1, y1, x2, y2]
    score: float
    cls_id: int
    cls_name: str
    track_id: Optional[int] = None  # filled by tracker


class YoloDetector:
    """
    Thin wrapper around Ultralytics YOLO for generic detection.

    Supports single-frame or batch inputs for speed.
    """

    def __init__(
        self,
        model_path: str | Path,
        device: str = "cpu",
        conf: float = 0.25,
        imgsz: int = 640,
    ) -> None:
        if YOLO is None:
            raise ImportError(
                "ultralytics is not installed. Please `pip install ultralytics`."
            )

        self.model = YOLO(str(model_path))
        self.device = device
        self.conf = conf
        self.imgsz = imgsz

    def detect(
        self,
        frame_or_batch,
        classes: Optional[Sequence[int]] = None,
    ):
        """
        Run YOLO detection on a single RGB frame or batch of frames.

        Args:
            frame_or_batch: np.ndarray [H,W,3] or list of frames
            classes: optional list of class ids to keep

        Returns:
            List[Detection] or List[List[Detection]]
        """
        is_batch = isinstance(frame_or_batch, list)
        source = frame_or_batch

        results = self.model.predict(
            source=source,
            device=self.device,
            conf=self.conf,
            classes=list(classes) if classes is not None else None,
            verbose=False,
            imgsz=self.imgsz,
        )

        if not is_batch:
            results = [results]

        all_dets: List[List[Detection]] = []
        for res in results:
            dets: List[Detection] = []
            if res.boxes is None:
                all_dets.append(dets)
                continue
            names = res.names
            for box in res.boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                score = float(box.conf[0].cpu().item())
                cls_id = int(box.cls[0].cpu().item())
                cls_name = names.get(cls_id, str(cls_id))
                dets.append(
                    Detection(
                        bbox=xyxy.astype(float),
                        score=score,
                        cls_id=cls_id,
                        cls_name=cls_name,
                        track_id=None,
                    )
                )
            all_dets.append(dets)

        return all_dets if is_batch else all_dets[0]


class PlayerDetector:
    """
    Person-only detector, built on top of YoloDetector.
    Assumes COCO-style classes where 'person' is class id 0.
    """

    def __init__(
        self,
        model_path: str | Path = "yolov8n.pt",
        device: str = "cpu",
        conf: float = 0.25,
        person_class_ids: Optional[Sequence[int]] = None,
        imgsz: int = 640,
    ) -> None:
        self.base = YoloDetector(
            model_path=model_path, device=device, conf=conf, imgsz=imgsz
        )
        self.person_class_ids = list(person_class_ids) if person_class_ids else [0]

    def detect_players(self, frame: np.ndarray) -> List[Detection]:
        return self.base.detect(frame, classes=self.person_class_ids)

    def detect(self, frames):
        """Alias to allow batch or single-frame detection."""
        return self.base.detect(frames, classes=self.person_class_ids)


def filter_by_class_name(
    detections: Iterable[Detection],
    allowed_names: Sequence[str],
) -> List[Detection]:
    allowed = {name.lower() for name in allowed_names}
    out: List[Detection] = []
    for det in detections:
        if det.cls_name.lower() in allowed:
            out.append(det)
    return out
