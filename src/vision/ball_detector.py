from __future__ import annotations

from typing import List, Sequence

import cv2
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
        allow_all_when_empty: bool = False,
        imgsz: int = 640,
    ) -> None:
        self.det = YoloDetector(model_path=model_path, device=device, conf=conf, imgsz=imgsz)
        self.allowed_class_names = (
            [name.lower() for name in allowed_class_names]
            if allowed_class_names is not None
            else ["shuttlecock", "badminton", "sports ball", "ball"]
        )
        self.allow_all_when_empty = allow_all_when_empty
        self._prev_gray: np.ndarray | None = None
        self._prev_center: tuple[float, float] | None = None
        # Filter out obvious false positives inside the lower body region of a detected player.
        # This reduces cases where YOLO predicts "sports ball" on shoes/rackets near the floor.
        self._exclude_player_lower_frac = 0.60

    @staticmethod
    def _filter_inside_player_lower_region(
        dets: List[Detection],
        exclude_bboxes: Sequence[Sequence[float]] | None,
        lower_frac: float = 0.60,
    ) -> List[Detection]:
        if not dets or not exclude_bboxes:
            return dets

        kept: List[Detection] = []
        for det in dets:
            x1, y1, x2, y2 = det.bbox
            cx = float((x1 + x2) / 2.0)
            cy = float((y1 + y2) / 2.0)
            drop = False
            for b in exclude_bboxes:
                if len(b) < 4:
                    continue
                px1, py1, px2, py2 = [float(v) for v in b[:4]]
                if px2 <= px1 or py2 <= py1:
                    continue
                if cx < px1 or cx > px2 or cy < py1 or cy > py2:
                    continue
                ph = py2 - py1
                if ph <= 1e-6:
                    continue
                # Only drop if the detection is inside the lower part of the player's box.
                if cy >= py1 + lower_frac * ph:
                    drop = True
                    break
            if not drop:
                kept.append(det)
        return kept

    @staticmethod
    def _fallback_small_objects(
        detections: List[Detection],
        frame_shape: tuple[int, int, int],
        max_area_ratio: float = 0.02,
        max_return: int = 10,
    ) -> List[Detection]:
        """
        Heuristic fallback when the model doesn't output a shuttle/ball class.

        We prefer small detections (shuttlecock is tiny) and avoid tracking large objects
        like players, which can otherwise dominate the scores.
        """
        if not detections:
            return []
        h, w = int(frame_shape[0]), int(frame_shape[1])
        frame_area = float(max(h * w, 1))

        def _area_ratio(det: Detection) -> float:
            x1, y1, x2, y2 = det.bbox
            area = max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))
            return area / frame_area

        # Drop obvious large objects first.
        candidates = [d for d in detections if d.cls_name.lower() != "person"]
        candidates.sort(key=_area_ratio)

        small = [d for d in candidates if _area_ratio(d) <= max_area_ratio]
        if small:
            return small[:max_return]

        # If nothing is small enough, do not fall back to tracking large boxes.
        return []

    def _fallback_motion_ball(
        self,
        frame: np.ndarray,
        exclude_bboxes: Sequence[Sequence[float]] | None = None,
    ) -> List[Detection]:
        """
        Motion + brightness based fallback to approximate shuttlecock location.

        This is intentionally simple and only used when YOLO does not return any
        ball-like candidates. It can still fail on some videos, but prevents the
        tracker from latching onto large objects like players.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        if self._prev_gray is None or self._prev_gray.shape != gray.shape:
            self._prev_gray = gray
            return []

        diff = cv2.absdiff(gray, self._prev_gray)
        self._prev_gray = gray

        # Motion mask
        _, motion = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        # White-ish mask in HSV (shuttlecock is typically low saturation + high value)
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]
        white = cv2.inRange(hsv, (0, 0, 160), (180, 90, 255))
        mask = cv2.bitwise_and(motion, white)

        # Exclude player regions to avoid latching onto shoes/rackets.
        if exclude_bboxes:
            for b in exclude_bboxes:
                if len(b) < 4:
                    continue
                x1, y1, x2, y2 = b[:4]
                pad = 10
                xi1 = max(0, int(float(x1) - pad))
                yi1 = max(0, int(float(y1) - pad))
                xi2 = min(mask.shape[1], int(float(x2) + pad))
                yi2 = min(mask.shape[0], int(float(y2) + pad))
                if xi2 > xi1 and yi2 > yi1:
                    mask[yi1:yi2, xi1:xi2] = 0

        # Clean up noise a bit
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return []

        h, w = gray.shape[:2]
        frame_area = float(max(h * w, 1))
        min_area = 2.0
        max_area = max(40.0, frame_area * 0.00012)  # ~250px @ 1080p

        candidates: list[tuple[float, float, tuple[float, float, float, float], tuple[float, float]]] = []
        for cnt in contours:
            area = float(cv2.contourArea(cnt))
            if area < min_area or area > max_area:
                continue
            x, y, ww, hh = cv2.boundingRect(cnt)
            if ww > 80 or hh > 80:
                continue
            x1 = max(0.0, float(x - 2))
            y1 = max(0.0, float(y - 2))
            x2 = min(float(w - 1), float(x + ww + 2))
            y2 = min(float(h - 1), float(y + hh + 2))
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            if self._prev_center is not None:
                px, py = self._prev_center
                dist = float(np.hypot(cx - px, cy - py))
            else:
                dist = 0.0
            candidates.append((dist, -area, (x1, y1, x2, y2), (cx, cy)))

        if not candidates:
            return []

        # Prefer continuity if we already have a track; otherwise prefer stronger blobs.
        if self._prev_center is not None:
            candidates.sort(key=lambda t: (t[0], t[1]))
        else:
            candidates.sort(key=lambda t: (t[1], t[0]))

        _, _, bbox_t, center = candidates[0]
        self._prev_center = center
        bbox = np.array(bbox_t, dtype=float)
        return [
            Detection(
                bbox=bbox,
                score=0.05,
                cls_id=-1,
                cls_name="motion_ball",
                track_id=None,
            )
        ]

    def detect_balls(self, frame: np.ndarray) -> List[Detection]:
        raw = self.det.detect(frame)
        filtered = filter_by_class_name(raw, self.allowed_class_names)
        if not filtered and self.allow_all_when_empty:
            filtered = self._fallback_small_objects(raw, frame.shape)
            if not filtered:
                filtered = self._fallback_motion_ball(frame)
            return filtered
        return filtered

    def detect(self, frames, exclude_bboxes_batch: Sequence[Sequence[Sequence[float]]] | None = None):
        """Alias to allow batch or single-frame detection."""
        raw = self.det.detect(frames)
        if isinstance(raw, list):
            out = []
            ex_iter = exclude_bboxes_batch if exclude_bboxes_batch is not None else [None] * len(raw)
            for f, r, ex in zip(frames, raw, ex_iter):
                filtered = filter_by_class_name(r, self.allowed_class_names)
                filtered = self._filter_inside_player_lower_region(
                    filtered,
                    exclude_bboxes=ex,
                    lower_frac=self._exclude_player_lower_frac,
                )
                if not filtered and self.allow_all_when_empty:
                    filtered = self._fallback_small_objects(r, f.shape)
                    if not filtered:
                        filtered = self._fallback_motion_ball(f, exclude_bboxes=ex)
                out.append(filtered)
            return out
        filtered = filter_by_class_name(raw, self.allowed_class_names)
        filtered = self._filter_inside_player_lower_region(
            filtered,
            exclude_bboxes=exclude_bboxes_batch,  # type: ignore[arg-type]
            lower_frac=self._exclude_player_lower_frac,
        )
        if not filtered and self.allow_all_when_empty:
            try:
                shape = frames.shape  # type: ignore[attr-defined]
            except Exception:
                shape = (1, 1, 3)
            filtered = self._fallback_small_objects(raw, shape)
            if not filtered:
                try:
                    filtered = self._fallback_motion_ball(frames, exclude_bboxes=exclude_bboxes_batch)  # type: ignore[arg-type]
                except Exception:
                    filtered = []
            return filtered
        return filtered
