from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from src.vision.detectors import Detection
from .utils import iou_xyxy


@dataclass
class Track:
    track_id: int
    bbox: np.ndarray  # [x1, y1, x2, y2]
    score: float
    cls_id: int
    cls_name: str
    age: int = 0
    hits: int = 0
    time_since_update: int = 0


class SimpleByteTrack:
    """
    Minimal ByteTrack-style tracker:
      - Greedy IoU matching per frame
      - New tracks for unmatched detections
      - Remove stale tracks
    """

    def __init__(
        self,
        iou_thresh: float = 0.3,
        max_time_since_update: int = 30,
        min_hits: int = 1,
    ) -> None:
        self.iou_thresh = iou_thresh
        self.max_time_since_update = max_time_since_update
        self.min_hits = min_hits

        self._next_id: int = 1
        self.tracks: Dict[int, Track] = {}

    def update(
        self,
        detections: List[Detection],
        frame_idx: int | None = None,
    ) -> List[Track]:
        # Age all tracks
        for t in self.tracks.values():
            t.age += 1
            t.time_since_update += 1

        unmatched_dets = list(range(len(detections)))
        unmatched_tracks = list(self.tracks.keys())
        matches: List[tuple[int, int]] = []

        # Greedy IoU matching
        for det_idx in list(unmatched_dets):
            best_iou = 0.0
            best_track_id: Optional[int] = None

            for track_id in unmatched_tracks:
                tr = self.tracks[track_id]
                iou = iou_xyxy(tr.bbox, detections[det_idx].bbox)
                if iou > self.iou_thresh and iou > best_iou:
                    best_iou = iou
                    best_track_id = track_id

            if best_track_id is not None:
                matches.append((det_idx, best_track_id))
                unmatched_dets.remove(det_idx)
                unmatched_tracks.remove(best_track_id)

        # Update matched tracks
        for det_idx, track_id in matches:
            det = detections[det_idx]
            tr = self.tracks[track_id]
            tr.bbox = det.bbox.copy()
            tr.score = det.score
            tr.cls_id = det.cls_id
            tr.cls_name = det.cls_name
            tr.time_since_update = 0
            tr.hits += 1

        # Create new tracks
        for det_idx in unmatched_dets:
            det = detections[det_idx]
            track_id = self._next_id
            self._next_id += 1
            self.tracks[track_id] = Track(
                track_id=track_id,
                bbox=det.bbox.copy(),
                score=det.score,
                cls_id=det.cls_id,
                cls_name=det.cls_name,
                age=1,
                hits=1,
                time_since_update=0,
            )

        # Remove stale tracks
        to_delete = [
            tid
            for tid, tr in self.tracks.items()
            if tr.time_since_update > self.max_time_since_update
        ]
        for tid in to_delete:
            del self.tracks[tid]

        active_tracks: List[Track] = []
        for tr in self.tracks.values():
            if tr.hits >= self.min_hits:
                active_tracks.append(tr)

        return active_tracks

    def predict_only(self) -> List[Track]:
        """Return current active tracks while aging them (no new detections)."""
        for t in self.tracks.values():
            t.age += 1
            t.time_since_update += 1
        to_delete = [
            tid
            for tid, tr in self.tracks.items()
            if tr.time_since_update > self.max_time_since_update
        ]
        for tid in to_delete:
            del self.tracks[tid]
        return [tr for tr in self.tracks.values() if tr.hits >= self.min_hits]
