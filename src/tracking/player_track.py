from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .utils import iou_xyxy


@dataclass
class PlayerState:
    track_id: int
    role: str  # "near" or "far"
    bboxes: List[Tuple[float, float, float, float]] = field(default_factory=list)
    frames: List[int] = field(default_factory=list)

    def last_bbox(self) -> Optional[np.ndarray]:
        if not self.bboxes:
            return None
        return np.array(self.bboxes[-1], dtype=float)


class PlayerTracker:
    """Maintain up to two long-lived player tracks (near / far)."""

    def __init__(self, max_iou_mismatch: float = 0.5) -> None:
        self.players: List[PlayerState] = []
        self.max_iou_mismatch = max_iou_mismatch
        self._next_id = 0

    def _select_top2(self, bboxes: List[Tuple[float, float, float, float]]) -> List[np.ndarray]:
        if len(bboxes) <= 2:
            return [np.array(b, dtype=float) for b in bboxes]
        areas = []
        for b in bboxes:
            x1, y1, x2, y2 = b
            areas.append((max(0.0, (x2 - x1) * (y2 - y1)), np.array(b, dtype=float)))
        areas.sort(key=lambda x: x[0], reverse=True)
        return [a[1] for a in areas[:2]]

    def _assign_roles(self) -> None:
        if len(self.players) < 2:
            return
        # Higher y-center => closer to camera => "near"
        centers = []
        for p in self.players:
            lb = p.last_bbox()
            if lb is None:
                centers.append((p.track_id, -1.0))
            else:
                cy = (lb[1] + lb[3]) / 2.0
                centers.append((p.track_id, cy))
        centers.sort(key=lambda x: x[1], reverse=True)
        if centers:
            self._set_role(centers[0][0], "near")
        if len(centers) > 1:
            self._set_role(centers[1][0], "far")

    def _set_role(self, track_id: int, role: str) -> None:
        for p in self.players:
            if p.track_id == track_id:
                p.role = role

    def update(self, frame_idx: int, player_bboxes: List[Tuple[float, float, float, float]]) -> None:
        """Update tracker with detections for this frame."""
        boxes = self._select_top2(player_bboxes)

        # IoU match to existing tracks
        unmatched_boxes = list(range(len(boxes)))
        matched_players: List[int] = []
        for pi, player in enumerate(self.players):
            best_iou = 0.0
            best_box_idx = None
            lb = player.last_bbox()
            if lb is None:
                continue
            for bi in unmatched_boxes:
                iou = iou_xyxy(lb, boxes[bi])
                if iou > self.max_iou_mismatch and iou > best_iou:
                    best_iou = iou
                    best_box_idx = bi
            if best_box_idx is not None:
                player.bboxes.append(tuple(boxes[best_box_idx]))
                player.frames.append(frame_idx)
                matched_players.append(player.track_id)
                unmatched_boxes.remove(best_box_idx)

        # Create new players for unmatched boxes (up to 2 total)
        for bi in unmatched_boxes:
            if len(self.players) >= 2:
                break
            self.players.append(
                PlayerState(
                    track_id=self._next_id,
                    role="unknown",
                    bboxes=[tuple(boxes[bi])],
                    frames=[frame_idx],
                )
            )
            self._next_id += 1

        # If fewer than 2 detections, keep last bbox (short-term hold) by duplicating last state
        if boxes == [] and self.players:
            for p in self.players:
                if p.frames and p.frames[-1] != frame_idx:
                    p.frames.append(frame_idx)
                    p.bboxes.append(tuple(p.bboxes[-1]))

        self._assign_roles()

    def get_players_at(self, frame_idx: int) -> List[PlayerState]:
        """Return player states with bbox at or before the given frame (latest known)."""
        states: List[PlayerState] = []
        for p in self.players:
            if not p.frames:
                continue
            # find latest frame <= frame_idx
            latest_idx = None
            for i, f in enumerate(p.frames):
                if f <= frame_idx:
                    latest_idx = i
                else:
                    break
            if latest_idx is None:
                continue
            states.append(
                PlayerState(
                    track_id=p.track_id,
                    role=p.role,
                    bboxes=[p.bboxes[latest_idx]],
                    frames=[p.frames[latest_idx]],
                )
            )
        return states
