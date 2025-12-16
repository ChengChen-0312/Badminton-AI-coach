from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from src.tracking.player_track import PlayerState


@dataclass
class HitterInfo:
    hitter_role: str  # "near" / "far"
    hitter_track_id: Optional[int]
    distance: float


def _bbox_center(bbox: Sequence[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (float((x1 + x2) / 2.0), float((y1 + y2) / 2.0))

def _point_to_bbox_distance(point_xy: Tuple[float, float], bbox: Sequence[float]) -> float:
    """Distance from a point to an axis-aligned bbox (0 if inside)."""
    px, py = point_xy
    x1, y1, x2, y2 = bbox
    dx = max(float(x1) - px, 0.0, px - float(x2))
    dy = max(float(y1) - py, 0.0, py - float(y2))
    return float(np.hypot(dx, dy))


def infer_hitter_for_stroke(
    contact_frame: int,
    ball_pos: Tuple[float, float],
    players_at_contact: List[PlayerState],
    max_distance: float = 200.0,
) -> Optional[HitterInfo]:
    """Choose the closest player to the ball at contact frame."""
    if not players_at_contact:
        return None

    best: Optional[HitterInfo] = None
    for p in players_at_contact:
        if not p.bboxes:
            continue
        # Use point-to-rect distance (more robust when the shuttle is above a player).
        dist = _point_to_bbox_distance(ball_pos, p.bboxes[-1])
        if best is None or dist < best.distance:
            best = HitterInfo(
                hitter_role=p.role if p.role else "unknown",
                hitter_track_id=p.track_id,
                distance=dist,
            )
    if best is None:
        return None
    if best.distance > max_distance:
        return HitterInfo(hitter_role="unknown", hitter_track_id=None, distance=best.distance)
    return best
