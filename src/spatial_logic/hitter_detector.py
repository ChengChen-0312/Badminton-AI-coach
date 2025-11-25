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


def infer_hitter_for_stroke(
    contact_frame: int,
    ball_pos: Tuple[float, float],
    players_at_contact: List[PlayerState],
    max_distance: float = 200.0,
) -> Optional[HitterInfo]:
    """Choose the closest player to the ball at contact frame."""
    if not players_at_contact:
        return None

    bx, by = ball_pos
    best: Optional[HitterInfo] = None
    for p in players_at_contact:
        if not p.bboxes:
            continue
        cx, cy = _bbox_center(p.bboxes[-1])
        dist = float(np.hypot(cx - bx, cy - by))
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
