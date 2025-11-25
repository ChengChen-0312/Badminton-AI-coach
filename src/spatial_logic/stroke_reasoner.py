"""High-level stroke reasoning."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

from .events import EventDetector
from .region_logic import court_zone
from .temporal_logic import smooth_events


class StrokeReasoner:
    def __init__(self) -> None:
        self.event_detector = EventDetector()

    def infer(self, player_pose_seq, ball_traj: Sequence[Tuple[float, float]]) -> Dict[str, Any]:
        """Return detected events and coarse stroke labels."""
        hit_indices = smooth_events(self.event_detector.detect_hit(player_pose_seq, ball_traj))
        stroke_labels: List[str] = []
        zones: List[str] = []
        for idx in hit_indices:
            if idx < len(ball_traj):
                x, y = ball_traj[idx]
            else:
                x, y = 0.5, 0.5
            zone = court_zone(x, y)
            zones.append(zone)
            stroke_labels.append(self._zone_to_stroke(zone))
        return {"hits": hit_indices, "zones": zones, "strokes": stroke_labels}

    @staticmethod
    def _zone_to_stroke(zone: str) -> str:
        if zone == "front":
            return "net_shot"
        if zone == "mid":
            return "drive"
        return "clear"
