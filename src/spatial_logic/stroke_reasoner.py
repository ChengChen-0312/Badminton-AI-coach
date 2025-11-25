"""High-level stroke reasoning."""

from __future__ import annotations

from typing import Any, Dict, List

from .events import EventDetector
from .region_logic import court_zone


class StrokeReasoner:
    def __init__(self) -> None:
        self.event_detector = EventDetector()

    def infer(self, player_pose_seq, ball_traj) -> Dict[str, Any]:
        """Return detected events and coarse stroke labels."""
        hit_indices = self.event_detector.detect_hit(player_pose_seq, ball_traj)
        # TODO: map hits + zones into stroke types (e.g., clear/net/smash)
        return {"hits": hit_indices, "zones": [court_zone(0.5, 0.5)]}
