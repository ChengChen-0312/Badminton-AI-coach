"""Event detectors (e.g., shuttle hits)."""

from __future__ import annotations

from typing import List


class EventDetector:
    def detect_hit(self, player_pose_seq, ball_traj) -> List[int]:
        """Return timestamps/indices when shuttle is hit."""
        # TODO: implement swing detection from pose + ball velocity changes
        return []
