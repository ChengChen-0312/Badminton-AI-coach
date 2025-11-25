"""Event detectors (e.g., shuttle hits)."""

from __future__ import annotations

from typing import List, Sequence, Tuple


class EventDetector:
    def detect_hit(self, player_pose_seq, ball_traj: Sequence[Tuple[float, float]]) -> List[int]:
        """Detect coarse hit events from ball velocity changes.

        Heuristic: mark an index when velocity magnitude drops sharply (possible contact).
        """
        if len(ball_traj) < 3:
            return []
        hits: List[int] = []
        prev_dx = 0.0
        prev_dy = 0.0
        for i in range(1, len(ball_traj)):
            dx = ball_traj[i][0] - ball_traj[i - 1][0]
            dy = ball_traj[i][1] - ball_traj[i - 1][1]
            speed = (dx**2 + dy**2) ** 0.5
            prev_speed = (prev_dx**2 + prev_dy**2) ** 0.5
            if prev_speed > 0 and speed < prev_speed * 0.5:
                hits.append(i)
            prev_dx, prev_dy = dx, dy
        return hits
