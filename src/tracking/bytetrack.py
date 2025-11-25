"""Player tracking with ByteTrack (placeholder)."""

from __future__ import annotations

from typing import List, Tuple


class PlayerTracker:
    def __init__(self) -> None:
        # TODO: load BYTETracker when dependency is available
        self.tracker = None

    def update(self, detections: List[Tuple]) -> List[Tuple]:
        """detections: list of (bbox, score)."""
        if self.tracker is None:
            # TODO: integrate actual tracker
            return detections
        return self.tracker.update(detections)
