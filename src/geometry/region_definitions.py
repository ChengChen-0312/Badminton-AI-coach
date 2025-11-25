from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CourtRegions:
    """
    Simple court region boundaries along y-axis (meters).
    """

    court_length: float = 13.4
    front_mid: float = 2.0
    mid_back: float = 4.5

    def classify_y(self, y: float) -> str:
        if y < self.front_mid:
            return "front"
        elif y < self.mid_back:
            return "mid"
        elif y <= self.court_length:
            return "back"
        return "out"


DEFAULT_REGIONS = CourtRegions()
