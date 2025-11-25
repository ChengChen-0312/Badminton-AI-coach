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


@dataclass
class CourtGrid9:
    """
    3x3 court grid region classifier.
    x in [0, court_width], y in [0, court_length], origin at left baseline, y increases toward opponent.
    Regions: left_front, mid_front, right_front, left_mid, mid_mid, right_mid, left_back, mid_back, right_back.
    """

    court_width: float = 6.1
    court_length: float = 13.4

    def classify_region(self, x: float, y: float) -> str:
        if x < 0 or y < 0 or x > self.court_width or y > self.court_length:
            return "out"

        x_thirds = [self.court_width / 3.0, 2 * self.court_width / 3.0]
        y_thirds = [self.court_length / 3.0, 2 * self.court_length / 3.0]

        # x bins
        if x < x_thirds[0]:
            col = "left"
        elif x < x_thirds[1]:
            col = "mid"
        else:
            col = "right"

        # y bins
        if y < y_thirds[0]:
            row = "front"
        elif y < y_thirds[1]:
            row = "mid"
        else:
            row = "back"

        return f"{col}_{row}"

    def classify_y(self, y: float) -> str:
        # fallback to row classification only
        if y < self.court_length / 3.0:
            return "front"
        if y < 2 * self.court_length / 3.0:
            return "mid"
        return "back"


DEFAULT_GRID9 = CourtGrid9()
