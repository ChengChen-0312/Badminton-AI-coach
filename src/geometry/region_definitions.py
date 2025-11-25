"""Court region definitions."""

from __future__ import annotations


def zone_from_real_court_pos(x: float, y: float) -> str:
    """Return front/mid/back zone label."""
    if y < 1 / 3:
        return "front"
    if y < 2 / 3:
        return "mid"
    return "back"
