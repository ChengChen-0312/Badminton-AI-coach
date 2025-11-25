"""Region logic for court zones."""

from __future__ import annotations

from src.geometry.region_definitions import zone_from_real_court_pos


def court_zone(x: float, y: float) -> str:
    return zone_from_real_court_pos(x, y)
