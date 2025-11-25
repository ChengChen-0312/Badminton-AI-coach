"""Spatial relations between tracked entities."""

from __future__ import annotations


def relative_left_right(p1, p2) -> str:
    if p1[0] < p2[0] - 5:
        return "left"
    if p1[0] > p2[0] + 5:
        return "right"
    return "aligned"
