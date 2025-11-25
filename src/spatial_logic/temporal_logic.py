"""Temporal reasoning utilities."""

from __future__ import annotations

from typing import List, Sequence


def smooth_events(events: Sequence[int], window: int = 3) -> List[int]:
    """Simple temporal smoothing over event indices."""
    if not events:
        return []
    smoothed = []
    for idx in events:
        if not smoothed or idx - smoothed[-1] > window:
            smoothed.append(idx)
    return smoothed
