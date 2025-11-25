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


def slice_segments(events: Sequence[int], total_len: int, pad: int = 2) -> List[tuple[int, int]]:
    """Create segments around events (for stroke extraction)."""
    segments = []
    for e in events:
        start = max(0, e - pad)
        end = min(total_len, e + pad)
        segments.append((start, end))
    return segments
