"""Extract labelled strokes from video or annotations (placeholder)."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from src.spatial_logic.temporal_logic import slice_segments


def extract_strokes(ball_traj, hit_indices: List[int]) -> List[Tuple[int, int]]:
    """Return stroke segments based on hit indices."""
    total_len = len(ball_traj)
    return slice_segments(hit_indices, total_len, pad=2)
