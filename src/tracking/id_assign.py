"""Assign stable IDs (P1/P2) to players."""

from __future__ import annotations

from typing import Dict, List, Tuple


def assign_ids(tracks: List[Tuple], prev_ids: Dict[int, str] | None = None) -> Dict[int, str]:
    """Map tracker IDs to human-readable labels (P1/P2)."""
    mapping: Dict[int, str] = {}
    for idx, track in enumerate(tracks):
        mapping[track[0]] = f"P{idx + 1}"
    return mapping
