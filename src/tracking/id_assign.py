from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .bytetrack import Track


def assign_player_roles(
    tracks: List[Track],
    frame_height: int,
) -> Dict[int, str]:
    """
    Assign roles 'near' (closer to camera) and 'far' (opposite side) based on bbox vertical position.
    """
    if not tracks:
        return {}

    scored: List[Tuple[int, float]] = []
    for tr in tracks:
        x1, y1, x2, y2 = tr.bbox
        center_y = (y1 + y2) / 2.0
        score = center_y / frame_height  # lower y => smaller score; higher => closer
        scored.append((tr.track_id, score))

    scored.sort(key=lambda x: x[1], reverse=True)

    roles: Dict[int, str] = {}
    if len(scored) >= 1:
        roles[scored[0][0]] = "near"
    if len(scored) >= 2:
        roles[scored[1][0]] = "far"
    return roles
