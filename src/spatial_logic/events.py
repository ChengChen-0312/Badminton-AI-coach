from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from .landing_detector import BallState, LandingPoint


@dataclass
class StrokeEvent:
    type: str  # clear / lift / net_shot / drive / smash / unknown
    start_frame: int
    end_frame: int
    hitter_role: Optional[str] = None  # near / far
    landing_region: Optional[str] = None
    landing_point: Optional[LandingPoint] = None


def infer_event_from_trajectory(
    track: Sequence[BallState],
    landing: Optional[LandingPoint],
    hitter_role: Optional[str] = None,
) -> StrokeEvent:
    """Heuristic event type inference from trajectory + landing."""
    if not track:
        return StrokeEvent(
            type="unknown",
            start_frame=0,
            end_frame=0,
            hitter_role=hitter_role,
        )

    start_frame = track[0].frame_idx
    end_frame = track[-1].frame_idx
    landing_region = landing.region if landing else None

    ys = [b.y for b in track]
    dy_total = ys[-1] - ys[0]

    stroke_type = "unknown"
    if landing_region == "front":
        stroke_type = "net_shot"
    elif landing_region == "back":
        if len(ys) > 1 and ys[0] < ys[-1]:
            stroke_type = "clear"
        else:
            stroke_type = "lift"
    else:
        if abs(dy_total) < 10:
            stroke_type = "drive"
        else:
            stroke_type = "smash"

    return StrokeEvent(
        type=stroke_type,
        start_frame=start_frame,
        end_frame=end_frame,
        hitter_role=hitter_role,
        landing_region=landing_region,
        landing_point=landing,
    )
