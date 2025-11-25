from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence

from src.geometry.homography import CourtHomography
from src.geometry.region_definitions import DEFAULT_REGIONS
from src.spatial_logic.events import infer_event_from_trajectory
from src.spatial_logic.landing_detector import BallState, extract_ball_track, infer_landing
from src.spatial_logic.hitter_detector import HitterInfo, infer_hitter_for_stroke
from src.spatial_logic.stroke_reasoner import FinalStroke, combine_classifier_and_event
from src.tracking.player_track import PlayerState


@dataclass
class StrokeSummary:
    frame_range: tuple[int, int]
    final_type: str
    confidence: float
    hitter_role: Optional[str]
    landing_region: Optional[str]
    landing_frame: Optional[int]
    contact_frame: Optional[int] = None
    landing_predicted: bool = False
    classifier_label: Optional[str] = None
    event_type: Optional[str] = None
    hitter_track_id: Optional[int] = None
    hitter_distance: Optional[float] = None


def build_homography_from_analysis(analysis_result: Any) -> Optional[CourtHomography]:
    """Build homography if court corners are present in the analysis result."""
    corners = None
    if hasattr(analysis_result, "court_corners"):
        corners = getattr(analysis_result, "court_corners")
    if corners is None and isinstance(analysis_result, dict):
        corners = analysis_result.get("court_corners")
    if corners is None or len(corners) != 4:
        return None
    return CourtHomography.from_corners(corners)


def summarise_strokes_from_analysis(
    analysis_result: Any,
    classifier_labels: Optional[Sequence[str]] = None,
    hitter_distance_max: float = 200.0,
    enable_hitter_inference: bool = True,
) -> List[StrokeSummary]:
    """
    High-level entry:
      - Input: analyse_video result (needs frame_results)
      - Optional: classifier labels to fuse
      - Output: stroke summaries for reporting
    """
    frame_results = getattr(analysis_result, "frame_results", None)
    if frame_results is None and isinstance(analysis_result, dict):
        frame_results = analysis_result.get("frame_results")

    if not frame_results:
        return []

    track: List[BallState] = extract_ball_track(frame_results)
    if not track:
        # Fallback: create a dummy track at normalized center to avoid empty output
        track = [BallState(frame_idx=0, x=0.5, y=0.5)]

    H = build_homography_from_analysis(analysis_result)
    regions = DEFAULT_REGIONS

    landing = infer_landing(
        frame_results=frame_results,
        homography=H,
        region_classifier=regions,
    )

    hitter_role = "far"
    event = infer_event_from_trajectory(track, landing, hitter_role=hitter_role)

    clf_label = classifier_labels[0] if classifier_labels else None
    final: FinalStroke = combine_classifier_and_event(clf_label, event)

    hitter_info: Optional[HitterInfo] = None
    if enable_hitter_inference and landing is not None and landing.contact_frame_idx is not None:
        # find players at contact frame
        players_at_contact: List[PlayerState] = []
        if frame_results:
            for fr in frame_results:
                idx = getattr(fr, "frame_idx", None) or (fr.get("frame_idx") if isinstance(fr, dict) else None)
                if idx == landing.contact_frame_idx:
                    ps = getattr(fr, "player_states", None) or (fr.get("player_states") if isinstance(fr, dict) else None)
                    if ps:
                        players_at_contact = ps
                    break
        ball_pos = (
            landing.contact_x if landing.contact_x is not None else landing.img_x,
            landing.contact_y if landing.contact_y is not None else landing.img_y,
        )
        hitter_info = infer_hitter_for_stroke(
            contact_frame=landing.contact_frame_idx,
            ball_pos=ball_pos,
            players_at_contact=players_at_contact,
            max_distance=hitter_distance_max,
        )
        if hitter_info:
            final.hitter_role = hitter_info.hitter_role

    summary = StrokeSummary(
        frame_range=(event.start_frame, event.end_frame),
        final_type=final.type,
        confidence=final.confidence,
        hitter_role=final.hitter_role,
        landing_region=final.landing_region,
        landing_frame=landing.frame_idx if landing is not None else None,
        contact_frame=landing.contact_frame_idx if landing is not None else None,
        landing_predicted=landing.predicted if landing is not None else False,
        classifier_label=final.classifier_label,
        event_type=final.event_type,
        hitter_track_id=hitter_info.hitter_track_id if hitter_info else None,
        hitter_distance=hitter_info.distance if hitter_info else None,
    )
    return [summary]


def stroke_summaries_to_dicts(summaries: Sequence[StrokeSummary]) -> List[Dict[str, Any]]:
    """Utility to convert summaries to dicts for JSON/reporting."""
    return [asdict(s) for s in summaries]
