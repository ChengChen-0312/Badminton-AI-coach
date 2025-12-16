from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence

from src.geometry.homography import CourtHomography
from src.geometry.region_definitions import DEFAULT_GRID9
from src.spatial_logic.events import infer_event_from_trajectory
from src.spatial_logic.landing_detector import BallState, extract_ball_track, infer_landing
from src.spatial_logic.hitter_detector import HitterInfo, infer_hitter_for_stroke
from src.spatial_logic.stroke_reasoner import FinalStroke, combine_classifier_and_event
from src.spatial_logic.pose_features import extract_pose_features
from src.tracking.player_track import PlayerState


@dataclass
class StrokeSummary:
    frame_range: tuple[int, int]
    final_type: str
    confidence: float
    hitter_role: Optional[str] = None
    hitter_x: Optional[float] = None
    hitter_y: Optional[float] = None
    landing_region: Optional[str] = None
    landing_x: Optional[float] = None
    landing_y: Optional[float] = None
    contact_x: Optional[float] = None
    contact_y: Optional[float] = None
    landing_frame: Optional[int] = None
    contact_frame: Optional[int] = None
    landing_predicted: bool = False
    contact_region: Optional[str] = None
    classifier_label: Optional[str] = None
    event_type: Optional[str] = None
    hitter_track_id: Optional[int] = None
    hitter_distance: Optional[float] = None
    pose_landmarks: Optional[Dict] = None
    pose_features: Optional[Dict] = None


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
    pose_window: int = 3,
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
    regions = DEFAULT_GRID9

    landing = infer_landing(frame_results=frame_results, homography=H, region_classifier=regions)

    hitter_role = "far"
    event = infer_event_from_trajectory(track, landing, hitter_role=hitter_role)

    clf_label = classifier_labels[0] if classifier_labels else None
    final: FinalStroke = combine_classifier_and_event(clf_label, event)

    hitter_info: Optional[HitterInfo] = None
    hitter_xy: tuple[float, float] | None = None
    if enable_hitter_inference and landing is not None and landing.contact_frame_idx is not None:
        # find players at contact frame
        players_at_contact: List[PlayerState] = []
        if frame_results:
            for fr in frame_results:
                idx = getattr(fr, "frame_idx", None) if hasattr(fr, "frame_idx") else (fr.get("frame_idx") if isinstance(fr, dict) else None)
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
        # Map hitter position to region if homography available
        if hitter_info and H is not None and players_at_contact:
            hitter = next((p for p in players_at_contact if p.track_id == hitter_info.hitter_track_id), None)
            if hitter and hitter.bboxes:
                hb = hitter.bboxes[-1]
                # Use bottom-center (foot point) as a better proxy for on-court hitter position.
                hx_img, hy_img = (hb[0] + hb[2]) / 2.0, hb[3]
                hx_c, hy_c = H.to_court((hx_img, hy_img))
                hitter_xy = (float(hx_c), float(hy_c))
                if hasattr(regions, "classify_region"):
                    hitter_region = regions.classify_region(hx_c, hy_c)
                else:
                    hitter_region = regions.classify_y(hy_c)
                if hitter_region:
                    landing.contact_region = hitter_region
        if hitter_info:
            final.hitter_role = hitter_info.hitter_role

    # Pose info at contact frame (if available on analysis_result)
    def _find_nearest_pose(pose_results: Dict[int, Dict], contact_frame: int, window: int) -> Optional[Dict]:
        best = None
        best_dist = 1e9
        for f_idx, pose in pose_results.items():
            dist = abs(f_idx - contact_frame)
            if dist <= window and dist < best_dist:
                best_dist = dist
                best = pose
        return best

    pose_landmarks = None
    pose_features = None
    contact_idx = landing.contact_frame_idx if landing is not None else None
    pose_results = None
    if hasattr(analysis_result, "pose_results"):
        pose_results = getattr(analysis_result, "pose_results")
    elif isinstance(analysis_result, dict):
        pose_results = analysis_result.get("pose_results")
    if pose_results and contact_idx is not None:
        if contact_idx in pose_results:
            pose_landmarks = pose_results[contact_idx]
        else:
            pose_landmarks = _find_nearest_pose(pose_results, contact_idx, pose_window)
        if pose_landmarks is not None:
            pose_features = extract_pose_features(pose_landmarks)

    summary = StrokeSummary(
        frame_range=(event.start_frame, event.end_frame),
        final_type=final.type,
        confidence=final.confidence,
        hitter_role=final.hitter_role,
        hitter_x=hitter_xy[0] if hitter_xy is not None else None,
        hitter_y=hitter_xy[1] if hitter_xy is not None else None,
        landing_region=final.landing_region,
        landing_x=landing.court_x if landing is not None else None,
        landing_y=landing.court_y if landing is not None else None,
        contact_x=landing.contact_x if landing is not None else None,
        contact_y=landing.contact_y if landing is not None else None,
        landing_frame=landing.frame_idx if landing is not None else None,
        contact_frame=landing.contact_frame_idx if landing is not None else None,
        landing_predicted=landing.predicted if landing is not None else False,
        contact_region=landing.contact_region if landing is not None else None,
        classifier_label=final.classifier_label,
        event_type=final.event_type,
        hitter_track_id=hitter_info.hitter_track_id if hitter_info else None,
        hitter_distance=hitter_info.distance if hitter_info else None,
        pose_landmarks=pose_landmarks,
        pose_features=pose_features,
    )
    return [summary]


def stroke_summaries_to_dicts(summaries: Sequence[StrokeSummary]) -> List[Dict[str, Any]]:
    """Utility to convert summaries to dicts for JSON/reporting."""
    return [asdict(s) for s in summaries]
