from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.geometry.homography import CourtHomography
from src.geometry.region_definitions import DEFAULT_GRID9
from src.spatial_logic.events import infer_event_from_trajectory
from src.spatial_logic.landing_detector import BallState, LandingPoint, extract_ball_track, infer_landing
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


def _frame_idx_of(frame_result: Any) -> Optional[int]:
    if hasattr(frame_result, "frame_idx"):
        return getattr(frame_result, "frame_idx")
    if isinstance(frame_result, dict):
        return frame_result.get("frame_idx")
    return None


def _player_states_of(frame_result: Any) -> List[PlayerState]:
    if hasattr(frame_result, "player_states"):
        players = getattr(frame_result, "player_states")
    elif isinstance(frame_result, dict):
        players = frame_result.get("player_states")
    else:
        players = None
    return players or []


def _point_to_bbox_distance(point_xy: Tuple[float, float], bbox: Sequence[float]) -> float:
    px, py = point_xy
    x1, y1, x2, y2 = bbox
    dx = max(float(x1) - px, 0.0, px - float(x2))
    dy = max(float(y1) - py, 0.0, py - float(y2))
    return float((dx * dx + dy * dy) ** 0.5)


def _nearest_player_distance(players: List[PlayerState], bx: float, by: float) -> Optional[float]:
    best: Optional[float] = None
    for p in players:
        if not p.bboxes:
            continue
        d = _point_to_bbox_distance((bx, by), p.bboxes[-1])
        if best is None or d < best:
            best = d
    return best


def _cluster_contact_candidates(
    candidates: List[Tuple[int, float, float, float]],
    min_sep_frames: int,
) -> List[Tuple[int, float, float, float]]:
    if not candidates:
        return []
    candidates = sorted(candidates, key=lambda v: v[0])
    out: List[Tuple[int, float, float, float]] = []
    cluster: List[Tuple[int, float, float, float]] = [candidates[0]]
    for cur in candidates[1:]:
        if int(cur[0]) - int(cluster[-1][0]) <= int(min_sep_frames):
            cluster.append(cur)
            continue
        # pick closest-ball-to-player frame in this local cluster
        out.append(min(cluster, key=lambda v: float(v[1])))
        cluster = [cur]
    if cluster:
        out.append(min(cluster, key=lambda v: float(v[1])))
    return out


def _find_nearest_pose(pose_results: Dict[int, Dict], contact_frame: int, window: int) -> Optional[Dict]:
    best = None
    best_dist = 1e9
    for f_idx, pose in pose_results.items():
        dist = abs(int(f_idx) - int(contact_frame))
        if dist <= int(window) and dist < best_dist:
            best_dist = dist
            best = pose
    return best


def summarise_strokes_from_analysis(
    analysis_result: Any,
    classifier_labels: Optional[Sequence[str]] = None,
    hitter_distance_max: float = 200.0,
    enable_hitter_inference: bool = True,
    pose_window: int = 3,
    contact_distance_px: float = 110.0,
    min_contact_separation_frames: int = 12,
    min_segment_frames: int = 6,
    min_segment_points: int = 3,
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

    raw_track: List[BallState] = extract_ball_track(frame_results)
    track: List[BallState] = sorted(raw_track, key=lambda b: int(b.frame_idx))
    if not track:
        return []

    H = build_homography_from_analysis(analysis_result)
    regions = DEFAULT_GRID9

    frame_by_idx: Dict[int, Any] = {}
    players_by_frame: Dict[int, List[PlayerState]] = {}
    for fr in frame_results:
        fi = _frame_idx_of(fr)
        if fi is None:
            continue
        fi = int(fi)
        frame_by_idx[fi] = fr
        players_by_frame[fi] = _player_states_of(fr)

    pose_results = None
    if hasattr(analysis_result, "pose_results"):
        pose_results = getattr(analysis_result, "pose_results")
    elif isinstance(analysis_result, dict):
        pose_results = analysis_result.get("pose_results")

    # Build contact candidates where shuttle is close to a player.
    contact_candidates: List[Tuple[int, float, float, float]] = []
    for b in track:
        fi = int(b.frame_idx)
        players = players_by_frame.get(fi, [])
        if not players:
            continue
        d = _nearest_player_distance(players, float(b.x), float(b.y))
        if d is None or d > float(contact_distance_px):
            continue
        contact_candidates.append((fi, float(d), float(b.x), float(b.y)))
    contacts = _cluster_contact_candidates(contact_candidates, int(min_contact_separation_frames))

    ball_by_frame = {int(b.frame_idx): b for b in track}
    summaries: List[StrokeSummary] = []
    last_track_frame = int(track[-1].frame_idx)

    # Build segment boundaries from consecutive contacts.
    segment_bounds: List[Tuple[int, int, bool]] = []
    if len(contacts) >= 2:
        for i in range(len(contacts) - 1):
            sf = int(contacts[i][0])
            ef = int(contacts[i + 1][0])
            if ef - sf >= int(min_segment_frames):
                # Intermediate segments end at next contact; not a true physical landing.
                segment_bounds.append((sf, ef, True))
        # Tail segment: last contact to last tracked frame.
        tail_s = int(contacts[-1][0])
        if last_track_frame - tail_s >= int(min_segment_frames):
            segment_bounds.append((tail_s, last_track_frame, False))
    elif len(contacts) == 1:
        sf = int(contacts[0][0])
        if last_track_frame - sf >= int(min_segment_frames):
            segment_bounds.append((sf, last_track_frame, False))

    for seg_idx, (start_f, end_f, forced_predicted) in enumerate(segment_bounds):
        seg_track = [b for b in track if start_f <= int(b.frame_idx) <= end_f]
        if len(seg_track) < int(min_segment_points):
            continue

        contact_state = ball_by_frame.get(start_f, seg_track[0])
        landing_state = seg_track[-1]
        landing = LandingPoint(
            frame_idx=int(landing_state.frame_idx),
            img_x=float(landing_state.x),
            img_y=float(landing_state.y),
            predicted=bool(forced_predicted),
            contact_frame_idx=int(contact_state.frame_idx),
            contact_x=float(contact_state.x),
            contact_y=float(contact_state.y),
        )

        if H is not None:
            lx, ly = H.to_court((landing.img_x, landing.img_y))
            landing.court_x = float(lx)
            landing.court_y = float(ly)
            if hasattr(regions, "classify_region"):
                landing.region = regions.classify_region(lx, ly)
            else:
                landing.region = regions.classify_y(ly)
            cx, cy = H.to_court((landing.contact_x, landing.contact_y))
            if hasattr(regions, "classify_region"):
                landing.contact_region = regions.classify_region(cx, cy)
            else:
                landing.contact_region = regions.classify_y(cy)

        players_at_contact = players_by_frame.get(int(contact_state.frame_idx), [])
        hitter_info: Optional[HitterInfo] = None
        hitter_xy: Optional[Tuple[float, float]] = None
        if enable_hitter_inference and players_at_contact:
            hitter_info = infer_hitter_for_stroke(
                contact_frame=int(contact_state.frame_idx),
                ball_pos=(float(contact_state.x), float(contact_state.y)),
                players_at_contact=players_at_contact,
                max_distance=float(hitter_distance_max),
            )
            if hitter_info and H is not None:
                hitter = next((p for p in players_at_contact if p.track_id == hitter_info.hitter_track_id), None)
                if hitter and hitter.bboxes:
                    hb = hitter.bboxes[-1]
                    hx_img, hy_img = (hb[0] + hb[2]) / 2.0, hb[3]
                    hx_c, hy_c = H.to_court((hx_img, hy_img))
                    hitter_xy = (float(hx_c), float(hy_c))
                    if hasattr(regions, "classify_region"):
                        hitter_region = regions.classify_region(hx_c, hy_c)
                    else:
                        hitter_region = regions.classify_y(hy_c)
                    if hitter_region:
                        landing.contact_region = hitter_region

        hitter_role_seed = hitter_info.hitter_role if hitter_info is not None else "far"
        event = infer_event_from_trajectory(seg_track, landing, hitter_role=hitter_role_seed)

        clf_label = None
        if classifier_labels:
            if len(classifier_labels) == 1:
                clf_label = classifier_labels[0]
            elif seg_idx < len(classifier_labels):
                clf_label = classifier_labels[seg_idx]
        final: FinalStroke = combine_classifier_and_event(clf_label, event)
        if hitter_info:
            final.hitter_role = hitter_info.hitter_role

        pose_landmarks = None
        pose_features = None
        contact_idx = int(contact_state.frame_idx)
        if pose_results and isinstance(pose_results, dict):
            if contact_idx in pose_results:
                pose_landmarks = pose_results[contact_idx]
            else:
                pose_landmarks = _find_nearest_pose(pose_results, contact_idx, int(pose_window))
            if pose_landmarks is not None:
                pose_features = extract_pose_features(pose_landmarks)

        summaries.append(
            StrokeSummary(
                frame_range=(int(seg_track[0].frame_idx), int(seg_track[-1].frame_idx)),
                final_type=final.type,
                confidence=final.confidence,
                hitter_role=final.hitter_role,
                hitter_x=hitter_xy[0] if hitter_xy is not None else None,
                hitter_y=hitter_xy[1] if hitter_xy is not None else None,
                landing_region=final.landing_region,
                landing_x=landing.court_x,
                landing_y=landing.court_y,
                contact_x=landing.contact_x,
                contact_y=landing.contact_y,
                landing_frame=landing.frame_idx,
                contact_frame=landing.contact_frame_idx,
                landing_predicted=bool(landing.predicted),
                contact_region=landing.contact_region,
                classifier_label=final.classifier_label,
                event_type=final.event_type,
                hitter_track_id=hitter_info.hitter_track_id if hitter_info else None,
                hitter_distance=hitter_info.distance if hitter_info else None,
                pose_landmarks=pose_landmarks,
                pose_features=pose_features,
            )
        )

    if summaries:
        return summaries

    # Safe fallback to legacy single-stroke behavior.
    landing = infer_landing(frame_results=frame_results, homography=H, region_classifier=regions)
    if landing is None:
        return []
    event = infer_event_from_trajectory(track, landing, hitter_role="far")
    clf_label = classifier_labels[0] if classifier_labels else None
    final = combine_classifier_and_event(clf_label, event)
    return [
        StrokeSummary(
            frame_range=(event.start_frame, event.end_frame),
            final_type=final.type,
            confidence=final.confidence,
            hitter_role=final.hitter_role,
            landing_region=final.landing_region,
            landing_x=landing.court_x,
            landing_y=landing.court_y,
            contact_x=landing.contact_x,
            contact_y=landing.contact_y,
            landing_frame=landing.frame_idx,
            contact_frame=landing.contact_frame_idx,
            landing_predicted=landing.predicted,
            contact_region=landing.contact_region,
            classifier_label=final.classifier_label,
            event_type=final.event_type,
        )
    ]


def stroke_summaries_to_dicts(summaries: Sequence[StrokeSummary]) -> List[Dict[str, Any]]:
    """Utility to convert summaries to dicts for JSON/reporting."""
    return [asdict(s) for s in summaries]
