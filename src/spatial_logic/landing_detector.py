from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Protocol, Sequence

import numpy as np


@dataclass
class BallState:
    frame_idx: int
    x: float
    y: float


@dataclass
class LandingPoint:
    frame_idx: int
    img_x: float
    img_y: float
    predicted: bool = False
    contact_frame_idx: Optional[int] = None
    contact_x: Optional[float] = None
    contact_y: Optional[float] = None
    court_x: Optional[float] = None
    court_y: Optional[float] = None
    region: Optional[str] = None
    contact_region: Optional[str] = None


class HomographyMapper(Protocol):
    """Interface that maps image coordinates to court coordinates."""

    def to_court(self, pt_xy: Sequence[float]) -> Sequence[float]:
        ...


class RegionClassifier(Protocol):
    """Interface that classifies a y-coordinate into a region label."""

    def classify_y(self, court_y: float) -> str:
        ...


def _get_ball_from_frame(fr: Any) -> Optional[Any]:
    """Extract ball object from frame result (supports attribute or dict)."""
    ball = None
    if hasattr(fr, "ball"):
        ball = getattr(fr, "ball")
    elif isinstance(fr, dict) and "ball" in fr:
        ball = fr["ball"]
    return ball


def _get_ball_center(ball: Any) -> Optional[tuple[float, float]]:
    """Extract ball center from various shapes (cx/cy or x/y/w/h)."""
    if ball is None:
        return None

    bbox = getattr(ball, "bbox", None)
    if bbox is not None and len(bbox) >= 4:
        return float((bbox[0] + bbox[2]) / 2.0), float((bbox[1] + bbox[3]) / 2.0)

    cx = getattr(ball, "cx", None)
    cy = getattr(ball, "cy", None)
    if cx is not None and cy is not None:
        return float(cx), float(cy)

    x = getattr(ball, "x", None)
    y = getattr(ball, "y", None)
    w = getattr(ball, "w", None)
    h = getattr(ball, "h", None)
    if None not in (x, y, w, h):
        return float(x + w / 2.0), float(y + h / 2.0)

    if isinstance(ball, dict):
        if "cx" in ball and "cy" in ball:
            return float(ball["cx"]), float(ball["cy"])
        if all(k in ball for k in ("x", "y", "w", "h")):
            return (
                float(ball["x"] + ball["w"] / 2.0),
                float(ball["y"] + ball["h"] / 2.0),
            )

    return None


def extract_ball_track(frame_results: Sequence[Any]) -> List[BallState]:
    """Build a ball track (pixel coordinates) from analyse_video frame results."""
    track: List[BallState] = []
    for fr in frame_results:
        idx = getattr(fr, "frame_idx", None)
        if idx is None and isinstance(fr, dict):
            idx = fr.get("frame_idx")
        if idx is None:
            idx = len(track)

        ball = _get_ball_from_frame(fr)
        if ball is None:
            continue
        # Ignore tracker-only predictions; use only observed detections for landing inference.
        if bool(getattr(ball, "predicted", False)) or (isinstance(ball, dict) and bool(ball.get("predicted", False))):
            continue
        center = _get_ball_center(ball)
        if center is None:
            continue
        x, y = center
        track.append(BallState(frame_idx=int(idx), x=float(x), y=float(y)))
    return track


def smooth_track(track: List[BallState], window: int = 3) -> List[BallState]:
    """Apply a simple moving average smoothing over the track."""
    if len(track) <= 2 or window <= 1:
        return track

    xs = np.array([b.x for b in track], dtype=float)
    ys = np.array([b.y for b in track], dtype=float)

    def _smooth(arr: np.ndarray, k: int) -> np.ndarray:
        pad = k // 2
        padded = np.pad(arr, (pad, pad), mode="edge")
        kernel = np.ones(k, dtype=float) / k
        return np.convolve(padded, kernel, mode="valid")

    xs_s = _smooth(xs, window)
    ys_s = _smooth(ys, window)

    smoothed: List[BallState] = []
    for b, sx, sy in zip(track, xs_s, ys_s):
        smoothed.append(BallState(frame_idx=b.frame_idx, x=float(sx), y=float(sy)))
    return smoothed


def detect_landing_point(
    track: List[BallState],
    missing_window: int = 3,
    min_drop_pixels: float = 5.0,
) -> tuple[Optional[BallState], Optional[BallState]]:
    """
    Heuristic landing detection in pixel space:
    - Assume increasing y means moving downward.
    - Find a frame after which ball is missing for `missing_window` frames and the prior delta-y is significant.
    - Fallback: pick the maximum y point (lowest on screen).
    """
    if not track:
        return None, None

    idx_to_state = {b.frame_idx: b for b in track}
    sorted_frames = sorted(idx_to_state.keys())
    present = set(sorted_frames)

    ys = [idx_to_state[f].y for f in sorted_frames]
    dys = np.diff(ys)

    for i in range(len(sorted_frames) - 1):
        f = sorted_frames[i]
        dy = dys[i] if i < len(dys) else 0.0

        missing = True
        for k in range(1, missing_window + 1):
            if f + k in present:
                missing = False
                break
        if missing and dy > min_drop_pixels:
            contact = idx_to_state[f]
            return contact, idx_to_state[f]

    # fallback
    contact = track[-1]
    return contact, max(track, key=lambda b: b.y)


def infer_landing(
    frame_results: Sequence[Any],
    homography: Optional[HomographyMapper] = None,
    region_classifier: Optional[RegionClassifier] = None,
) -> Optional[LandingPoint]:
    """End-to-end landing inference from frame results."""
    raw_track = extract_ball_track(frame_results)
    if not raw_track:
        return None

    smoothed = smooth_track(raw_track, window=3)
    if not smoothed:
        return None

    # --- Contact (racket hit) inference ---
    ball_by_frame = {b.frame_idx: b for b in smoothed}

    def _frame_idx(fr: Any) -> Optional[int]:
        if hasattr(fr, "frame_idx"):
            return getattr(fr, "frame_idx")
        if isinstance(fr, dict):
            return fr.get("frame_idx")
        return None

    def _player_states(fr: Any) -> List[Any]:
        ps = getattr(fr, "player_states", None) if hasattr(fr, "player_states") else None
        if ps is None and isinstance(fr, dict):
            ps = fr.get("player_states")
        return ps or []

    def _point_to_bbox_dist(px: float, py: float, bbox: Sequence[float]) -> float:
        x1, y1, x2, y2 = bbox
        dx = max(float(x1) - px, 0.0, px - float(x2))
        dy = max(float(y1) - py, 0.0, py - float(y2))
        return float(np.hypot(dx, dy))

    best = None  # (dist, frame_idx, bx, by)
    last_frame_idx = None
    for fr in frame_results:
        fi = _frame_idx(fr)
        if fi is None:
            continue
        last_frame_idx = fi
        ball = ball_by_frame.get(int(fi))
        if ball is None:
            continue
        players = _player_states(fr)
        if not players:
            continue
        bx, by = float(ball.x), float(ball.y)
        for p in players:
            bboxes = getattr(p, "bboxes", None) if hasattr(p, "bboxes") else None
            if bboxes is None and isinstance(p, dict):
                bboxes = p.get("bboxes")
            if not bboxes:
                continue
            bbox = bboxes[-1]
            dist = _point_to_bbox_dist(bx, by, bbox)
            if best is None or dist < best[0]:
                best = (dist, int(fi), bx, by)

    if best is not None:
        contact_frame_idx = int(best[1])
        contact_x = float(best[2])
        contact_y = float(best[3])
    else:
        ys = [b.y for b in smoothed]
        dy_total = (ys[-1] - ys[0]) if len(ys) >= 2 else 0.0
        contact_state = max(smoothed, key=lambda b: b.y) if dy_total < 0 else min(smoothed, key=lambda b: b.y)
        contact_frame_idx = int(contact_state.frame_idx)
        contact_x = float(contact_state.x)
        contact_y = float(contact_state.y)

    # --- Landing inference ---
    # For this lightweight demo pipeline: treat the last observed ball position as landing candidate.
    # If the ball disappears for `missing_window` frames at the end, consider it an observed landing.
    landing_state = smoothed[-1]
    last_frame = int(last_frame_idx) if last_frame_idx is not None else int(landing_state.frame_idx)
    missing_window = 3
    end_gap = max(0, last_frame - int(landing_state.frame_idx))
    landing_predicted = end_gap < missing_window or int(landing_state.frame_idx) <= contact_frame_idx

    lp = LandingPoint(
        frame_idx=int(landing_state.frame_idx),
        img_x=float(landing_state.x),
        img_y=float(landing_state.y),
        predicted=bool(landing_predicted),
        contact_frame_idx=contact_frame_idx,
        contact_x=contact_x,
        contact_y=contact_y,
    )

    if homography is not None:
        cx, cy = homography.to_court((lp.img_x, lp.img_y))
        lp.court_x = float(cx)
        lp.court_y = float(cy)
        if region_classifier is not None:
            if hasattr(region_classifier, "classify_region"):
                lp.region = region_classifier.classify_region(cx, cy)
            elif lp.court_y is not None:
                lp.region = region_classifier.classify_y(lp.court_y)

        if lp.contact_x is not None and lp.contact_y is not None and region_classifier is not None:
            ccx, ccy = homography.to_court((lp.contact_x, lp.contact_y))
            if hasattr(region_classifier, "classify_region"):
                lp.contact_region = region_classifier.classify_region(ccx, ccy)
            else:
                lp.contact_region = region_classifier.classify_y(ccy)

    return lp
