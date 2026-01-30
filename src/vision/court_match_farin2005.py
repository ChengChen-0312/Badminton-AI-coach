from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import itertools
import math
import os

import numpy as np
import cv2


# --------------------------
# Real badminton dimensions (meters)
# --------------------------
L_COURT = 13.40
W_DOUBLES = 6.10
W_SINGLES = 5.18
LINE_W_M = 0.040  # 40mm
NET_H_M = 1.524   # not used in planar homography
BUF_BASE_M = 2.0  # safety buffer outside baselines
BUF_SIDE_M = 1.0  # safety buffer outside sidelines


# --------------------------
# Data structures
# --------------------------

@dataclass
class LineSeg:
    p0: np.ndarray  # (2,) float32, image coords (px)
    p1: np.ndarray  # (2,) float32, image coords (px)
    # infinite line ax+by+c=0 normalized
    a: float
    b: float
    c: float
    theta: float  # radians
    support: float
    n_inliers: int


@dataclass
class MatchResult:
    H: Optional[np.ndarray]          # 3x3, model(m) -> image(px)
    corners: Optional[np.ndarray]    # (4,2) float32, image corners of doubles boundary: lb,rb,rt,lt
    E: float
    reason: str
    metrics: Dict[str, Any]


# --------------------------
# Model (meters)
# --------------------------

def _P(x: float, y: float) -> np.ndarray:
    return np.array([x, y], np.float32)


def badminton_model_segments_m(include_safety: bool = False) -> List[Tuple[np.ndarray, np.ndarray, str]]:
    """
    Court-line model in meters.
    We include only lines you provided explicitly (dimensions + net + singles/doubles boundaries).
    (Service lines etc. can be added later if you want.)
    """
    dx = (W_DOUBLES - W_SINGLES) / 2.0  # 0.46m

    segs: List[Tuple[np.ndarray, np.ndarray, str]] = []

    # Doubles outer boundary
    segs += [
        (_P(0, 0), _P(W_DOUBLES, 0), "doubles_near_baseline"),
        (_P(W_DOUBLES, 0), _P(W_DOUBLES, L_COURT), "doubles_right_sideline"),
        (_P(W_DOUBLES, L_COURT), _P(0, L_COURT), "doubles_far_baseline"),
        (_P(0, L_COURT), _P(0, 0), "doubles_left_sideline"),
    ]

    # Singles sidelines
    segs += [
        (_P(dx, 0), _P(dx, L_COURT), "singles_left_sideline"),
        (_P(W_DOUBLES - dx, 0), _P(W_DOUBLES - dx, L_COURT), "singles_right_sideline"),
    ]

    # Net line (on the floor projection)
    segs += [
        (_P(0, L_COURT / 2.0), _P(W_DOUBLES, L_COURT / 2.0), "net_line"),
    ]

    if include_safety:
        segs += [
            (_P(-BUF_SIDE_M, -BUF_BASE_M), _P(W_DOUBLES + BUF_SIDE_M, -BUF_BASE_M), "safety_near"),
            (_P(W_DOUBLES + BUF_SIDE_M, -BUF_BASE_M), _P(W_DOUBLES + BUF_SIDE_M, L_COURT + BUF_BASE_M), "safety_right"),
            (_P(W_DOUBLES + BUF_SIDE_M, L_COURT + BUF_BASE_M), _P(-BUF_SIDE_M, L_COURT + BUF_BASE_M), "safety_far"),
            (_P(-BUF_SIDE_M, L_COURT + BUF_BASE_M), _P(-BUF_SIDE_M, -BUF_BASE_M), "safety_left"),
        ]

    return segs


def model_corners_m() -> np.ndarray:
    # Doubles boundary corners in model coords (meters), order lb, rb, rt, lt (image y down)
    return np.array(
        [
            [0.0, L_COURT],         # lb
            [W_DOUBLES, L_COURT],   # rb
            [W_DOUBLES, 0.0],       # rt
            [0.0, 0.0],             # lt
        ],
        np.float32,
    )


def model_aspect_ratio_m() -> float:
    # height/width = length/width for doubles boundary
    return L_COURT / W_DOUBLES


def _is_close(a: float, b: float, eps: float = 1e-4) -> bool:
    return abs(a - b) <= eps


def _outer_boundary_indices(
    model_segs: List[Tuple[np.ndarray, np.ndarray, str]],
) -> Optional[Tuple[int, int, int, int]]:
    """
    Identify outer boundary lines via endpoints (meters), independent of tag names.
      y=0   near baseline
      y=L   far baseline
      x=0   left doubles sideline
      x=W   right doubles sideline
    """
    idx_y0: list[int] = []
    idx_yL: list[int] = []
    idx_x0: list[int] = []
    idx_xW: list[int] = []

    for i, (p, q, _tag) in enumerate(model_segs):
        x1, y1 = float(p[0]), float(p[1])
        x2, y2 = float(q[0]), float(q[1])
        if _is_close(y1, 0.0) and _is_close(y2, 0.0):
            idx_y0.append(i)
        if _is_close(y1, L_COURT) and _is_close(y2, L_COURT):
            idx_yL.append(i)
        if _is_close(x1, 0.0) and _is_close(x2, 0.0):
            idx_x0.append(i)
        if _is_close(x1, W_DOUBLES) and _is_close(x2, W_DOUBLES):
            idx_xW.append(i)

    if idx_y0 and idx_yL and idx_x0 and idx_xW:
        return idx_y0[0], idx_yL[0], idx_x0[0], idx_xW[0]
    return None


def _quad_edge_ar_lb_rb_rt_lt(quad: np.ndarray) -> float:
    """
    Aspect ratio from edge lengths (more stable than bbox AR under perspective).
    quad order: lb, rb, rt, lt.
    """
    q = np.asarray(quad, np.float32).reshape(4, 2)
    lb, rb, rt, lt = q[0], q[1], q[2], q[3]
    len1 = float(np.linalg.norm(lb - lt))
    len2 = float(np.linalg.norm(rb - rt))
    wid1 = float(np.linalg.norm(lb - rb))
    wid2 = float(np.linalg.norm(lt - rt))
    Lm = 0.5 * (len1 + len2)
    Wm = 0.5 * (wid1 + wid2)
    return float(Lm / max(1e-6, Wm))


def _quad_oob_too_far(quad: np.ndarray, img_wh: Tuple[int, int]) -> bool:
    """Allow corners outside, but reject if they fly too far."""
    w, h = img_wh
    q = np.asarray(quad, np.float32).reshape(4, 2)
    frac = float(os.environ.get("BADC_FARIN_OOB_FRAC", "0.25"))
    xmin_ok = -frac * w
    xmax_ok = (1.0 + frac) * w
    ymin_ok = -frac * h
    ymax_ok = (1.0 + frac) * h
    if np.any(q[:, 0] < xmin_ok) or np.any(q[:, 0] > xmax_ok) or np.any(q[:, 1] < ymin_ok) or np.any(q[:, 1] > ymax_ok):
        return True
    return False

# --------------------------
# Geometry helpers
# --------------------------

def _line_from_points(p0: np.ndarray, p1: np.ndarray) -> Tuple[float, float, float, float]:
    x0, y0 = float(p0[0]), float(p0[1])
    x1, y1 = float(p1[0]), float(p1[1])
    a = y0 - y1
    b = x1 - x0
    c = x0 * y1 - x1 * y0
    n = math.hypot(a, b) + 1e-9
    a /= n
    b /= n
    c /= n
    # direction angle of the segment (not normal)
    theta = math.atan2((y1 - y0), (x1 - x0))
    return a, b, c, theta


def _angle_mod_pi(t: float) -> float:
    return t % math.pi


def _angle_diff(a: float, b: float) -> float:
    d = abs((a - b) % math.pi)
    return min(d, math.pi - d)


def _project_points(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, np.float32).reshape(-1, 2)
    ones = np.ones((pts.shape[0], 1), np.float32)
    p = np.concatenate([pts, ones], axis=1)  # (N,3)
    q = (H @ p.T).T
    z = q[:, 2:3]
    z = np.where(np.abs(z) < 1e-9, 1e-9, z)
    return q[:, :2] / z


def _poly_area(quad: np.ndarray) -> float:
    q = np.asarray(quad, np.float32).reshape(4, 2)
    x = q[:, 0]
    y = q[:, 1]
    return 0.5 * float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _extent_and_ar(quad: np.ndarray) -> Tuple[float, float, float]:
    q = np.asarray(quad, np.float32).reshape(4, 2)
    xmin, xmax = float(q[:, 0].min()), float(q[:, 0].max())
    ymin, ymax = float(q[:, 1].min()), float(q[:, 1].max())
    bw = max(1e-6, xmax - xmin)
    bh = max(1e-6, ymax - ymin)
    return bw, bh, (bh / bw)


def compute_dt_to_lines(mask_noblob: np.ndarray) -> np.ndarray:
    """
    Distance Transform to nearest white pixel (court line).
    dt[y,x] = distance (pixels) from (x,y) to nearest white pixel.
    """
    m = (mask_noblob > 0).astype(np.uint8)
    inv = (1 - m).astype(np.uint8)
    dt = cv2.distanceTransform(inv, cv2.DIST_L2, 3)
    return dt.astype(np.float32)


def _order_corners_lb_rb_rt_lt(pts_xy: np.ndarray) -> np.ndarray:
    pts = np.array(pts_xy, dtype=np.float32).reshape(4, 2)
    sums = pts[:, 0] + pts[:, 1]
    diffs = pts[:, 0] - pts[:, 1]
    tl = pts[np.argmin(sums)]
    br = pts[np.argmax(sums)]
    tr = pts[np.argmin(diffs)]
    bl = pts[np.argmax(diffs)]
    return np.stack([bl, br, tr, tl], axis=0)


def _is_convex_quad(quad: np.ndarray) -> bool:
    q = np.array(quad, dtype=np.float32).reshape(4, 2)
    cross = []
    for i in range(4):
        p0 = q[i]
        p1 = q[(i + 1) % 4]
        p2 = q[(i + 2) % 4]
        v1 = p1 - p0
        v2 = p2 - p1
        cross.append(np.cross(v1, v2))
    cross = np.array(cross, dtype=np.float32)
    return bool(np.all(cross >= 0) or np.all(cross <= 0))


def _intersect_lines(l1: Tuple[float, float, float], l2: Tuple[float, float, float]) -> Optional[np.ndarray]:
    a1, b1, c1 = l1
    a2, b2, c2 = l2
    d = a1 * b2 - a2 * b1
    if abs(d) < 1e-9:
        return None
    x = (b1 * c2 - b2 * c1) / d
    y = (c1 * a2 - c2 * a1) / d
    if not (math.isfinite(x) and math.isfinite(y)):
        return None
    return np.array([x, y], np.float32)


def _line_tuple(seg: LineSeg) -> Tuple[float, float, float]:
    return (float(seg.a), float(seg.b), float(seg.c))


def dt_error_E(
    H: np.ndarray,
    dt: np.ndarray,
    img_wh: Tuple[int, int],
    samples_per_seg: int = 30,
    dt_cap_px: float = 12.0,
    normalize_by: float = 3.0,
) -> Tuple[float, Dict[str, float]]:
    """
    Project model line samples to image, read dt at those pixels.
    Sum capped dt. Lower is better.
    Returns (E_dt, stats).
    """
    w, h = img_wh
    model_segs = badminton_model_segments_m(include_safety=False)

    total = 0.0
    inside = 0
    count = 0

    for p, q, tag in model_segs:
        if tag.startswith("safety_"):
            continue
        ts = np.linspace(0.0, 1.0, samples_per_seg, dtype=np.float32)
        pts_m = (p[None, :] * (1.0 - ts[:, None]) + q[None, :] * ts[:, None]).astype(np.float32)
        pts_i = _project_points(H, pts_m)

        for (x, y) in pts_i:
            xi = int(round(float(x)))
            yi = int(round(float(y)))
            count += 1
            if 0 <= xi < w and 0 <= yi < h:
                inside += 1
                d = float(dt[yi, xi])
                total += min(d, dt_cap_px)
            else:
                total += dt_cap_px

    inside_ratio = inside / max(1, count)
    E = total / max(1e-6, float(normalize_by))
    return float(E), {"inside_ratio": float(inside_ratio), "dt_cap_px": float(dt_cap_px)}


def _quick_reject_allow_outside(H: np.ndarray, img_wh: Tuple[int, int]) -> Tuple[bool, Dict[str, Any]]:
    w, h = img_wh
    quad = _project_points(H, model_corners_m())

    # reject if corners fly too far
    oob_far = _quad_oob_too_far(quad, img_wh)
    if oob_far:
        return False, {"oob_far": True}

    xmin, xmax = float(quad[:, 0].min()), float(quad[:, 0].max())
    ymin, ymax = float(quad[:, 1].min()), float(quad[:, 1].max())
    inter_w = max(0.0, min(xmax, w - 1.0) - max(xmin, 0.0))
    inter_h = max(0.0, min(ymax, h - 1.0) - max(ymin, 0.0))
    inter_area = inter_w * inter_h
    overlap_ratio = inter_area / float(w * h)

    min_overlap = float(os.environ.get("BADC_FARIN_MIN_OVERLAP_RATIO", "0.08"))
    ok_overlap = overlap_ratio >= min_overlap

    # edge-length aspect ratio gate (semantic court shape)
    ar_model = model_aspect_ratio_m()
    ar_edge = _quad_edge_ar_lb_rb_rt_lt(quad)
    ar_lo = float(os.environ.get("BADC_FARIN_AR_LO", "0.60"))
    ar_hi = float(os.environ.get("BADC_FARIN_AR_HI", "1.60"))
    ok_ar = (ar_edge >= ar_model * ar_lo) and (ar_edge <= ar_model * ar_hi)

    return (ok_overlap and ok_ar), {
        "oob_far": False,
        "overlap_ratio": float(overlap_ratio),
        "min_overlap_ratio": float(min_overlap),
        "ok_overlap": bool(ok_overlap),
        "ar_edge": float(ar_edge),
        "ar_model": float(ar_model),
        "ar_lo": float(ar_lo),
        "ar_hi": float(ar_hi),
        "ok_ar": bool(ok_ar),
    }


def _cluster_two_orientations(segs: List[LineSeg], tol_deg: float = 12.0) -> Tuple[List[LineSeg], List[LineSeg]]:
    if not segs:
        return [], []
    segs_sorted = sorted(segs, key=lambda s: s.support, reverse=True)
    base = _angle_mod_pi(segs_sorted[0].theta)
    tol = math.radians(tol_deg)
    grpA: List[LineSeg] = []
    grpB: List[LineSeg] = []
    baseB: Optional[float] = None
    for s in segs_sorted:
        ang = _angle_mod_pi(s.theta)
        dA = _angle_diff(ang, base)
        if dA <= tol:
            grpA.append(s)
            continue
        if baseB is None:
            baseB = ang
            grpB.append(s)
            continue
        dB = _angle_diff(ang, baseB)
        if dB <= tol or dB < dA:
            grpB.append(s)
        else:
            grpA.append(s)
    return grpA, grpB


def _pick_extreme_parallel_pair(segs: List[LineSeg], img_wh: Tuple[int, int]) -> Optional[Tuple[LineSeg, LineSeg]]:
    if len(segs) < 2:
        return None
    best = None
    best_sep = -1.0
    for s0, s1 in itertools.combinations(segs, 2):
        sep = abs(float(s0.c) - float(s1.c))
        if sep > best_sep:
            best_sep = sep
            best = (s0, s1)
    return best


def _order_quad_from_lines(
    v1: Tuple[float, float, float],
    v2: Tuple[float, float, float],
    h1: Tuple[float, float, float],
    h2: Tuple[float, float, float],
    img_wh: Tuple[int, int],
) -> Optional[np.ndarray]:
    p00 = _intersect_lines(v1, h1)
    p01 = _intersect_lines(v2, h1)
    p11 = _intersect_lines(v2, h2)
    p10 = _intersect_lines(v1, h2)
    if any(pt is None for pt in (p00, p01, p11, p10)):
        return None
    quad = np.stack([p00, p01, p11, p10], axis=0)
    ordered = _order_corners_lb_rb_rt_lt(quad)
    if not _is_convex_quad(ordered):
        return None
    if _poly_area(ordered) < 1.0:
        return None
    return ordered


# --------------------------
# Tau (pixels) tied to real line width 40mm via candidate H
# --------------------------

def tau_from_H(H: np.ndarray, line_w_m: float = LINE_W_M) -> float:
    """
    Estimate the expected line width in pixels around court center, then set tau accordingly.
    """
    p = np.array([[W_DOUBLES / 2.0, L_COURT / 2.0]], np.float32)
    p2 = np.array([[W_DOUBLES / 2.0 + line_w_m, L_COURT / 2.0]], np.float32)
    u = _project_points(H, p)[0]
    v = _project_points(H, p2)[0]
    px = float(np.linalg.norm(u - v))
    # tau: allow some tolerance beyond half line width
    tau = max(2.0, 0.6 * px)
    return float(np.clip(tau, 2.0, 8.0))


# --------------------------
# Paper Step 2: RANSAC line segment extraction from mask pixels
# score = sum(max(tau - dist, 0)) over points
# --------------------------

def _ransac_dominant_line(points: np.ndarray, tau_px: float, n_hyp: int, rng: np.random.Generator):
    if points.shape[0] < 80:
        return None, 0.0

    best = None
    best_score = 0.0
    N = points.shape[0]

    for _ in range(n_hyp):
        i, j = rng.integers(0, N, size=2)
        if i == j:
            continue
        p0 = points[i]
        p1 = points[j]
        a, b, c, theta = _line_from_points(p0, p1)
        d = np.abs(a * points[:, 0] + b * points[:, 1] + c)
        s = float(np.sum(np.maximum(tau_px - d, 0.0)))
        if s > best_score:
            best_score = s
            best = (a, b, c, theta)

    return best, best_score


def _pca_refine_line(inliers: np.ndarray) -> Tuple[float, float, float, float]:
    pts = inliers.astype(np.float32)
    mu = np.mean(pts, axis=0)
    X = pts - mu
    cov = (X.T @ X) / max(1, pts.shape[0])
    w, V = np.linalg.eigh(cov)
    v = V[:, np.argmax(w)]  # direction
    n = np.array([-v[1], v[0]], np.float32)  # normal
    n = n / (np.linalg.norm(n) + 1e-9)
    a, b = float(n[0]), float(n[1])
    c = -(a * float(mu[0]) + b * float(mu[1]))
    # normalize
    nn = math.hypot(a, b) + 1e-9
    a /= nn
    b /= nn
    c /= nn
    theta = math.atan2(float(v[1]), float(v[0]))
    return a, b, c, theta


def _segment_boundaries_maxsubarray(mask_u8: np.ndarray, a: float, b: float, c: float) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Paper-style boundary detection along the line:
    choose [start,end] minimizing (#black inside + #white outside).
    Equivalent to maximum subarray where white=+1 black=-1.
    """
    h, w = mask_u8.shape[:2]

    # direction along line
    v = np.array([b, -a], np.float32)
    v = v / (np.linalg.norm(v) + 1e-9)

    # anchor: projection of image center onto line
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    t = -(a * cx + b * cy + c)  # since (a,b) normalized
    anchor = np.array([cx + a * t, cy + b * t], np.float32)

    # sample along both directions
    pts = []
    for sign in (-1.0, 1.0):
        p = anchor.copy()
        for _ in range(max(h, w) * 2):
            x, y = float(p[0]), float(p[1])
            if x < -2 or x > w + 2 or y < -2 or y > h + 2:
                break
            pts.append((x, y))
            p += sign * v

    if len(pts) < 80:
        return None

    pts = np.array(pts, np.float32)
    xs = np.rint(pts[:, 0]).astype(np.int32)
    ys = np.rint(pts[:, 1]).astype(np.int32)
    inside = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    xs = xs[inside]
    ys = ys[inside]
    if xs.size < 80:
        return None

    seq = (mask_u8[ys, xs] > 0).astype(np.int32)
    val = np.where(seq == 1, 1, -1).astype(np.int32)

    best_sum = -10**9
    best_i = 0
    best_j = 0
    cur_sum = 0
    cur_i = 0
    for j in range(val.size):
        if cur_sum <= 0:
            cur_sum = int(val[j])
            cur_i = j
        else:
            cur_sum += int(val[j])
        if cur_sum > best_sum:
            best_sum = cur_sum
            best_i = cur_i
            best_j = j

    if best_j - best_i < 25:
        return None

    p0 = np.array([xs[best_i], ys[best_i]], np.float32)
    p1 = np.array([xs[best_j], ys[best_j]], np.float32)
    return p0, p1


def extract_line_segments_from_mask(
    mask: np.ndarray,
    tau_px: float,
    max_lines: int = 12,
    min_support: float = 800.0,
    n_hyp: int = 25,
    rng_seed: int = 0,
) -> List[LineSeg]:
    """
    Extract dominant line segments iteratively (paper Step 2).
    tau_px here is pixel tolerance used in RANSAC scoring/inliers.
    """
    work_mask = (mask > 0).astype(np.uint8) * 255
    rng = np.random.default_rng(rng_seed)

    segs: List[LineSeg] = []

    for _ in range(max_lines):
        ys, xs = np.where(work_mask > 0)
        if xs.size < 250:
            break
        pts = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)

        best, score = _ransac_dominant_line(pts, tau_px=tau_px, n_hyp=n_hyp, rng=rng)
        if best is None or score < min_support:
            break

        a, b, c, theta = best
        d = np.abs(a * pts[:, 0] + b * pts[:, 1] + c)
        inlier_idx = np.where(d <= tau_px)[0]
        if inlier_idx.size < 100:
            break

        inliers = pts[inlier_idx]
        a, b, c, theta = _pca_refine_line(inliers)

        bounds = _segment_boundaries_maxsubarray(work_mask, a, b, c)
        if bounds is None:
            # fallback: extremes along direction
            v = np.array([b, -a], np.float32)
            v = v / (np.linalg.norm(v) + 1e-9)
            tproj = inliers @ v
            p0 = inliers[int(np.argmin(tproj))]
            p1 = inliers[int(np.argmax(tproj))]
        else:
            p0, p1 = bounds

        seg = LineSeg(
            p0=p0.astype(np.float32),
            p1=p1.astype(np.float32),
            a=float(a),
            b=float(b),
            c=float(c),
            theta=float(theta),
            support=float(score),
            n_inliers=int(inlier_idx.size),
        )
        segs.append(seg)

        # remove pixels along this segment (thickness tied to tau)
        erase = np.zeros_like(work_mask)
        thickness = int(max(3, round(tau_px * 2.0)))
        cv2.line(erase, tuple(np.int32(seg.p0)), tuple(np.int32(seg.p1)), 255, thickness=thickness)
        work_mask[erase > 0] = 0

    return segs


# --------------------------
# Paper Step 3: Model fitting
# - fast: parallel segment pairs -> 4 endpoints -> H
# - quick rejection: aspect ratio + area ratio
# - error E: sample model segments -> distance to nearest image line (orientation-gated), capped
# --------------------------

def _parallel_pairs(segs: List[LineSeg], ang_tol_deg: float = 10.0) -> List[Tuple[int, int]]:
    ang_tol = math.radians(ang_tol_deg)
    pairs = []
    for i, j in itertools.combinations(range(len(segs)), 2):
        ai = _angle_mod_pi(segs[i].theta)
        aj = _angle_mod_pi(segs[j].theta)
        if _angle_diff(ai, aj) <= ang_tol:
            pairs.append((i, j))
    return pairs


def _model_parallel_pairs(model_segs: List[Tuple[np.ndarray, np.ndarray, str]]) -> List[Tuple[int, int]]:
    """
    Prefer doubles outer boundary; fallback to all parallels if disabled/missing.
    """
    allow_interior = os.environ.get("BADC_FARIN_ALLOW_INTERIOR", "0") == "1"
    if not allow_interior:
        out = _outer_boundary_indices(model_segs)
        if out is not None:
            i_y0, i_yL, i_x0, i_xW = out
            return [(i_y0, i_yL), (i_x0, i_xW)]
        # if failed to identify, fall back to interior mode
    angs = []
    for p, q, _ in model_segs:
        v = q - p
        th = math.atan2(float(v[1]), float(v[0]))
        angs.append(_angle_mod_pi(th))
    pairs = []
    for i, j in itertools.combinations(range(len(model_segs)), 2):
        if _angle_diff(angs[i], angs[j]) <= math.radians(1.0):
            pairs.append((i, j))
    return pairs


def _compute_H_from_2seg_pairs(
    model_pair: List[Tuple[np.ndarray, np.ndarray, str]],
    img_pair: Tuple[LineSeg, LineSeg],
) -> List[np.ndarray]:
    """
    4 endpoints define H (paper fast fitting).
    Try flips + swapping.
    """
    (mp0, mq0, _t0) = model_pair[0]
    (mp1, mq1, _t1) = model_pair[1]
    s0, s1 = img_pair

    src = np.stack([mp0, mq0, mp1, mq1], axis=0).astype(np.float32)

    Hs: List[np.ndarray] = []
    for flip0 in (False, True):
        for flip1 in (False, True):
            p0 = s0.p0 if not flip0 else s0.p1
            q0 = s0.p1 if not flip0 else s0.p0
            p1 = s1.p0 if not flip1 else s1.p1
            q1 = s1.p1 if not flip1 else s1.p0

            dst = np.stack([p0, q0, p1, q1], axis=0).astype(np.float32)

            # also try swapping segment assignment
            for perm in ((0, 1, 2, 3), (2, 3, 0, 1)):
                dst2 = dst[list(perm)]
                H = cv2.getPerspectiveTransform(src, dst2)
                if np.isfinite(H).all():
                    Hs.append(H)

    return Hs


def _quick_reject(H: np.ndarray, img_wh: Tuple[int, int]) -> Tuple[bool, Dict[str, Any]]:
    w, h = img_wh
    qc = _project_points(H, model_corners_m())

    # reject if corners fly too far
    oob_far = _quad_oob_too_far(qc, img_wh)
    if oob_far:
        return False, {"oob_far": True}

    # Use edge-length AR instead of bbox AR (more semantic)
    ar_model = model_aspect_ratio_m()
    ar_edge = _quad_edge_ar_lb_rb_rt_lt(qc)
    ar_lo = float(os.environ.get("BADC_FARIN_AR_LO", "0.60"))
    ar_hi = float(os.environ.get("BADC_FARIN_AR_HI", "1.60"))
    ok_ar = (ar_edge >= ar_model * ar_lo) and (ar_edge <= ar_model * ar_hi)

    area = _poly_area(qc)
    area_ratio = area / float(w * h)
    min_area_ratio = float(os.environ.get("BADC_FARIN_MIN_AREA_RATIO", "0.10"))
    ok_area = area_ratio >= min_area_ratio

    return (ok_ar and ok_area), {
        "oob_far": False,
        "ar_edge": float(ar_edge),
        "ar_model": float(ar_model),
        "ok_ar": bool(ok_ar),
        "ar_lo": float(ar_lo),
        "ar_hi": float(ar_hi),
        "area": float(area),
        "area_ratio": float(area_ratio),
        "min_area_ratio": float(min_area_ratio),
        "ok_area": bool(ok_area),
    }


def _pt_line_dist(seg: LineSeg, pt: np.ndarray) -> float:
    return abs(seg.a * float(pt[0]) + seg.b * float(pt[1]) + seg.c)


def model_error_E(
    H: np.ndarray,
    img_segs: List[LineSeg],
    tau_px: float,
    em: float,
    samples_per_seg: int = 30,
    ang_gate_deg: float = 25.0,
) -> float:
    """
    Error E: sample points along each model segment, project to image,
    take distance to nearest *orientation-compatible* image line, capped by em.
    Smaller is better.

    tau_px enters as the meaningful distance scale (linked to 40mm).
    """
    model_segs = badminton_model_segments_m(include_safety=False)
    ang_gate = math.radians(ang_gate_deg)

    total = 0.0
    for p, q, tag in model_segs:
        if tag.startswith("safety_"):
            continue

        ts = np.linspace(0.0, 1.0, samples_per_seg, dtype=np.float32)
        pts_m = (p[None, :] * (1.0 - ts[:, None]) + q[None, :] * ts[:, None]).astype(np.float32)
        pts_i = _project_points(H, pts_m)

        # model segment angle in model space
        v = q - p
        th_m = _angle_mod_pi(math.atan2(float(v[1]), float(v[0])))

        for pt in pts_i:
            best = em
            for s in img_segs:
                th_i = _angle_mod_pi(s.theta)
                if _angle_diff(th_m, th_i) > ang_gate:
                    continue
                d = _pt_line_dist(s, pt)
                if d < best:
                    best = d
            # scale/cap: treat >tau as quickly saturating
            total += min(best, em)

    # normalize by tau to keep numbers comparable across zoom levels
    return float(total / max(1e-6, tau_px))


# --------------------------
# Main entry
# --------------------------

def fit_model_farin2005_m(
    mask_noblob: np.ndarray,
    img_shape: Tuple[int, int],
) -> MatchResult:
    """
    Full meters-based Farin-style matcher (from noblob mask).
    Returns H (model meters -> image pixels) + doubles corners.
    """
    h, w = img_shape
    img_wh = (w, h)

    dt = compute_dt_to_lines(mask_noblob)

    # --- Step 2: extract image line segments (needs tau before H exists) ---
    tau0 = float(os.environ.get("BADC_FARIN_TAU0_PX", "3.0"))  # initial tau for extraction
    n_hyp = int(os.environ.get("BADC_FARIN_N_HYP", "25"))
    min_support = float(os.environ.get("BADC_FARIN_MIN_SUPPORT", "800.0"))
    max_lines = int(os.environ.get("BADC_FARIN_MAX_LINES", "12"))

    y0_focus = int(h * float(os.environ.get("BADC_FARIN_FOCUS_Y0", "0.55")))
    mask_focus = np.zeros_like(mask_noblob)
    mask_focus[y0_focus:, :] = mask_noblob[y0_focus:, :]

    img_segs = extract_line_segments_from_mask(
        mask_focus,
        tau_px=tau0,
        max_lines=max_lines,
        min_support=min_support,
        n_hyp=n_hyp,
        rng_seed=0,
    )
    if len(img_segs) < 4:
        return MatchResult(
            H=None,
            corners=None,
            E=float("inf"),
            reason="R_not_enough_lines",
            metrics={"n_img_segs": len(img_segs), "tau0": tau0},
        )

    # --- Step 3: fast fitting from parallel pairs ---
    model_segs = badminton_model_segments_m(include_safety=False)
    img_pairs = _parallel_pairs(img_segs, ang_tol_deg=float(os.environ.get("BADC_FARIN_PAR_TOL_DEG", "10.0")))
    model_pairs = _model_parallel_pairs(model_segs)

    if not img_pairs:
        return MatchResult(
            H=None,
            corners=None,
            E=float("inf"),
            reason="R_no_parallel_pairs",
            metrics={"n_img_segs": len(img_segs), "tau0": tau0},
        )

    best_H = None
    best_E = float("inf")
    best_meta: Dict[str, Any] = {"mode": None}
    reject_stats = {
        "tried": 0,
        "rej_quick": 0,
        "rej_oob": 0,
        "rej_ar": 0,
        "rej_area": 0,
        "rej_overlap": 0,
        "rej_dt": 0,
        "kept": 0,
    }

    # em is a cap in pixel distance (before normalization by tau)
    # make it adaptive to image size (avoid too small/too big)
    em = float(os.environ.get("BADC_FARIN_EM_PX", str(max(12.0, 0.02 * max(w, h)))))

    for (mi, mj) in model_pairs:
        model_pair = [model_segs[mi], model_segs[mj]]
        for (ii, ij) in img_pairs:
            img_pair = (img_segs[ii], img_segs[ij])
            for H in _compute_H_from_2seg_pairs(model_pair, img_pair):
                reject_stats["tried"] += 1
                ok, rej = _quick_reject(H, img_wh)
                if not ok:
                    reject_stats["rej_quick"] += 1
                    if rej.get("oob_far"):
                        reject_stats["rej_oob"] += 1
                    if not rej.get("ok_ar", True):
                        reject_stats["rej_ar"] += 1
                    if not rej.get("ok_area", True):
                        reject_stats["rej_area"] += 1
                    continue
                if not rej.get("ok_overlap", True):
                    reject_stats["rej_overlap"] += 1
                    continue
                tauH = tau_from_H(H, LINE_W_M)
                E_line = model_error_E(H, img_segs, tau_px=tauH, em=em)
                dt_cap = float(os.environ.get("BADC_FARIN_DT_CAP_PX", "12.0"))
                E_dt, dt_stats = dt_error_E(
                    H,
                    dt,
                    img_wh,
                    samples_per_seg=30,
                    dt_cap_px=dt_cap,
                    normalize_by=tauH,
                )
                dt_gate = float(os.environ.get("BADC_FARIN_DT_GATE", "250.0"))
                if E_dt > dt_gate:
                    reject_stats["rej_dt"] += 1
                    continue
                w_line = float(os.environ.get("BADC_FARIN_W_LINE", "1.0"))
                w_dt = float(os.environ.get("BADC_FARIN_W_DT", "2.0"))
                E = w_line * E_line + w_dt * E_dt
                if E < best_E:
                    best_E = E
                    best_H = H
                    best_meta = {
                        "mode": "fast",
                        "mi": int(mi),
                        "mj": int(mj),
                        "ii": int(ii),
                        "ij": int(ij),
                        "tauH": float(tauH),
                        "em": float(em),
                        "E_line": float(E_line),
                        "E_dt": float(E_dt),
                        "dt_gate": float(dt_gate),
                        "w_line": float(w_line),
                        "w_dt": float(w_dt),
                        **dt_stats,
                        **rej,
                        "E": float(E),
                    }
                reject_stats["kept"] += 1

    # --- Robust fallback (very small, still paper-shaped) ---
    # If fast fails completely, try 2+2 from top segments (orthogonal groups).
    if best_H is None:
        top = sorted(img_segs, key=lambda s: s.support, reverse=True)[:8]
        base = _angle_mod_pi(top[0].theta)
        grpA = [s for s in top if _angle_diff(_angle_mod_pi(s.theta), base) <= math.radians(12)]
        grpB = [s for s in top if _angle_diff(_angle_mod_pi(s.theta), base) > math.radians(12)]

        def intersect(l1: Tuple[float, float, float], l2: Tuple[float, float, float]) -> Optional[np.ndarray]:
            a1, b1, c1 = l1
            a2, b2, c2 = l2
            D = a1 * b2 - a2 * b1
            if abs(D) < 1e-9:
                return None
            x = (b1 * c2 - b2 * c1) / D
            y = (c1 * a2 - c2 * a1) / D
            return np.array([x, y], np.float32)

        # model infinite lines: x=0, x=W, y=0, y=L
        mH0 = _line_from_points(_P(0, 0), _P(W_DOUBLES, 0))[:3]
        mH1 = _line_from_points(_P(0, L_COURT), _P(W_DOUBLES, L_COURT))[:3]
        mV0 = _line_from_points(_P(0, 0), _P(0, L_COURT))[:3]
        mV1 = _line_from_points(_P(W_DOUBLES, 0), _P(W_DOUBLES, L_COURT))[:3]

        m00 = intersect(mH0, mV0)
        m01 = intersect(mH0, mV1)
        m11 = intersect(mH1, mV1)
        m10 = intersect(mH1, mV0)
        if all(v is not None for v in (m00, m01, m11, m10)) and len(grpA) >= 2 and len(grpB) >= 2:
            src = np.stack([m00, m01, m11, m10], axis=0).astype(np.float32)
            for sH0, sH1 in itertools.combinations(grpA, 2):
                for sV0, sV1 in itertools.combinations(grpB, 2):
                    i00 = intersect((sH0.a, sH0.b, sH0.c), (sV0.a, sV0.b, sV0.c))
                    i01 = intersect((sH0.a, sH0.b, sH0.c), (sV1.a, sV1.b, sV1.c))
                    i11 = intersect((sH1.a, sH1.b, sH1.c), (sV1.a, sV1.b, sV1.c))
                    i10 = intersect((sH1.a, sH1.b, sH1.c), (sV0.a, sV0.b, sV0.c))
                    if any(v is None for v in (i00, i01, i11, i10)):
                        continue
                    dst = np.stack([i00, i01, i11, i10], axis=0).astype(np.float32)
                    H = cv2.getPerspectiveTransform(src, dst)
                    ok, rej = _quick_reject(H, img_wh)
                    if not ok:
                        continue
                    tauH = tau_from_H(H, LINE_W_M)
                    E = model_error_E(H, img_segs, tau_px=tauH, em=em)
                    if E < best_E:
                        best_E = E
                        best_H = H
                        best_meta = {
                            "mode": "robust",
                            "tauH": float(tauH),
                            "em": float(em),
                            **rej,
                            "E": float(E),
                        }

    # --- Robust lines -> quad (corners may be outside) ---
    K = int(os.environ.get("BADC_FARIN_TOPK", "8"))
    top = sorted(img_segs, key=lambda s: s.support, reverse=True)[:K]
    grpA, grpB = _cluster_two_orientations(top, tol_deg=float(os.environ.get("BADC_FARIN_ORI_TOL_DEG", "12")))
    pairA = _pick_extreme_parallel_pair(grpA, img_wh) if len(grpA) >= 2 else None
    pairB = _pick_extreme_parallel_pair(grpB, img_wh) if len(grpB) >= 2 else None

    if pairA and pairB:
        LA1, LA2 = _line_tuple(pairA[0]), _line_tuple(pairA[1])
        LB1, LB2 = _line_tuple(pairB[0]), _line_tuple(pairB[1])

        quad = _order_quad_from_lines(LA1, LA2, LB1, LB2, img_wh)
        if quad is not None:
            H = cv2.getPerspectiveTransform(model_corners_m().astype(np.float32), quad.astype(np.float32))
            ok, rej = _quick_reject_allow_outside(H, img_wh)
            if ok:
                tauH = tau_from_H(H, LINE_W_M)
                E_line = model_error_E(H, img_segs, tau_px=tauH, em=em)
                dt_cap = float(os.environ.get("BADC_FARIN_DT_CAP_PX", "12.0"))
                E_dt, dt_stats = dt_error_E(
                    H,
                    dt,
                    img_wh,
                    samples_per_seg=40,
                    dt_cap_px=dt_cap,
                    normalize_by=tauH,
                )
                dt_gate = float(os.environ.get("BADC_FARIN_DT_GATE", "250.0"))
                if E_dt > dt_gate:
                    reject_stats["rej_dt"] += 1
                else:
                    w_line = float(os.environ.get("BADC_FARIN_W_LINE", "1.0"))
                    w_dt = float(os.environ.get("BADC_FARIN_W_DT", "2.0"))
                    E = w_line * E_line + w_dt * E_dt
                    if E < best_E:
                        best_E = E
                        best_H = H
                        best_meta = {
                            "mode": "robust_lines",
                            "tauH": float(tauH),
                            "E_line": float(E_line),
                            "E_dt": float(E_dt),
                            "dt_gate": float(dt_gate),
                            "w_line": float(w_line),
                            "w_dt": float(w_dt),
                            **dt_stats,
                            **rej,
                            "E": float(E),
                        }

    if best_H is None:
        return MatchResult(
            H=None,
            corners=None,
            E=float("inf"),
            reason="R_fit_poor",
            metrics={"n_img_segs": len(img_segs), "tau0": tau0, "reject_stats": reject_stats},
        )

    corners = _project_points(best_H, model_corners_m()).astype(np.float32)

    return MatchResult(
        H=best_H,
        corners=corners,
        E=float(best_E),
        reason="R_ok",
        metrics={"n_img_segs": len(img_segs), "tau0": tau0, "reject_stats": reject_stats, **best_meta},
    )
