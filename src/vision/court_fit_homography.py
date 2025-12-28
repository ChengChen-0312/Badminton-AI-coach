from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from src.vision.court_model_bwf import MODEL_ID, get_bwf_corners, get_bwf_lines, sample_model_points

try:  # optional SciPy-based LM refine
    from scipy.optimize import least_squares

    _HAS_SCIPY = True
except Exception:
    least_squares = None
    _HAS_SCIPY = False


@dataclass
class FitMetrics:
    score: float
    inlier_ratio: float
    mean_dist_px: float
    p90_dist_px: float
    num_inliers: int
    num_valid_samples: int
    num_samples: int
    tau_px: float


@dataclass
class CourtFitResult:
    H: Optional[np.ndarray]
    corners: Optional[np.ndarray]
    confidence: float
    metrics: Dict[str, Any]
    reason: str
    debug_image: Optional[np.ndarray] = None
    white_mask: Optional[np.ndarray] = None


def _order_corners_lb_rb_rt_lt(pts_xy: np.ndarray) -> np.ndarray:
    pts = np.array(pts_xy, dtype=np.float32).reshape(4, 2)
    idx = np.argsort(pts[:, 1])
    top = pts[idx[:2]]
    bottom = pts[idx[2:]]
    bottom = bottom[np.argsort(bottom[:, 0])]
    top = top[np.argsort(top[:, 0])]
    lb, rb = bottom[0], bottom[1]
    lt, rt = top[0], top[1]
    return np.stack([lb, rb, rt, lt], axis=0)


def _quad_area(pts_xy: np.ndarray) -> float:
    pts = np.array(pts_xy, dtype=np.float32).reshape(-1, 1, 2)
    return float(abs(cv2.contourArea(pts)))


def _is_convex_quad(pts_xy: np.ndarray) -> bool:
    pts = np.array(pts_xy, dtype=np.float32).reshape(-1, 1, 2)
    return bool(cv2.isContourConvex(pts))


def build_distance_transform(white_mask: np.ndarray) -> np.ndarray:
    inv = (white_mask == 0).astype(np.uint8)
    return cv2.distanceTransform(inv, distanceType=cv2.DIST_L2, maskSize=3)


def build_white_mask(
    frame_bgr: np.ndarray,
    white_s_max: int = 80,
    white_v_min: int = 180,
    close_kernel: int = 5,
    open_kernel: int = 3,
    close_iter: int = 1,
    open_iter: int = 1,
) -> np.ndarray:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    hsv_mask = cv2.inRange(hsv, (0, 0, white_v_min), (180, white_s_max, 255))
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    lab_mask = (l > 200) & ((np.abs(a.astype(np.int16) - 128) + np.abs(b.astype(np.int16) - 128)) < 25)
    white = cv2.bitwise_and(hsv_mask, lab_mask.astype(np.uint8) * 255)
    kernel_open = np.ones((open_kernel, open_kernel), np.uint8)
    kernel_close = np.ones((close_kernel, close_kernel), np.uint8)
    if open_iter > 0:
        white = cv2.morphologyEx(white, cv2.MORPH_OPEN, kernel_open, iterations=open_iter)
    if close_iter > 0:
        white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, kernel_close, iterations=close_iter)
    return white


def _white_mask_stats(white_mask: np.ndarray) -> Tuple[int, float, Tuple[int, int, int, int]]:
    h, w = white_mask.shape[:2]
    ys, xs = np.where(white_mask > 0)
    count = int(xs.size)
    ratio = float(count) / float(max(h * w, 1))
    if xs.size == 0:
        return count, ratio, (0, 0, w - 1, h - 1)
    x1 = int(xs.min())
    x2 = int(xs.max())
    y1 = int(ys.min())
    y2 = int(ys.max())
    return count, ratio, (x1, y1, x2, y2)


def _sample_white_points(
    white_mask: np.ndarray,
    max_points: int,
    rng: np.random.Generator,
) -> np.ndarray:
    ys, xs = np.where(white_mask > 0)
    if xs.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    idx = np.arange(xs.size)
    if xs.size > max_points:
        idx = rng.choice(idx, size=int(max_points), replace=False)
    pts = np.stack([xs[idx], ys[idx]], axis=1).astype(np.float32)
    return pts


def _line_from_rho_theta(rho: float, theta: float) -> Tuple[float, float, float]:
    a = float(np.cos(theta))
    b = float(np.sin(theta))
    c = -float(rho)
    return a, b, c


def _intersect_lines(l1: Tuple[float, float, float], l2: Tuple[float, float, float]) -> Optional[np.ndarray]:
    a1, b1, c1 = l1
    a2, b2, c2 = l2
    d = a1 * b2 - a2 * b1
    if abs(d) < 1e-6:
        return None
    x = (b1 * c2 - b2 * c1) / d
    y = (c1 * a2 - c2 * a1) / d
    return np.array([x, y], dtype=np.float32)


def _split_lines_by_angle(
    lines: np.ndarray,
    angle_thresh_deg: float = 20.0,
) -> Tuple[list[Tuple[float, float, float]], list[Tuple[float, float, float]]]:
    horiz: list[Tuple[float, float, float]] = []
    vert: list[Tuple[float, float, float]] = []
    if lines is None:
        return horiz, vert
    thresh = np.deg2rad(float(angle_thresh_deg))
    for rho_theta in lines:
        rho, theta = float(rho_theta[0][0]), float(rho_theta[0][1])
        if abs(theta - np.pi / 2.0) <= thresh:
            horiz.append(_line_from_rho_theta(rho, theta))
        elif abs(theta) <= thresh or abs(theta - np.pi) <= thresh:
            vert.append(_line_from_rho_theta(rho, theta))
    return horiz, vert


def _estimate_vanishing_point(
    lines: list[Tuple[float, float, float]],
    rng: np.random.Generator,
    samples: int = 100,
) -> Optional[np.ndarray]:
    if len(lines) < 2:
        return None
    points = []
    for _ in range(int(samples)):
        idx = rng.choice(len(lines), size=2, replace=False)
        l1 = lines[int(idx[0])]
        l2 = lines[int(idx[1])]
        p = _intersect_lines(l1, l2)
        if p is None or not np.all(np.isfinite(p)):
            continue
        points.append(p)
    if not points:
        return None
    pts = np.stack(points, axis=0)
    return np.median(pts, axis=0)


def _project_line_segments(H: np.ndarray) -> np.ndarray:
    segs = []
    for ln in get_bwf_lines():
        pts = np.array([ln.p1, ln.p2], dtype=np.float32)
        uv = project_points(H, pts)
        segs.append([uv[0, 0], uv[0, 1], uv[1, 0], uv[1, 1]])
    return np.array(segs, dtype=np.float32)


def _dist_points_to_segments(points: np.ndarray, segs: np.ndarray) -> np.ndarray:
    if points.size == 0 or segs.size == 0:
        return np.zeros((points.shape[0],), dtype=np.float32)
    px = points[:, 0:1]
    py = points[:, 1:2]
    min_dist = np.full((points.shape[0],), np.inf, dtype=np.float32)
    for seg in segs:
        x1, y1, x2, y2 = seg
        vx = x2 - x1
        vy = y2 - y1
        denom = float(vx * vx + vy * vy + 1e-9)
        t = ((px - x1) * vx + (py - y1) * vy) / denom
        t = np.clip(t, 0.0, 1.0)
        proj_x = x1 + t * vx
        proj_y = y1 + t * vy
        dx = px - proj_x
        dy = py - proj_y
        dist = np.sqrt(dx * dx + dy * dy).reshape(-1)
        min_dist = np.minimum(min_dist, dist)
    return min_dist


def _area_ratio_and_penalty(
    corners_uv: np.ndarray,
    floor_bbox: Tuple[int, int, int, int],
    min_ratio: float = 0.15,
    max_ratio: float = 0.90,
) -> Tuple[float, float]:
    x1, y1, x2, y2 = floor_bbox
    area_floor = float(max(1, (x2 - x1) * (y2 - y1)))
    area_court = float(_quad_area(corners_uv))
    ratio = area_court / area_floor if area_floor > 0 else 0.0
    penalty = 0.0
    if ratio < min_ratio:
        penalty += (min_ratio - ratio) / max(min_ratio, 1e-6)
    if ratio > max_ratio:
        penalty += (ratio - max_ratio) / max(max_ratio, 1e-6)
    return float(ratio), float(penalty)


def _outside_bbox_penalty(
    corners_uv: np.ndarray,
    floor_bbox: Tuple[int, int, int, int],
) -> float:
    x1, y1, x2, y2 = floor_bbox
    dx1 = np.maximum(float(x1) - corners_uv[:, 0], 0.0)
    dx2 = np.maximum(corners_uv[:, 0] - float(x2), 0.0)
    dy1 = np.maximum(float(y1) - corners_uv[:, 1], 0.0)
    dy2 = np.maximum(corners_uv[:, 1] - float(y2), 0.0)
    dist = np.sqrt((dx1 + dx2) ** 2 + (dy1 + dy2) ** 2)
    return float(np.mean(dist)) if dist.size > 0 else 0.0


def _H_to_p(H: np.ndarray) -> np.ndarray:
    return np.array(
        [H[0, 0], H[0, 1], H[0, 2], H[1, 0], H[1, 1], H[1, 2], H[2, 0], H[2, 1]],
        dtype=np.float64,
    )


def _p_to_H(p: np.ndarray) -> np.ndarray:
    return np.array(
        [
            [p[0], p[1], p[2]],
            [p[3], p[4], p[5]],
            [p[6], p[7], 1.0],
        ],
        dtype=np.float64,
    )


def _residual_vector(
    p: np.ndarray,
    dt: np.ndarray,
    Xw: np.ndarray,
    weights: np.ndarray,
    cover_points: np.ndarray,
    floor_bbox: Tuple[int, int, int, int],
    dist_weight: float = 1.0,
    cover_weight: float = 0.7,
    reg_weight: float = 0.2,
    oob_penalty: float = 6.0,
) -> np.ndarray:
    H = _p_to_H(p)
    uv = project_points(H, Xw)
    if not np.all(np.isfinite(uv)):
        return {
            "loss": float("inf"),
            "l_dist": float("inf"),
            "l_cover": float("inf"),
            "l_reg": float("inf"),
            "area_ratio": 0.0,
            "inlier_ratio": 0.0,
            "inlier_dist_p90": None,
            "num_inliers": 0,
            "num_valid_samples": 0,
            "num_samples": int(uv.shape[0]),
            "sample_dist_mean": None,
            "sample_dist_p50": None,
            "sample_dist_p90": None,
            "sample_dist_max": None,
            "sample_dists": np.full((uv.shape[0],), float("inf"), dtype=np.float32),
            "sample_uv": uv,
        }
    h, w = dt.shape[:2]
    u = np.rint(uv[:, 0]).astype(np.int32)
    v = np.rint(uv[:, 1]).astype(np.int32)
    valid = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    r = np.full((uv.shape[0],), float(oob_penalty), dtype=np.float64)
    if np.any(valid):
        r[valid] = dt[v[valid], u[valid]].astype(np.float64)
    residuals = [np.sqrt(float(dist_weight)) * np.sqrt(weights.astype(np.float64)) * r]

    if cover_points is not None and cover_points.size > 0:
        segs = _project_line_segments(H)
        cover_dist = _dist_points_to_segments(cover_points, segs).astype(np.float64)
        residuals.append(np.sqrt(float(cover_weight)) * cover_dist)

    if reg_weight > 0:
        corners_uv = project_points(H, get_bwf_corners())
        area_ratio, penalty_area = _area_ratio_and_penalty(corners_uv, floor_bbox)
        diag = float(np.hypot(w, h))
        penalty_inside = _outside_bbox_penalty(corners_uv, floor_bbox) / float(max(diag, 1.0))
        residuals.append(
            np.sqrt(float(reg_weight)) * np.array([penalty_area, penalty_inside], dtype=np.float64)
        )

    return np.concatenate(residuals, axis=0)


def project_points(H: np.ndarray, Xw: np.ndarray) -> np.ndarray:
    ones = np.ones((Xw.shape[0], 1), dtype=np.float32)
    X = np.concatenate([Xw.astype(np.float32), ones], axis=1)
    if (not np.all(np.isfinite(H))) or float(np.max(np.abs(H))) > 1e6:
        return np.full((Xw.shape[0], 2), np.nan, dtype=np.float32)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        proj = (H @ X.T).T
    w = proj[:, 2:3]
    uv = proj[:, :2] / (w + 1e-9)
    return uv.astype(np.float32)


def _huber(residuals: np.ndarray, delta: float) -> np.ndarray:
    abs_r = np.abs(residuals)
    quad = abs_r <= delta
    out = np.empty_like(residuals, dtype=np.float32)
    out[quad] = 0.5 * (residuals[quad] ** 2)
    out[~quad] = delta * (abs_r[~quad] - 0.5 * delta)
    return out


def compute_fit_metrics(
    dt: np.ndarray,
    uv: np.ndarray,
    weights: np.ndarray,
    tau_px: float = 3.0,
    huber_delta: float = 3.0,
) -> FitMetrics:
    h, w = dt.shape[:2]
    u = np.rint(uv[:, 0]).astype(np.int32)
    v = np.rint(uv[:, 1]).astype(np.int32)
    valid = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    num_samples = int(uv.shape[0])
    if not np.any(valid):
        return FitMetrics(
            score=float("inf"),
            inlier_ratio=0.0,
            mean_dist_px=float("inf"),
            p90_dist_px=float("inf"),
            num_inliers=0,
            num_valid_samples=0,
            num_samples=num_samples,
            tau_px=float(tau_px),
        )
    u_valid = u[valid]
    v_valid = v[valid]
    r = dt[v_valid, u_valid].astype(np.float32)
    w_valid = weights[valid].astype(np.float32)
    num_valid = int(r.shape[0])
    num_inliers = int(np.count_nonzero(r < float(tau_px)))
    inlier_ratio = float(num_inliers) / float(max(num_valid, 1))
    mean_dist = float(np.average(r, weights=w_valid))
    p90 = float(np.percentile(r, 90))
    huber_vals = _huber(r, float(huber_delta))
    score = float(np.average(huber_vals, weights=w_valid))
    return FitMetrics(
        score=score,
        inlier_ratio=inlier_ratio,
        mean_dist_px=mean_dist,
        p90_dist_px=p90,
        num_inliers=num_inliers,
        num_valid_samples=num_valid,
        num_samples=num_samples,
        tau_px=float(tau_px),
    )


def _compute_loss_terms(
    H: np.ndarray,
    dt: np.ndarray,
    Xw: np.ndarray,
    weights: np.ndarray,
    cover_points: np.ndarray,
    floor_bbox: Tuple[int, int, int, int],
    tau_px: float = 3.0,
    huber_delta: float = 3.0,
    w_dist: float = 1.0,
    w_cover: float = 0.7,
    w_reg: float = 0.2,
) -> Dict[str, Any]:
    uv = project_points(H, Xw)
    h, w = dt.shape[:2]
    u = np.rint(uv[:, 0]).astype(np.int32)
    v = np.rint(uv[:, 1]).astype(np.int32)
    valid = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    r = np.full((uv.shape[0],), 6.0, dtype=np.float32)
    if np.any(valid):
        r[valid] = dt[v[valid], u[valid]].astype(np.float32)
    w_valid = weights.astype(np.float32)
    huber_vals = _huber(r, float(huber_delta))
    l_dist = float(np.average(huber_vals, weights=w_valid))

    num_valid = int(np.count_nonzero(valid))
    inlier_mask = (r < float(tau_px)) & valid
    num_inliers = int(np.count_nonzero(inlier_mask))
    inlier_ratio = float(num_inliers) / float(max(num_valid, 1))
    inlier_dists = r[inlier_mask]
    inlier_dist_p90 = float(np.percentile(inlier_dists, 90)) if inlier_dists.size > 0 else None

    sample_dist_mean = float(np.mean(r[valid])) if np.any(valid) else None
    sample_dist_p50 = float(np.percentile(r[valid], 50)) if np.any(valid) else None
    sample_dist_p90 = float(np.percentile(r[valid], 90)) if np.any(valid) else None
    sample_dist_max = float(np.max(r[valid])) if np.any(valid) else None

    l_cover = 0.0
    if cover_points is not None and cover_points.size > 0:
        segs = _project_line_segments(H)
        cover_dist = _dist_points_to_segments(cover_points, segs)
        l_cover = float(np.mean(_huber(cover_dist, float(huber_delta))))

    corners_uv = project_points(H, get_bwf_corners())
    area_ratio, penalty_area = _area_ratio_and_penalty(corners_uv, floor_bbox)
    diag = float(np.hypot(w, h))
    penalty_inside = _outside_bbox_penalty(corners_uv, floor_bbox) / float(max(diag, 1.0))
    l_reg = float(penalty_area + penalty_inside)

    total_loss = float(w_dist * l_dist + w_cover * l_cover + w_reg * l_reg)
    return {
        "loss": total_loss,
        "l_dist": float(l_dist),
        "l_cover": float(l_cover),
        "l_reg": float(l_reg),
        "area_ratio": float(area_ratio),
        "inlier_ratio": float(inlier_ratio),
        "inlier_dist_p90": inlier_dist_p90,
        "num_inliers": num_inliers,
        "num_valid_samples": num_valid,
        "num_samples": int(uv.shape[0]),
        "sample_dist_mean": sample_dist_mean,
        "sample_dist_p50": sample_dist_p50,
        "sample_dist_p90": sample_dist_p90,
        "sample_dist_max": sample_dist_max,
        "sample_dists": r,
        "sample_uv": uv,
    }


def ransac_init_homography(
    white_mask: np.ndarray,
    dt: np.ndarray,
    world_corners: np.ndarray,
    Xw: np.ndarray,
    weights: np.ndarray,
    cover_points: np.ndarray,
    floor_bbox: Tuple[int, int, int, int],
    iters: int = 2000,
    tau_px: float = 3.0,
    min_area_ratio: float = 0.01,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[Optional[np.ndarray], Optional[Dict[str, Any]]]:
    ys, xs = np.where(white_mask > 0)
    if xs.size < 4:
        return None, None
    coords = np.stack([xs, ys], axis=1).astype(np.float32)
    rng = rng or np.random.default_rng()
    h, w = white_mask.shape[:2]
    min_area = float(h * w) * float(min_area_ratio)
    best_H = None
    best_info: Optional[Dict[str, Any]] = None
    best_score = float("inf")
    world = world_corners.astype(np.float32)

    edges = cv2.Canny(white_mask, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 180.0, 120)
    horiz, vert = _split_lines_by_angle(lines)
    vp_h = _estimate_vanishing_point(horiz, rng)
    vp_v = _estimate_vanishing_point(vert, rng)

    def _sample_from_lines() -> Optional[np.ndarray]:
        if len(horiz) < 2 or len(vert) < 2:
            return None
        h_idx = rng.choice(len(horiz), size=2, replace=False)
        v_idx = rng.choice(len(vert), size=2, replace=False)
        h1 = horiz[int(h_idx[0])]
        h2 = horiz[int(h_idx[1])]
        v1 = vert[int(v_idx[0])]
        v2 = vert[int(v_idx[1])]
        p1 = _intersect_lines(h1, v1)
        p2 = _intersect_lines(h1, v2)
        p3 = _intersect_lines(h2, v2)
        p4 = _intersect_lines(h2, v1)
        if p1 is None or p2 is None or p3 is None or p4 is None:
            return None
        pts = np.stack([p1, p2, p3, p4], axis=0)
        return pts

    def _sample_from_quadrants() -> Optional[np.ndarray]:
        h_mid = float(h) * 0.5
        w_mid = float(w) * 0.5
        bins = [
            coords[(coords[:, 0] <= w_mid) & (coords[:, 1] >= h_mid)],
            coords[(coords[:, 0] > w_mid) & (coords[:, 1] >= h_mid)],
            coords[(coords[:, 0] > w_mid) & (coords[:, 1] < h_mid)],
            coords[(coords[:, 0] <= w_mid) & (coords[:, 1] < h_mid)],
        ]
        if any(b.shape[0] == 0 for b in bins):
            return None
        pts = np.stack([b[rng.integers(0, b.shape[0])] for b in bins], axis=0)
        return pts

    for _ in range(int(iters)):
        pts = _sample_from_lines()
        if pts is None:
            pts = _sample_from_quadrants()
        if pts is None:
            idx = rng.choice(coords.shape[0], size=4, replace=False)
            pts = coords[idx]
        ordered = _order_corners_lb_rb_rt_lt(pts)
        if not _is_convex_quad(ordered):
            continue
        area = _quad_area(ordered)
        if area < min_area:
            continue
        H = cv2.getPerspectiveTransform(world, ordered.astype(np.float32))
        if not np.all(np.isfinite(H)):
            continue
        info = _compute_loss_terms(
            H,
            dt=dt,
            Xw=Xw,
            weights=weights,
            cover_points=cover_points,
            floor_bbox=floor_bbox,
            tau_px=tau_px,
        )
        score = float(info.get("loss", float("inf")))
        if score < best_score:
            best_score = score
            best_H = H
            best_info = info

    if best_info is None:
        return None, None
    best_info["vp_h"] = vp_h.tolist() if isinstance(vp_h, np.ndarray) else None
    best_info["vp_v"] = vp_v.tolist() if isinstance(vp_v, np.ndarray) else None
    best_info["num_hough_lines_h"] = int(len(horiz))
    best_info["num_hough_lines_v"] = int(len(vert))
    return best_H, best_info


def _lm_numeric(
    p0: np.ndarray,
    dt: np.ndarray,
    Xw: np.ndarray,
    weights: np.ndarray,
    cover_points: np.ndarray,
    floor_bbox: Tuple[int, int, int, int],
    max_iter: int = 30,
    lam: float = 1e-3,
    eps: float = 1e-4,
) -> Tuple[np.ndarray, float]:
    p = p0.astype(np.float64).copy()
    for _ in range(int(max_iter)):
        r = _residual_vector(p, dt, Xw, weights, cover_points, floor_bbox)
        cost = 0.5 * float(np.dot(r, r))
        J = np.zeros((r.shape[0], 8), dtype=np.float64)
        for i in range(8):
            p_eps = p.copy()
            p_eps[i] += float(eps)
            r_eps = _residual_vector(p_eps, dt, Xw, weights, cover_points, floor_bbox)
            J[:, i] = (r_eps - r) / float(eps)
        A = J.T @ J + float(lam) * np.eye(8, dtype=np.float64)
        g = J.T @ r
        try:
            delta = -np.linalg.solve(A, g)
        except np.linalg.LinAlgError:
            break
        p_new = p + delta
        r_new = _residual_vector(p_new, dt, Xw, weights, cover_points, floor_bbox)
        new_cost = 0.5 * float(np.dot(r_new, r_new))
        if new_cost < cost:
            p = p_new
            lam *= 0.7
        else:
            lam *= 2.0
    final_r = _residual_vector(p, dt, Xw, weights, cover_points, floor_bbox)
    final_cost = 0.5 * float(np.dot(final_r, final_r))
    return p, final_cost


def refine_homography_lm(
    H_init: np.ndarray,
    dt: np.ndarray,
    Xw: np.ndarray,
    weights: np.ndarray,
    cover_points: np.ndarray,
    floor_bbox: Tuple[int, int, int, int],
    max_nfev: int = 50,
) -> Tuple[np.ndarray, float]:
    p0 = _H_to_p(H_init)
    if _HAS_SCIPY and least_squares is not None:
        res = least_squares(
            lambda p: _residual_vector(p, dt, Xw, weights, cover_points, floor_bbox),
            p0,
            loss="huber",
            f_scale=3.0,
            max_nfev=int(max_nfev),
        )
        return _p_to_H(res.x), float(res.cost)
    p_ref, cost = _lm_numeric(p0, dt, Xw, weights, cover_points, floor_bbox, max_iter=int(max_nfev))
    return _p_to_H(p_ref), float(cost)


def draw_debug_overlay(
    frame_bgr: np.ndarray,
    white_mask: np.ndarray,
    H: np.ndarray,
    conf: float,
    inlier_ratio: float,
    mean_dist_px: float,
    p90_dist_px: float,
    white_mask_ratio: float,
    l_dist: float,
    l_cover: float,
    l_reg: float,
    area_ratio: float,
    dt_min: Optional[float],
    dt_mean: Optional[float],
    dt_p90: Optional[float],
    sample_dist_mean: Optional[float],
    sample_dist_p50: Optional[float],
    sample_dist_p90: Optional[float],
    sample_dist_max: Optional[float],
    sample_uv: np.ndarray,
    sample_dist: np.ndarray,
    tau_px: float,
) -> np.ndarray:
    out = frame_bgr.copy()
    rng = np.random.default_rng(0)

    # White mask pixels (downsampled).
    ys, xs = np.where(white_mask > 0)
    if xs.size > 0:
        idx = np.arange(xs.size)
        if xs.size > 3000:
            idx = rng.choice(idx, size=3000, replace=False)
        pts = np.stack([xs[idx], ys[idx]], axis=1).astype(np.int32)
        for x, y in pts:
            cv2.circle(out, (int(x), int(y)), 1, (220, 220, 220), -1)

    lines = get_bwf_lines()
    for ln in lines:
        pts = np.array([ln.p1, ln.p2], dtype=np.float32)
        uv = project_points(H, pts)
        p1 = (int(round(float(uv[0, 0]))), int(round(float(uv[0, 1]))))
        p2 = (int(round(float(uv[1, 0]))), int(round(float(uv[1, 1]))))
        if ln.weight >= 0.9:
            color = (0, 255, 0)
        elif ln.weight >= 0.6:
            color = (0, 200, 200)
        else:
            color = (128, 200, 255)
        cv2.line(out, p1, p2, color, 2, lineType=cv2.LINE_AA)

    corners_w = get_bwf_corners()
    corners_uv = project_points(H, corners_w)
    labels = ["LB", "RB", "RT", "LT"]
    for (x, y), label in zip(corners_uv, labels):
        px = int(round(float(x)))
        py = int(round(float(y)))
        cv2.circle(out, (px, py), 6, (0, 0, 255), -1)
        cv2.putText(out, label, (px + 8, py - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
        cv2.putText(out, label, (px + 8, py - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)

    # Sample points colored by dt distance (green=near, red=far).
    if sample_uv is not None and sample_uv.size > 0:
        max_draw = min(sample_uv.shape[0], 2500)
        idx = np.arange(sample_uv.shape[0])
        if sample_uv.shape[0] > max_draw:
            idx = rng.choice(idx, size=max_draw, replace=False)
        dmax = max(float(np.max(sample_dist)), 1e-6) if sample_dist.size > 0 else 1.0
        dmax = min(dmax, float(tau_px) * 3.0)
        for i in idx:
            x, y = sample_uv[i]
            d = float(sample_dist[i])
            if d <= float(tau_px):
                color = (0, 255, 0)
            else:
                t = min(d / max(dmax, 1e-6), 1.0)
                r = int(255 * t)
                g = int(255 * (1.0 - t))
                color = (0, g, r)
            cv2.circle(out, (int(round(float(x))), int(round(float(y)))), 2, color, -1)

    title = f"conf={conf:.2f} inlier={inlier_ratio:.2f} mean={mean_dist_px:.2f}px p90={p90_dist_px:.2f}px"
    cv2.putText(out, title, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
    cv2.putText(out, title, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    row2 = f"mask={white_mask_ratio:.3f} Ld={l_dist:.3f} Lc={l_cover:.3f} Lr={l_reg:.3f} area={area_ratio:.3f}"
    cv2.putText(out, row2, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
    cv2.putText(out, row2, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    row3 = f"dt_min={dt_min:.2f} dt_mean={dt_mean:.2f} dt_p90={dt_p90:.2f}" if dt_min is not None else "dt_min=NA dt_mean=NA dt_p90=NA"
    cv2.putText(out, row3, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
    cv2.putText(out, row3, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    if sample_dist_mean is not None:
        row4 = (
            f"samp_mean={sample_dist_mean:.2f} p50={sample_dist_p50:.2f} "
            f"p90={sample_dist_p90:.2f} max={sample_dist_max:.2f}"
        )
    else:
        row4 = "samp_mean=NA p50=NA p90=NA max=NA"
    cv2.putText(out, row4, (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
    cv2.putText(out, row4, (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return out


def fit_court_homography(frame_bgr: np.ndarray) -> CourtFitResult:
    rng = np.random.default_rng(0)
    white_mask = build_white_mask(frame_bgr)
    white_mask_sum, white_mask_ratio, floor_bbox = _white_mask_stats(white_mask)
    dt = build_distance_transform(white_mask)
    dt_min = float(np.min(dt)) if dt.size > 0 else None
    dt_mean = float(np.mean(dt)) if dt.size > 0 else None
    dt_p90 = float(np.percentile(dt, 90)) if dt.size > 0 else None

    corners_world = get_bwf_corners()
    Xw_ransac, w_ransac, _ = sample_model_points(points_per_meter=30.0, min_weight=0.6)
    Xw_full, w_full, _ = sample_model_points(points_per_meter=30.0, min_weight=0.3)

    base_metrics = {
        "model": MODEL_ID,
        "dt_from": "white_mask_frame",
        "white_mask_sum": int(white_mask_sum),
        "white_mask_ratio": float(white_mask_ratio),
        "dt_min": dt_min,
        "dt_mean": dt_mean,
        "dt_p90": dt_p90,
        "debug_image_path": None,
        "white_mask_path": None,
    }

    if white_mask_ratio < 0.001 or white_mask_ratio > 0.25:
        metrics = {
            **base_metrics,
            "H": None,
            "inlier_ratio": 0.0,
            "mean_dist_px": None,
            "p90_dist_px": None,
            "num_inliers": 0,
            "num_valid_samples": 0,
            "num_samples": int(Xw_ransac.shape[0]),
            "tau_px": 3.0,
            "lm_cost": None,
        }
        return CourtFitResult(
            H=None,
            corners=None,
            confidence=0.0,
            metrics=metrics,
            reason="R_mask_invalid",
            debug_image=None,
            white_mask=white_mask,
        )

    cover_points_ransac = _sample_white_points(white_mask, max_points=600, rng=rng)
    cover_points = _sample_white_points(white_mask, max_points=1500, rng=rng)

    H_init, init_info = ransac_init_homography(
        white_mask=white_mask,
        dt=dt,
        world_corners=corners_world,
        Xw=Xw_ransac,
        weights=w_ransac,
        cover_points=cover_points_ransac,
        floor_bbox=floor_bbox,
        iters=2000,
        tau_px=3.0,
        min_area_ratio=0.01,
    )
    if H_init is None or init_info is None:
        metrics = {
            **base_metrics,
            "H": None,
            "inlier_ratio": 0.0,
            "mean_dist_px": None,
            "p90_dist_px": None,
            "num_inliers": 0,
            "num_valid_samples": 0,
            "num_samples": int(Xw_ransac.shape[0]),
            "tau_px": 3.0,
            "lm_cost": None,
        }
        return CourtFitResult(
            H=None,
            corners=None,
            confidence=0.0,
            metrics=metrics,
            reason="R_fit_poor",
            debug_image=None,
            white_mask=white_mask,
        )

    H_ref, lm_cost = refine_homography_lm(
        H_init,
        dt,
        Xw_full,
        w_full,
        cover_points=cover_points,
        floor_bbox=floor_bbox,
        max_nfev=50,
    )
    loss_info = _compute_loss_terms(
        H_ref,
        dt=dt,
        Xw=Xw_full,
        weights=w_full,
        cover_points=cover_points,
        floor_bbox=floor_bbox,
        tau_px=3.0,
    )
    mean_dist = loss_info.get("sample_dist_mean")
    p90_dist = loss_info.get("sample_dist_p90")
    inlier_ratio = float(loss_info.get("inlier_ratio", 0.0))
    conf = float(np.clip(inlier_ratio * np.exp(-float(mean_dist or 0.0) / 6.0), 0.0, 1.0))
    reason = "OK" if (inlier_ratio > 0.25 and mean_dist is not None and mean_dist < 5.0) else "R_fit_poor"

    corners_uv = project_points(H_ref, corners_world)
    corners_out = corners_uv.astype(np.float32) if reason == "OK" else None

    debug_image = draw_debug_overlay(
        frame_bgr,
        white_mask,
        H_ref,
        conf=float(conf),
        inlier_ratio=float(inlier_ratio),
        mean_dist_px=float(mean_dist) if mean_dist is not None else 0.0,
        p90_dist_px=float(p90_dist) if p90_dist is not None else 0.0,
        white_mask_ratio=float(white_mask_ratio),
        l_dist=float(loss_info.get("l_dist", 0.0)),
        l_cover=float(loss_info.get("l_cover", 0.0)),
        l_reg=float(loss_info.get("l_reg", 0.0)),
        area_ratio=float(loss_info.get("area_ratio", 0.0)),
        dt_min=dt_min,
        dt_mean=dt_mean,
        dt_p90=dt_p90,
        sample_dist_mean=loss_info.get("sample_dist_mean"),
        sample_dist_p50=loss_info.get("sample_dist_p50"),
        sample_dist_p90=loss_info.get("sample_dist_p90"),
        sample_dist_max=loss_info.get("sample_dist_max"),
        sample_uv=loss_info.get("sample_uv", np.zeros((0, 2), dtype=np.float32)),
        sample_dist=loss_info.get("sample_dists", np.zeros((0,), dtype=np.float32)),
        tau_px=3.0,
    )

    metrics = {
        **base_metrics,
        "H": [float(v) for v in H_ref.reshape(-1)],
        "inlier_ratio": float(inlier_ratio),
        "mean_dist_px": float(mean_dist) if mean_dist is not None else None,
        "p90_dist_px": float(p90_dist) if p90_dist is not None else None,
        "num_inliers": int(loss_info.get("num_inliers", 0)),
        "num_valid_samples": int(loss_info.get("num_valid_samples", 0)),
        "num_samples": int(loss_info.get("num_samples", Xw_full.shape[0])),
        "tau_px": 3.0,
        "lm_cost": float(lm_cost),
        "l_dist": float(loss_info.get("l_dist", 0.0)),
        "l_cover": float(loss_info.get("l_cover", 0.0)),
        "l_reg": float(loss_info.get("l_reg", 0.0)),
        "area_ratio": float(loss_info.get("area_ratio", 0.0)),
        "inlier_dist_p90": loss_info.get("inlier_dist_p90"),
        "sample_dist_mean": loss_info.get("sample_dist_mean"),
        "sample_dist_p50": loss_info.get("sample_dist_p50"),
        "sample_dist_p90": loss_info.get("sample_dist_p90"),
        "sample_dist_max": loss_info.get("sample_dist_max"),
        "vp_h": init_info.get("vp_h") if isinstance(init_info, dict) else None,
        "vp_v": init_info.get("vp_v") if isinstance(init_info, dict) else None,
        "num_hough_lines_h": init_info.get("num_hough_lines_h") if isinstance(init_info, dict) else None,
        "num_hough_lines_v": init_info.get("num_hough_lines_v") if isinstance(init_info, dict) else None,
    }
    return CourtFitResult(
        H=H_ref,
        corners=corners_out,
        confidence=conf,
        metrics=metrics,
        reason=reason,
        debug_image=debug_image,
        white_mask=white_mask,
    )
