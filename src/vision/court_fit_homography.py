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
    white_s_max: int = 110,
    white_v_min: int = 160,
    morph_kernel: int = 9,
    close_iter: int = 2,
    open_iter: int = 1,
) -> np.ndarray:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, (0, 0, white_v_min), (180, white_s_max, 255))
    kernel = np.ones((morph_kernel, morph_kernel), np.uint8)
    if close_iter > 0:
        white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, kernel, iterations=close_iter)
    if open_iter > 0:
        white = cv2.morphologyEx(white, cv2.MORPH_OPEN, kernel, iterations=open_iter)
    return white


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
    oob_penalty: float = 6.0,
) -> np.ndarray:
    H = _p_to_H(p)
    uv = project_points(H, Xw)
    h, w = dt.shape[:2]
    u = np.rint(uv[:, 0]).astype(np.int32)
    v = np.rint(uv[:, 1]).astype(np.int32)
    valid = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    r = np.full((uv.shape[0],), float(oob_penalty), dtype=np.float64)
    if np.any(valid):
        r[valid] = dt[v[valid], u[valid]].astype(np.float64)
    return np.sqrt(weights.astype(np.float64)) * r


def project_points(H: np.ndarray, Xw: np.ndarray) -> np.ndarray:
    ones = np.ones((Xw.shape[0], 1), dtype=np.float32)
    X = np.concatenate([Xw.astype(np.float32), ones], axis=1)
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


def score_homography_dt(
    H: np.ndarray,
    dt: np.ndarray,
    Xw: np.ndarray,
    weights: np.ndarray,
    tau_px: float = 3.0,
) -> Tuple[FitMetrics, np.ndarray]:
    uv = project_points(H, Xw)
    metrics = compute_fit_metrics(dt, uv, weights, tau_px=tau_px)
    return metrics, uv


def ransac_init_homography(
    white_mask: np.ndarray,
    dt: np.ndarray,
    world_corners: np.ndarray,
    Xw: np.ndarray,
    weights: np.ndarray,
    iters: int = 2000,
    tau_px: float = 3.0,
    min_area_ratio: float = 0.01,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[Optional[np.ndarray], Optional[FitMetrics]]:
    ys, xs = np.where(white_mask > 0)
    if xs.size < 4:
        return None, None
    coords = np.stack([xs, ys], axis=1).astype(np.float32)
    rng = rng or np.random.default_rng()
    h, w = white_mask.shape[:2]
    min_area = float(h * w) * float(min_area_ratio)
    best_H = None
    best_metrics = None
    best_score = float("inf")
    world = world_corners.astype(np.float32)
    for _ in range(int(iters)):
        idx = rng.choice(coords.shape[0], size=4, replace=False)
        pts = coords[idx]
        ordered = _order_corners_lb_rb_rt_lt(pts)
        if not _is_convex_quad(ordered):
            continue
        area = _quad_area(ordered)
        if area < min_area:
            continue
        H = cv2.getPerspectiveTransform(world, ordered.astype(np.float32))
        metrics, _ = score_homography_dt(H, dt, Xw, weights, tau_px=tau_px)
        if metrics.score < best_score:
            best_score = metrics.score
            best_H = H
            best_metrics = metrics
    return best_H, best_metrics


def _lm_numeric(
    p0: np.ndarray,
    dt: np.ndarray,
    Xw: np.ndarray,
    weights: np.ndarray,
    max_iter: int = 30,
    lam: float = 1e-3,
    eps: float = 1e-4,
) -> Tuple[np.ndarray, float]:
    p = p0.astype(np.float64).copy()
    for _ in range(int(max_iter)):
        r = _residual_vector(p, dt, Xw, weights)
        cost = 0.5 * float(np.dot(r, r))
        J = np.zeros((r.shape[0], 8), dtype=np.float64)
        for i in range(8):
            p_eps = p.copy()
            p_eps[i] += float(eps)
            r_eps = _residual_vector(p_eps, dt, Xw, weights)
            J[:, i] = (r_eps - r) / float(eps)
        A = J.T @ J + float(lam) * np.eye(8, dtype=np.float64)
        g = J.T @ r
        try:
            delta = -np.linalg.solve(A, g)
        except np.linalg.LinAlgError:
            break
        p_new = p + delta
        r_new = _residual_vector(p_new, dt, Xw, weights)
        new_cost = 0.5 * float(np.dot(r_new, r_new))
        if new_cost < cost:
            p = p_new
            lam *= 0.7
        else:
            lam *= 2.0
    final_r = _residual_vector(p, dt, Xw, weights)
    final_cost = 0.5 * float(np.dot(final_r, final_r))
    return p, final_cost


def refine_homography_lm(
    H_init: np.ndarray,
    dt: np.ndarray,
    Xw: np.ndarray,
    weights: np.ndarray,
    max_nfev: int = 50,
) -> Tuple[np.ndarray, float]:
    p0 = _H_to_p(H_init)
    if _HAS_SCIPY and least_squares is not None:
        res = least_squares(
            lambda p: _residual_vector(p, dt, Xw, weights),
            p0,
            loss="huber",
            f_scale=3.0,
            max_nfev=int(max_nfev),
        )
        return _p_to_H(res.x), float(res.cost)
    p_ref, cost = _lm_numeric(p0, dt, Xw, weights, max_iter=int(max_nfev))
    return _p_to_H(p_ref), float(cost)


def draw_debug_overlay(
    frame_bgr: np.ndarray,
    H: np.ndarray,
    conf: float,
    inlier_ratio: float,
    mean_dist_px: float,
    p90_dist_px: float,
) -> np.ndarray:
    out = frame_bgr.copy()
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

    title = f"conf={conf:.2f} inlier={inlier_ratio:.2f} mean={mean_dist_px:.2f}px p90={p90_dist_px:.2f}px"
    cv2.putText(out, title, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
    cv2.putText(out, title, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return out


def fit_court_homography(frame_bgr: np.ndarray) -> CourtFitResult:
    white_mask = build_white_mask(frame_bgr)
    dt = build_distance_transform(white_mask)
    corners_world = get_bwf_corners()
    Xw_ransac, w_ransac, _ = sample_model_points(points_per_meter=30.0, min_weight=0.6)
    Xw_full, w_full, _ = sample_model_points(points_per_meter=30.0, min_weight=0.3)

    H_init, init_metrics = ransac_init_homography(
        white_mask=white_mask,
        dt=dt,
        world_corners=corners_world,
        Xw=Xw_ransac,
        weights=w_ransac,
        iters=2000,
        tau_px=3.0,
        min_area_ratio=0.01,
    )
    if H_init is None or init_metrics is None:
        metrics = {
            "model": MODEL_ID,
            "H": None,
            "inlier_ratio": 0.0,
            "mean_dist_px": None,
            "p90_dist_px": None,
            "num_inliers": 0,
            "num_valid_samples": 0,
            "num_samples": int(Xw_ransac.shape[0]),
            "tau_px": 3.0,
            "lm_cost": None,
            "debug_image_path": None,
        }
        return CourtFitResult(H=None, corners=None, confidence=0.0, metrics=metrics, reason="R_fit_poor")

    H_ref, lm_cost = refine_homography_lm(H_init, dt, Xw_full, w_full, max_nfev=50)
    uv_full = project_points(H_ref, Xw_full)
    metrics_final = compute_fit_metrics(dt, uv_full, w_full, tau_px=3.0)
    mean_dist = metrics_final.mean_dist_px if np.isfinite(metrics_final.mean_dist_px) else None
    p90_dist = metrics_final.p90_dist_px if np.isfinite(metrics_final.p90_dist_px) else None
    conf = float(
        np.clip(
            metrics_final.inlier_ratio * np.exp(-metrics_final.mean_dist_px / 6.0),
            0.0,
            1.0,
        )
    )
    reason = (
        "OK"
        if (metrics_final.inlier_ratio > 0.25 and mean_dist is not None and float(mean_dist) < 5.0)
        else "R_fit_poor"
    )
    corners_uv = project_points(H_ref, corners_world)
    corners_out = corners_uv.astype(np.float32) if reason == "OK" else None
    metrics = {
        "model": MODEL_ID,
        "H": [float(v) for v in H_ref.reshape(-1)],
        "inlier_ratio": float(metrics_final.inlier_ratio),
        "mean_dist_px": float(mean_dist) if mean_dist is not None else None,
        "p90_dist_px": float(p90_dist) if p90_dist is not None else None,
        "num_inliers": int(metrics_final.num_inliers),
        "num_valid_samples": int(metrics_final.num_valid_samples),
        "num_samples": int(metrics_final.num_samples),
        "tau_px": float(metrics_final.tau_px),
        "lm_cost": float(lm_cost),
        "debug_image_path": None,
    }
    return CourtFitResult(H=H_ref, corners=corners_out, confidence=conf, metrics=metrics, reason=reason)
