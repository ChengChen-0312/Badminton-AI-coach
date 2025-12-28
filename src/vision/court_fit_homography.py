from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


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
