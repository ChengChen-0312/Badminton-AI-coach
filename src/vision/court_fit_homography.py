from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from typing import Any, Dict, Optional, Sequence, Tuple

import cv2
import numpy as np

from src.vision.court_model_bwf import MODEL_ID, get_bwf_corners, get_bwf_lines, sample_model_points

# Mixer weights for candidate legitimacy scoring (lower is better).
MIXER_WEIGHTS = {
    "dt": 1.0,
    "aspect": 0.8,
    "vp": 0.5,
    "floor": 2.0,
    "cover": 2.0,
    "pos": 0.3,
    "area": 8.0,
}

CANONICAL_POINTS_NORM = {
    "y_net": 0.5,
    "y_short_near": 0.352239,
    "y_short_far": 0.647761,
    "x_center": 0.5,
    "x_singles_L": 0.075410,
    "x_singles_R": 0.924590,
    "SSN_L_out": (0.0, 0.352239),
    "SSN_R_out": (1.0, 0.352239),
    "SSF_L_out": (0.0, 0.647761),
    "SSF_R_out": (1.0, 0.647761),
    "C_SSN": (0.5, 0.352239),
    "C_SSF": (0.5, 0.647761),
    "NetPost_L": (0.0, 0.5),
    "NetPost_R": (1.0, 0.5),
    "NetMid": (0.5, 0.5),
    "SSN_L_in": (0.075410, 0.352239),
    "SSN_R_in": (0.924590, 0.352239),
    "SSF_L_in": (0.075410, 0.647761),
    "SSF_R_in": (0.924590, 0.647761),
    "corners_lb_rb_rt_lt": (
        (0.0, 0.0),
        (1.0, 0.0),
        (1.0, 1.0),
        (0.0, 1.0),
    ),
    "service_short_sideline_intersections": (
        (0.0, 0.352238806),
        (1.0, 0.352238806),
        (0.0, 0.647761194),
        (1.0, 0.647761194),
    ),
    "service_short_center_intersections": (
        (0.5, 0.352238806),
        (0.5, 0.647761194),
    ),
    "net_posts": (
        (0.0, 0.5),
        (1.0, 0.5),
    ),
    "net_mid": (0.5, 0.5),
}


def get_canonical_points_norm() -> Dict[str, Any]:
    """Return canonical court points in normalized coordinates for debug/metrics."""
    points: Dict[str, Any] = {}
    for key, val in CANONICAL_POINTS_NORM.items():
        if isinstance(val, (tuple, list)):
            if val and isinstance(val[0], (tuple, list)):
                points[key] = [[float(p[0]), float(p[1])] for p in val]  # type: ignore[index]
            else:
                points[key] = [float(val[0]), float(val[1])]  # type: ignore[index]
        else:
            points[key] = float(val)
    return points


# ---------- Raw-floor Hough env knobs ----------
def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)).strip())
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)).strip())
    except Exception:
        return default


def _env_str(name: str, default: str = "") -> str:
    v = os.environ.get(name, None)
    if v is None:
        return str(default)
    return str(v)


def _env_flag(name: str, default: bool = False) -> bool:
    """
    Accepts common boolean strings: 1/0, true/false, yes/no, on/off (case-insensitive).
    """
    v = os.environ.get(name, None)
    if v is None:
        return bool(default)
    v = str(v).strip().lower()
    if v in ("1", "true", "yes", "y", "on"):
        return True
    if v in ("0", "false", "no", "n", "off"):
        return False
    return bool(default)


def _apply_legacy_env_aliases() -> None:
    """
    Backward-compatible env aliases for historical tuning scripts.
    If the new key is already provided, keep it as the source of truth.
    """
    alias_pairs = {
        "BADC_ADV_POSTBLOB": "BADC_FORCE_LINEPIX_POSTBLOB",
        "BADC_RANSAC_MAX_LINES": "BADC_RAW_FLOOR_RANSAC_MAX_LINES",
        "BADC_RANSAC_MIN_INLIERS": "BADC_RAW_FLOOR_RANSAC_MIN_INLIERS",
        "BADC_RANSAC_MIN_LENGTH_RATIO": "BADC_RAW_FLOOR_RANSAC_MINLEN_RT",
        "BADC_RANSAC_DISABLE_FLOOR_GATE": "BADC_DISABLE_FLOOR_GATE",
        "BADC_ORI_USE_PEAK_CENTERS": "BADC_RAW_FLOOR_ANGLE_RESPLIT",
        "BADC_ORI_PEAK_RESPLIT": "BADC_RAW_FLOOR_ANGLE_RESPLIT",
        "BADC_RANSAC_RIGHT_MAX_LINES": "BADC_RANSAC_RIGHT_RESCUE_MAX_LINES",
        "BADC_RANSAC_RIGHT_MIN_LENGTH_RATIO": "BADC_RANSAC_RIGHT_RESCUE_MIN_LENGTH_RATIO",
        # Legacy right-rescue controls mapped to closest modern debug-rescue knobs.
        "BADC_RANSAC_RIGHT_RESCUE_MARGIN_PX": "BADC_RANSAC_DEBUG_RIGHT_RESCUE_BASELINE_SUPPORT_PAD_PX",
        "BADC_RANSAC_RIGHT_ITERS": "BADC_RANSAC_DEBUG_RIGHT_RESCUE_BASELINE_SCAN_SAMPLES",
        "BADC_RANSAC_RIGHT_INLIER_THR": "BADC_RANSAC_DEBUG_RIGHT_RESCUE_BASELINE_SUPPORT_THR",
        "BADC_RANSAC_RIGHT_MIN_INLIERS": "BADC_RANSAC_DEBUG_RIGHT_RESCUE_BASELINE_MIN_RUN",
        "BADC_RANSAC_RIGHT_RESCUE_INLIER_THR": "BADC_RANSAC_DEBUG_RIGHT_RESCUE_BASELINE_SUPPORT_THR",
        "BADC_RANSAC_RIGHT_RESCUE_MIN_INLIERS": "BADC_RANSAC_DEBUG_RIGHT_RESCUE_BASELINE_MIN_RUN",
        "BADC_ORI_EXTREME_MAX_ANGLE_DEG": "BADC_ORI_BOUNDARY_FALLBACK_MAX_ANGLE_DEG",
    }
    for legacy_key, new_key in alias_pairs.items():
        if new_key in os.environ:
            continue
        legacy_val = os.environ.get(legacy_key)
        if legacy_val is None:
            continue
        os.environ[new_key] = legacy_val


_apply_legacy_env_aliases()


def _model_role_weight_params() -> Dict[str, float]:
    return {
        # Defaults prefer outer-frame semantics (baseline/doubles sidelines) for robustness.
        "baseline": float(_env_float("BADC_MODEL_W_BASELINE", 1.30)),
        "sideline_d": float(_env_float("BADC_MODEL_W_SIDELINE_D", 1.25)),
        "service_short": float(_env_float("BADC_MODEL_W_SERVICE_SHORT", 0.70)),
        "service_long": float(_env_float("BADC_MODEL_W_SERVICE_LONG", 0.80)),
        "sideline_s": float(_env_float("BADC_MODEL_W_SIDELINE_S", 0.75)),
        "center": float(_env_float("BADC_MODEL_W_CENTER", 0.55)),
    }


def _apply_model_role_weights(
    names: Sequence[str],
    weights: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Apply optional role-aware line weights (baseline / service / sideline / center).
    Defaults keep behavior unchanged (all multipliers = 1.0).
    """
    w = np.asarray(weights, dtype=np.float32).copy()
    if w.size == 0 or not names or len(names) != int(w.size):
        return w, _model_role_weight_params()
    cfg = _model_role_weight_params()

    def _mul_for_name(name: str) -> float:
        n = str(name)
        if n.startswith("baseline_"):
            return float(cfg["baseline"])
        if n in ("sideline_left_d", "sideline_right_d"):
            return float(cfg["sideline_d"])
        if n in ("short_near", "short_far"):
            return float(cfg["service_short"])
        if n in ("long_near_d", "long_far_d"):
            return float(cfg["service_long"])
        if n in ("sideline_left_s", "sideline_right_s"):
            return float(cfg["sideline_s"])
        if n.startswith("center_"):
            return float(cfg["center"])
        return 1.0

    mul = np.array([_mul_for_name(nm) for nm in names], dtype=np.float32)
    # Keep positive numeric weights.
    mul = np.clip(mul, 0.0, 10.0)
    w = w * mul
    return w, cfg


def _seg_len_px(seg) -> float:
    x1, y1, x2, y2 = seg
    return math.hypot(float(x2) - float(x1), float(y2) - float(y1))


def _filter_segments_minlen(segments, min_len_px: float, max_keep: int = 0):
    """Filter segments by minimum pixel length; optionally keep only the longest max_keep."""
    if segments is None:
        return []
    segs = [s for s in segments if _seg_len_px(s) >= float(min_len_px)]
    if max_keep and max_keep > 0 and len(segs) > max_keep:
        segs.sort(key=_seg_len_px, reverse=True)
        segs = segs[:max_keep]
    return segs


def _filter_segments_minlen_with_boundary_rescue(
    segments,
    min_len_px: float,
    floor_bbox_xyxy: Optional[Tuple[int, int, int, int]] = None,
    max_keep: int = 0,
    resq_enable: bool = True,
    resq_margin_frac: float = 0.08,
    resq_minlen_ratio: float = 0.35,
    resq_vert_cos_max: float = 0.35,
):
    """
    Keep long segments by min length and optionally rescue shorter near-vertical
    segments near left/right floor boundaries.
    """
    if segments is None:
        return []

    min_len = float(max(1.0, min_len_px))
    rescued_min = float(max(1.0, resq_minlen_ratio * min_len))
    out = []

    x0 = y0 = x1 = y1 = None
    margin = 0.0
    if floor_bbox_xyxy is not None:
        try:
            x0, y0, x1, y1 = [float(v) for v in floor_bbox_xyxy]
            bw = max(1.0, x1 - x0)
            margin = float(max(4.0, resq_margin_frac * bw))
        except Exception:
            x0 = y0 = x1 = y1 = None
            margin = 0.0

    for seg in segments:
        x1s, y1s, x2s, y2s = seg
        dx = float(x2s - x1s)
        dy = float(y2s - y1s)
        ll = float(math.hypot(dx, dy))
        if ll >= min_len:
            out.append(seg)
            continue
        if not resq_enable or ll < rescued_min or x0 is None or x1 is None:
            continue

        # Near-vertical segments: |dx| / len should be small.
        cos_abs = abs(dx) / max(ll, 1e-6)
        if cos_abs > float(max(0.0, resq_vert_cos_max)):
            continue

        near_left = (min(float(x1s), float(x2s)) <= float(x0) + margin)
        near_right = (max(float(x1s), float(x2s)) >= float(x1) - margin)
        if near_left or near_right:
            out.append(seg)

    if max_keep and max_keep > 0 and len(out) > max_keep:
        out.sort(key=_seg_len_px, reverse=True)
        out = out[:max_keep]
    return out


_RAW_HOUGH_THRESHOLD = _env_int("BADC_RAW_HOUGH_THRESHOLD", 45)
_RAW_HOUGH_MINLEN_RT = _env_float("BADC_RAW_HOUGH_MINLEN_RATIO", 0.05)
_RAW_SEG_MINLEN_PX = _env_float("BADC_RAW_SEG_MINLEN_PX", 40.0)
_RAW_SEG_MAX_KEEP = _env_int("BADC_RAW_SEG_MAX_KEEP", 0)
# -----------------------------------------------

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
    method_used: Optional[str] = None
    debug_image: Optional[np.ndarray] = None
    white_mask: Optional[np.ndarray] = None
    debug_image_init: Optional[np.ndarray] = None
    white_mask_raw: Optional[np.ndarray] = None
    white_mask_clean: Optional[np.ndarray] = None
    white_mask_raw_full: Optional[np.ndarray] = None
    white_mask_raw_floor: Optional[np.ndarray] = None
    white_mask_raw_floor_noblob: Optional[np.ndarray] = None
    white_mask_raw_floor_preblob: Optional[np.ndarray] = None
    white_mask_raw_floor_postblob: Optional[np.ndarray] = None
    floor_roi_mask: Optional[np.ndarray] = None
    floor_roi_overlay: Optional[np.ndarray] = None
    seed_bottom_mask: Optional[np.ndarray] = None
    green_mask: Optional[np.ndarray] = None
    largest_cc_mask: Optional[np.ndarray] = None
    exg_row_plot: Optional[np.ndarray] = None
    raw_floor_hough_lines_img: Optional[np.ndarray] = None
    raw_floor_hough_lines_a: Optional[np.ndarray] = None
    raw_floor_hough_lines_b: Optional[np.ndarray] = None
    raw_floor_dt_debug: Optional[np.ndarray] = None
    raw_floor_model_overlay: Optional[np.ndarray] = None
    raw_floor_top5_overlay: Optional[np.ndarray] = None
    raw_floor_preprocessed: Optional[np.ndarray] = None
    raw_floor_edges: Optional[np.ndarray] = None
    frame_model_overlay: Optional[np.ndarray] = None
    linepix_mask: Optional[np.ndarray] = None
    linepix_mask_pre: Optional[np.ndarray] = None
    floor_gate_mask: Optional[np.ndarray] = None
    linepix_overlay: Optional[np.ndarray] = None
    ransac_lines_img: Optional[np.ndarray] = None
    lsd_lines_a: Optional[np.ndarray] = None
    lsd_lines_b: Optional[np.ndarray] = None
    dt_debug: Optional[np.ndarray] = None
    ori_mask_a: Optional[np.ndarray] = None
    ori_mask_b: Optional[np.ndarray] = None
    hough_lines_img: Optional[np.ndarray] = None


@dataclass
class LineSeg:
    p1: np.ndarray
    p2: np.ndarray
    theta: float
    length: float
    support: float
    weight: float
    line: Tuple[float, float, float]


@dataclass
class RansacLineSeg:
    x1: float
    y1: float
    x2: float
    y2: float
    a: float
    b: float
    c: float
    support: int
    length: float


def _order_corners_lb_rb_rt_lt(pts_xy: np.ndarray) -> np.ndarray:
    pts = np.array(pts_xy, dtype=np.float32).reshape(4, 2)
    idx = np.argsort(pts[:, 1])
    ys = pts[idx, 1]
    y_span = float(np.max(ys) - np.min(ys))
    mid_gap = float(abs(ys[1] - ys[2]))
    if y_span < 1e-3 or mid_gap < max(5.0, 0.10 * y_span):
        tltrbrbl = _order_corners_tl_tr_br_bl(pts)
        tl, tr, br, bl = tltrbrbl
        return np.stack([bl, br, tr, tl], axis=0)
    top = pts[idx[:2]]
    bottom = pts[idx[2:]]
    bottom = bottom[np.argsort(bottom[:, 0])]
    top = top[np.argsort(top[:, 0])]
    lb, rb = bottom[0], bottom[1]
    lt, rt = top[0], top[1]
    return np.stack([lb, rb, rt, lt], axis=0)


def _quad_span_ok(
    pts_xy: np.ndarray,
    img_w: int,
    img_h: int,
    *,
    min_w_ratio: float = 0.18,
    min_h_ratio: float = 0.10,
    min_edge_ratio: float = 0.06,
) -> Tuple[bool, Dict[str, float]]:
    pts = np.array(pts_xy, dtype=np.float32).reshape(4, 2)
    minx = float(np.min(pts[:, 0]))
    maxx = float(np.max(pts[:, 0]))
    miny = float(np.min(pts[:, 1]))
    maxy = float(np.max(pts[:, 1]))
    bbox_w = float(maxx - minx)
    bbox_h = float(maxy - miny)
    edges = [
        float(np.linalg.norm(pts[i] - pts[(i + 1) % 4]))
        for i in range(4)
    ]
    min_edge = float(min(edges)) if edges else 0.0
    ok = True
    if bbox_w < float(min_w_ratio) * float(img_w):
        ok = False
    if bbox_h < float(min_h_ratio) * float(img_h):
        ok = False
    if min_edge < float(min_edge_ratio) * float(img_w):
        ok = False
    return ok, {"quad_bbox_w": bbox_w, "quad_bbox_h": bbox_h, "quad_min_edge": min_edge}


def _bottom_span_floor_ratio(
    pts_xy: np.ndarray,
    floor_bbox: Sequence[int],
) -> float:
    pts = np.array(pts_xy, dtype=np.float32).reshape(4, 2)
    lb = pts[0]
    rb = pts[1]
    span = float(np.linalg.norm(rb - lb))
    floor_w = float(max(1.0, float(floor_bbox[2]) - float(floor_bbox[0])))
    return float(span / floor_w)


def _top_edge_floor_ratio(
    pts_xy: np.ndarray,
    floor_bbox: Sequence[int],
) -> float:
    pts = np.array(pts_xy, dtype=np.float32).reshape(4, 2)
    lt = pts[3]
    rt = pts[2]
    y_top = float(0.5 * (lt[1] + rt[1]))
    floor_y0 = float(floor_bbox[1])
    floor_h = float(max(1.0, float(floor_bbox[3]) - float(floor_bbox[1])))
    return float((y_top - floor_y0) / floor_h)


def _find_peaks_1d(
    values: np.ndarray,
    *,
    min_sep: int,
    min_value: float,
    topk: int,
) -> list[int]:
    vals = np.asarray(values, dtype=np.float32).reshape(-1)
    if vals.size == 0:
        return []
    order = np.argsort(vals)[::-1]
    peaks: list[int] = []
    for idx in order:
        i = int(idx)
        v = float(vals[i])
        if v < float(min_value):
            break
        if all(abs(i - j) >= int(min_sep) for j in peaks):
            peaks.append(i)
            if len(peaks) >= int(topk):
                break
    return peaks


def _semantic_line_support_in_warp(
    warp_mask: np.ndarray,
    ppm: float,
) -> Dict[str, Any]:
    if warp_mask is None or warp_mask.size == 0:
        return {
            "valid": False,
            "score": -1e9,
            "near_pair_found": False,
            "near_pair_gap_m": None,
            "near_pair_gap_err_m": None,
            "near_pair_soft_fallback": False,
            "far_pair_found": False,
            "far_pair_gap_m": None,
            "far_pair_gap_err_m": None,
            "near_observable": False,
            "near_peak": 0.0,
            "far_observable": False,
            "far_peak": 0.0,
            "near_lower_stronger": False,
            "near_lower_stronger_margin": 0.0,
            "support_near_baseline": 0.0,
            "support_near_dls": 0.0,
            "support_far_baseline": 0.0,
            "support_far_dls": 0.0,
        }
    mask = (np.asarray(warp_mask) > 0).astype(np.uint8)
    hh, ww = mask.shape[:2]
    if hh < 8 or ww < 8:
        return {
            "valid": False,
            "score": -1e9,
            "near_pair_found": False,
            "near_pair_gap_m": None,
            "near_pair_gap_err_m": None,
            "near_pair_soft_fallback": False,
            "far_pair_found": False,
            "far_pair_gap_m": None,
            "far_pair_gap_err_m": None,
            "near_observable": False,
            "near_peak": 0.0,
            "far_observable": False,
            "far_peak": 0.0,
            "near_lower_stronger": False,
            "near_lower_stronger_margin": 0.0,
            "support_near_baseline": 0.0,
            "support_near_dls": 0.0,
            "support_far_baseline": 0.0,
            "support_far_dls": 0.0,
        }

    x0 = int(max(0, min(ww - 1, round(0.08 * ww))))
    x1 = int(max(x0 + 1, min(ww, round(0.92 * ww))))
    region = mask[:, x0:x1]
    row_cov = np.mean(region, axis=1).astype(np.float32)
    if row_cov.size >= 5:
        row_cov = cv2.GaussianBlur(row_cov.reshape(-1, 1), (1, 9), 0).reshape(-1)

    court_h_m = 13.40
    near_baseline_m = 13.40
    near_dls_m = 12.64
    far_baseline_m = 0.0
    far_dls_m = 0.76
    y_near = int(np.clip(round(near_baseline_m * ppm), 0, hh - 1))
    y_near_dls = int(np.clip(round(near_dls_m * ppm), 0, hh - 1))
    y_far = int(np.clip(round(far_baseline_m * ppm), 0, hh - 1))
    y_far_dls = int(np.clip(round(far_dls_m * ppm), 0, hh - 1))
    band = int(max(2, round(0.06 * ppm)))

    def _band_support(yc: int) -> float:
        y0 = int(max(0, yc - band))
        y1 = int(min(hh - 1, yc + band))
        if y1 < y0:
            return 0.0
        return float(np.mean(region[y0 : y1 + 1]))

    support_near_baseline = _band_support(y_near)
    support_near_dls = _band_support(y_near_dls)
    support_far_baseline = _band_support(y_far)
    support_far_dls = _band_support(y_far_dls)

    y_min = int(max(0, round((court_h_m - 2.6) * ppm)))
    y_max = int(hh - 1)
    near_rows = row_cov[y_min : y_max + 1]
    near_peak = float(np.max(near_rows)) if near_rows.size > 0 else 0.0
    near_observe_thr = float(max(0.015, min(0.25, _env_float("BADC_SEMANTIC_NEAR_OBSERVE_THR", 0.03))))
    near_observable = bool(near_peak >= near_observe_thr)
    near_pair_found = False
    near_pair_soft_fallback = False
    near_gap_m = None
    near_gap_err = None
    near_pair_y0 = None
    near_pair_y1 = None
    near_pair_base_sup = 0.0
    near_pair_dls_sup = 0.0
    near_pair_base_row: Optional[int] = None
    target_gap_px = float(0.76 * ppm)
    if near_rows.size > 0:
        near_max = float(np.max(near_rows))
        min_keep = float(max(0.02, 0.20 * near_max))
        peaks_local = _find_peaks_1d(
            near_rows,
            min_sep=int(max(4, round(0.14 * ppm))),
            min_value=min_keep,
            topk=12,
        )
        peaks = [int(y_min + p) for p in peaks_local]
        if peaks:
            peaks_sorted = sorted(peaks)
            pair_best = None
            min_gap_px = float(0.40 * ppm)
            max_gap_px = float(1.30 * ppm)
            near_span = float(max(1, y_max - y_min))
            for i in range(len(peaks_sorted)):
                for j in range(i + 1, len(peaks_sorted)):
                    p1 = int(peaks_sorted[i])
                    p0 = int(peaks_sorted[j])  # lower line (larger y)
                    gap_px = float(p0 - p1)
                    if gap_px < min_gap_px or gap_px > max_gap_px:
                        continue
                    e0 = float(row_cov[p0])
                    e1 = float(row_cov[p1])
                    baseline_rank = float(p0 - y_min) / near_span
                    score = float(
                        2.0 * e0
                        + 1.5 * e1
                        - abs(gap_px - target_gap_px) / max(1.0, 0.30 * ppm)
                        + 0.45 * baseline_rank
                    )
                    if pair_best is None or score > pair_best[0]:
                        pair_best = (score, p0, p1, gap_px, e0, e1)
            if pair_best is not None:
                near_pair_found = True
                near_pair_y0 = float(pair_best[1] / max(1e-6, ppm))
                near_pair_y1 = float(pair_best[2] / max(1e-6, ppm))
                near_gap_m = float(pair_best[3] / max(1e-6, ppm))
                near_gap_err = float(abs(near_gap_m - 0.76))
                near_pair_base_sup = float(pair_best[4])
                near_pair_dls_sup = float(pair_best[5])
                near_pair_base_row = int(pair_best[1])
        # Soft fallback: if DLS is weak/broken, keep semantic pair using strongest low baseline + expected 0.76m.
        if (not near_pair_found) and near_rows.size > 0:
            soft_rank = near_rows.astype(np.float32).copy()
            if soft_rank.size > 0:
                soft_rank += np.linspace(0.0, 0.06 * max(near_peak, 0.05), soft_rank.size, dtype=np.float32)
            p0_soft = int(y_min + int(np.argmax(soft_rank)))
            p1_soft = int(round(float(p0_soft) - target_gap_px))
            if 0 <= p1_soft < hh and p1_soft < p0_soft:
                e0_soft = float(row_cov[p0_soft])
                e1_soft = float(_band_support(p1_soft))
                min_e0_soft = float(max(0.03, 0.28 * near_peak))
                min_e1_soft = float(max(0.003, 0.05 * near_peak))
                if e0_soft >= min_e0_soft and e1_soft >= min_e1_soft:
                    near_pair_found = True
                    near_pair_soft_fallback = True
                    near_pair_base_row = int(p0_soft)
                    near_pair_base_sup = float(e0_soft)
                    near_pair_dls_sup = float(e1_soft)
                    near_gap_m = float((p0_soft - p1_soft) / max(1e-6, ppm))
                    near_gap_err = float(abs(near_gap_m - 0.76))
                    near_pair_y0 = float(p0_soft / max(1e-6, ppm))
                    near_pair_y1 = float(p1_soft / max(1e-6, ppm))

    support_near_baseline = float(max(float(support_near_baseline), float(near_pair_base_sup)))
    support_near_dls = float(max(float(support_near_dls), float(near_pair_dls_sup)))

    near_lower_stronger = False
    near_lower_stronger_margin = 0.0
    if near_pair_found and near_pair_base_row is not None:
        lower_sep_px = int(max(2, round(0.08 * ppm)))
        y_low0 = int(min(hh - 1, int(near_pair_base_row) + lower_sep_px))
        if y_low0 < hh:
            lower_peak = float(np.max(row_cov[y_low0:]))
            lower_ratio = float(max(1.0, _env_float("BADC_SEMANTIC_NEAR_LOWER_STRONG_RATIO", 1.10)))
            lower_margin_abs = float(max(0.0, _env_float("BADC_SEMANTIC_NEAR_LOWER_STRONG_MARGIN", 0.006)))
            threshold = max(lower_ratio * float(near_pair_base_sup), float(near_pair_base_sup) + lower_margin_abs)
            if lower_peak > threshold:
                near_lower_stronger = True
                near_lower_stronger_margin = float(lower_peak - float(near_pair_base_sup))

    near_gap_score = 0.0
    if near_pair_found and near_gap_m is not None:
        sigma = 0.22
        near_gap_score = float(math.exp(-((near_gap_m - 0.76) ** 2) / max(1e-6, 2.0 * sigma * sigma)))

    y_far_max = int(min(hh - 1, max(y_far_dls + int(round(2.2 * ppm)), int(round(3.0 * ppm)))))
    far_rows = row_cov[0 : y_far_max + 1]
    far_peak = float(np.max(far_rows)) if far_rows.size > 0 else 0.0
    far_observe_thr = float(max(0.015, min(0.20, _env_float("BADC_SEMANTIC_FAR_OBSERVE_THR", 0.03))))
    far_observable = bool(far_peak >= far_observe_thr)
    far_pair_found = False
    far_gap_m = None
    far_gap_err = None
    far_pair_y0 = None
    far_pair_y1 = None
    if far_rows.size > 0:
        far_max = float(np.max(far_rows))
        min_keep_far = float(max(0.015, 0.18 * far_max))
        peaks_local_far = _find_peaks_1d(
            far_rows,
            min_sep=int(max(4, round(0.14 * ppm))),
            min_value=min_keep_far,
            topk=10,
        )
        peaks_far = [int(p) for p in peaks_local_far]
        if peaks_far:
            peaks_far_sorted = sorted(peaks_far)
            pair_best_far = None
            target_gap_px_far = float(0.76 * ppm)
            min_gap_px_far = float(0.35 * ppm)
            max_gap_px_far = float(1.35 * ppm)
            for i in range(len(peaks_far_sorted)):
                for j in range(i + 1, len(peaks_far_sorted)):
                    p0 = int(peaks_far_sorted[i])  # upper line (smaller y): far baseline
                    p1 = int(peaks_far_sorted[j])  # lower line (larger y): far DLS
                    gap_px = float(p1 - p0)
                    if gap_px < min_gap_px_far or gap_px > max_gap_px_far:
                        continue
                    e0 = float(row_cov[p0])
                    e1 = float(row_cov[p1])
                    score = float(2.2 * e0 + 1.4 * e1 - abs(gap_px - target_gap_px_far) / max(1.0, 0.32 * ppm))
                    if pair_best_far is None or score > pair_best_far[0]:
                        pair_best_far = (score, p0, p1, gap_px, e0, e1)
            if pair_best_far is not None:
                far_pair_found = True
                far_pair_y0 = float(pair_best_far[1] / max(1e-6, ppm))
                far_pair_y1 = float(pair_best_far[2] / max(1e-6, ppm))
                far_gap_m = float(pair_best_far[3] / max(1e-6, ppm))
                far_gap_err = float(abs(far_gap_m - 0.76))

    far_gap_score = 0.0
    if far_pair_found and far_gap_m is not None:
        sigma_far = 0.22
        far_gap_score = float(math.exp(-((far_gap_m - 0.76) ** 2) / max(1e-6, 2.0 * sigma_far * sigma_far)))

    semantic_score = float(
        3.0 * support_near_baseline
        + 2.2 * support_near_dls
        + 1.3 * support_far_baseline
        + 0.8 * support_far_dls
        + 1.8 * near_gap_score
        + 1.3 * far_gap_score
        + (0.6 if near_pair_found else -0.8)
        + (0.35 if far_pair_found else (-0.20 if far_observable else 0.0))
        - (0.9 if near_lower_stronger else 0.0)
    )
    return {
        "valid": True,
        "score": float(semantic_score),
        "near_pair_found": bool(near_pair_found),
        "near_pair_soft_fallback": bool(near_pair_soft_fallback),
        "near_pair_gap_m": float(near_gap_m) if near_gap_m is not None else None,
        "near_pair_gap_err_m": float(near_gap_err) if near_gap_err is not None else None,
        "near_pair_baseline_y_m": float(near_pair_y0) if near_pair_y0 is not None else None,
        "near_pair_dls_y_m": float(near_pair_y1) if near_pair_y1 is not None else None,
        "near_observable": bool(near_observable),
        "near_peak": float(near_peak),
        "far_pair_found": bool(far_pair_found),
        "far_pair_gap_m": float(far_gap_m) if far_gap_m is not None else None,
        "far_pair_gap_err_m": float(far_gap_err) if far_gap_err is not None else None,
        "far_pair_baseline_y_m": float(far_pair_y0) if far_pair_y0 is not None else None,
        "far_pair_dls_y_m": float(far_pair_y1) if far_pair_y1 is not None else None,
        "far_observable": bool(far_observable),
        "far_peak": float(far_peak),
        "near_lower_stronger": bool(near_lower_stronger),
        "near_lower_stronger_margin": float(near_lower_stronger_margin),
        "support_near_baseline": float(support_near_baseline),
        "support_near_dls": float(support_near_dls),
        "support_far_baseline": float(support_far_baseline),
        "support_far_dls": float(support_far_dls),
        "warp_shape": [int(hh), int(ww)],
        "warp_crop_x": [int(x0), int(x1)],
    }


def _semantic_refine_homography(
    H_world_to_img: np.ndarray,
    line_mask_u8: Optional[np.ndarray],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    H0 = np.asarray(H_world_to_img, dtype=np.float64).reshape(3, 3)
    meta: Dict[str, Any] = {
        "enabled": False,
        "applied": False,
    }
    if line_mask_u8 is None or not isinstance(line_mask_u8, np.ndarray) or int(np.count_nonzero(line_mask_u8)) == 0:
        meta["skip_reason"] = "no_line_mask"
        return H0, meta
    if not np.all(np.isfinite(H0)):
        meta["skip_reason"] = "invalid_H"
        return H0, meta

    enable = bool(_env_flag("BADC_SEMANTIC_REFINE_ENABLE", True))
    meta["enabled"] = bool(enable)
    if not enable:
        meta["skip_reason"] = "disabled"
        return H0, meta

    ppm = float(max(60.0, _env_float("BADC_SEMANTIC_WARP_PPM", 120.0)))
    court_w_m = 6.10
    court_h_m = 13.40
    out_w = int(max(32, round(court_w_m * ppm)))
    out_h = int(max(64, round(court_h_m * ppm)))
    A = np.array([[ppm, 0.0, 0.0], [0.0, ppm, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    mask_u8 = (np.asarray(line_mask_u8) > 0).astype(np.uint8) * 255

    def _eval(Hm: np.ndarray) -> Dict[str, Any]:
        if not np.all(np.isfinite(Hm)):
            return {"valid": False, "score": -1e9}
        try:
            invH = np.linalg.inv(Hm)
        except Exception:
            return {"valid": False, "score": -1e9}
        M = A @ invH
        if not np.all(np.isfinite(M)):
            return {"valid": False, "score": -1e9}
        warped = cv2.warpPerspective(
            mask_u8,
            M.astype(np.float32),
            (out_w, out_h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        info = _semantic_line_support_in_warp(warped, ppm)
        info["valid"] = bool(info.get("valid", False))
        return info

    base_info = _eval(H0)
    best_H = H0
    best_info = dict(base_info)
    best_score = float(best_info.get("score", -1e9))
    best_a = 1.0
    best_b = 0.0

    a_grid = _parse_float_csv(_env_str("BADC_SEMANTIC_YSNAP_A_GRID", "0.96,0.98,1.00,1.02,1.04"), [0.96, 0.98, 1.0, 1.02, 1.04])
    b_grid = _parse_float_csv(
        _env_str("BADC_SEMANTIC_YSNAP_B_GRID", "-0.45,-0.35,-0.25,-0.15,-0.05,0.00,0.05,0.15,0.25,0.35,0.45"),
        [-0.45, -0.35, -0.25, -0.15, -0.05, 0.0, 0.05, 0.15, 0.25, 0.35, 0.45],
    )
    min_gain = float(_env_float("BADC_SEMANTIC_YSNAP_MIN_GAIN", 0.04))

    hard_enable = bool(_env_flag("BADC_SEMANTIC_HARD_ENABLE", True))
    hard_require_pair = bool(_env_flag("BADC_SEMANTIC_REQUIRE_PAIR", True))
    hard_require_pair_conditional = bool(_env_flag("BADC_SEMANTIC_REQUIRE_PAIR_CONDITIONAL", True))
    hard_require_dls_support = bool(_env_flag("BADC_SEMANTIC_REQUIRE_DLS_SUPPORT", False))
    hard_reject_near_lower_stronger = bool(_env_flag("BADC_SEMANTIC_REJECT_NEAR_LOWER_STRONGER", True))
    hard_require_far_pair = bool(_env_flag("BADC_SEMANTIC_REQUIRE_FAR_PAIR", False))
    hard_require_far_baseline_support = bool(_env_flag("BADC_SEMANTIC_REQUIRE_FAR_BASELINE_SUPPORT", True))
    hard_require_far_baseline_anchor = bool(_env_flag("BADC_SEMANTIC_REQUIRE_FAR_BASELINE_ANCHOR", True))
    hard_require_far_conditional = bool(_env_flag("BADC_SEMANTIC_REQUIRE_FAR_CONDITIONAL", True))
    min_sup_base = float(_env_float("BADC_SEMANTIC_MIN_BASELINE_SUPPORT", 0.025))
    min_sup_dls = float(_env_float("BADC_SEMANTIC_MIN_DLS_SUPPORT", 0.012))
    min_sup_far_base = float(_env_float("BADC_SEMANTIC_MIN_FAR_BASELINE_SUPPORT", 0.010))
    min_pair_dls_sup = float(_env_float("BADC_SEMANTIC_REQUIRE_PAIR_MIN_DLS_SUPPORT", 0.015))
    min_far_gap_dls_sup = float(_env_float("BADC_SEMANTIC_MIN_FAR_DLS_SUPPORT_FOR_GAP", 0.015))
    max_gap_err = float(_env_float("BADC_SEMANTIC_MAX_GAP_ERR_M", 0.40))
    max_far_gap_err = float(_env_float("BADC_SEMANTIC_MAX_FAR_GAP_ERR_M", 0.40))
    min_far_baseline_margin = float(_env_float("BADC_SEMANTIC_MIN_FAR_BASELINE_MARGIN", 0.002))
    max_far_baseline_y_m = float(_env_float("BADC_SEMANTIC_MAX_FAR_BASELINE_Y_M", 0.55))
    max_far_dls_y_m = float(_env_float("BADC_SEMANTIC_MAX_FAR_DLS_Y_M", 1.45))

    def _semantic_hard_check(info: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        hard_pass_i = True
        hard_reason_i = None
        if not hard_enable:
            return True, None
        near_observable_i = bool(info.get("near_observable", False))
        near_dls_sup_i = float(info.get("support_near_dls", 0.0))
        should_require_pair_i = bool(
            hard_require_pair
            and (
                not hard_require_pair_conditional
                or (near_observable_i and near_dls_sup_i >= min_pair_dls_sup)
            )
        )
        if should_require_pair_i and not bool(info.get("near_pair_found", False)):
            hard_pass_i = False
            hard_reason_i = "R_semantic_pair_missing"
        elif info.get("near_pair_gap_err_m") is not None and float(info.get("near_pair_gap_err_m")) > max_gap_err:
            hard_pass_i = False
            hard_reason_i = "R_semantic_gap"
        elif hard_reject_near_lower_stronger and bool(info.get("near_lower_stronger", False)):
            hard_pass_i = False
            hard_reason_i = "R_semantic_baseline_not_lowest"
        elif float(info.get("support_near_baseline", 0.0)) < min_sup_base:
            hard_pass_i = False
            hard_reason_i = "R_semantic_baseline_support"
        elif hard_require_dls_support and should_require_pair_i and float(info.get("support_near_dls", 0.0)) < min_sup_dls:
            hard_pass_i = False
            hard_reason_i = "R_semantic_dls_support"
        if hard_pass_i:
            far_observable_i = bool(info.get("far_observable", False))
            should_check_far_i = bool(hard_require_far_pair or hard_require_far_baseline_support)
            if should_check_far_i and (not hard_require_far_conditional or far_observable_i):
                if hard_require_far_pair and not bool(info.get("far_pair_found", False)):
                    hard_pass_i = False
                    hard_reason_i = "R_semantic_far_pair_missing"
                if hard_pass_i:
                    far_gap_err_i = info.get("far_pair_gap_err_m")
                    far_dls_sup_i = float(info.get("support_far_dls", 0.0))
                    if (
                        far_gap_err_i is not None
                        and far_dls_sup_i >= min_far_gap_dls_sup
                        and float(far_gap_err_i) > max_far_gap_err
                    ):
                        hard_pass_i = False
                        hard_reason_i = "R_semantic_far_gap"
                if hard_pass_i and hard_require_far_baseline_support and float(info.get("support_far_baseline", 0.0)) < min_sup_far_base:
                    hard_pass_i = False
                    hard_reason_i = "R_semantic_far_baseline_support"
                if hard_pass_i and hard_require_far_baseline_support and bool(info.get("far_pair_found", False)):
                    far_base_sup_i = float(info.get("support_far_baseline", 0.0))
                    far_dls_sup_i = float(info.get("support_far_dls", 0.0))
                    if far_base_sup_i + min_far_baseline_margin < far_dls_sup_i:
                        hard_pass_i = False
                        hard_reason_i = "R_semantic_far_baseline_vs_dls"
                if hard_pass_i and hard_require_far_baseline_anchor and bool(info.get("far_pair_found", False)):
                    far_base_y_i = info.get("far_pair_baseline_y_m")
                    far_dls_y_i = info.get("far_pair_dls_y_m")
                    if far_base_y_i is not None and float(far_base_y_i) > max_far_baseline_y_m:
                        hard_pass_i = False
                        hard_reason_i = "R_semantic_far_baseline_anchor"
                    elif far_dls_y_i is not None and float(far_dls_y_i) > max_far_dls_y_m:
                        hard_pass_i = False
                        hard_reason_i = "R_semantic_far_dls_anchor"
        return bool(hard_pass_i), hard_reason_i

    eval_bank: list[Tuple[np.ndarray, Dict[str, Any], float, float]] = [(H0, dict(base_info), 1.0, 0.0)]

    for a in a_grid:
        aa = float(max(0.90, min(1.10, float(a))))
        for b in b_grid:
            bb = float(max(-0.60, min(0.60, float(b))))
            if abs(aa - 1.0) < 1e-6 and abs(bb) < 1e-6:
                continue
            S = np.array([[1.0, 0.0, 0.0], [0.0, aa, bb], [0.0, 0.0, 1.0]], dtype=np.float64)
            Hc = H0 @ S
            info = _eval(Hc)
            eval_bank.append((Hc, dict(info), aa, bb))
            sc = float(info.get("score", -1e9))
            if sc > best_score + min_gain:
                best_score = sc
                best_H = Hc
                best_info = dict(info)
                best_a = aa
                best_b = bb

    hard_pass, hard_reason = _semantic_hard_check(best_info)
    hard_rescue_used = False
    hard_rescue_reason = None
    if hard_enable and not hard_pass:
        best_hard_item = None
        best_hard_score = float("-inf")
        for Hc_i, info_i, aa_i, bb_i in eval_bank:
            hp_i, hr_i = _semantic_hard_check(info_i)
            if not hp_i:
                continue
            sc_i = float(info_i.get("score", -1e9))
            if sc_i > best_hard_score:
                best_hard_score = sc_i
                best_hard_item = (Hc_i, info_i, aa_i, bb_i, hr_i)
        if best_hard_item is not None:
            Hc_i, info_i, aa_i, bb_i, _ = best_hard_item
            best_H = np.asarray(Hc_i, dtype=np.float64)
            best_info = dict(info_i)
            best_score = float(best_info.get("score", -1e9))
            best_a = float(aa_i)
            best_b = float(bb_i)
            hard_pass, hard_reason = _semantic_hard_check(best_info)
            hard_rescue_used = True
            hard_rescue_reason = "picked_hard_pass_candidate"

    meta.update(
        {
            "enabled": True,
            "warp_ppm": float(ppm),
            "initial_score": float(base_info.get("score", -1e9)),
            "final_score": float(best_score),
            "improved": bool(best_score > float(base_info.get("score", -1e9)) + min_gain),
            "ysnap_a": float(best_a),
            "ysnap_b_m": float(best_b),
            "min_gain": float(min_gain),
            "hard_enable": bool(hard_enable),
            "hard_require_pair": bool(hard_require_pair),
            "hard_require_pair_conditional": bool(hard_require_pair_conditional),
            "hard_require_dls_support": bool(hard_require_dls_support),
            "hard_reject_near_lower_stronger": bool(hard_reject_near_lower_stronger),
            "hard_require_far_pair": bool(hard_require_far_pair),
            "hard_require_far_baseline_support": bool(hard_require_far_baseline_support),
            "hard_require_far_baseline_anchor": bool(hard_require_far_baseline_anchor),
            "hard_require_far_conditional": bool(hard_require_far_conditional),
            "hard_pass": bool(hard_pass),
            "hard_reason": hard_reason,
            "hard_rescue_used": bool(hard_rescue_used),
            "hard_rescue_reason": hard_rescue_reason,
            "hard_min_baseline_support": float(min_sup_base),
            "hard_min_dls_support": float(min_sup_dls),
            "hard_min_far_baseline_support": float(min_sup_far_base),
            "hard_pair_min_dls_support": float(min_pair_dls_sup),
            "hard_far_gap_min_dls_support": float(min_far_gap_dls_sup),
            "hard_max_gap_err_m": float(max_gap_err),
            "hard_max_far_gap_err_m": float(max_far_gap_err),
            "hard_max_far_baseline_y_m": float(max_far_baseline_y_m),
            "hard_max_far_dls_y_m": float(max_far_dls_y_m),
            "support_near_baseline": float(best_info.get("support_near_baseline", 0.0)),
            "support_near_dls": float(best_info.get("support_near_dls", 0.0)),
            "near_observable": bool(best_info.get("near_observable", False)),
            "near_peak": float(best_info.get("near_peak", 0.0)),
            "support_far_baseline": float(best_info.get("support_far_baseline", 0.0)),
            "support_far_dls": float(best_info.get("support_far_dls", 0.0)),
            "near_pair_found": bool(best_info.get("near_pair_found", False)),
            "near_pair_soft_fallback": bool(best_info.get("near_pair_soft_fallback", False)),
            "near_pair_gap_m": best_info.get("near_pair_gap_m"),
            "near_pair_gap_err_m": best_info.get("near_pair_gap_err_m"),
            "near_pair_baseline_y_m": best_info.get("near_pair_baseline_y_m"),
            "near_pair_dls_y_m": best_info.get("near_pair_dls_y_m"),
            "near_lower_stronger": bool(best_info.get("near_lower_stronger", False)),
            "near_lower_stronger_margin": float(best_info.get("near_lower_stronger_margin", 0.0)),
            "far_pair_found": bool(best_info.get("far_pair_found", False)),
            "far_pair_gap_m": best_info.get("far_pair_gap_m"),
            "far_pair_gap_err_m": best_info.get("far_pair_gap_err_m"),
            "far_pair_baseline_y_m": best_info.get("far_pair_baseline_y_m"),
            "far_pair_dls_y_m": best_info.get("far_pair_dls_y_m"),
            "far_observable": bool(best_info.get("far_observable", False)),
            "far_peak": float(best_info.get("far_peak", 0.0)),
        }
    )
    meta["applied"] = bool(meta.get("improved", False))
    return best_H, meta


def _quad_area(pts_xy: np.ndarray) -> float:
    pts = np.array(pts_xy, dtype=np.float32).reshape(-1, 1, 2)
    return float(abs(cv2.contourArea(pts)))


def _quad_edges(pts_xy: np.ndarray) -> np.ndarray:
    pts = np.array(pts_xy, dtype=np.float32).reshape(4, 2)
    d = pts - np.roll(pts, -1, axis=0)
    return np.sqrt(np.sum(d * d, axis=1))


def _edge_vp_angle(p1: np.ndarray, p2: np.ndarray, vp: Optional[np.ndarray]) -> Optional[float]:
    if vp is None or not isinstance(vp, np.ndarray):
        return None
    v_edge = p2 - p1
    v_edge_norm = float(np.linalg.norm(v_edge))
    if v_edge_norm < 1e-6:
        return None
    mid = 0.5 * (p1 + p2)
    v_vp = vp.astype(np.float32) - mid
    v_vp_norm = float(np.linalg.norm(v_vp))
    if v_vp_norm < 1e-6:
        return None
    cosang = abs(float(np.dot(v_edge, v_vp)) / float(v_edge_norm * v_vp_norm))
    cosang = float(np.clip(cosang, -1.0, 1.0))
    return float(math.degrees(math.acos(cosang)))


def _aspect_penalty(ordered_xy: np.ndarray) -> float:
    pts = np.array(ordered_xy, dtype=np.float32).reshape(4, 2)
    lb, rb, rt, lt = pts[0], pts[1], pts[2], pts[3]
    w1 = float(np.linalg.norm(rb - lb))
    w2 = float(np.linalg.norm(rt - lt))
    h1 = float(np.linalg.norm(lt - lb))
    h2 = float(np.linalg.norm(rt - rb))
    width = max(1e-6, 0.5 * (w1 + w2))
    height = max(1e-6, 0.5 * (h1 + h2))
    ratio = height / width
    target = 13.4 / 6.1
    return float(math.log(max(ratio, 1e-6) / target) ** 2)


def _vanishing_point_penalty(
    ordered_xy: np.ndarray,
    vp_long: Optional[np.ndarray],
    vp_short: Optional[np.ndarray],
) -> float:
    pts = np.array(ordered_xy, dtype=np.float32).reshape(4, 2)
    edges = [
        (pts[0], pts[1]),
        (pts[1], pts[2]),
        (pts[2], pts[3]),
        (pts[3], pts[0]),
    ]
    angles = []
    for p1, p2 in edges:
        a1 = _edge_vp_angle(p1, p2, vp_long)
        a2 = _edge_vp_angle(p1, p2, vp_short)
        cand = [a for a in [a1, a2] if a is not None]
        if cand:
            angles.append(min(cand))
    if not angles:
        return 0.0
    return float(np.mean(angles) / 45.0)


def _out_of_floor_penalty(sample_uv: Optional[np.ndarray], floor_bbox: Sequence[int]) -> float:
    if sample_uv is None or len(sample_uv) == 0:
        return 1.0
    x0, y0, x1, y1 = [float(v) for v in floor_bbox]
    xs = sample_uv[:, 0]
    ys = sample_uv[:, 1]
    inside = (xs >= x0) & (xs <= x1) & (ys >= y0) & (ys <= y1)
    return float(1.0 - float(np.mean(inside)))


def _bottom_support_from_loss(
    loss_info: Dict[str, Any],
    ymax_ratio: float,
    img_h: int,
    tau_px: float,
) -> float:
    uv = loss_info.get("sample_uv")
    dists = loss_info.get("sample_dists")
    if (
        isinstance(uv, np.ndarray)
        and isinstance(dists, np.ndarray)
        and uv.ndim == 2
        and dists.ndim == 1
        and uv.shape[0] == dists.shape[0]
        and uv.shape[0] > 0
    ):
        inlier = dists < float(tau_px)
        if np.any(inlier):
            bottom = uv[inlier, 1] > (0.55 * float(img_h))
            return float(np.mean(bottom))
    return float(max(0.0, min(1.0, (float(ymax_ratio) - 0.55) / 0.45)))


def _position_penalty(ordered_xy: np.ndarray, floor_bbox: Sequence[int], target_ratio: float = 0.75) -> float:
    pts = np.array(ordered_xy, dtype=np.float32).reshape(4, 2)
    x0, y0, x1, y1 = [float(v) for v in floor_bbox]
    floor_h = max(1.0, y1 - y0)
    bottom_y = float(np.max(pts[:, 1]))
    bottom_norm = (bottom_y - y0) / floor_h
    return float(max(0.0, float(target_ratio) - float(bottom_norm)))


def _area_soft_penalty(area_ratio: float, min_ratio: float = 0.08) -> float:
    if float(area_ratio) >= float(min_ratio):
        return 0.0
    safe_ratio = max(float(area_ratio), 1e-6)
    scale = float(min_ratio) / safe_ratio
    return float((scale - 1.0) ** 2)


def _mixer_score_candidate(
    ordered_xy: np.ndarray,
    loss_info: Dict[str, Any],
    floor_bbox: Sequence[int],
    *,
    vp_long: Optional[np.ndarray] = None,
    vp_short: Optional[np.ndarray] = None,
    weights: Optional[Dict[str, float]] = None,
) -> Tuple[float, Dict[str, float]]:
    w = weights or MIXER_WEIGHTS
    dt_mean = float(loss_info.get("sample_dist_mean") or 0.0)
    dt_p90 = float(
        loss_info.get("sample_dist_p90_raw_weighted")
        or loss_info.get("sample_dist_p90_raw")
        or loss_info.get("sample_dist_p90")
        or 0.0
    )
    dt_score = float(dt_mean + 0.5 * dt_p90)
    cover_ratio = float(loss_info.get("cover_ratio", loss_info.get("inlier_ratio", 0.0)))
    aspect_pen = _aspect_penalty(ordered_xy)
    vp_pen = _vanishing_point_penalty(ordered_xy, vp_long, vp_short)
    out_floor = _out_of_floor_penalty(loss_info.get("sample_uv"), floor_bbox)
    cover_pen = float(1.0 - cover_ratio)
    pos_pen = _position_penalty(ordered_xy, floor_bbox)
    area_pen = _area_soft_penalty(float(loss_info.get("area_ratio", 0.0)))

    mixer = (
        float(w.get("dt", 1.0)) * dt_score
        + float(w.get("aspect", 0.0)) * aspect_pen
        + float(w.get("vp", 0.0)) * vp_pen
        + float(w.get("floor", 0.0)) * out_floor
        + float(w.get("cover", 0.0)) * cover_pen
        + float(w.get("pos", 0.0)) * pos_pen
        + float(w.get("area", 0.0)) * area_pen
    )
    parts = {
        "mixer_dt": dt_score,
        "mixer_aspect": aspect_pen,
        "mixer_vp": vp_pen,
        "mixer_floor": out_floor,
        "mixer_cover": cover_pen,
        "mixer_pos": pos_pen,
        "mixer_area": area_pen,
        "mixer_score": float(mixer),
    }
    return float(mixer), parts


def _passes_geom_gates(
    quad_xy: np.ndarray,
    floor_bbox: Sequence[int],
    *,
    min_edge_px: float = 2.0,
) -> Tuple[bool, str, Dict[str, float]]:
    pts = np.array(quad_xy, dtype=np.float32).reshape(4, 2)
    minx, miny = float(np.min(pts[:, 0])), float(np.min(pts[:, 1]))
    maxx, maxy = float(np.max(pts[:, 0])), float(np.max(pts[:, 1]))
    bbox_w = maxx - minx
    bbox_h = maxy - miny

    area = _quad_area(pts)
    edges = _quad_edges(pts)
    min_edge = float(np.min(edges)) if edges.size else 0.0

    metrics = {
        "cand_bbox_w": float(bbox_w),
        "cand_bbox_h": float(bbox_h),
        "cand_min_edge": float(min_edge),
    }
    if not np.isfinite(area) or area < 1.0:
        return False, "area_degenerate", metrics
    if not np.isfinite(bbox_w) or not np.isfinite(bbox_h):
        return False, "bbox_invalid", metrics
    if min_edge < float(min_edge_px):
        return False, "min_edge_too_small", metrics
    return True, "OK", metrics


def _is_convex_quad(pts_xy: np.ndarray) -> bool:
    pts = np.array(pts_xy, dtype=np.float32).reshape(-1, 1, 2)
    return bool(cv2.isContourConvex(pts))


def build_distance_transform(white_mask: np.ndarray) -> np.ndarray:
    inv = (white_mask == 0).astype(np.uint8)
    return cv2.distanceTransform(inv, distanceType=cv2.DIST_L2, maskSize=3)


def build_white_mask_raw(
    frame_bgr: np.ndarray,
    white_s_max: int = 140,
    white_v_min: int = 155,
    l_min: int = 170,
    chroma_max: int = 55,
    roi_mask_u8: Optional[np.ndarray] = None,
    soft_s_max: int = 120,
    soft_chroma_max: int = 45,
    soft_v_margin: int = 12,
    soft_l_margin: int = 6,
) -> np.ndarray:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    hsv_mask = cv2.inRange(hsv, (0, 0, white_v_min), (180, white_s_max, 255))
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    chroma = np.abs(a.astype(np.int16) - 128) + np.abs(b.astype(np.int16) - 128)
    lab_mask = (l >= int(l_min)) & (chroma <= int(chroma_max))
    strict = cv2.bitwise_and(hsv_mask, lab_mask.astype(np.uint8) * 255)
    if roi_mask_u8 is None or int(np.count_nonzero(roi_mask_u8)) == 0:
        return strict
    roi = roi_mask_u8 > 0
    if not np.any(roi):
        return strict
    v_p90 = float(np.percentile(v[roi], 90))
    l_p90 = float(np.percentile(l[roi], 90))
    v_soft = int(np.clip(v_p90 - int(soft_v_margin), 145, white_v_min))
    l_soft = int(np.clip(l_p90 - int(soft_l_margin), 170, 200))
    soft_hsv = (s <= int(soft_s_max)) & (v >= v_soft)
    soft_lab = (l >= l_soft) & (chroma <= int(soft_chroma_max))
    soft = (soft_hsv & soft_lab & roi).astype(np.uint8) * 255
    return cv2.bitwise_or(strict, soft)


def filter_blobs_keep_lines(
    mask_u8: np.ndarray,
    min_blob_area_ratio: float = 0.002,
    max_aspect_for_blob: float = 2.0,
    min_fill_ratio: float = 0.6,
    line_kernel_len: int = 31,
    line_kernel_thickness: int = 1,
    line_thickness_max: float = 4.0,
    close_kernel_lens: Tuple[int, ...] = (9, 13),
    open_kernel_lens: Tuple[int, ...] = (15, 21, 31),
    close_thickness: int = 2,
    speckle_area: int = 12,
    blocky_extent: float = 0.55,
    blocky_aspect_max: float = 2.3,
    blocky_area: int = 80,
    return_stats: bool = False,
) -> np.ndarray:
    """Remove blocky blobs while preserving line-like structures."""
    if mask_u8.ndim == 3:
        mask_u8 = cv2.cvtColor(mask_u8, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(mask_u8, 127, 255, cv2.THRESH_BINARY)
    lens = (max(15, int(line_kernel_len // 2)), int(line_kernel_len))
    line_like = _line_like_mask(bw, lengths=lens, thickness=int(line_kernel_thickness)) > 0
    dt = cv2.distanceTransform((bw > 0).astype(np.uint8), cv2.DIST_L2, 3)
    dt_dil = cv2.dilate(dt, np.ones((3, 3), np.uint8))
    ridge = (dt >= 1.0) & (dt >= (dt_dil - 0.01))
    line_like = line_like & ridge & (dt <= float(line_thickness_max))
    h, w = bw.shape[:2]
    min_blob_area = float(min_blob_area_ratio) * float(h * w)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    removed = np.zeros_like(bw)
    removed_stats: list[Dict[str, float]] = []
    for idx in range(1, num):
        area = float(stats[idx, cv2.CC_STAT_AREA])
        bw_cc = float(stats[idx, cv2.CC_STAT_WIDTH])
        bh_cc = float(stats[idx, cv2.CC_STAT_HEIGHT])
        if bw_cc <= 0.0 or bh_cc <= 0.0 or area <= 0.0:
            continue
        aspect = max(bw_cc, bh_cc) / max(min(bw_cc, bh_cc), 1.0)
        fill = area / max(bw_cc * bh_cc, 1.0)
        is_blob = (area >= min_blob_area) and (fill >= float(min_fill_ratio)) and (
            aspect <= float(max_aspect_for_blob)
        )
        cc_mask = labels == idx
        cc_line_like = bool(np.any(line_like & cc_mask))
        if area < int(speckle_area):
            if not cc_line_like:
                removed[cc_mask] = 255
                removed_stats.append(
                    {
                        "area": float(area),
                        "aspect": float(aspect),
                        "fill": float(fill),
                        "bbox_w": float(bw_cc),
                        "bbox_h": float(bh_cc),
                        "kind": "speckle",
                    }
                )
            continue
        if is_blob:
            cc_line_pixels = int(np.count_nonzero(line_like & cc_mask))
            if cc_line_pixels < int(0.02 * area):
                removed[cc_mask] = 255
                removed_stats.append(
                    {
                        "area": float(area),
                        "aspect": float(aspect),
                        "fill": float(fill),
                        "bbox_w": float(bw_cc),
                        "bbox_h": float(bh_cc),
                        "kind": "blob_drop_full",
                    }
                )
            else:
                # Remove only the non-line-like portion if blob is connected to lines.
                removed_part = cc_mask & (~line_like)
                if np.any(removed_part):
                    removed[removed_part] = 255
                    removed_stats.append(
                        {
                            "area": float(area),
                            "aspect": float(aspect),
                            "fill": float(fill),
                            "bbox_w": float(bw_cc),
                            "bbox_h": float(bh_cc),
                            "kind": "blob_drop_core",
                        }
                    )
    removed_stats.sort(key=lambda item: item["area"], reverse=True)
    keep = cv2.bitwise_and(bw, cv2.bitwise_not(removed))
    if return_stats:
        return keep, {
            "removed_count": int(np.count_nonzero(removed > 0)),
            "removed_topk": removed_stats[:5],
        }
    return keep


def detect_thin_white_line_pixels(
    frame_bgr: np.ndarray,
    roi_mask_u8: np.ndarray,
    *,
    y_thr: int = 200,
    dark_thr: int = 25,
    tau: int = 6,
    floor_bbox: Optional[Tuple[int, int, int, int]] = None,
    floor_y_cut: Optional[int] = None,
    enable_green_support: bool = False,
    enable_green_floor_gate: bool = False,
) -> np.ndarray:
    y_channel = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    h, w = y_channel.shape[:2]
    roi = np.zeros((h, w), dtype=bool)
    if floor_bbox is not None:
        x0, y0, x1, y1 = floor_bbox
        y0 = int(np.clip(y0, 0, h - 1))
        y1 = int(np.clip(y1, y0 + 1, h))
        x0 = int(np.clip(x0, 0, w - 1))
        x1 = int(np.clip(x1, x0 + 1, w))
        roi[y0:y1, x0:x1] = True
    elif roi_mask_u8 is not None:
        roi = roi_mask_u8 > 0
    else:
        roi[:, :] = True
    if floor_y_cut is not None:
        roi[: int(floor_y_cut), :] = False
    if np.any(roi):
        sigma_l = float(np.percentile(y_channel[roi], 75))
    else:
        sigma_l = float(y_thr)
    sigma_l = float(np.clip(sigma_l, 160.0, 235.0))
    bright = (y_channel >= sigma_l) & roi
    y = y_channel.astype(np.int16)
    b, g, r = cv2.split(frame_bgr)
    green_score = g.astype(np.int16) - np.maximum(r, b).astype(np.int16)
    g_delta = 20
    green = (green_score > g_delta) & roi
    green_left = np.zeros((h, w), dtype=bool)
    green_right = np.zeros((h, w), dtype=bool)
    green_up = np.zeros((h, w), dtype=bool)
    green_down = np.zeros((h, w), dtype=bool)
    for d in range(1, int(tau) + 1):
        green_left[:, d:] |= green[:, :-d]
        green_right[:, :-d] |= green[:, d:]
        green_up[d:, :] |= green[:-d, :]
        green_down[:-d, :] |= green[d:, :]
    green_support = green_left | green_right | green_up | green_down
    min_left = np.full((h, w), 255, dtype=np.int16)
    min_right = np.full((h, w), 255, dtype=np.int16)
    min_up = np.full((h, w), 255, dtype=np.int16)
    min_down = np.full((h, w), 255, dtype=np.int16)
    for d in range(1, int(tau) + 1):
        min_left[:, d:] = np.minimum(min_left[:, d:], y[:, :-d])
        min_right[:, :-d] = np.minimum(min_right[:, :-d], y[:, d:])
        min_up[d:, :] = np.minimum(min_up[d:, :], y[:-d, :])
        min_down[:-d, :] = np.minimum(min_down[:-d, :], y[d:, :])
    delta = int(dark_thr)
    y_minus = y - delta
    cond_any = (min_left <= y_minus) | (min_right <= y_minus) | (min_up <= y_minus) | (min_down <= y_minus)
    line = bright & cond_any
    if enable_green_support:
        line &= green_support
    if enable_green_floor_gate:
        green_mask = (green_score > g_delta).astype(np.uint8) * 255
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        d = int(tau) + 3
        k = np.ones((2 * d + 1, 2 * d + 1), np.uint8)
        green_dil = cv2.dilate(green_mask, k, iterations=1)
        if roi_mask_u8 is not None:
            green_dil = cv2.bitwise_and(green_dil, roi_mask_u8)
        if int(np.count_nonzero(green_dil)) > 0:
            line &= green_dil > 0
    pad = int(tau) + 2
    if pad > 0:
        line[:pad, :] = False
        line[-pad:, :] = False
        line[:, :pad] = False
        line[:, -pad:] = False
    line_u8 = (line.astype(np.uint8) * 255).astype(np.uint8)
    line_u8 = cv2.morphologyEx(line_u8, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return line_u8


def _person_mask_from_boxes(
    shape: Tuple[int, int],
    boxes: Sequence[Sequence[float]],
    pad: int = 10,
) -> Optional[np.ndarray]:
    if not boxes:
        return None
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    for box in boxes:
        if box is None or len(box) < 4:
            continue
        x1, y1, x2, y2 = box[:4]
        x1 = int(round(float(x1))) - pad
        y1 = int(round(float(y1))) - pad
        x2 = int(round(float(x2))) + pad
        y2 = int(round(float(y2))) + pad
        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w - 1, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h - 1, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        mask[y1 : y2 + 1, x1 : x2 + 1] = 255
    if int(np.count_nonzero(mask)) == 0:
        return None
    return mask


def _foot_points_from_boxes(
    boxes: Optional[Sequence[Sequence[float]]],
    img_h: Optional[int] = None,
    offset: float = 2.0,
) -> list[Tuple[float, float]]:
    """Return foot points (center of bottom edge) for person boxes."""
    pts: list[Tuple[float, float]] = []
    if boxes is None:
        return pts
    for box in boxes:
        if box is None or len(box) < 4:
            continue
        try:
            x1, y1, x2, y2 = [float(v) for v in box[:4]]
        except Exception:
            continue
        cx = 0.5 * (x1 + x2)
        fy = float(y2 + float(offset))
        if img_h is not None:
            fy = float(np.clip(fy, 0.0, float(max(img_h - 1, 0))))
        pts.append((cx, fy))
    return pts


def _count_points_inside_quad(
    quad_xy: np.ndarray,
    points_xy: Sequence[Tuple[float, float]],
) -> Tuple[int, list[bool]]:
    if quad_xy is None or len(points_xy) == 0:
        return 0, []
    quad = np.asarray(quad_xy, dtype=np.float32).reshape(4, 2)
    pts = []
    for pt in points_xy:
        if pt is None or len(pt) < 2:
            continue
        try:
            px, py = float(pt[0]), float(pt[1])
        except Exception:
            continue
        pts.append((px, py))
    if not pts:
        return 0, []
    contour = quad.reshape(-1, 1, 2)
    inside_flags: list[bool] = []
    for px, py in pts:
        res = cv2.pointPolygonTest(contour, (float(px), float(py)), measureDist=False)
        inside_flags.append(bool(res >= 0))
    count = sum(inside_flags)
    return int(count), inside_flags


def _filter_lines_for_completion(
    lines: list[Tuple[float, float, float, float]],
    line_ids: list[Optional[int]],
    quad_xy: np.ndarray,
    *,
    margin_px: float = 8.0,
    margin_x_px: Optional[float] = None,
    margin_top_px: Optional[float] = None,
    margin_bottom_px: Optional[float] = None,
    expand_span_axis: Optional[str] = None,
    min_span_ratio: float = 0.55,
    min_span_px: float = 220.0,
    extreme_keep: int = 2,
) -> Tuple[list[Tuple[float, float, float, float]], list[Optional[int]]]:
    if quad_xy is None or quad_xy.size == 0 or not lines:
        return lines, line_ids
    quad = np.asarray(quad_xy, dtype=np.float32).reshape(4, 2)
    xs = quad[:, 0]
    ys = quad[:, 1]
    mx = float(margin_x_px) if margin_x_px is not None else float(margin_px)
    my_top = float(margin_top_px) if margin_top_px is not None else float(margin_px)
    my_bottom = float(margin_bottom_px) if margin_bottom_px is not None else float(margin_px)
    x0 = float(np.min(xs)) - mx
    x1 = float(np.max(xs)) + mx
    y0 = float(np.min(ys)) - my_top
    y1 = float(np.max(ys)) + my_bottom
    bbox = (x0, y0, x1, y1)
    filtered_lines: list[Tuple[float, float, float, float]] = []
    filtered_ids: list[Optional[int]] = []
    for i, seg in enumerate(lines):
        lid = line_ids[i] if i < len(line_ids) else None
        x1s, y1s, x2s, y2s = seg
        seg_x0 = min(float(x1s), float(x2s))
        seg_x1 = max(float(x1s), float(x2s))
        seg_y0 = min(float(y1s), float(y2s))
        seg_y1 = max(float(y1s), float(y2s))
        # Use segment AABB overlap instead of midpoint-only gating so long boundary lines
        # crossing the completion ROI are preserved even when their midpoint is outside.
        if seg_x1 < bbox[0] or seg_x0 > bbox[2] or seg_y1 < bbox[1] or seg_y0 > bbox[3]:
            continue
        filtered_lines.append(seg)
        filtered_ids.append(lid)
    if not filtered_lines:
        return lines, line_ids

    if expand_span_axis in ("x", "y") and len(lines) >= 3 and len(filtered_lines) >= 2:
        axis = 0 if expand_span_axis == "x" else 1
        centers_src = [
            (0.5 * (float(seg[0]) + float(seg[2])), 0.5 * (float(seg[1]) + float(seg[3])))
            for seg in lines
        ]
        centers_keep = [
            (0.5 * (float(seg[0]) + float(seg[2])), 0.5 * (float(seg[1]) + float(seg[3])))
            for seg in filtered_lines
        ]
        vals_src = [float(c[axis]) for c in centers_src]
        vals_keep = [float(c[axis]) for c in centers_keep]
        src_span = float(max(vals_src) - min(vals_src)) if vals_src else 0.0
        keep_span = float(max(vals_keep) - min(vals_keep)) if vals_keep else 0.0
        need_span = float(max(float(min_span_px), float(min_span_ratio) * src_span))
        if src_span > 1.0 and keep_span < need_span:
            k = int(max(1, int(extreme_keep)))
            order = sorted(range(len(lines)), key=lambda idx: float(centers_src[idx][axis]))
            add_idx = order[: min(k, len(order))] + order[max(0, len(order) - k) :]
            seen: set[Tuple[int, int, int, int]] = set()
            merged_lines: list[Tuple[float, float, float, float]] = []
            merged_ids: list[Optional[int]] = []
            for seg, lid in zip(filtered_lines, filtered_ids):
                x1s, y1s, x2s, y2s = [float(v) for v in seg]
                p1 = (x1s, y1s)
                p2 = (x2s, y2s)
                if (p2[0], p2[1]) < (p1[0], p1[1]):
                    p1, p2 = p2, p1
                key = (
                    int(round(p1[0] / 2.0)),
                    int(round(p1[1] / 2.0)),
                    int(round(p2[0] / 2.0)),
                    int(round(p2[1] / 2.0)),
                )
                if key in seen:
                    continue
                seen.add(key)
                merged_lines.append((x1s, y1s, x2s, y2s))
                merged_ids.append(lid)
            for idx in add_idx:
                seg = lines[idx]
                lid = line_ids[idx] if idx < len(line_ids) else None
                x1s, y1s, x2s, y2s = [float(v) for v in seg]
                p1 = (x1s, y1s)
                p2 = (x2s, y2s)
                if (p2[0], p2[1]) < (p1[0], p1[1]):
                    p1, p2 = p2, p1
                key = (
                    int(round(p1[0] / 2.0)),
                    int(round(p1[1] / 2.0)),
                    int(round(p2[0] / 2.0)),
                    int(round(p2[1] / 2.0)),
                )
                if key in seen:
                    continue
                seen.add(key)
                merged_lines.append((x1s, y1s, x2s, y2s))
                merged_ids.append(lid)
            filtered_lines = merged_lines
            filtered_ids = merged_ids
    return filtered_lines, filtered_ids


def _fit_line_tls(points: np.ndarray) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=np.float64)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if pts.shape[0] < 2:
        v = np.array([1.0, 0.0], dtype=np.float64)
        mean = np.array([0.0, 0.0], dtype=np.float64)
        return 0.0, 1.0, 0.0, v, mean
    mean = np.mean(pts, axis=0)
    centered = pts - mean
    cov = centered.T @ centered
    if not np.isfinite(cov).all():
        v = np.array([1.0, 0.0], dtype=np.float64)
        return 0.0, 1.0, 0.0, v, mean
    eigvals, eigvecs = np.linalg.eigh(cov)
    v = eigvecs[:, int(np.argmax(eigvals))]
    norm_v = float(np.linalg.norm(v))
    if not math.isfinite(norm_v) or norm_v < 1e-6:
        v = np.array([1.0, 0.0], dtype=np.float64)
        return 0.0, 1.0, 0.0, v, mean
    v = v / norm_v
    a = -float(v[1])
    b = float(v[0])
    c = -(a * float(mean[0]) + b * float(mean[1]))
    norm = math.hypot(a, b)
    if norm > 1e-6 and math.isfinite(norm):
        a /= norm
        b /= norm
        c /= norm
    return a, b, c, v, mean


def ransac_line_segments_from_mask(
    line_mask_u8: np.ndarray,
    floor_mask_u8: np.ndarray,
    *,
    max_lines: int = 30,
    iters: int = 800,
    inlier_thr: float = 2.0,
    min_inliers: int = 250,
    min_length_ratio: float = 0.08,
    seed: int = 0,
) -> Tuple[list[RansacLineSeg], Dict[str, int]]:
    ys, xs = np.where((line_mask_u8 > 0) & (floor_mask_u8 > 0))
    if xs.size == 0:
        return [], {"tls_num_input_pts": 0, "tls_num_finite_pts": 0, "tls_skipped_degenerate": 0}
    pts = np.stack([xs, ys], axis=1).astype(np.float32)
    rng = np.random.default_rng(int(seed))
    max_points = 20000
    if pts.shape[0] > max_points:
        idx = rng.choice(pts.shape[0], size=max_points, replace=False)
        pts = pts[idx]
    h, w = line_mask_u8.shape[:2]
    min_len = float(min(h, w)) * float(min_length_ratio)
    active = np.ones((pts.shape[0],), dtype=bool)
    segments: list[RansacLineSeg] = []
    # --- NEW: dedup similar lines so max_lines isn't consumed by the same boundary ---
    dedup_enable = os.getenv("BADC_RANSAC_DEDUP", "1").strip() != "0"
    dedup_rho = float(os.getenv("BADC_RANSAC_DEDUP_RHO", "4.0"))
    dedup_theta_deg = float(os.getenv("BADC_RANSAC_DEDUP_THETA_DEG", "1.5"))
    dedup_theta = math.radians(dedup_theta_deg)
    accepted_normals: list[Tuple[float, float]] = []  # (theta, rho)
    tls_num_input_pts = 0
    tls_num_finite_pts = 0
    tls_skipped_degenerate = 0
    for _ in range(int(max_lines)):
        active_idx = np.where(active)[0]
        if active_idx.size < int(min_inliers):
            break
        pts_active = pts[active_idx]
        best_inliers = None
        best_count = 0
        best_line = None
        for _ in range(int(iters)):
            if pts_active.shape[0] < 2:
                break
            idx = rng.choice(pts_active.shape[0], size=2, replace=False)
            p1 = pts_active[idx[0]]
            p2 = pts_active[idx[1]]
            if float(np.linalg.norm(p1 - p2)) < 1.0:
                continue
            a = float(p1[1] - p2[1])
            b = float(p2[0] - p1[0])
            c = float(p1[0] * p2[1] - p2[0] * p1[1])
            norm = math.hypot(a, b)
            if norm < 1e-6:
                continue
            a /= norm
            b /= norm
            c /= norm
            dist = np.abs(a * pts_active[:, 0] + b * pts_active[:, 1] + c)
            inliers = dist < float(inlier_thr)
            count = int(np.count_nonzero(inliers))
            if count > best_count:
                best_count = count
                best_inliers = inliers
                best_line = (a, b, c)
        if best_inliers is None or best_count < int(min_inliers):
            break
        inlier_pts = pts_active[best_inliers]
        tls_num_input_pts += int(inlier_pts.shape[0])
        a, b, c, v, mean = _fit_line_tls(inlier_pts)
        pts64 = np.asarray(inlier_pts, dtype=np.float64)
        pts64 = pts64[np.isfinite(pts64).all(axis=1)]
        tls_num_finite_pts += int(pts64.shape[0])
        if pts64.shape[0] < 2 or not np.isfinite(mean).all() or not np.isfinite(v).all():
            tls_skipped_degenerate += 1
            continue
        centered = pts64 - mean
        if not np.isfinite(centered).all():
            tls_skipped_degenerate += 1
            continue
        max_abs = float(np.max(np.abs(centered))) if centered.size > 0 else 0.0
        if not math.isfinite(max_abs) or max_abs > 1e6:
            tls_skipped_degenerate += 1
            continue
        with np.errstate(all="ignore"):
            proj = centered @ v
        t_min = float(np.min(proj))
        t_max = float(np.max(proj))
        p1 = mean + t_min * v
        p2 = mean + t_max * v
        x1 = float(np.clip(p1[0], 0.0, float(w - 1)))
        y1 = float(np.clip(p1[1], 0.0, float(h - 1)))
        x2 = float(np.clip(p2[0], 0.0, float(w - 1)))
        y2 = float(np.clip(p2[1], 0.0, float(h - 1)))
        length = float(math.hypot(x2 - x1, y2 - y1))
        support = int(best_count)
        active[active_idx[best_inliers]] = False
        if length < float(min_len):
            continue

        # --- NEW: canonicalize (theta,rho) and dedup ---
        # line: a*x + b*y + c = 0 with (a,b) normalized.
        # rho = -c, theta = atan2(b,a) in [0,pi)
        if dedup_enable:
            theta = float(np.mod(math.atan2(b, a), math.pi))
            rho = float(-c)
            dup = False
            for (t0, r0) in accepted_normals:
                dt = abs(theta - t0)
                dt = min(dt, math.pi - dt)
                if dt <= dedup_theta and abs(rho - r0) <= dedup_rho:
                    dup = True
                    break
            if dup:
                # We still removed inliers from active, so next iterations explore new lines
                continue
            accepted_normals.append((theta, rho))
        segments.append(
            RansacLineSeg(
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                a=a,
                b=b,
                c=c,
                support=support,
                length=length,
            )
        )
    return segments, {
        "tls_num_input_pts": int(tls_num_input_pts),
        "tls_num_finite_pts": int(tls_num_finite_pts),
        "tls_skipped_degenerate": int(tls_skipped_degenerate),
    }


def _merge_ransac_with_raw_floor_segs(
    ransac_segs: list[RansacLineSeg],
    raw_floor_segs: list[Tuple[float, float, float, float]],
    *,
    rho_tol: float = 4.0,
    theta_tol_deg: float = 2.0,
) -> list[RansacLineSeg]:
    """Merge raw_floor hough segs into ransac segs (as pseudo segments), with dedup."""
    if not raw_floor_segs:
        return ransac_segs
    theta_tol = math.radians(theta_tol_deg)
    out = list(ransac_segs)

    def _seg_to_theta_rho(x1, y1, x2, y2):
        dx = float(x2 - x1)
        dy = float(y2 - y1)
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return None
        ang = float(np.mod(math.atan2(dy, dx), math.pi))
        # normal angle = ang + pi/2
        theta = float(np.mod(ang + math.pi / 2.0, math.pi))
        # rho = x*cos(theta) + y*sin(theta)
        rho = float(x1 * math.cos(theta) + y1 * math.sin(theta))
        return theta, rho

    existing = []
    for s in out:
        tr = _seg_to_theta_rho(s.x1, s.y1, s.x2, s.y2)
        if tr:
            existing.append(tr)

    for (x1, y1, x2, y2) in raw_floor_segs:
        tr = _seg_to_theta_rho(x1, y1, x2, y2)
        if not tr:
            continue
        theta, rho = tr
        dup = False
        for (t0, r0) in existing:
            dt = abs(theta - t0)
            dt = min(dt, math.pi - dt)
            if dt <= theta_tol and abs(rho - r0) <= rho_tol:
                dup = True
                break
        if dup:
            continue
        existing.append((theta, rho))
        length = float(math.hypot(x2 - x1, y2 - y1))
        out.append(
            RansacLineSeg(
                x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2),
                a=0.0, b=0.0, c=0.0, support=0, length=length
            )
        )
    return out


def build_white_mask(
    frame_bgr: np.ndarray,
    white_s_max: int = 80,
    white_v_min: int = 180,
    close_kernel: int = 2,
    open_kernel: int = 3,
    close_iter: int = 1,
    open_iter: int = 1,
    floor_y_min_ratio: float = 0.45,
    floor_x_max_ratio: float = 0.92,
    max_blob_area_ratio: float = 0.002,
    min_aspect_ratio: float = 6.0,
    max_line_width_px: int = 20,
    max_fill_ratio: float = 0.6,
    green_delta_min: int = 12,
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int], int, int]:
    white_raw = build_white_mask_raw(frame_bgr, white_s_max=white_s_max, white_v_min=white_v_min)
    h, w = white_raw.shape[:2]
    y1 = int(round(float(h) * float(floor_y_min_ratio)))
    x2 = int(round(float(w) * float(floor_x_max_ratio)))
    y1 = int(np.clip(y1, 0, h - 1))
    x2 = int(np.clip(x2, 1, w))
    roi_mask = np.zeros_like(white_raw)
    roi_mask[y1:h, 0:x2] = 255
    white = cv2.bitwise_and(white_raw, roi_mask)
    bgr_blur = cv2.GaussianBlur(frame_bgr, (5, 5), 0)
    b, g, r = cv2.split(bgr_blur)
    green_score = g.astype(np.int16) - np.maximum(r, b).astype(np.int16)
    green_mask = (green_score > int(green_delta_min)).astype(np.uint8) * 255
    green_mask = cv2.bitwise_and(green_mask, roi_mask)
    white = cv2.bitwise_and(white, green_mask)
    if int(np.count_nonzero(white)) == 0:
        white = cv2.bitwise_and(white_raw, roi_mask)
    kernel_open = np.ones((open_kernel, open_kernel), np.uint8)
    if open_iter > 0:
        white = cv2.morphologyEx(white, cv2.MORPH_OPEN, kernel_open, iterations=open_iter)
    kernel_dilate = np.ones((close_kernel, close_kernel), np.uint8)
    if close_iter > 0:
        white = cv2.dilate(white, kernel_dilate, iterations=close_iter)
    roi_area = float(max(1, (h - y1) * x2))
    max_area = float(max_blob_area_ratio) * roi_area
    max_cc_area_raw = 0
    max_cc_area_clean = 0
    if max_area > 0:
        num, labels, stats, _ = cv2.connectedComponentsWithStats(white, connectivity=8)
        keep = np.zeros_like(white)
        for idx in range(1, num):
            area = float(stats[idx, cv2.CC_STAT_AREA])
            max_cc_area_raw = max(max_cc_area_raw, int(area))
            if area <= 0:
                continue
            bw = float(stats[idx, cv2.CC_STAT_WIDTH])
            bh = float(stats[idx, cv2.CC_STAT_HEIGHT])
            aspect = float(max(bw, bh)) / float(max(min(bw, bh), 1.0))
            min_side = float(min(bw, bh))
            fill = float(area) / float(max(bw * bh, 1.0))
            is_line_like = aspect >= float(min_aspect_ratio) and min_side <= float(max_line_width_px)
            keep_blob = area <= float(max_area) and fill <= float(max_fill_ratio)
            if is_line_like or keep_blob:
                keep[labels == idx] = 255
                max_cc_area_clean = max(max_cc_area_clean, int(area))
        white = keep
    floor_bbox = (0, y1, x2 - 1, h - 1)
    return white_raw, white, floor_bbox, int(max_cc_area_raw), int(max_cc_area_clean)


def _render_exg_row_plot(exg_row: np.ndarray, y_cut: Optional[int]) -> Optional[np.ndarray]:
    if exg_row is None or exg_row.size == 0:
        return None
    h = int(exg_row.size)
    w = 120
    vmin = float(np.percentile(exg_row, 5))
    vmax = float(np.percentile(exg_row, 95))
    denom = max(vmax - vmin, 1e-6)
    norm = np.clip((exg_row - vmin) / denom, 0.0, 1.0)
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        val = int(round(norm[y] * 255.0))
        img[y, :] = (val, val, val)
    if y_cut is not None:
        y_cut_i = int(np.clip(y_cut, 0, h - 1))
        cv2.line(img, (0, y_cut_i), (w - 1, y_cut_i), (0, 0, 255), 2)
    return img


def _floor_mask_from_seed_lab(
    frame_bgr: np.ndarray,
    *,
    seed_ratio: float = 0.40,
    patch: int = 9,
    delta_lab: float = 18.0,
    min_area_ratio: float = 0.06,
    seed_x_ratios: Sequence[float] = (0.2, 0.4, 0.5, 0.6, 0.8),
    dilate_ksize: int = 19,
    bbox_pad_ratio: float = 0.02,
) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]], Dict[str, Any]]:
    h, w = frame_bgr.shape[:2]
    roi_y0 = int(round(float(h) * float(seed_ratio)))
    roi_y0 = max(0, min(h - 1, roi_y0))
    bgr_roi = frame_bgr[roi_y0:h, :]
    if bgr_roi.size == 0:
        return None, None, {"lab_seed_ok": False}
    lab = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2LAB).astype(np.int16)
    rh, rw = lab.shape[:2]
    sy_full = int(np.clip(int(round(0.95 * float(h))), 0, h - 1))
    sy = int(np.clip(sy_full - roi_y0, 0, rh - 1))
    r = max(1, int(patch) // 2)
    sx_vals: list[int] = []
    for rx in seed_x_ratios:
        try:
            sx_vals.append(int(np.clip(int(round(float(rx) * float(w))), 0, rw - 1)))
        except Exception:
            continue
    if not sx_vals:
        sx_vals = [int(np.clip(w // 2, 0, rw - 1))]
    seeds = []
    for sx in sx_vals:
        x0 = max(0, sx - r)
        x1 = min(rw, sx + r + 1)
        y0 = max(0, sy - r)
        y1 = min(rh, sy + r + 1)
        seeds.append(lab[y0:y1, x0:x1].mean(axis=(0, 1)))
    seed = np.mean(np.stack(seeds, axis=0), axis=0)
    d = lab - seed[None, None, :]
    dist = np.sqrt((d * d).sum(axis=2)).astype(np.float32)
    mask = dist < float(delta_lab)
    mask &= (lab[..., 0] < 245)
    mask_u8 = (mask.astype(np.uint8) * 255)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, k, iterations=1)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, k, iterations=1)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    label_set = set()
    for sx in sx_vals:
        if 0 <= sy < rh and 0 <= sx < rw:
            lbl = int(labels[sy, sx])
            if lbl != 0:
                label_set.add(lbl)
    if not label_set:
        return None, None, {"lab_seed_ok": False}
    cc = np.zeros((rh, rw), dtype=np.uint8)
    for lbl in label_set:
        cc[labels == lbl] = 1
    area = int(cc.sum())
    if area < int(min_area_ratio * float(rw * rh)):
        return None, None, {"lab_seed_ok": False, "lab_seed_area": area}
    mask_full = np.zeros((h, w), dtype=np.uint8)
    mask_full[roi_y0:h, :] = cc.astype(np.uint8) * 255
    if int(dilate_ksize) > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(dilate_ksize), int(dilate_ksize)))
        mask_full = cv2.dilate(mask_full, k, iterations=1)
    ys, xs = np.where(mask_full > 0)
    if xs.size == 0:
        return None, None, {"lab_seed_ok": False}
    pad = int(round(float(bbox_pad_ratio) * float(w)))
    x_min = max(0, int(xs.min()) - pad)
    x_max = min(w - 1, int(xs.max()) + pad)
    y_min = max(0, int(ys.min()) - pad)
    y_max = min(h - 1, int(ys.max()) + pad)
    bbox = (x_min, y_min, x_max, y_max)
    return mask_full, bbox, {
        "lab_seed_ok": True,
        "lab_seed_bbox": [int(x_min), int(y_min), int(x_max), int(y_max)],
        "lab_seed_y0": int(roi_y0),
    }


def _load_manual_vp_corners(path: str) -> Optional[Dict[str, Any]]:
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _line_point_dist(line: Tuple[float, float, float], pt: Tuple[float, float]) -> float:
    a, b, c = line
    x, y = pt
    return float(abs(a * x + b * y + c))


def _format_predicted_corners(
    corners_uv: np.ndarray,
    width: int,
    height: int,
) -> list[Dict[str, Any]]:
    labels = ["LB", "RB", "RT", "LT"]
    out: list[Dict[str, Any]] = []
    for idx, lab in enumerate(labels):
        x = float(corners_uv[idx, 0])
        y = float(corners_uv[idx, 1])
        visible = 0.0 <= x < float(width) and 0.0 <= y < float(height)
        out.append({"label": lab, "x": x, "y": y, "visible": bool(visible)})
    return out


def _corner_errors_to_manual(
    corners_uv: np.ndarray,
    manual_pts: Dict[str, Any],
) -> Dict[str, Optional[float]]:
    if corners_uv is None or manual_pts is None:
        return {"err_LT": None, "err_RT": None, "err_RB": None}
    mapping = {"LT": 3, "RT": 2, "RB": 1}
    out: Dict[str, Optional[float]] = {"err_LT": None, "err_RT": None, "err_RB": None}
    for key, idx in mapping.items():
        pt = manual_pts.get(key)
        if not isinstance(pt, dict):
            continue
        try:
            x = float(pt.get("x"))
            y = float(pt.get("y"))
        except Exception:
            continue
        dx = float(corners_uv[idx, 0]) - x
        dy = float(corners_uv[idx, 1]) - y
        out[f"err_{key}"] = float(math.hypot(dx, dy))
    return out

def _tighten_floor_roi_xrange(
    green_mask: np.ndarray,
    floor_mask: np.ndarray,
    *,
    strip_ratio: float = 0.06,
    min_density: float = 0.20,
    min_width_ratio: float = 0.20,
    margin_ratio: float = 0.02,
    support_mask: Optional[np.ndarray] = None,
    min_keep_ratio: float = 0.60,
    min_abs_width_ratio: float = 0.60,
    court_corners: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    h, w = floor_mask.shape[:2]
    green_work = green_mask
    floor_roi_source = "floor_mask"
    court_bbox = None
    court_bbox_pad = None
    if court_corners is not None:
        try:
            cc = np.asarray(court_corners, dtype=np.float32).reshape(-1, 2)
        except Exception:
            cc = None
        if cc is not None and cc.shape == (4, 2) and np.isfinite(cc).all():
            in_bounds = (
                (cc[:, 0] >= 0.0)
                & (cc[:, 0] <= float(w - 1))
                & (cc[:, 1] >= 0.0)
                & (cc[:, 1] <= float(h - 1))
            )
            if bool(np.all(in_bounds)):
                lb, rb, rt, lt = cc
                court_w = float(np.linalg.norm(rb - lb))
                court_h = float(np.linalg.norm(lt - lb))
                mx = 0.25 * court_w
                my = 0.20 * court_h
                minx = float(np.min(cc[:, 0]) - mx)
                maxx = float(np.max(cc[:, 0]) + mx)
                miny = float(np.min(cc[:, 1]) - my)
                maxy = float(np.max(cc[:, 1]) + my)
                x0c = max(0, int(round(minx)))
                x1c = min(w - 1, int(round(maxx)))
                y0c = max(0, int(round(miny)))
                y1c = min(h - 1, int(round(maxy)))
                court_bbox = (x0c, y0c, x1c, y1c)
                court_bbox_pad = (float(mx), float(my))
                floor_roi_source = "corners_bbox"
                floor_mask = floor_mask.copy()
                floor_mask[:y0c, :] = 0
                floor_mask[y1c + 1 :, :] = 0
                floor_mask[:, :x0c] = 0
                floor_mask[:, x1c + 1 :] = 0
                if green_mask is not None:
                    green_work = green_mask.copy()
                    green_work[:y0c, :] = 0
                    green_work[y1c + 1 :, :] = 0
                    green_work[:, :x0c] = 0
                    green_work[:, x1c + 1 :] = 0
    ys, xs = np.where(floor_mask > 0)
    if xs.size == 0:
        return floor_mask, {
            "floor_x_tightened": False,
            "floor_roi_source": floor_roi_source,
            "court_bbox": list(court_bbox) if court_bbox is not None else None,
            "court_bbox_pad": list(court_bbox_pad) if court_bbox_pad is not None else None,
        }
    orig_x0 = int(xs.min())
    orig_x1 = int(xs.max())
    orig_w = max(1, int(orig_x1 - orig_x0 + 1))
    y_top = int(ys.min())
    y_bot = int(ys.max())
    strip_h1 = max(3, int(round((y_bot - y_top + 1) * float(strip_ratio))))
    strip_h2 = max(3, int(round((y_bot - y_top + 1) * 0.08)))
    y_strip1 = max(y_top, y_bot - strip_h1 + 1)
    y_strip0 = max(y_top, y_bot - strip_h1 - strip_h2 + 1)
    strip1 = green_work[y_strip1 : y_bot + 1, :]
    strip2 = green_work[y_strip0:y_strip1, :]
    if strip1.size == 0:
        return floor_mask, {
            "floor_x_tightened": False,
            "floor_strip_y0": int(y_strip0),
            "floor_strip_y1": int(y_strip1),
            "floor_roi_source": floor_roi_source,
            "court_bbox": list(court_bbox) if court_bbox is not None else None,
            "court_bbox_pad": list(court_bbox_pad) if court_bbox_pad is not None else None,
        }
    density1 = (strip1 > 0).mean(axis=0)
    if strip2.size > 0:
        density2 = (strip2 > 0).mean(axis=0)
    else:
        density2 = np.zeros_like(density1)
    col_density = 0.6 * density1 + 0.4 * density2
    thr = float(np.percentile(col_density, 70.0))
    thr = max(thr, float(min_density))
    keep = col_density >= thr
    best_len = 0
    best_a = best_b = None
    run_a = None
    for i, v in enumerate(keep):
        if v and run_a is None:
            run_a = i
        if (not v or i == len(keep) - 1) and run_a is not None:
            run_b = i if not v else i + 1
            run_len = run_b - run_a
            if run_len > best_len:
                best_len = run_len
                best_a, best_b = run_a, run_b
            run_a = None
    min_width = int(round(float(min_width_ratio) * float(w)))
    if best_a is None or best_b is None or best_len < max(5, min_width):
        fallback_bbox = court_bbox
        if fallback_bbox is None:
            fallback_bbox = (orig_x0, y_top, orig_x1, y_bot)
        x0f, y0f, x1f, y1f = fallback_bbox
        fallback = np.zeros_like(floor_mask)
        fallback[y0f : y1f + 1, x0f : x1f + 1] = 255
        return fallback, {
            "floor_x_tightened": False,
            "floor_x_range": [int(x0f), int(x1f)],
            "floor_strip_thr": float(thr),
            "floor_strip_y0": int(y_strip0),
            "floor_strip_y1": int(y_strip1),
            "floor_x_reject_reason": "width_too_narrow_keep_bbox",
            "floor_roi_source": floor_roi_source,
            "court_bbox": list(court_bbox) if court_bbox is not None else None,
            "court_bbox_pad": list(court_bbox_pad) if court_bbox_pad is not None else None,
        }
    margin = int(round(float(margin_ratio) * float(w)))
    x0 = max(0, best_a - margin)
    x1 = min(w - 1, best_b - 1 + margin)
    cand_w = max(1, int(x1 - x0 + 1))
    min_width_abs = int(round(float(min_abs_width_ratio) * float(w)))
    min_width_orig = int(round(float(min_keep_ratio) * float(orig_w)))
    if cand_w < max(min_width_abs, min_width_orig):
        fallback_bbox = court_bbox
        if fallback_bbox is None:
            fallback_bbox = (orig_x0, y_top, orig_x1, y_bot)
        x0f, y0f, x1f, y1f = fallback_bbox
        fallback = np.zeros_like(floor_mask)
        fallback[y0f : y1f + 1, x0f : x1f + 1] = 255
        return fallback, {
            "floor_x_tightened": False,
            "floor_x_range": [int(x0f), int(x1f)],
            "floor_strip_thr": float(thr),
            "floor_strip_y0": int(y_strip0),
            "floor_strip_y1": int(y_strip1),
            "floor_x_reject_reason": "width_too_narrow_keep_bbox",
            "floor_roi_source": floor_roi_source,
            "court_bbox": list(court_bbox) if court_bbox is not None else None,
            "court_bbox_pad": list(court_bbox_pad) if court_bbox_pad is not None else None,
        }
    tightened = floor_mask.copy()
    if x0 > 0:
        tightened[:, :x0] = 0
    if x1 + 1 < w:
        tightened[:, x1 + 1 :] = 0
    floor_x_support_ratio = None
    if support_mask is not None:
        support_total = int(np.count_nonzero((support_mask > 0) & (floor_mask > 0)))
        support_keep = int(np.count_nonzero((support_mask > 0) & (tightened > 0)))
        keep_ratio = float(support_keep) / float(max(support_total, 1))
        floor_x_support_ratio = float(keep_ratio)
        if support_total > 0 and keep_ratio < float(min_keep_ratio):
            fallback_bbox = court_bbox
            if fallback_bbox is None:
                fallback_bbox = (orig_x0, y_top, orig_x1, y_bot)
            x0f, y0f, x1f, y1f = fallback_bbox
            fallback = np.zeros_like(floor_mask)
            fallback[y0f : y1f + 1, x0f : x1f + 1] = 255
            return fallback, {
                "floor_x_tightened": False,
                "floor_x_range": [int(x0f), int(x1f)],
                "floor_strip_thr": float(thr),
                "floor_strip_y0": int(y_strip0),
                "floor_strip_y1": int(y_strip1),
                "floor_x_support_ratio": float(keep_ratio),
                "floor_x_reject_reason": "support_drop_keep_bbox",
                "floor_roi_source": floor_roi_source,
                "court_bbox": list(court_bbox) if court_bbox is not None else None,
                "court_bbox_pad": list(court_bbox_pad) if court_bbox_pad is not None else None,
            }
    return tightened, {
        "floor_x_tightened": True,
        "floor_x_range": [int(x0), int(x1)],
        "floor_strip_thr": float(thr),
        "floor_strip_y0": int(y_strip0),
        "floor_strip_y1": int(y_strip1),
        "floor_x_support_ratio": floor_x_support_ratio,
        "floor_roi_source": floor_roi_source,
        "court_bbox": list(court_bbox) if court_bbox is not None else None,
        "court_bbox_pad": list(court_bbox_pad) if court_bbox_pad is not None else None,
    }


def get_floor_roi_mask_debug(
    frame_bgr: np.ndarray,
    fallback_y_ratio: float = 0.55,
    seed_ratio: float = 0.40,
    min_cc_ratio: float = 0.04,
    green_threshold_percentile: float = 56.0,
    morphology_kernel_size: int = 11,
    closing_iterations: int = 4,
    court_corners: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    seed_ratio = float(np.clip(_env_float("BADC_FLOOR_ROI_SEED_RATIO", seed_ratio), 0.10, 0.90))
    h, w = frame_bgr.shape[:2]
    b, g, r = cv2.split(frame_bgr)
    green_score = g.astype(np.float32) - np.maximum(r, b).astype(np.float32)
    white_support = build_white_mask_raw(frame_bgr)
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    hue_ch, sat_ch, val_ch = cv2.split(hsv)
    rgb_sum = r.astype(np.float32) + g.astype(np.float32) + b.astype(np.float32)
    green_ratio_map = g.astype(np.float32) / np.maximum(rgb_sum, 1.0)
    floor_roi_top_guard_cfg = {
        "min_bbox_h_frac": float(np.clip(_env_float("BADC_FLOOR_ROI_MIN_BBOX_H_FRAC", 0.40), 0.05, 1.0)),
        "max_y1_frac": float(np.clip(_env_float("BADC_FLOOR_ROI_MAX_Y1_FRAC", 0.35), 0.05, 0.95)),
        "max_y1_hard_frac": float(np.clip(_env_float("BADC_FLOOR_ROI_MAX_Y1_HARD_FRAC", 0.50), 0.05, 0.98)),
        "y1_wide_min_w_frac": float(np.clip(_env_float("BADC_FLOOR_ROI_Y1_WIDE_MIN_W_FRAC", 0.90), 0.10, 1.00)),
        "fallback_top_frac": float(np.clip(_env_float("BADC_FLOOR_ROI_FALLBACK_TOP_FRAC", 0.23), 0.05, 0.95)),
    }

    def _build_adaptive_green_mask(
        seed_mask_u8: np.ndarray,
        top_boost: bool = False,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        roi = seed_mask_u8 > 0
        if not np.any(roi):
            return np.zeros((h, w), dtype=np.uint8), {"green_adapt_reason": "empty_seed"}
        base_apply_top_frac = float(np.clip(_env_float("BADC_FLOOR_ROI_GREEN_APPLY_TOP_FRAC", seed_ratio), 0.0, 0.95))
        boost_apply_top_frac = float(
            np.clip(_env_float("BADC_FLOOR_ROI_GREEN_TOP_BOOST_APPLY_TOP_FRAC", 0.22), 0.0, 0.95)
        )
        apply_top_frac = float(boost_apply_top_frac if top_boost else base_apply_top_frac)
        apply_y0 = int(max(0, min(h - 1, round(apply_top_frac * float(h)))))
        apply_roi = np.zeros((h, w), dtype=bool)
        apply_roi[apply_y0:h, :] = True
        if _env_flag("BADC_FLOOR_ROI_GREEN_APPLY_INTERSECT_SEED", False):
            apply_roi = apply_roi & roi
        if not np.any(apply_roi):
            apply_roi = roi.copy()
        score_vals = green_score[roi]
        ratio_vals = green_ratio_map[roi]
        sat_vals = sat_ch[roi]
        val_vals = val_ch[roi]
        p_hi = float(np.clip(float(green_threshold_percentile), 30.0, 90.0))
        delta_pct = float(_env_float("BADC_FLOOR_ROI_GREEN_ADAPT_DELTA_PCT", 16.0))
        p_lo = float(max(18.0, p_hi - max(6.0, delta_pct)))
        thr_hi = float(np.percentile(score_vals, p_hi))
        thr_lo = float(np.percentile(score_vals, p_lo))
        ratio_thr = float(np.percentile(ratio_vals, 35.0))
        sat_thr = float(np.clip(np.percentile(sat_vals, 18.0), 18.0, 140.0))
        val_thr = float(np.clip(np.percentile(val_vals, 10.0), 12.0, 170.0))

        rough = roi & ((green_score >= thr_lo) | (green_ratio_map >= ratio_thr))
        hue_lo, hue_hi = 28, 98
        if int(np.count_nonzero(rough)) > 256:
            hue_vals = hue_ch[rough]
            hue_lo = int(np.clip(float(np.percentile(hue_vals, 3.0)) - 6.0, 18.0, 100.0))
            hue_hi = int(np.clip(float(np.percentile(hue_vals, 97.0)) + 6.0, max(hue_lo + 6, 30), 125))
        hue_ok = (hue_ch >= hue_lo) & (hue_ch <= hue_hi)

        strict = (
            ((green_score >= thr_hi) | ((green_score >= thr_lo) & (green_ratio_map >= ratio_thr)))
            & hue_ok
            & (sat_ch >= sat_thr)
            & (val_ch >= val_thr)
        )
        green_mask_local = (strict & apply_roi).astype(np.uint8) * 255
        k = max(3, int(morphology_kernel_size))
        if k % 2 == 0:
            k += 1
        kernel = np.ones((k, k), np.uint8)
        green_mask_local = cv2.morphologyEx(
            green_mask_local,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=max(1, int(closing_iterations)),
        )
        green_mask_local = cv2.morphologyEx(green_mask_local, cv2.MORPH_OPEN, kernel)

        top_relax_used = False
        if top_boost:
            boost_band_frac = float(np.clip(_env_float("BADC_FLOOR_ROI_GREEN_TOP_BOOST_BAND_FRAC", 0.62), 0.10, 0.95))
            top_y1 = int(max(apply_y0, min(h - 1, round(apply_y0 + boost_band_frac * float(max(1, h - apply_y0 - 1))))))
            top_band = np.zeros((h, w), dtype=bool)
            top_band[apply_y0 : top_y1 + 1, :] = True
            top_val_relax = float(np.clip(_env_float("BADC_FLOOR_ROI_GREEN_TOP_BOOST_VAL_RELAX", 26.0), 0.0, 90.0))
            top_sat_relax = float(np.clip(_env_float("BADC_FLOOR_ROI_GREEN_TOP_BOOST_SAT_RELAX", 14.0), 0.0, 70.0))
            top_score_relax = float(np.clip(_env_float("BADC_FLOOR_ROI_GREEN_TOP_BOOST_SCORE_RELAX", 5.0), 0.0, 24.0))
            top_ratio_mul = float(np.clip(_env_float("BADC_FLOOR_ROI_GREEN_TOP_BOOST_RATIO_MUL", 0.94), 0.70, 1.20))
            boost_relaxed = (
                top_band
                & apply_roi
                & hue_ok
                & (
                    (green_score >= (thr_lo - top_score_relax))
                    | (green_ratio_map >= max(0.14, ratio_thr * top_ratio_mul))
                )
                & (val_ch >= max(6.0, val_thr - top_val_relax))
                & (sat_ch >= max(8.0, sat_thr - top_sat_relax))
            )
            boost_mask = (boost_relaxed & apply_roi).astype(np.uint8) * 255
            top_close_w = int(max(0, _env_int("BADC_FLOOR_ROI_GREEN_TOP_BOOST_CLOSE_W", 41)))
            top_close_h = int(max(0, _env_int("BADC_FLOOR_ROI_GREEN_TOP_BOOST_CLOSE_H", 9)))
            if top_close_w > 0 and top_close_h > 0:
                if top_close_w % 2 == 0:
                    top_close_w += 1
                if top_close_h % 2 == 0:
                    top_close_h += 1
                boost_mask = cv2.morphologyEx(
                    boost_mask,
                    cv2.MORPH_CLOSE,
                    cv2.getStructuringElement(cv2.MORPH_RECT, (top_close_w, top_close_h)),
                    iterations=1,
                )
            boost_mask[top_y1 + 1 :, :] = 0
            if int(np.count_nonzero(boost_mask)) > 0:
                green_mask_local = cv2.bitwise_or(green_mask_local, boost_mask)
                top_relax_used = True

        cover_ratio = float(np.mean(green_mask_local[apply_roi] > 0)) if np.any(apply_roi) else 0.0
        min_cover = float(_env_float("BADC_FLOOR_ROI_GREEN_MIN_COVER", 0.22))
        relaxed_used = False
        if cover_ratio < min_cover:
            relaxed = (
                ((green_score >= thr_lo) | (green_ratio_map >= max(0.18, ratio_thr * 0.96)))
                & hue_ok
                & (val_ch >= max(8.0, val_thr - 20.0))
            )
            green_mask_local = (relaxed & apply_roi).astype(np.uint8) * 255
            k_rel = max(3, int(round(0.85 * float(k))))
            if k_rel % 2 == 0:
                k_rel += 1
            kernel_rel = np.ones((k_rel, k_rel), np.uint8)
            green_mask_local = cv2.morphologyEx(green_mask_local, cv2.MORPH_CLOSE, kernel_rel, iterations=1)
            green_mask_local = cv2.morphologyEx(
                green_mask_local, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1
            )
            cover_ratio = float(np.mean(green_mask_local[apply_roi] > 0)) if np.any(apply_roi) else 0.0
            relaxed_used = True
        return green_mask_local, {
            "green_adapt_thr_hi": float(thr_hi),
            "green_adapt_thr_lo": float(thr_lo),
            "green_adapt_ratio_thr": float(ratio_thr),
            "green_adapt_sat_thr": float(sat_thr),
            "green_adapt_val_thr": float(val_thr),
            "green_adapt_hue_lo": int(hue_lo),
            "green_adapt_hue_hi": int(hue_hi),
            "green_adapt_cover_ratio": float(cover_ratio),
            "green_adapt_min_cover": float(min_cover),
            "green_adapt_relaxed_used": bool(relaxed_used),
            "green_adapt_top_boost": bool(top_boost),
            "green_adapt_apply_y0": int(apply_y0),
            "green_adapt_apply_top_frac": float(apply_top_frac),
            "green_adapt_top_relax_used": bool(top_relax_used),
        }

    def _guard_floor_roi_top(
        floor_mask_u8: np.ndarray,
        y_cut: Optional[int],
        source: str,
    ) -> Tuple[np.ndarray, Optional[int], Dict[str, Any]]:
        meta: Dict[str, Any] = {
            "source": str(source),
            "applied": False,
            "reason": None,
            "y1_raw": int(y_cut) if y_cut is not None else None,
            "y1_final": int(y_cut) if y_cut is not None else None,
            "y2": None,
            "bbox_w": None,
            "bbox_w_frac": None,
            "bbox_h": None,
            "bbox_h_frac": None,
            "cfg": dict(floor_roi_top_guard_cfg),
        }
        if y_cut is None:
            meta["reason"] = "missing_y_cut"
            return floor_mask_u8, y_cut, meta

        ys, xs = np.where(floor_mask_u8 > 0)
        if xs.size == 0:
            meta["reason"] = "empty_floor_mask"
            return floor_mask_u8, int(y_cut), meta

        y1_raw = int(max(0, min(h - 1, int(y_cut))))
        y2 = int(ys.max())
        if y2 < y1_raw:
            y2 = y1_raw
        x1 = int(xs.min())
        x2 = int(xs.max())
        bbox_w = int(max(1, x2 - x1 + 1))
        bbox_h = int(max(1, y2 - y1_raw + 1))
        min_bbox_h = int(round(float(floor_roi_top_guard_cfg["min_bbox_h_frac"]) * float(h)))
        max_y1 = int(round(float(floor_roi_top_guard_cfg["max_y1_frac"]) * float(h)))
        max_y1_hard = int(round(float(floor_roi_top_guard_cfg["max_y1_hard_frac"]) * float(h)))
        y1_wide_min_w = float(floor_roi_top_guard_cfg["y1_wide_min_w_frac"])
        fallback_top_y = int(round(float(floor_roi_top_guard_cfg["fallback_top_frac"]) * float(h)))
        fallback_top_y = int(max(0, min(h - 1, fallback_top_y)))
        bbox_w_frac = float(bbox_w) / float(max(w, 1))

        reasons: list[str] = []
        if bbox_h < max(1, min_bbox_h):
            reasons.append("bbox_h_small")
        if y1_raw > max_y1 and bbox_w_frac >= y1_wide_min_w:
            reasons.append("y1_too_low_wide_bbox")
        if y1_raw > max_y1_hard:
            reasons.append("y1_too_low_hard")

        y1_final = y1_raw
        floor_mask_out = floor_mask_u8
        if reasons:
            y1_final = int(max(0, min(y1_raw, fallback_top_y)))
            if y1_final < y1_raw:
                floor_mask_out = floor_mask_u8.copy()
                floor_mask_out[y1_final:y1_raw, x1 : x2 + 1] = 255
                meta["applied"] = True
            meta["reason"] = "+".join(reasons)
        else:
            meta["reason"] = "pass"

        meta["y1_raw"] = int(y1_raw)
        meta["y1_final"] = int(y1_final)
        meta["y2"] = int(y2)
        meta["bbox_w"] = int(bbox_w)
        meta["bbox_w_frac"] = float(bbox_w_frac)
        meta["bbox_h"] = int(bbox_h)
        meta["bbox_h_frac"] = float(bbox_h) / float(max(h, 1))
        return floor_mask_out, int(y1_final), meta
    prepass_corners = None
    prepass_area_ratio = None
    prepass_ymax_ratio = None
    prepass_bottom_support = None
    prepass_gate_passed = False
    prepass_line = white_support.copy()
    try:
        if not _env_flag("BADC_FLOOR_PREPASS_ENABLE", False):
            raise RuntimeError("floor prepass disabled")
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        k = int(max(21, round(min(h, w) * 0.03)))
        if k % 2 == 0:
            k += 1
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1))
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, k))
        tophat_h = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel_h)
        tophat_v = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel_v)
        tophat = cv2.max(tophat_h, tophat_v)
        roi_vals = tophat[white_support > 0]
        if roi_vals.size > 0:
            thresh = max(float(np.percentile(roi_vals, 75.0)), 10.0)
            thin_line_pre = (tophat >= thresh).astype(np.uint8) * 255
            prepass_line = cv2.bitwise_and(white_support, thin_line_pre)
        else:
            prepass_line = white_support.copy()
        full_mask = np.ones((h, w), dtype=np.uint8) * 255
        pre_H, _pre_metrics, _pre_debug = _fit_court_homography_from_raw_floor_debug(
            prepass_line,
            frame_bgr.shape,
            floor_roi_mask=full_mask,
        )
        if pre_H is not None and np.all(np.isfinite(pre_H)):
            pre_uv = project_points(pre_H, get_bwf_corners())
            if np.all(np.isfinite(pre_uv)):
                pre_ordered = _order_corners_lb_rb_rt_lt(pre_uv)
                area = float(_quad_area(pre_ordered))
                bbox_w = float(np.max(pre_ordered[:, 0]) - np.min(pre_ordered[:, 0]))
                bbox_h = float(np.max(pre_ordered[:, 1]) - np.min(pre_ordered[:, 1]))
                min_edge = float(np.min(_quad_edges(pre_ordered)))
                if bbox_w < 0.08 * float(w) or bbox_h < 0.06 * float(h) or min_edge < 0.05 * float(min(h, w)):
                    prepass_area_ratio = None
                    prepass_ymax_ratio = None
                    prepass_gate_passed = False
                else:
                    pre_clamp = pre_ordered.copy()
                    pre_clamp[:, 0] = np.clip(pre_clamp[:, 0], 0.0, float(w - 1))
                    pre_clamp[:, 1] = np.clip(pre_clamp[:, 1], 0.0, float(h - 1))
                    prepass_area_ratio = float(_quad_area(pre_clamp)) / float(max(h * w, 1))
                    y_max = float(np.max(pre_ordered[:, 1]))
                    prepass_ymax_ratio = max(0.0, min(1.0, y_max / float(max(h - 1, 1))))
                    prepass_bottom_support = max(
                        0.0,
                        min(1.0, (prepass_ymax_ratio - 0.55) / 0.45),
                    )
                    if prepass_area_ratio >= 0.06 and (prepass_bottom_support or 0.0) >= 0.35:
                        prepass_gate_passed = True
                        prepass_corners = pre_ordered
                    prepass_gate_passed = True
                    prepass_corners = pre_ordered
    except Exception:
        prepass_corners = None
        prepass_gate_passed = False

    def _select_top_components(
        mask_u8: np.ndarray,
        support_u8: np.ndarray,
        top_k: int = 2,
    ) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]], list[Dict[str, float]]]:
        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
        comps = []
        for idx in range(1, num):
            area = int(stats[idx, cv2.CC_STAT_AREA])
            if area <= 0:
                continue
            support_cnt = int(np.count_nonzero(support_u8[labels == idx]))
            score = float(area) * (1.0 + float(support_cnt) / float(max(area, 1)))
            comps.append({"idx": idx, "area": float(area), "support": float(support_cnt), "score": score})
        if not comps:
            return None, None, []
        comps.sort(key=lambda item: item["score"], reverse=True)
        keep = comps[: max(1, int(top_k))]
        keep_ids = [int(item["idx"]) for item in keep]
        union = np.isin(labels, keep_ids).astype(np.uint8) * 255
        ys_u, xs_u = np.where(union > 0)
        if xs_u.size == 0:
            return None, None, keep
        bbox = (int(xs_u.min()), int(ys_u.min()), int(xs_u.max()), int(ys_u.max()))
        return union, bbox, keep
    seed_y = int(round(float(h) * float(seed_ratio)))
    seed = np.zeros((h, w), dtype=np.uint8)
    seed[seed_y:h, :] = 255
    bottom_scores = green_score[seed > 0]
    fallback_floor_roi = False
    fallback_mode = "cc"
    floor_y_cut = None
    floor_top_guard_meta: Optional[Dict[str, Any]] = None
    largest_cc_mask = np.zeros((h, w), dtype=np.uint8)
    # Default to direct green-mask ROI to avoid largest-CC truncation on far court.
    use_green_mask_direct = _env_flag("BADC_FLOOR_ROI_USE_GREEN_MASK_DIRECT", True)
    lab_mask, lab_bbox, lab_debug = _floor_mask_from_seed_lab(
        frame_bgr,
        seed_ratio=seed_ratio,
        patch=9,
        delta_lab=18.0,
        min_area_ratio=0.06,
    )
    if (not use_green_mask_direct) and lab_mask is not None and lab_bbox is not None:
        cc_scores = []
        union_mask, union_bbox, cc_scores = _select_top_components(
            (lab_mask > 0).astype(np.uint8) * 255,
            white_support,
            top_k=2,
        )
        if union_mask is not None and union_bbox is not None:
            x1, y1, x2, y2 = union_bbox
            margin = int(0.03 * float(min(h, w)))
            x1 = max(0, x1 - margin)
            y1 = max(0, y1 - margin)
            x2 = min(w - 1, x2 + margin)
            y2 = min(h - 1, y2 + margin)
            floor_mask = np.zeros((h, w), dtype=np.uint8)
            floor_mask[y1 : y2 + 1, x1 : x2 + 1] = 255
            floor_y_cut = int(y1)
        else:
            floor_mask = (lab_mask > 0).astype(np.uint8) * 255
            ys_cc, xs_cc = np.where(floor_mask > 0)
            if xs_cc.size > 0:
                floor_y_cut = int(ys_cc.min())
        floor_mask, floor_y_cut, floor_top_guard_meta = _guard_floor_roi_top(
            floor_mask,
            floor_y_cut,
            source="lab_seed",
        )
        fallback_mode = "lab_seed"
        if bottom_scores.size > 0:
            use_top_boost = bool((floor_top_guard_meta or {}).get("applied", False))
            green_mask, green_adapt_meta = _build_adaptive_green_mask(seed, top_boost=use_top_boost)
        else:
            green_mask = np.zeros((h, w), dtype=np.uint8)
            green_adapt_meta = {"green_adapt_reason": "empty_bottom_scores"}
        floor_mask, tighten_info = _tighten_floor_roi_xrange(
            green_mask,
            floor_mask,
            support_mask=white_support,
            court_corners=prepass_corners if prepass_gate_passed else court_corners,
            min_density=0.09,
            min_keep_ratio=0.45,
            margin_ratio=0.075,
        )
        return floor_mask, {
            "green_mask": green_mask,
            "seed_bottom_mask": seed,
            "largest_cc_mask": lab_mask,
            "fallback_floor_roi": False,
            "fallback_mode": fallback_mode,
            "floor_y_cut": int(floor_y_cut) if floor_y_cut is not None else None,
            "floor_roi_top_guard": floor_top_guard_meta,
            "floor_roi_top_guard_cfg": dict(floor_roi_top_guard_cfg),
            "exg_row_plot": None,
            "floor_cc_scores": cc_scores,
            "prepass_area_ratio": prepass_area_ratio,
            "prepass_ymax_ratio": prepass_ymax_ratio,
            "prepass_gate_passed": prepass_gate_passed,
            "green_adapt_meta": green_adapt_meta,
            **lab_debug,
            **tighten_info,
        }
    if bottom_scores.size == 0:
        fallback_floor_roi = True
        fallback_mode = "bottom_band"
        floor_y_cut = int(round(float(h) * float(fallback_y_ratio)))
        floor_mask = np.zeros((h, w), dtype=np.uint8)
        floor_mask[floor_y_cut:h, :] = 255
        floor_mask, tighten_info = _tighten_floor_roi_xrange(
            np.zeros((h, w), dtype=np.uint8),
            floor_mask,
            support_mask=white_support,
            court_corners=prepass_corners if prepass_gate_passed else court_corners,
            min_density=0.09,
            min_keep_ratio=0.45,
            margin_ratio=0.075,
        )
        return floor_mask, {
            "green_mask": np.zeros((h, w), dtype=np.uint8),
            "seed_bottom_mask": seed,
            "largest_cc_mask": largest_cc_mask,
            "fallback_floor_roi": fallback_floor_roi,
            "fallback_mode": fallback_mode,
            "floor_y_cut": int(floor_y_cut),
            "floor_roi_top_guard": floor_top_guard_meta,
            "floor_roi_top_guard_cfg": dict(floor_roi_top_guard_cfg),
            "exg_row_plot": None,
            "prepass_area_ratio": prepass_area_ratio,
            "prepass_ymax_ratio": prepass_ymax_ratio,
            "prepass_gate_passed": prepass_gate_passed,
            **tighten_info,
        }
    green_mask, green_adapt_meta = _build_adaptive_green_mask(seed, top_boost=False)
    if use_green_mask_direct:
        green_bin = ((green_mask > 0).astype(np.uint8) * 255)
        seed_y_frac = float(os.getenv("BADC_FLOOR_ROI_SEED_Y_FRAC", "0.93") or 0.93)
        seed_y = int(round(seed_y_frac * float(max(h - 1, 0))))
        seed_xs_str = os.getenv("BADC_FLOOR_ROI_SEED_XS", "0.20,0.40,0.60,0.80")
        seed_x_fracs: list[float] = []
        for s in seed_xs_str.split(","):
            s = s.strip()
            if not s:
                continue
            try:
                seed_x_fracs.append(float(s))
            except Exception:
                pass
        if not seed_x_fracs:
            seed_x_fracs = [0.2, 0.4, 0.6, 0.8]
        radius = int(os.getenv("BADC_FLOOR_ROI_SEED_RADIUS", "60") or 60)
        radius = max(5, radius)
        picked = np.zeros_like(green_bin)

        def _find_nearest_green_direct(xc: int, yc: int, r: int):
            x0p = max(0, xc - r)
            x1p = min(w - 1, xc + r)
            y0p = max(0, yc - r)
            y1p = min(h - 1, yc + r)
            win = green_bin[y0p : y1p + 1, x0p : x1p + 1]
            ys, xs = np.where(win > 0)
            if len(xs) == 0:
                return None
            ix = int(np.median(xs)) + x0p
            iy = int(np.median(ys)) + y0p
            return (ix, iy)

        for fx in seed_x_fracs:
            sx = int(round(fx * float(max(w - 1, 0))))
            pt = _find_nearest_green_direct(sx, seed_y, radius)
            if pt is None:
                continue
            ff_img = green_bin.copy()
            ff_mask = np.zeros((h + 2, w + 2), np.uint8)
            cv2.floodFill(ff_img, ff_mask, seedPoint=pt, newVal=255)
            comp = (ff_mask[1:-1, 1:-1] > 0).astype(np.uint8) * 255
            picked = cv2.bitwise_or(picked, comp)

        picked_area = int(np.count_nonzero(picked))
        min_area_frac = float(os.getenv("BADC_FLOOR_ROI_PICK_MIN_AREA_FRAC", "0.01") or 0.01)
        if picked_area < int(min_area_frac * h * w):
            floor_mask = green_bin
            floor_direct_source = "green_direct_full"
        else:
            floor_mask = picked
            floor_direct_source = "green_direct_flood"

        ys_g, xs_g = np.where(floor_mask > 0)
        if xs_g.size == 0:
            fallback_floor_roi = True
            fallback_mode = "bottom_band"
            floor_y_cut = int(round(float(h) * float(fallback_y_ratio)))
            floor_mask = np.zeros((h, w), dtype=np.uint8)
            floor_mask[floor_y_cut:h, :] = 255
        else:
            floor_y_cut = int(ys_g.min())
            floor_x1_raw = int(xs_g.min())
            floor_x2_raw = int(xs_g.max())
            floor_mask, floor_y_cut, floor_top_guard_meta = _guard_floor_roi_top(
                floor_mask,
                floor_y_cut,
                source=floor_direct_source,
            )
            if bool((floor_top_guard_meta or {}).get("applied", False)):
                y_guard = int((floor_top_guard_meta or {}).get("y1_final", floor_y_cut if floor_y_cut is not None else 0))
                x_pad_frac = float(np.clip(_env_float("BADC_FLOOR_ROI_GREEN_DIRECT_X_PAD_FRAC", 0.02), 0.0, 0.30))
                x_pad = int(round(x_pad_frac * float(max(w, 1))))
                x1_keep = int(max(0, floor_x1_raw - x_pad))
                x2_keep = int(min(w - 1, floor_x2_raw + x_pad))
                keep_strip = np.zeros((h, w), dtype=np.uint8)
                keep_strip[:, x1_keep : x2_keep + 1] = 255
                rebuilt = cv2.bitwise_and(green_bin, keep_strip)
                rebuilt[:y_guard, :] = 0
                if int(np.count_nonzero(rebuilt)) > 0:
                    floor_mask = rebuilt
                    if isinstance(floor_top_guard_meta, dict):
                        floor_top_guard_meta["green_direct_rebuild_from_green"] = True
                        floor_top_guard_meta["green_direct_rebuild_x_range"] = [int(x1_keep), int(x2_keep)]
            if bool((floor_top_guard_meta or {}).get("applied", False)):
                green_mask_boost, green_adapt_meta_boost = _build_adaptive_green_mask(seed, top_boost=True)
                if int(np.count_nonzero(green_mask_boost)) > 0:
                    green_mask = green_mask_boost
                    green_bin = ((green_mask > 0).astype(np.uint8) * 255)
                    green_adapt_meta = green_adapt_meta_boost
                    floor_mask = green_bin
                    ys_gb, xs_gb = np.where(floor_mask > 0)
                    if xs_gb.size > 0:
                        floor_y_cut = int(ys_gb.min())
                        floor_x1_raw = int(xs_gb.min())
                        floor_x2_raw = int(xs_gb.max())
                        floor_mask, floor_y_cut, floor_top_guard_meta = _guard_floor_roi_top(
                            floor_mask,
                            floor_y_cut,
                            source="green_direct_boost",
                        )
                        if bool((floor_top_guard_meta or {}).get("applied", False)):
                            y_guard = int(
                                (floor_top_guard_meta or {}).get("y1_final", floor_y_cut if floor_y_cut is not None else 0)
                            )
                            x_pad_frac = float(np.clip(_env_float("BADC_FLOOR_ROI_GREEN_DIRECT_X_PAD_FRAC", 0.02), 0.0, 0.30))
                            x_pad = int(round(x_pad_frac * float(max(w, 1))))
                            x1_keep = int(max(0, floor_x1_raw - x_pad))
                            x2_keep = int(min(w - 1, floor_x2_raw + x_pad))
                            keep_strip = np.zeros((h, w), dtype=np.uint8)
                            keep_strip[:, x1_keep : x2_keep + 1] = 255
                            rebuilt = cv2.bitwise_and(green_bin, keep_strip)
                            rebuilt[:y_guard, :] = 0
                            if int(np.count_nonzero(rebuilt)) > 0:
                                floor_mask = rebuilt
                                if isinstance(floor_top_guard_meta, dict):
                                    floor_top_guard_meta["green_direct_rebuild_from_green"] = True
                                    floor_top_guard_meta["green_direct_rebuild_x_range"] = [int(x1_keep), int(x2_keep)]
            min_top_frac = float(np.clip(_env_float("BADC_FLOOR_ROI_GREEN_DIRECT_MIN_TOP_FRAC", 0.23), 0.05, 0.95))
            min_top_y = int(max(0, min(h - 1, round(min_top_frac * float(h)))))
            if floor_y_cut is not None and floor_y_cut < min_top_y:
                floor_y_cut = int(min_top_y)
                floor_mask[: int(floor_y_cut), :] = 0
                if isinstance(floor_top_guard_meta, dict):
                    floor_top_guard_meta["green_direct_min_top_applied"] = True
                    floor_top_guard_meta["green_direct_min_top_y"] = int(min_top_y)
            fallback_mode = "green_direct"

        skip_x_tighten = _env_flag("BADC_FLOOR_ROI_GREEN_DIRECT_SKIP_X_TIGHTEN", True)
        if skip_x_tighten:
            ys_t, xs_t = np.where(floor_mask > 0)
            if xs_t.size > 0:
                x0_t = int(xs_t.min())
                x1_t = int(xs_t.max())
            else:
                x0_t, x1_t = 0, w - 1
            tighten_info = {
                "floor_x_tightened": False,
                "floor_x_range": [int(x0_t), int(x1_t)],
                "floor_x_reject_reason": "green_direct_skip_tighten",
            }
        else:
            floor_mask, tighten_info = _tighten_floor_roi_xrange(
                green_mask,
                floor_mask,
                support_mask=white_support,
                court_corners=prepass_corners if prepass_gate_passed else court_corners,
                min_density=0.09,
                min_keep_ratio=0.45,
                margin_ratio=0.075,
            )
        return floor_mask, {
            "green_mask": green_mask,
            "seed_bottom_mask": seed,
            "largest_cc_mask": green_mask.copy(),
            "fallback_floor_roi": fallback_floor_roi,
            "fallback_mode": fallback_mode,
            "floor_y_cut": int(floor_y_cut) if floor_y_cut is not None else None,
            "floor_roi_top_guard": floor_top_guard_meta,
            "floor_roi_top_guard_cfg": dict(floor_roi_top_guard_cfg),
            "floor_roi_use_green_mask_direct": True,
            "floor_roi_green_direct_skip_x_tighten": bool(skip_x_tighten),
            "exg_row_plot": None,
            "floor_cc_scores": [],
            "prepass_area_ratio": prepass_area_ratio,
            "prepass_ymax_ratio": prepass_ymax_ratio,
            "prepass_gate_passed": prepass_gate_passed,
            "green_adapt_meta": green_adapt_meta,
            **tighten_info,
        }

    union_mask, union_bbox, cc_scores = _select_top_components(green_mask, white_support, top_k=2)
    if union_mask is None or union_bbox is None:
        fallback_floor_roi = True
        fallback_mode = "bottom_band"
        floor_y_cut = int(round(float(h) * float(fallback_y_ratio)))
        floor_mask = np.zeros((h, w), dtype=np.uint8)
        floor_mask[floor_y_cut:h, :] = 255
        floor_mask, tighten_info = _tighten_floor_roi_xrange(
            green_mask,
            floor_mask,
            support_mask=white_support,
            court_corners=prepass_corners if prepass_gate_passed else court_corners,
            min_density=0.09,
            min_keep_ratio=0.45,
            margin_ratio=0.075,
        )
        return floor_mask, {
            "green_mask": green_mask,
            "seed_bottom_mask": seed,
            "largest_cc_mask": largest_cc_mask,
            "fallback_floor_roi": fallback_floor_roi,
            "fallback_mode": fallback_mode,
            "floor_y_cut": int(floor_y_cut),
            "floor_roi_top_guard": floor_top_guard_meta,
            "floor_roi_top_guard_cfg": dict(floor_roi_top_guard_cfg),
            "exg_row_plot": None,
            "floor_cc_scores": cc_scores,
            "prepass_area_ratio": prepass_area_ratio,
            "prepass_ymax_ratio": prepass_ymax_ratio,
            "prepass_gate_passed": prepass_gate_passed,
            "green_adapt_meta": green_adapt_meta,
            **tighten_info,
        }
    largest_cc_mask = union_mask
    cc_area = int(np.count_nonzero(union_mask))
    cc_ratio = float(cc_area) / float(max(h * w, 1))
    if cc_ratio < float(min_cc_ratio):
        fallback_floor_roi = True
        fallback_mode = "bottom_band"
        floor_y_cut = int(round(float(h) * float(fallback_y_ratio)))
        floor_mask = np.zeros((h, w), dtype=np.uint8)
        floor_mask[floor_y_cut:h, :] = 255
    else:
        ys_cc, xs_cc = np.where(largest_cc_mask > 0)
        if xs_cc.size > 0:
            x1 = int(xs_cc.min())
            x2 = int(xs_cc.max())
            y1 = int(ys_cc.min())
            y2 = int(ys_cc.max())
            margin = int(0.03 * float(min(h, w)))
            x1 = max(0, x1 - margin)
            y1 = max(0, y1 - margin)
            x2 = min(w - 1, x2 + margin)
            y2 = min(h - 1, y2 + margin)
            floor_mask = np.zeros((h, w), dtype=np.uint8)
            floor_mask[y1 : y2 + 1, x1 : x2 + 1] = 255
            floor_y_cut = int(y1)
        else:
            floor_mask = largest_cc_mask
    if not fallback_floor_roi and fallback_mode == "cc":
        floor_mask, floor_y_cut, floor_top_guard_meta = _guard_floor_roi_top(
            floor_mask,
            floor_y_cut,
            source="cc",
        )
        if bool((floor_top_guard_meta or {}).get("applied", False)):
            green_mask, green_adapt_meta = _build_adaptive_green_mask(seed, top_boost=True)
    floor_mask, tighten_info = _tighten_floor_roi_xrange(
        green_mask,
        floor_mask,
        support_mask=white_support,
        court_corners=prepass_corners if prepass_gate_passed else court_corners,
        min_density=0.09,
        min_keep_ratio=0.45,
        margin_ratio=0.075,
    )
    return floor_mask, {
        "green_mask": green_mask,
        "seed_bottom_mask": seed,
        "largest_cc_mask": largest_cc_mask,
        "fallback_floor_roi": fallback_floor_roi,
        "fallback_mode": fallback_mode,
        "floor_y_cut": int(floor_y_cut) if floor_y_cut is not None else None,
        "floor_roi_top_guard": floor_top_guard_meta,
        "floor_roi_top_guard_cfg": dict(floor_roi_top_guard_cfg),
        "exg_row_plot": None,
        "floor_cc_scores": cc_scores,
        "prepass_area_ratio": prepass_area_ratio,
        "prepass_ymax_ratio": prepass_ymax_ratio,
        "prepass_gate_passed": prepass_gate_passed,
        "green_adapt_meta": green_adapt_meta,
        **tighten_info,
    }


def get_floor_roi_mask(
    frame_bgr: np.ndarray,
    fallback_y_ratio: float = 0.55,
) -> np.ndarray:
    floor_mask, _ = get_floor_roi_mask_debug(
        frame_bgr,
        fallback_y_ratio=fallback_y_ratio,
    )
    return floor_mask


def hough_lines(
    mask: np.ndarray,
    floor_mask: np.ndarray,
    min_length_ratio: float = 0.08,
    threshold: int = 80,
) -> list[Tuple[float, float, float, float]]:
    h, w = mask.shape[:2]
    edges = cv2.Canny(mask, 50, 150)
    min_len = max(1, int(float(w) * float(min_length_ratio)))
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180.0, threshold, minLineLength=min_len, maxLineGap=20)
    segs: list[Tuple[float, float, float, float]] = []
    if lines is None:
        return segs
    for ln in lines:
        x1, y1, x2, y2 = [float(v) for v in ln[0]]
        length = float(math.hypot(x2 - x1, y2 - y1))
        if length < float(min_len):
            continue
        mx = int(round(0.5 * (x1 + x2)))
        my = int(round(0.5 * (y1 + y2)))
        if mx < 0 or mx >= w or my < 0 or my >= h:
            continue
        if floor_mask[my, mx] == 0:
            continue
        segs.append((x1, y1, x2, y2))
    return segs


def _line_support(
    seg: Tuple[float, float, float, float],
    mask: np.ndarray,
    samples: int = 30,
) -> float:
    x1, y1, x2, y2 = seg
    h, w = mask.shape[:2]
    t = np.linspace(0.0, 1.0, int(samples), dtype=np.float32)
    xs = x1 + (x2 - x1) * t
    ys = y1 + (y2 - y1) * t
    xi = np.clip(np.rint(xs).astype(np.int32), 0, w - 1)
    yi = np.clip(np.rint(ys).astype(np.int32), 0, h - 1)
    return float(np.mean(mask[yi, xi] > 0))


def candidate_outer_pairs(
    lines: list[Tuple[float, float, float, float]],
    mask: np.ndarray,
    angle_ref: float,
    top_k: int = 6,
    max_angle_deg: float = 15.0,
    max_pairs: int = 3,
) -> list[Tuple[Tuple[Tuple[float, float, float], Tuple[float, float, float]], float]]:
    if not lines:
        return []
    max_angle = math.radians(float(max_angle_deg))
    scored = []
    for seg in lines:
        x1, y1, x2, y2 = seg
        angle = math.atan2(y2 - y1, x2 - x1) % math.pi
        if _angle_distance(angle, float(angle_ref)) > max_angle:
            continue
        length = float(math.hypot(x2 - x1, y2 - y1))
        support = _line_support(seg, mask)
        score = float(length * (0.5 + 0.5 * support))
        line = _line_from_points((x1, y1), (x2, y2))
        scored.append((score, line, angle))
    if not scored:
        return []
    scored.sort(key=lambda item: item[0], reverse=True)
    top = scored[: max(2, int(top_k))]
    pairs = []
    for i in range(len(top)):
        for j in range(i + 1, len(top)):
            si, li, ai = top[i]
            sj, lj, aj = top[j]
            if _angle_distance(ai, aj) > max_angle:
                continue
            dist = abs(li[2] - lj[2])
            pair_score = float(si + sj)
            pairs.append(((li, lj), pair_score, dist))
    pairs.sort(key=lambda item: (item[2], item[1]), reverse=True)
    return [(pair, score) for pair, score, _ in pairs[: max_pairs]]


def _x_at_y(line: Tuple[float, float, float], y: float) -> Optional[float]:
    a, b, c = line
    if abs(a) < 1e-6:
        return None
    return float(-(b * y + c) / a)


def _y_at_x(line: Tuple[float, float, float], x: float) -> Optional[float]:
    a, b, c = line
    if abs(b) < 1e-6:
        return None
    return float(-(a * x + c) / b)


def candidate_outer_pairs_vp(
    lines: list[Tuple[float, float, float, float]],
    mask: np.ndarray,
    floor_bbox: Tuple[int, int, int, int],
    frame_shape: Tuple[int, int],
    top_k: int = 8,
    max_pairs: int = 3,
) -> list[Tuple[Tuple[Tuple[float, float, float], Tuple[float, float, float]], float]]:
    if not lines:
        return []
    scored = []
    for seg in lines:
        x1, y1, x2, y2 = seg
        length = float(math.hypot(x2 - x1, y2 - y1))
        support = _line_support(seg, mask)
        score = float(length * (0.5 + 0.5 * support))
        line = _line_from_points((x1, y1), (x2, y2))
        scored.append((score, line))
    scored.sort(key=lambda item: item[0], reverse=True)
    top = scored[: max(2, int(top_k))]
    x0, y0, x1b, y1b = floor_bbox
    h, w = frame_shape
    y_ref = float(y0 + 0.75 * max(1.0, (y1b - y0)))
    x_ref = float(x0 + 0.50 * max(1.0, (x1b - x0)))
    pairs = []
    for i in range(len(top)):
        for j in range(i + 1, len(top)):
            si, li = top[i]
            sj, lj = top[j]
            vp = _intersect_lines(li, lj)
            if vp is None or not np.all(np.isfinite(vp)):
                continue
            if vp[1] > float(y0 + 0.20 * max(1.0, (y1b - y0))):
                continue
            if vp[0] < -0.5 * float(w) or vp[0] > 1.5 * float(w):
                continue
            sep = None
            if abs(li[0]) > abs(li[1]) and abs(lj[0]) > abs(lj[1]):
                xi = _x_at_y(li, y_ref)
                xj = _x_at_y(lj, y_ref)
                if xi is not None and xj is not None:
                    sep = abs(xi - xj)
            else:
                yi = _y_at_x(li, x_ref)
                yj = _y_at_x(lj, x_ref)
                if yi is not None and yj is not None:
                    sep = abs(yi - yj)
            if sep is None:
                continue

            # Prefer geometrically plausible "outer" pairs:
            # - reject pairs that are too close (often inner+outer parallel lines on the same side)
            # - make separation dominate; support acts as a weak tie-breaker
            if abs(li[0]) > abs(li[1]) and abs(lj[0]) > abs(lj[1]):
                sep_scale = float(max(1.0, (x1b - x0)))  # vertical-ish -> x separation
            else:
                sep_scale = float(max(1.0, (y1b - y0)))  # horizontal-ish -> y separation
            sep_norm = float(sep) / sep_scale
            if sep_norm < 0.08:
                continue

            pair_score = float(10.0 * sep_norm + 0.05 * math.sqrt(max(si, 0.0) * max(sj, 0.0)) / sep_scale)
            pairs.append(((li, lj), pair_score))
    pairs.sort(key=lambda item: item[1], reverse=True)
    return pairs[: max_pairs]


def _intersections_from_pairs(
    pair_a: Tuple[Tuple[float, float, float], Tuple[float, float, float]],
    pair_b: Tuple[Tuple[float, float, float], Tuple[float, float, float]],
) -> Optional[np.ndarray]:
    a1, a2 = pair_a
    b1, b2 = pair_b
    p00 = _intersect_lines(a1, b1)
    p01 = _intersect_lines(a1, b2)
    p10 = _intersect_lines(a2, b1)
    p11 = _intersect_lines(a2, b2)
    if p00 is None or p01 is None or p10 is None or p11 is None:
        return None
    return np.stack([p00, p01, p10, p11], axis=0)


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


def _point_in_bbox(pt_xy: Tuple[float, float], bbox: Tuple[int, int, int, int]) -> bool:
    x1, y1, x2, y2 = bbox
    x, y = pt_xy
    return (float(x1) <= float(x) <= float(x2)) and (float(y1) <= float(y) <= float(y2))


def _line_from_points(p1: Tuple[float, float], p2: Tuple[float, float]) -> Tuple[float, float, float]:
    x1, y1 = p1
    x2, y2 = p2
    a = float(y1 - y2)
    b = float(x2 - x1)
    c = float(x1 * y2 - x2 * y1)
    norm = math.hypot(a, b)
    if norm > 1e-6:
        a /= norm
        b /= norm
        c /= norm
    return a, b, c


def _canonicalize_line_abc(a: float, b: float, c: float) -> Tuple[float, float, float]:
    """
    Canonicalize ax+by+c=0 (with sqrt(a^2+b^2)=1) so that the normal (a,b) has a consistent sign.
    This avoids averaging (a,b,c) with (-a,-b,-c) which would cancel out and drift.
    Rule: force (a>0) or (a==0 and b>0).
    """
    if a < 0.0 or (abs(a) < 1e-12 and b < 0.0):
        return -a, -b, -c
    return a, b, c


def _angle_distance(a: float, b: float) -> float:
    """Angular distance on [0, pi)."""
    d = abs(float(a) - float(b)) % math.pi
    return float(min(d, math.pi - d))


def _seg_theta_undirected(x1: float, y1: float, x2: float, y2: float) -> float:
    theta = float(math.atan2(y2 - y1, x2 - x1)) % math.pi
    return theta


def _median_angle_segments(
    segments: list[Tuple[float, float, float, float]],
) -> Optional[float]:
    """Robust orientation estimate (weighted median) in radians, modulo pi."""
    if not segments:
        return None
    angles = []
    weights = []
    for x1, y1, x2, y2 in segments:
        ang = float(np.mod(math.atan2(y2 - y1, x2 - x1), math.pi))
        w = float(math.hypot(x2 - x1, y2 - y1))
        angles.append(ang)
        weights.append(max(1e-6, w))
    order = np.argsort(angles)
    ang_sorted = np.array([angles[i] for i in order], dtype=np.float32)
    w_sorted = np.array([weights[i] for i in order], dtype=np.float32)
    cdf = np.cumsum(w_sorted)
    cutoff = 0.5 * float(cdf[-1])
    j = int(np.searchsorted(cdf, cutoff))
    j = max(0, min(j, len(ang_sorted) - 1))
    return float(ang_sorted[j])


def _two_angle_peaks(
    thetas: np.ndarray,
    weights: Optional[np.ndarray] = None,
    bins: int = 90,
    min_sep_deg: float = 55.0,
) -> Optional[Tuple[float, float]]:
    """Find two dominant undirected angle peaks on [0, pi).

    Returns (t0, t1) in radians, or None if a stable pair can't be found.
    """
    if thetas is None or len(thetas) < 8:
        return None
    thetas = np.asarray(thetas, dtype=np.float64) % math.pi
    if weights is None:
        weights = np.ones_like(thetas, dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape != thetas.shape:
            weights = np.ones_like(thetas, dtype=np.float64)

    bins = int(max(18, bins))
    # Histogram on [0, pi)
    hist, edges = np.histogram(thetas, bins=bins, range=(0.0, math.pi), weights=weights)
    if hist.max() <= 0:
        return None
    i0 = int(np.argmax(hist))
    # Peak center
    t0 = float(0.5 * (edges[i0] + edges[i0 + 1]))

    # Second peak: best weighted count sufficiently far from t0
    min_sep = float(min_sep_deg) * math.pi / 180.0
    best_i1 = None
    best_v1 = -1.0
    for i in range(len(hist)):
        t = float(0.5 * (edges[i] + edges[i + 1]))
        if _angle_distance(t, t0) < min_sep:
            continue
        v = float(hist[i])
        if v > best_v1:
            best_v1 = v
            best_i1 = i
    if best_i1 is None or best_v1 <= 0:
        return None
    t1 = float(0.5 * (edges[best_i1] + edges[best_i1 + 1]))

    # Prefer near-orthogonal pairs if possible: adjust by scanning around t1
    # (helps when there are multiple close secondary peaks).
    target = (t0 + 0.5 * math.pi) % math.pi
    best_i1b = best_i1
    best_cost = 1e9
    for i in range(len(hist)):
        if hist[i] <= 0:
            continue
        t = float(0.5 * (edges[i] + edges[i + 1]))
        if _angle_distance(t, t0) < min_sep:
            continue
        # cost = distance to orthogonal target minus small reward for count
        cost = float(_angle_distance(t, target)) - 0.002 * float(hist[i])
        if cost < best_cost:
            best_cost = cost
            best_i1b = i
    t1 = float(0.5 * (edges[best_i1b] + edges[best_i1b + 1]))
    return (t0, t1)


def _resplit_segments_by_angle_peaks(
    seg_a: list[Tuple[float, float, float, float]],
    score_a: np.ndarray,
    seg_b: list[Tuple[float, float, float, float]],
    score_b: np.ndarray,
    min_keep_each: int = 8,
    min_sep_deg: float = 55.0,
) -> Tuple[list[Tuple[float, float, float, float]], np.ndarray, list[Tuple[float, float, float, float]], np.ndarray, Dict[str, Any]]:
    """Fix A/B bucket role mix-ups by re-clustering segments using 2 dominant angle peaks."""
    meta: Dict[str, Any] = {"used": False}
    seg_all = list(seg_a) + list(seg_b)
    if not seg_all:
        return seg_a, score_a, seg_b, score_b, meta
    scores_all = np.concatenate([np.asarray(score_a, dtype=np.float32), np.asarray(score_b, dtype=np.float32)], axis=0)
    thetas = []
    for (x1, y1, x2, y2) in seg_all:
        thetas.append(_seg_theta_undirected(float(x1), float(y1), float(x2), float(y2)))
    thetas = np.asarray(thetas, dtype=np.float64)
    peaks = _two_angle_peaks(thetas, weights=scores_all.astype(np.float64), bins=90, min_sep_deg=min_sep_deg)
    if peaks is None:
        return seg_a, score_a, seg_b, score_b, meta
    t0, t1 = peaks
    d0 = np.array([_angle_distance(float(t), float(t0)) for t in thetas], dtype=np.float32)
    d1 = np.array([_angle_distance(float(t), float(t1)) for t in thetas], dtype=np.float32)
    assign0 = d0 <= d1
    seg0 = [seg_all[i] for i in range(len(seg_all)) if assign0[i]]
    seg1 = [seg_all[i] for i in range(len(seg_all)) if not assign0[i]]
    sc0 = scores_all[assign0]
    sc1 = scores_all[~assign0]
    if len(seg0) < int(min_keep_each) or len(seg1) < int(min_keep_each):
        return seg_a, score_a, seg_b, score_b, meta

    # Decide which becomes A vs B: keep closer to original means to reduce jitter.
    # Compute mean theta for original A and original B.
    def _mean_theta(segs: list[Tuple[float, float, float, float]]) -> float:
        if not segs:
            return 0.0
        th = np.array([_seg_theta_undirected(float(x1), float(y1), float(x2), float(y2)) for (x1, y1, x2, y2) in segs], dtype=np.float64)
        # circular mean on [0, pi)
        return float(math.atan2(np.mean(np.sin(2.0 * th)), np.mean(np.cos(2.0 * th))) / 2.0) % math.pi

    mean_a = _mean_theta(seg_a)
    mean_b = _mean_theta(seg_b)
    mean0 = _mean_theta(seg0)
    mean1 = _mean_theta(seg1)

    # Two possible mappings: (0->A,1->B) vs swapped
    cost_keep = _angle_distance(mean0, mean_a) + _angle_distance(mean1, mean_b)
    cost_swap = _angle_distance(mean0, mean_b) + _angle_distance(mean1, mean_a)
    if cost_swap < cost_keep:
        seg_a2, score_a2 = seg1, sc1
        seg_b2, score_b2 = seg0, sc0
        meta["swapped"] = True
    else:
        seg_a2, score_a2 = seg0, sc0
        seg_b2, score_b2 = seg1, sc1
        meta["swapped"] = False

    meta.update(
        {
            "used": True,
            "peaks_deg": [float(t0 * 180.0 / math.pi), float(t1 * 180.0 / math.pi)],
            "mean_before_deg": [float(mean_a * 180.0 / math.pi), float(mean_b * 180.0 / math.pi)],
            "mean_after_deg": [float(_mean_theta(seg_a2) * 180.0 / math.pi), float(_mean_theta(seg_b2) * 180.0 / math.pi)],
            "n_before": [int(len(seg_a)), int(len(seg_b))],
            "n_after": [int(len(seg_a2)), int(len(seg_b2))],
        }
    )
    return seg_a2, score_a2, seg_b2, score_b2, meta


def _mean_theta_segments(segs: list[tuple[int, int, int, int]]) -> Optional[float]:
    if not segs:
        return None
    angles = np.array([_seg_theta_undirected(x1, y1, x2, y2) for (x1, y1, x2, y2) in segs], dtype=np.float32)
    weights = np.array([math.hypot(x2 - x1, y2 - y1) for (x1, y1, x2, y2) in segs], dtype=np.float32)
    s = float(np.sum(weights * np.sin(2.0 * angles)))
    c = float(np.sum(weights * np.cos(2.0 * angles)))
    if abs(s) < 1e-6 and abs(c) < 1e-6:
        return None
    ang = 0.5 * math.atan2(s, c)
    if ang < 0:
        ang += math.pi
    return float(ang)


def _filter_segments_by_angle_keep_min(
    segs: list[tuple[int, int, int, int]],
    ang_tol_deg: float,
    min_keep: int,
) -> tuple[list[tuple[int, int, int, int]], Dict[str, Any]]:
    """Filter segments by closeness to the mean undirected angle.

    Keeps at least `min_keep` by selecting the closest ones if the strict filter would drop too much.
    """
    if not segs:
        return [], {"mean_theta": None, "kept": 0, "dropped": 0, "forced_keep": 0}
    mean_theta = _mean_theta_segments(segs)
    if mean_theta is None:
        return segs, {"mean_theta": None, "kept": len(segs), "dropped": 0, "forced_keep": 0}

    tol = math.radians(float(ang_tol_deg))
    scored = []
    for s in segs:
        x1, y1, x2, y2 = s
        th = _seg_theta_undirected(x1, y1, x2, y2)
        d = _angle_distance(th, float(mean_theta))
        scored.append((d, s))
    scored.sort(key=lambda t: t[0])

    strict = [s for (d, s) in scored if d <= tol]
    forced_keep = 0
    if len(strict) < int(min_keep):
        forced_keep = int(min_keep) - len(strict)
        strict = [s for (_, s) in scored[: int(min_keep)]]
    kept = len(strict)
    dropped = max(0, len(segs) - kept)
    return strict, {
        "mean_theta": float(mean_theta),
        "kept": int(kept),
        "dropped": int(dropped),
        "forced_keep": int(forced_keep),
        "ang_tol_deg": float(ang_tol_deg),
        "min_keep": int(min_keep),
    }


def _merge_collinear_segments_by_rho(
    segs: list[tuple[int, int, int, int]],
    rho_bin_px: float,
    gap_px: float,
    min_len_px: float,
) -> tuple[list[tuple[int, int, int, int]], Dict[str, Any]]:
    """Merge collinear (nearly parallel) segments into longer segments, per-rho bin.

    Strategy inspired by common lane/court line post-processing:
    1) Estimate dominant direction (mean undirected theta)
    2) Compute each segment's signed distance (rho) to that direction's normal
    3) Bin by rho (so different court markings don't merge)
    4) Within each bin, project endpoints onto the direction axis and merge 1D intervals with small gaps
    """
    if not segs:
        return [], {"bins": 0, "in": 0, "out": 0}
    mean_theta = _mean_theta_segments(segs)
    if mean_theta is None:
        return segs, {"bins": 0, "in": len(segs), "out": len(segs), "mean_theta": None}

    u = np.array([math.cos(mean_theta), math.sin(mean_theta)], dtype=np.float32)  # direction
    n = np.array([-u[1], u[0]], dtype=np.float32)  # normal

    # bin by rho
    bins: Dict[int, list[tuple[float, float]]] = {}
    for (x1, y1, x2, y2) in segs:
        p1 = np.array([float(x1), float(y1)], dtype=np.float32)
        p2 = np.array([float(x2), float(y2)], dtype=np.float32)
        mid = 0.5 * (p1 + p2)
        rho = float(np.dot(n, mid))
        b = int(round(rho / float(rho_bin_px)))
        t1 = float(np.dot(u, p1))
        t2 = float(np.dot(u, p2))
        tmin, tmax = (t1, t2) if t1 <= t2 else (t2, t1)
        bins.setdefault(b, []).append((tmin, tmax))

    merged: list[tuple[int, int, int, int]] = []
    out_bins = 0
    total_intervals = 0
    for b, intervals in bins.items():
        if not intervals:
            continue
        out_bins += 1
        total_intervals += len(intervals)
        intervals.sort(key=lambda t: t[0])
        cur0, cur1 = intervals[0]
        for t0, t1 in intervals[1:]:
            if t0 <= cur1 + float(gap_px):
                cur1 = max(cur1, t1)
            else:
                if (cur1 - cur0) >= float(min_len_px):
                    rho = float(b) * float(rho_bin_px)
                    pA = u * cur0 + n * rho
                    pB = u * cur1 + n * rho
                    merged.append((int(round(pA[0])), int(round(pA[1])), int(round(pB[0])), int(round(pB[1]))))
                cur0, cur1 = t0, t1
        if (cur1 - cur0) >= float(min_len_px):
            rho = float(b) * float(rho_bin_px)
            pA = u * cur0 + n * rho
            pB = u * cur1 + n * rho
            merged.append((int(round(pA[0])), int(round(pA[1])), int(round(pB[0])), int(round(pB[1]))))

    return merged, {
        "mean_theta": float(mean_theta),
        "bins": int(len(bins)),
        "nonempty_bins": int(out_bins),
        "intervals_in": int(total_intervals),
        "segs_in": int(len(segs)),
        "segs_out": int(len(merged)),
        "rho_bin_px": float(rho_bin_px),
        "gap_px": float(gap_px),
        "min_len_px": float(min_len_px),
    }


def _mean_angle(lines: list[LineSeg]) -> Optional[float]:
    if not lines:
        return None
    angles = np.array([ln.theta for ln in lines], dtype=np.float32)
    weights = np.array([ln.weight for ln in lines], dtype=np.float32)
    s = float(np.sum(weights * np.sin(2.0 * angles)))
    c = float(np.sum(weights * np.cos(2.0 * angles)))
    if abs(s) < 1e-6 and abs(c) < 1e-6:
        return None
    ang = 0.5 * math.atan2(s, c)
    if ang < 0:
        ang += math.pi
    return float(ang)


def _detect_lsd_lines(
    edge_mask: np.ndarray,
    support_mask: np.ndarray,
    floor_bbox: Tuple[int, int, int, int],
    min_length: float = 20.0,
    support_thresh: float = 0.08,
    support_thresh_long: float = 0.03,
    long_length: float = 80.0,
    support_samples: int = 30,
) -> Tuple[list[LineSeg], int]:
    edges = cv2.Canny(edge_mask, 50, 150)
    lsd = cv2.createLineSegmentDetector(0)
    lines = lsd.detect(edges)[0]
    segs: list[LineSeg] = []
    num_raw = 0
    if lines is None:
        return segs, num_raw
    num_raw = int(len(lines))
    h, w = support_mask.shape[:2]
    for ln in lines:
        x1, y1, x2, y2 = [float(v) for v in ln[0]]
        length = float(math.hypot(x2 - x1, y2 - y1))
        if length < float(min_length):
            continue
        mx = 0.5 * (x1 + x2)
        my = 0.5 * (y1 + y2)
        if floor_bbox is not None and not _point_in_bbox((mx, my), floor_bbox):
            continue
        t = np.linspace(0.0, 1.0, int(support_samples), dtype=np.float32)
        xs = x1 + (x2 - x1) * t
        ys = y1 + (y2 - y1) * t
        xi = np.clip(np.rint(xs).astype(np.int32), 0, w - 1)
        yi = np.clip(np.rint(ys).astype(np.int32), 0, h - 1)
        support = float(np.mean(support_mask[yi, xi] > 0))
        keep = support > float(support_thresh) or (
            length > float(long_length) and support > float(support_thresh_long)
        )
        if not keep:
            continue
        theta = math.atan2(y2 - y1, x2 - x1)
        theta = float(theta % math.pi)
        line = _line_from_points((x1, y1), (x2, y2))
        weight = float(length * (0.5 + 0.5 * support))
        segs.append(
            LineSeg(
                p1=np.array([x1, y1], dtype=np.float32),
                p2=np.array([x2, y2], dtype=np.float32),
                theta=theta,
                length=length,
                support=support,
                weight=weight,
                line=line,
            )
        )
    return segs, num_raw


def _cluster_lines_by_angle(
    lines: list[LineSeg],
    min_ratio: float = 0.25,
    fallback_angle_deg: float = 15.0,
) -> Tuple[list[LineSeg], list[LineSeg], Dict[str, Any]]:
    if not lines:
        return [], [], {"mean_angle_a": None, "mean_angle_b": None, "cluster_ratio": 0.0}
    angles = np.array([ln.theta for ln in lines], dtype=np.float32)
    weights = np.array([ln.weight for ln in lines], dtype=np.float32)
    bins = 36
    hist, edges = np.histogram(angles, bins=bins, range=(0.0, math.pi), weights=weights)
    idx = np.argsort(hist)[::-1]
    idx1 = int(idx[0])
    idx2 = int(idx[1]) if idx.size > 1 and hist[idx[1]] > 0 else int(idx[0])
    mu1 = float((edges[idx1] + edges[idx1 + 1]) * 0.5)
    if idx2 == idx1:
        mu2 = float((mu1 + math.pi * 0.5) % math.pi)
    else:
        mu2 = float((edges[idx2] + edges[idx2 + 1]) * 0.5)
    cluster_a: list[LineSeg] = []
    cluster_b: list[LineSeg] = []
    for ln in lines:
        if _angle_distance(ln.theta, mu1) <= _angle_distance(ln.theta, mu2):
            cluster_a.append(ln)
        else:
            cluster_b.append(ln)
    sum_a = float(np.sum([ln.weight for ln in cluster_a])) if cluster_a else 0.0
    sum_b = float(np.sum([ln.weight for ln in cluster_b])) if cluster_b else 0.0
    denom = max(sum_a, sum_b, 1e-6)
    ratio = float(min(sum_a, sum_b) / denom)

    if ratio < float(min_ratio):
        main = cluster_a if sum_a >= sum_b else cluster_b
        main_angle = _mean_angle(main)
        if main_angle is None:
            main_angle = mu1
        perp = float((main_angle + math.pi * 0.5) % math.pi)
        alt = [ln for ln in lines if _angle_distance(ln.theta, perp) < math.radians(float(fallback_angle_deg))]
        if not alt:
            alt = [ln for ln in lines if _angle_distance(ln.theta, perp) < math.radians(25.0)]
        if alt:
            if sum_a >= sum_b:
                cluster_b = alt
            else:
                cluster_a = alt
            sum_a = float(np.sum([ln.length for ln in cluster_a])) if cluster_a else 0.0
            sum_b = float(np.sum([ln.length for ln in cluster_b])) if cluster_b else 0.0
            denom = max(sum_a, sum_b, 1e-6)
            ratio = float(min(sum_a, sum_b) / denom)

    return cluster_a, cluster_b, {
        "mean_angle_a": _mean_angle(cluster_a),
        "mean_angle_b": _mean_angle(cluster_b),
        "sum_len_a": sum_a,
        "sum_len_b": sum_b,
        "cluster_ratio": ratio,
    }


def _assign_cluster_roles(
    cluster_a: list[LineSeg],
    cluster_b: list[LineSeg],
) -> Tuple[Optional[list[LineSeg]], Optional[list[LineSeg]], Dict[str, Any]]:
    mean_a = _mean_angle(cluster_a)
    mean_b = _mean_angle(cluster_b)
    if mean_a is None or mean_b is None:
        return None, None, {"mean_angle_a": mean_a, "mean_angle_b": mean_b}
    score_a = abs(float(mean_a) - math.pi * 0.5)
    score_b = abs(float(mean_b) - math.pi * 0.5)
    if score_a < score_b:
        vert = cluster_a
        horiz = cluster_b
    else:
        vert = cluster_b
        horiz = cluster_a
    return horiz, vert, {"mean_angle_a": mean_a, "mean_angle_b": mean_b}


def _line_signed_distance(line: Tuple[float, float, float], center: Tuple[float, float]) -> float:
    a, b, c = line
    x, y = center
    return float(a * x + b * y + c)


def _quad_inside_ratio(quad_xy: np.ndarray, bbox: Tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = bbox
    inside = 0
    for x, y in quad_xy:
        if float(x1) <= float(x) <= float(x2) and float(y1) <= float(y) <= float(y2):
            inside += 1
    return float(inside) / 4.0


def _points_inside_ratio(points_xy: np.ndarray, bbox: Tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = bbox
    inside = 0
    for x, y in points_xy:
        if float(x1) <= float(x) <= float(x2) and float(y1) <= float(y) <= float(y2):
            inside += 1
    return float(inside) / float(max(points_xy.shape[0], 1))


def _corners_inside_floor_roi(
    corners_xy: np.ndarray,
    floor_bbox_xyxy: Tuple[int, int, int, int],
    margin: int = 10,
    margin_x: Optional[int] = None,
    margin_y: Optional[int] = None,
) -> bool:
    x0, y0, x1, y1 = floor_bbox_xyxy
    xs = corners_xy[:, 0]
    ys = corners_xy[:, 1]
    if margin_x is None:
        margin_x = margin
    if margin_y is None:
        margin_y = margin
    return (
        float(xs.min()) >= float(x0 - margin_x)
        and float(xs.max()) <= float(x1 + margin_x)
        and float(ys.min()) >= float(y0 - margin_y)
        and float(ys.max()) <= float(y1 + margin_y)
    )


def _bottom_corners_in_roi(
    quad_xy: np.ndarray,
    roi_mask: Optional[np.ndarray] = None,
    roi_rect: Optional[Tuple[int, int, int, int]] = None,
    margin: int = 6,
) -> bool:
    pts = np.asarray(quad_xy, dtype=np.float32).reshape(-1, 2)
    idx = np.argsort(pts[:, 1])[::-1][:2]
    bottoms = pts[idx]
    if roi_mask is not None:
        h, w = roi_mask.shape[:2]
        for x, y in bottoms:
            xi = int(np.clip(round(float(x)), 0, w - 1))
            yi = int(np.clip(round(float(y)), 0, h - 1))
            if roi_mask[yi, xi] == 0:
                return False
        return True
    if roi_rect is not None:
        x0, y0, x1, y1 = roi_rect
        x0 -= margin
        y0 -= margin
        x1 += margin
        y1 += margin
        for x, y in bottoms:
            if not (float(x0) <= float(x) <= float(x1) and float(y0) <= float(y) <= float(y1)):
                return False
        return True
    return True


def _quad_height_ratio(quad_xy: np.ndarray, h: int) -> float:
    pts = np.asarray(quad_xy, dtype=np.float32).reshape(-1, 2)
    height = float(np.max(pts[:, 1]) - np.min(pts[:, 1]))
    return height / float(max(h, 1))


def _filter_hough_segments_for_scoring(
    segments: list[Tuple[float, float, float, float]],
    roi_rect: Tuple[int, int, int, int],
    upper_band: Tuple[float, float] = (0.05, 0.40),
    near_horiz_deg: float = 10.0,
    min_len: float = 50.0,
) -> list[Tuple[float, float, float, float]]:
    if not segments:
        return []
    x0, y0, x1, y1 = roi_rect
    roi_h = max(1.0, float(y1 - y0))
    y_a = float(y0) + float(upper_band[0]) * roi_h
    y_b = float(y0) + float(upper_band[1]) * roi_h
    th = math.tan(math.radians(float(near_horiz_deg)))
    out = []
    for x1s, y1s, x2s, y2s in segments:
        dx = float(x2s - x1s)
        dy = float(y2s - y1s)
        seg_len = math.hypot(dx, dy)
        if seg_len < float(min_len):
            continue
        y_mid = 0.5 * (float(y1s) + float(y2s))
        near_horiz = abs(dx) > 1e-6 and abs(dy / dx) < th
        in_upper = y_a <= y_mid <= y_b
        if in_upper and near_horiz:
            continue
        out.append((x1s, y1s, x2s, y2s))
    return out


def _suppress_bright_band_on_edges(
    edges: np.ndarray,
    floor_y0: int,
    search_height_frac: float = 0.25,
    row_white_ratio_thr: float = 0.22,
    min_run: int = 12,
    pad: int = 6,
) -> Tuple[np.ndarray, Optional[Tuple[int, int]]]:
    e = edges.copy()
    h, w = e.shape[:2]
    y1 = max(0, int(floor_y0))
    y2 = min(h, int(floor_y0 + (h - floor_y0) * float(search_height_frac)))
    if y2 <= y1 + 5:
        return e, None
    roi = e[y1:y2, :]
    row_ratio = (roi > 0).mean(axis=1)
    if row_ratio.size > 0:
        perc = float(np.percentile(row_ratio, 90))
        thr = max(float(row_white_ratio_thr), perc)
    else:
        thr = float(row_white_ratio_thr)
    bad = row_ratio >= float(thr)
    best_len = 0
    best_a = None
    best_b = None
    a = None
    for i, v in enumerate(bad):
        if v and a is None:
            a = i
        if (not v or i == len(bad) - 1) and a is not None:
            b = i if not v else i + 1
            length = b - a
            if length > best_len:
                best_len = length
                best_a, best_b = a, b
            a = None
    if best_len >= int(min_run):
        yy1 = max(y1, y1 + int(best_a) - int(pad))
        yy2 = min(y2, y1 + int(best_b) + int(pad))
        e[yy1:yy2, :] = 0
        return e, (yy1, yy2)
    return e, None


def _segment_y_weight(seg_mid_y: float, floor_y0: int, h: int, gamma: float = 2.5) -> float:
    t = (float(seg_mid_y) - float(floor_y0)) / float(max(1.0, h - floor_y0))
    t = float(np.clip(t, 0.0, 1.0))
    return float(t**float(gamma))


def _edge_aligns_vp(p1: np.ndarray, p2: np.ndarray, vp: Optional[np.ndarray], max_angle_deg: float = 15.0) -> bool:
    if vp is None or not isinstance(vp, np.ndarray):
        return True
    v_edge = p2 - p1
    v_edge_norm = float(np.linalg.norm(v_edge))
    if v_edge_norm < 1e-6:
        return False
    mid = 0.5 * (p1 + p2)
    v_vp = vp.astype(np.float32) - mid
    v_vp_norm = float(np.linalg.norm(v_vp))
    if v_vp_norm < 1e-6:
        return False
    cosang = abs(float(np.dot(v_edge, v_vp)) / float(v_edge_norm * v_vp_norm))
    cosang = float(np.clip(cosang, -1.0, 1.0))
    angle = math.degrees(math.acos(cosang))
    return angle <= float(max_angle_deg)


def _draw_lsd_clusters(
    frame_bgr: np.ndarray,
    lines_a: list[LineSeg],
    lines_b: list[LineSeg],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if frame_bgr is None:
        return None, None
    img_a = frame_bgr.copy()
    img_b = frame_bgr.copy()
    for ln in lines_a:
        p1 = (int(round(float(ln.p1[0]))), int(round(float(ln.p1[1]))))
        p2 = (int(round(float(ln.p2[0]))), int(round(float(ln.p2[1]))))
        cv2.line(img_a, p1, p2, (0, 0, 255), 2, lineType=cv2.LINE_AA)
    for ln in lines_b:
        p1 = (int(round(float(ln.p1[0]))), int(round(float(ln.p1[1]))))
        p2 = (int(round(float(ln.p2[0]))), int(round(float(ln.p2[1]))))
        cv2.line(img_b, p1, p2, (0, 255, 255), 2, lineType=cv2.LINE_AA)
    return img_a, img_b


def _draw_lsd_segment_lists(
    frame_bgr: np.ndarray,
    lines_a: Optional[list],
    lines_b: Optional[list],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if frame_bgr is None:
        return None, None
    img_a = frame_bgr.copy()
    img_b = frame_bgr.copy()
    if isinstance(lines_a, list):
        for seg in lines_a:
            if not isinstance(seg, (list, tuple)) or len(seg) != 4:
                continue
            x1, y1, x2, y2 = [float(v) for v in seg]
            cv2.line(
                img_a,
                (int(round(x1)), int(round(y1))),
                (int(round(x2)), int(round(y2))),
                (0, 0, 255),
                2,
                lineType=cv2.LINE_AA,
            )
    if isinstance(lines_b, list):
        for seg in lines_b:
            if not isinstance(seg, (list, tuple)) or len(seg) != 4:
                continue
            x1, y1, x2, y2 = [float(v) for v in seg]
            cv2.line(
                img_b,
                (int(round(x1)), int(round(y1))),
                (int(round(x2)), int(round(y2))),
                (0, 255, 255),
                2,
                lineType=cv2.LINE_AA,
            )
    return img_a, img_b


def _detect_blob_mask(
    mask_u8: np.ndarray,
    roi_mask: Optional[np.ndarray] = None,
    dt_thr: float = 5.5,
    min_core_area: int = 800,
    dilate_ksize: int = 17,
) -> np.ndarray:
    if mask_u8.ndim == 3:
        mask_u8 = cv2.cvtColor(mask_u8, cv2.COLOR_BGR2GRAY)
    bw = (mask_u8 > 127).astype(np.uint8)
    if roi_mask is not None:
        bw = bw & (roi_mask > 0).astype(np.uint8)
    dt = cv2.distanceTransform(bw, cv2.DIST_L2, 3)
    core = (dt >= float(dt_thr)).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(core, 8)
    core_f = np.zeros_like(core)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= int(min_core_area):
            core_f[labels == i] = 1
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(dilate_ksize), int(dilate_ksize)))
    blob = cv2.dilate(core_f, k, iterations=1)
    blob = (blob & bw).astype(np.uint8) * 255
    if roi_mask is not None:
        blob = cv2.bitwise_and(blob, roi_mask)
    return blob


def _keep_thin_structures_in_blob(
    mask_u8: np.ndarray,
    blob_mask_u8: np.ndarray,
) -> np.ndarray:
    if mask_u8.ndim == 3:
        mask_u8 = cv2.cvtColor(mask_u8, cv2.COLOR_BGR2GRAY)
    bw = (mask_u8 > 127).astype(np.uint8) * 255
    in_blob = cv2.bitwise_and(bw, blob_mask_u8)
    keep = np.zeros_like(in_blob)
    for ang in [0, 45, 90, 135]:
        k = _line_kernel(length=31, thickness=1, angle_deg=ang)
        opened = cv2.morphologyEx(in_blob, cv2.MORPH_OPEN, k)
        keep = cv2.bitwise_or(keep, opened)
    keep = cv2.dilate(keep, np.ones((3, 3), np.uint8), iterations=1)
    return keep


def _line_kernel(length: int, thickness: int, angle_deg: float) -> np.ndarray:
    k = np.zeros((length, length), np.uint8)
    c = length // 2
    rad = np.deg2rad(angle_deg)
    dx = int(round(np.cos(rad) * (length // 2 - 1)))
    dy = int(round(np.sin(rad) * (length // 2 - 1)))
    cv2.line(k, (c - dx, c - dy), (c + dx, c + dy), 1, thickness=thickness)
    return k


def _line_like_mask(
    mask_u8: np.ndarray,
    *,
    lengths: Tuple[int, ...] = (9, 15),
    thickness: int = 1,
    angles: Tuple[int, ...] = (0, 45, 90, 135),
) -> np.ndarray:
    if mask_u8.ndim == 3:
        mask_u8 = cv2.cvtColor(mask_u8, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(mask_u8, 127, 255, cv2.THRESH_BINARY)
    out = np.zeros_like(bw)
    for length in lengths:
        for ang in angles:
            k = _line_kernel(int(length), int(thickness), float(ang))
            opened = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k)
            out = cv2.bitwise_or(out, opened)
    return out


def _remove_small_speckles(
    mask_u8: np.ndarray,
    *,
    min_area: int = 8,
    min_aspect: float = 3.0,
    min_long: int = 10,
) -> np.ndarray:
    if mask_u8.ndim == 3:
        mask_u8 = cv2.cvtColor(mask_u8, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(mask_u8, 127, 255, cv2.THRESH_BINARY)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    keep = np.zeros_like(bw)
    for idx in range(1, num):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        x, y, bw_cc, bh_cc, _ = stats[idx]
        bw_cc = max(1, int(bw_cc))
        bh_cc = max(1, int(bh_cc))
        aspect = float(max(bw_cc, bh_cc)) / float(max(1, min(bw_cc, bh_cc)))
        long_side = max(bw_cc, bh_cc)
        if area < int(min_area) and aspect < float(min_aspect) and long_side < int(min_long):
            continue
        keep[labels == idx] = 255
    return keep


def _filter_postblob_non_line(
    mask_u8: np.ndarray,
    *,
    line_lengths: Tuple[int, ...] = (15, 21, 31),
    thickness_thr: float = 3.5,
    speckle_area: int = 8,
    min_aspect: float = 3.0,
    min_long: int = 10,
    density_win: int = 15,
    dense_soft: float = 0.35,
    dense_hard: float = 0.60,
    line_density_max: float = 0.35,
) -> np.ndarray:
    if mask_u8.ndim == 3:
        mask_u8 = cv2.cvtColor(mask_u8, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(mask_u8, 127, 255, cv2.THRESH_BINARY)
    if int(np.count_nonzero(bw)) == 0:
        return bw
    line_like = _line_like_mask(bw, lengths=line_lengths, thickness=1) > 0
    dt = cv2.distanceTransform((bw > 0).astype(np.uint8), cv2.DIST_L2, 3)
    dt_dil = cv2.dilate(dt, np.ones((3, 3), np.uint8))
    ridge = (dt >= 1.0) & (dt >= (dt_dil - 0.01))
    line_keep = line_like & ridge
    thick = dt > float(thickness_thr)
    filtered = bw.copy()
    filtered[thick] = 0
    if int(density_win) >= 3:
        win = int(density_win)
        if win % 2 == 0:
            win += 1
        kernel = (win, win)
        bw_f = (bw > 0).astype(np.float32)
        line_f = line_like.astype(np.float32)
        density = cv2.blur(bw_f, kernel)
        line_density = cv2.blur(line_f, kernel)
        dense_non_line = (density > float(dense_soft)) & (line_density < float(line_density_max))
        dense_block = density > float(dense_hard)
        filtered[dense_non_line] = 0
        line_keep = line_keep & (~dense_block)
    # Restore only thin line-like structure (skeleton), not full blobs.
    filtered = cv2.bitwise_or(filtered, line_keep.astype(np.uint8) * 255)
    filtered = _remove_small_speckles(
        filtered,
        min_area=speckle_area,
        min_aspect=min_aspect,
        min_long=min_long,
    )
    filtered = cv2.bitwise_or(filtered, line_keep.astype(np.uint8) * 255)
    return filtered


def _bridge_edges_across_hole(
    edges_u8: np.ndarray,
    hole_u8: np.ndarray,
    angles_deg: list[float],
    length: int = 41,
    thickness: int = 3,
) -> np.ndarray:
    edges = edges_u8.copy()
    edges = cv2.bitwise_and(edges, cv2.bitwise_not(hole_u8))
    for ang in angles_deg:
        k = _line_kernel(int(length), int(thickness), float(ang))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k)
        newpix = cv2.subtract(closed, edges)
        newpix = cv2.bitwise_and(newpix, hole_u8)
        edges = cv2.bitwise_or(edges, newpix)
    return edges


def _dominant_angles_from_segments(
    segments: list[Tuple[float, float, float, float]],
) -> Optional[Tuple[float, float]]:
    if len(segments) < 2:
        return None
    angles = []
    lengths = []
    for x1, y1, x2, y2 in segments:
        ang = math.atan2(y2 - y1, x2 - x1)
        ang = float(np.mod(ang, math.pi))
        angles.append(ang)
        lengths.append(float(math.hypot(x2 - x1, y2 - y1)))
    angles_arr = np.array(angles, dtype=np.float32)
    weights = np.array(lengths, dtype=np.float32)
    feats = np.stack([np.cos(2.0 * angles_arr), np.sin(2.0 * angles_arr)], axis=1)
    idx0 = int(np.argmax(weights))
    c1 = feats[idx0]
    dists = np.sum((feats - c1) ** 2, axis=1)
    idx1 = int(np.argmax(dists))
    c2 = feats[idx1]
    for _ in range(10):
        d1 = np.sum((feats - c1) ** 2, axis=1)
        d2 = np.sum((feats - c2) ** 2, axis=1)
        assign_a = d1 <= d2
        if not np.any(assign_a) or np.all(assign_a):
            break
        w_a = weights[assign_a][:, None]
        w_b = weights[~assign_a][:, None]
        c1 = np.sum(feats[assign_a] * w_a, axis=0) / max(np.sum(w_a), 1e-6)
        c2 = np.sum(feats[~assign_a] * w_b, axis=0) / max(np.sum(w_b), 1e-6)

    assign_a = np.sum((feats - c1) ** 2, axis=1) <= np.sum((feats - c2) ** 2, axis=1)
    if not np.any(assign_a) or np.all(assign_a):
        return None
    ang_a = angles_arr[assign_a]
    ang_b = angles_arr[~assign_a]
    w_a = weights[assign_a]
    w_b = weights[~assign_a]
    sa = float(np.sum(w_a * np.sin(2.0 * ang_a)))
    ca = float(np.sum(w_a * np.cos(2.0 * ang_a)))
    sb = float(np.sum(w_b * np.sin(2.0 * ang_b)))
    cb = float(np.sum(w_b * np.cos(2.0 * ang_b)))
    if abs(sa) < 1e-6 and abs(ca) < 1e-6:
        return None
    if abs(sb) < 1e-6 and abs(cb) < 1e-6:
        return None
    a1 = 0.5 * math.atan2(sa, ca)
    a2 = 0.5 * math.atan2(sb, cb)
    if a1 < 0:
        a1 += math.pi
    if a2 < 0:
        a2 += math.pi
    return float(a1), float(a2)


def _mean_angle_segments(
    segments: list[Tuple[float, float, float, float]],
) -> Optional[float]:
    if not segments:
        return None
    angles = []
    weights = []
    for x1, y1, x2, y2 in segments:
        ang = float(np.mod(math.atan2(y2 - y1, x2 - x1), math.pi))
        angles.append(ang)
        weights.append(float(math.hypot(x2 - x1, y2 - y1)))
    ang_arr = np.array(angles, dtype=np.float32)
    w_arr = np.array(weights, dtype=np.float32)
    sa = float(np.sum(w_arr * np.sin(2.0 * ang_arr)))
    ca = float(np.sum(w_arr * np.cos(2.0 * ang_arr)))
    if abs(sa) < 1e-6 and abs(ca) < 1e-6:
        return None
    ang = 0.5 * math.atan2(sa, ca)
    if ang < 0:
        ang += math.pi
    return float(ang)
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
    if not (math.isfinite(x) and math.isfinite(y)):
        return None
    if abs(x) > 1e6 or abs(y) > 1e6:
        return None
    return np.array([x, y], dtype=np.float32)


def _pick_rep_line(
    segments: list[Tuple[float, float, float, float]],
    *,
    y_ref: float,
    x_ref: float,
    pick: str,
    angle_ref: Optional[float],
    max_angle_deg: float = 15.0,
    min_len: float = 0.0,
    line_mask: Optional[np.ndarray] = None,
    min_support: float = 0.0,
    top_k: int = 8,
    line_ids: Optional[Sequence[Optional[int]]] = None,
) -> Tuple[Optional[Tuple[float, float, float]], Dict[str, Any]]:
    best_line = None
    best_val: Optional[float] = None
    best_meta: Dict[str, Any] = {}
    max_angle = math.radians(float(max_angle_deg))
    candidates = []
    for idx, (x1, y1, x2, y2) in enumerate(segments):
        dx = float(x2 - x1)
        dy = float(y2 - y1)
        length = float(math.hypot(dx, dy))
        if length < float(min_len):
            continue
        angle = float(np.mod(math.atan2(dy, dx), math.pi))
        if angle_ref is not None and _angle_distance(angle, float(angle_ref)) > max_angle:
            continue
        line = _line_from_points((x1, y1), (x2, y2))
        support = None
        if line_mask is not None:
            support = _line_support((x1, y1, x2, y2), line_mask)
            if support is not None and float(support) < float(min_support):
                continue
        score = float(length * (0.5 + 0.5 * (support if support is not None else 0.0)))
        lid = None
        if line_ids is not None and idx < len(line_ids):
            lid = line_ids[idx]
        candidates.append((score, line, angle, length, support, lid))
    if not candidates:
        return None, {}
    candidates.sort(key=lambda item: item[0], reverse=True)
    top = candidates[: max(1, int(top_k))]
    for score, line, angle, length, support, lid in top:
        if pick in ("min_x", "max_x"):
            x_at = _x_at_y(line, y_ref)
            if x_at is None:
                continue
            val = float(x_at)
            if best_val is None or (pick == "min_x" and val < best_val) or (
                pick == "max_x" and val > best_val
            ):
                best_val = val
                best_line = line
                best_meta = {
                    "theta_deg": float(math.degrees(angle)),
                    "length": float(length),
                    "support": float(support) if support is not None else None,
                    "line_id": lid,
                    "score": float(score),
                }
        elif pick in ("min_y", "max_y"):
            y_at = _y_at_x(line, x_ref)
            if y_at is None:
                continue
            val = float(y_at)
            if best_val is None or (pick == "min_y" and val < best_val) or (
                pick == "max_y" and val > best_val
            ):
                best_val = val
                best_line = line
                best_meta = {
                    "theta_deg": float(math.degrees(angle)),
                    "length": float(length),
                    "support": float(support) if support is not None else None,
                    "line_id": lid,
                    "score": float(score),
                }
    return best_line, best_meta


def _complete_corners_from_lines(
    ordered_lb_rb_rt_lt: np.ndarray,
    lines_left: list[Tuple[float, float, float, float]],
    lines_bottom: list[Tuple[float, float, float, float]],
    floor_bbox: Tuple[int, int, int, int],
    img_w: int,
    img_h: int,
    line_mask: Optional[np.ndarray] = None,
    line_ids_left: Optional[Sequence[Optional[int]]] = None,
    line_ids_bottom: Optional[Sequence[Optional[int]]] = None,
    component_id: Optional[int] = None,
) -> Tuple[np.ndarray, bool, Dict[str, Any]]:
    ordered = np.asarray(ordered_lb_rb_rt_lt, dtype=np.float32).reshape(4, 2)
    x0, y0, x1, y1 = [float(v) for v in floor_bbox]
    y_ref = float(y1 - 1.0)
    x_ref = float(0.5 * (x0 + x1))
    sideline_angle_raw = _median_angle_segments(lines_left) or _mean_angle_segments(lines_left)
    sideline_angle = sideline_angle_raw
    baseline_angle_lines = _median_angle_segments(lines_bottom) or _mean_angle_segments(lines_bottom)
    baseline_angle_quad: Optional[float] = None
    baseline_vec = np.asarray(ordered[1] - ordered[0], dtype=np.float32)
    if float(np.linalg.norm(baseline_vec)) >= 1.0:
        baseline_angle_quad = float(
            np.mod(math.atan2(float(baseline_vec[1]), float(baseline_vec[0])), math.pi)
        )
    baseline_ref_disagree_deg = float(_env_float("BADC_COMPLETION_BASELINE_REF_DISAGREE_DEG", 14.0))
    baseline_angle = baseline_angle_lines
    baseline_angle_source = "lines"
    if baseline_angle_quad is not None:
        if baseline_angle is None:
            baseline_angle = baseline_angle_quad
            baseline_angle_source = "quad"
        else:
            ref_diff_deg = float(
                math.degrees(abs(_angle_distance(float(baseline_angle), float(baseline_angle_quad))))
            )
            if ref_diff_deg > baseline_ref_disagree_deg:
                baseline_angle = baseline_angle_quad
                baseline_angle_source = "quad_override"
    sideline_perp_tol_deg = _env_float("BADC_COMPLETION_SIDELINE_PERP_TOL_DEG", 30.0)
    sideline_perp_filter = _env_flag("BADC_COMPLETION_SIDELINE_PERP_FILTER", True)
    sideline_perp_repick = _env_flag("BADC_COMPLETION_SIDELINE_PERP_REPICK", True)
    sideline_snap_tol_deg = _env_float("BADC_COMPLETION_SIDELINE_PERP_SNAP_TOL_DEG", sideline_perp_tol_deg)
    sideline_perp_min_keep = int(max(2, _env_int("BADC_COMPLETION_SIDELINE_PERP_MIN_KEEP", 6)))
    sideline_rep_top_k = int(max(4, _env_int("BADC_COMPLETION_SIDELINE_TOP_K", 16)))
    sideline_bad_perp_deg = float(_env_float("BADC_COMPLETION_SIDELINE_BAD_PERP_DEG", 24.0))
    sideline_lr_parallel_tol_deg = float(
        _env_float("BADC_COMPLETION_SIDELINE_LR_PARALLEL_TOL_DEG", 12.0)
    )
    sideline_repick_perp_tol_deg = float(
        _env_float(
            "BADC_COMPLETION_SIDELINE_REPICK_PERP_TOL_DEG",
            max(10.0, min(36.0, sideline_perp_tol_deg + 4.0)),
        )
    )
    sideline_right_strict_perp_deg = float(
        _env_float("BADC_COMPLETION_RIGHT_STRICT_PERP_TOL_DEG", max(14.0, min(30.0, sideline_bad_perp_deg)))
    )
    sideline_right_strict_parallel_deg = float(
        _env_float("BADC_COMPLETION_RIGHT_STRICT_PARALLEL_TOL_DEG", max(8.0, sideline_lr_parallel_tol_deg))
    )
    sideline_right_min_sep_px = float(_env_float("BADC_COMPLETION_RIGHT_MIN_SEP_PX", 60.0))
    sideline_projective_parallel_tol_deg = float(
        _env_float(
            "BADC_COMPLETION_PROJECTIVE_PARALLEL_TOL_DEG",
            max(8.0, sideline_lr_parallel_tol_deg),
        )
    )
    sideline_projective_parallel_hard_deg = float(
        _env_float("BADC_COMPLETION_PROJECTIVE_PARALLEL_HARD_DEG", 78.0)
    )
    sideline_projective_perp_tol_deg = float(
        _env_float(
            "BADC_COMPLETION_PROJECTIVE_PERP_TOL_DEG",
            max(16.0, sideline_bad_perp_deg + 4.0),
        )
    )
    sideline_projective_perp_hard_deg = float(
        _env_float("BADC_COMPLETION_PROJECTIVE_PERP_HARD_DEG", 78.0)
    )
    sideline_projective_parallel_penalty_w = float(
        _env_float("BADC_COMPLETION_PROJECTIVE_PARALLEL_PENALTY_W", 3.5)
    )
    sideline_projective_perp_penalty_w = float(
        _env_float("BADC_COMPLETION_PROJECTIVE_PERP_PENALTY_W", 2.5)
    )
    sideline_projective_base_y_spread_frac = float(
        max(
            0.05,
            min(0.95, _env_float("BADC_COMPLETION_PROJECTIVE_BASE_Y_SPREAD_FRAC", 0.32)),
        )
    )
    sideline_min_support = float(
        max(0.0, min(1.0, _env_float("BADC_COMPLETION_SIDELINE_MIN_SUPPORT", 0.08)))
    )
    baseline_min_support = float(
        max(0.0, min(1.0, _env_float("BADC_COMPLETION_BASELINE_MIN_SUPPORT", 0.05)))
    )
    # Perspective-heavy views can make sidelines and baselines look far from orthogonal in image space.
    # Prefer a projective pair-pick (based on baseline/top intersections + quad geometry) over angle-perp assumptions.
    sideline_projective_pick = bool(_env_flag("BADC_COMPLETION_PROJECTIVE_SIDELINE_PICK", True))
    sideline_angle_perp: Optional[float] = None
    sideline_snapped_to_perp = False
    if baseline_angle is not None:
        sideline_angle_perp = float((float(baseline_angle) + 0.5 * math.pi) % math.pi)
    lines_left_use = list(lines_left)
    line_ids_left_use = list(line_ids_left) if line_ids_left is not None else [None] * len(lines_left_use)
    if sideline_perp_filter and (not sideline_projective_pick) and baseline_angle is not None and lines_left_use:
        tol = math.radians(float(max(5.0, sideline_perp_tol_deg)))
        scored_idx: list[Tuple[float, int]] = []
        keep_idx: list[int] = []
        for i, (x1s, y1s, x2s, y2s) in enumerate(lines_left_use):
            ang = float(np.mod(math.atan2(float(y2s - y1s), float(x2s - x1s)), math.pi))
            d_perp = abs(_angle_distance(ang, float(baseline_angle)) - 0.5 * math.pi)
            scored_idx.append((float(d_perp), int(i)))
            if d_perp <= tol:
                keep_idx.append(int(i))
        if len(keep_idx) >= 2:
            lines_left_use = [lines_left_use[i] for i in keep_idx]
            line_ids_left_use = [line_ids_left_use[i] for i in keep_idx]
        elif len(scored_idx) >= 2:
            scored_idx.sort(key=lambda t: t[0])
            pick = [idx for _, idx in scored_idx[: min(len(scored_idx), sideline_perp_min_keep)]]
            lines_left_use = [lines_left_use[i] for i in pick]
            line_ids_left_use = [line_ids_left_use[i] for i in pick]
    sideline_angle = _median_angle_segments(lines_left_use) or _mean_angle_segments(lines_left_use) or sideline_angle
    if sideline_angle is None and sideline_angle_perp is not None:
        sideline_angle = float(sideline_angle_perp)
        sideline_snapped_to_perp = True
    elif sideline_angle is not None and baseline_angle is not None and sideline_angle_perp is not None:
        perp_err = abs(_angle_distance(float(sideline_angle), float(baseline_angle)) - 0.5 * math.pi)
        if perp_err > math.radians(float(max(5.0, sideline_snap_tol_deg))):
            sideline_angle = float(sideline_angle_perp)
            sideline_snapped_to_perp = True
    sideline_max_angle_deg = _env_float("BADC_REP_MAX_ANGLE_SIDELINE_DEG", 30.0)
    baseline_max_angle_deg = _env_float("BADC_REP_MAX_ANGLE_BASELINE_DEG", 20.0)
    baseline_pick_angle_deg = float(
        max(
            8.0,
            min(
                35.0,
                _env_float(
                    "BADC_COMPLETION_BASELINE_PICK_MAX_ANGLE_DEG",
                    min(float(baseline_max_angle_deg), 16.0),
                ),
            ),
        )
    )
    baseline_neighbor_gap_min_px = float(max(6.0, _env_float("BADC_COMPLETION_BASELINE_NEIGHBOR_GAP_MIN_PX", 14.0)))
    baseline_neighbor_gap_max_px = float(
        max(
            baseline_neighbor_gap_min_px + 4.0,
            _env_float("BADC_COMPLETION_BASELINE_NEIGHBOR_GAP_MAX_PX", 92.0),
        )
    )
    min_len = 0.06 * float(min(img_w, img_h))
    left_line, left_meta = _pick_rep_line(
        lines_left_use,
        y_ref=y_ref,
        x_ref=x_ref,
        pick="min_x",
        angle_ref=sideline_angle,
        max_angle_deg=sideline_max_angle_deg,
        min_len=min_len,
        line_mask=line_mask,
        min_support=sideline_min_support,
        top_k=sideline_rep_top_k,
        line_ids=line_ids_left_use,
    )
    right_line, right_meta = _pick_rep_line(
        lines_left_use,
        y_ref=y_ref,
        x_ref=x_ref,
        pick="max_x",
        angle_ref=sideline_angle,
        max_angle_deg=sideline_max_angle_deg,
        min_len=min_len,
        line_mask=line_mask,
        min_support=sideline_min_support,
        top_k=sideline_rep_top_k,
        line_ids=line_ids_left_use,
    )
    left_line_initial = left_line
    right_line_initial = right_line
    right_meta_initial = dict(right_meta or {})

    def _line_angle_from_abc(line_abc: Optional[Tuple[float, float, float]]) -> Optional[float]:
        if line_abc is None:
            return None
        a, b, _ = line_abc
        ang = float(np.mod(math.atan2(-float(a), float(b)), math.pi))
        if not math.isfinite(ang):
            return None
        return ang

    def _perp_error_to_baseline(line_abc: Optional[Tuple[float, float, float]]) -> Optional[float]:
        if line_abc is None or baseline_angle is None:
            return None
        ang = _line_angle_from_abc(line_abc)
        if ang is None:
            return None
        return float(abs(_angle_distance(ang, float(baseline_angle)) - 0.5 * math.pi))

    def _repick_sideline_if_needed(
        current_line: Optional[Tuple[float, float, float]],
        current_meta: Dict[str, Any],
        pick_mode: str,
    ) -> Tuple[Optional[Tuple[float, float, float]], Dict[str, Any]]:
        meta = dict(current_meta or {})
        if not sideline_perp_repick:
            err = _perp_error_to_baseline(current_line)
            if err is not None:
                meta["perp_err_deg"] = float(math.degrees(err))
            return current_line, meta
        if baseline_angle is None or not lines_left_use:
            err = _perp_error_to_baseline(current_line)
            if err is not None:
                meta["perp_err_deg"] = float(math.degrees(err))
            return current_line, meta
        tol = math.radians(float(max(5.0, sideline_perp_tol_deg)))
        err = _perp_error_to_baseline(current_line)
        if err is not None:
            meta["perp_err_deg"] = float(math.degrees(err))
        if err is None or err <= tol:
            return current_line, meta

        keep_idx: list[int] = []
        for i, seg in enumerate(lines_left_use):
            x1s, y1s, x2s, y2s = seg
            seg_ang = float(np.mod(math.atan2(float(y2s - y1s), float(x2s - x1s)), math.pi))
            d_perp = abs(_angle_distance(seg_ang, float(baseline_angle)) - 0.5 * math.pi)
            if d_perp <= tol:
                keep_idx.append(int(i))
        meta["perp_repick_candidates"] = int(len(keep_idx))
        if len(keep_idx) < 1:
            return current_line, meta

        segs = [lines_left_use[i] for i in keep_idx]
        ids = [line_ids_left_use[i] for i in keep_idx]
        repick_angle_ref = sideline_angle_perp if sideline_angle_perp is not None else sideline_angle
        alt_line, alt_meta = _pick_rep_line(
            segs,
            y_ref=y_ref,
            x_ref=x_ref,
            pick=pick_mode,
            angle_ref=repick_angle_ref,
            max_angle_deg=max(sideline_max_angle_deg, sideline_perp_tol_deg + 8.0),
            min_len=min_len,
            line_mask=line_mask,
            min_support=sideline_min_support,
            top_k=sideline_rep_top_k,
            line_ids=ids,
        )
        if alt_line is None:
            return current_line, meta
        out_meta = dict(alt_meta or {})
        out_meta["perp_repicked"] = True
        out_err = _perp_error_to_baseline(alt_line)
        if out_err is not None:
            out_meta["perp_err_deg"] = float(math.degrees(out_err))
        return alt_line, out_meta

    left_line, left_meta = _repick_sideline_if_needed(left_line, left_meta, "min_x")
    right_line, right_meta = _repick_sideline_if_needed(right_line, right_meta, "max_x")

    def _lines_too_close(
        line_a: Optional[Tuple[float, float, float]],
        line_b: Optional[Tuple[float, float, float]],
    ) -> bool:
        if line_a is None or line_b is None:
            return False
        ang_a = _line_angle_from_abc(line_a)
        ang_b = _line_angle_from_abc(line_b)
        if ang_a is None or ang_b is None:
            return False
        if _angle_distance(float(ang_a), float(ang_b)) > math.radians(5.0):
            return False
        xa = _x_at_y(line_a, y_ref)
        xb = _x_at_y(line_b, y_ref)
        if xa is not None and xb is not None and math.isfinite(float(xa)) and math.isfinite(float(xb)):
            return abs(float(xa) - float(xb)) <= 14.0
        aa, bb, cc = line_b
        if abs(float(aa)) >= abs(float(bb)):
            py = 0.0
            px = -float(cc) / float(aa) if abs(float(aa)) > 1e-6 else 0.0
        else:
            px = 0.0
            py = -float(cc) / float(bb) if abs(float(bb)) > 1e-6 else 0.0
        return abs(_line_signed_distance(line_a, (float(px), float(py)))) <= 14.0

    if _lines_too_close(left_line, right_line):
        distinct_angle_ref = sideline_angle_perp if sideline_angle_perp is not None else sideline_angle
        cand_right, cand_meta = _pick_rep_line(
            lines_left_use,
            y_ref=y_ref,
            x_ref=x_ref,
            pick="max_x",
            angle_ref=distinct_angle_ref,
            max_angle_deg=max(sideline_max_angle_deg, sideline_perp_tol_deg + 10.0),
            min_len=min_len * 0.6,
            line_mask=line_mask,
            min_support=0.0,
            line_ids=line_ids_left_use,
            top_k=max(12, len(lines_left_use)),
        )
        if cand_right is not None and not _lines_too_close(left_line, cand_right):
            right_line = cand_right
            right_meta = dict(cand_meta or {})
            right_meta["distinct_repick"] = True
            right_meta["distinct_source"] = "lines_left_use"
        elif right_line_initial is not None and not _lines_too_close(left_line, right_line_initial):
            right_line = right_line_initial
            right_meta = dict(right_meta_initial or {})
            right_meta["distinct_repick"] = True
            right_meta["distinct_source"] = "initial_pick"
        elif right_line is not None:
            right_meta = dict(right_meta or {})
            right_meta["distinct_repick"] = False
            right_meta["distinct_source"] = "same_as_left"

    if right_line is None and lines_left_use:
        right_line, right_meta = _pick_rep_line(
            lines_left_use,
            y_ref=y_ref,
            x_ref=x_ref,
            pick="max_x",
            angle_ref=sideline_angle_perp if sideline_angle_perp is not None else sideline_angle,
            max_angle_deg=max(sideline_max_angle_deg, 45.0),
            min_len=min_len * 0.6,
            line_mask=line_mask,
            min_support=0.0,
            line_ids=line_ids_left_use,
        )

    def _pick_right_sideline_strict(
        left_abc: Optional[Tuple[float, float, float]],
        current_right_abc: Optional[Tuple[float, float, float]],
    ) -> Tuple[Optional[Tuple[float, float, float]], Dict[str, Any]]:
        if not lines_left_use:
            return current_right_abc, dict(right_meta or {})
        left_ang = _line_angle_from_abc(left_abc)
        strict_perp_tol = math.radians(float(max(6.0, sideline_right_strict_perp_deg)))
        strict_parallel_tol = math.radians(float(max(3.0, sideline_right_strict_parallel_deg)))
        x_left = _x_at_y(left_abc, y_ref) if left_abc is not None else None
        x_right_curr = _x_at_y(current_right_abc, y_ref) if current_right_abc is not None else None
        candidates: list[Tuple[float, Tuple[float, float, float], Dict[str, Any]]] = []
        for i, seg in enumerate(lines_left_use):
            x1s, y1s, x2s, y2s = seg
            line_i = _line_from_points((x1s, y1s), (x2s, y2s))
            xi = _x_at_y(line_i, y_ref)
            if xi is None or (not math.isfinite(float(xi))):
                continue
            ang_i = _line_angle_from_abc(line_i)
            if ang_i is None:
                continue
            if baseline_angle is not None:
                d_perp = abs(_angle_distance(float(ang_i), float(baseline_angle)) - 0.5 * math.pi)
                if d_perp > strict_perp_tol:
                    continue
            if left_ang is not None and _angle_distance(float(ang_i), float(left_ang)) > strict_parallel_tol:
                continue
            if x_left is not None and float(xi) <= float(x_left) + float(sideline_right_min_sep_px):
                continue
            seg_len = float(math.hypot(float(x2s - x1s), float(y2s - y1s)))
            sup = _line_support((x1s, y1s, x2s, y2s), line_mask) if line_mask is not None else None
            score = float(xi + 0.04 * seg_len + 20.0 * (float(sup) if sup is not None else 0.0))
            lid = line_ids_left_use[i] if i < len(line_ids_left_use) else None
            candidates.append(
                (
                    score,
                    line_i,
                    {
                        "theta_deg": float(math.degrees(float(ang_i))),
                        "length": float(seg_len),
                        "support": float(sup) if sup is not None else None,
                        "line_id": lid,
                        "score": float(score),
                        "strict_right_pick": True,
                        "strict_right_x_at_y": float(xi),
                    },
                )
            )
        if not candidates:
            out = dict(right_meta or {})
            out["strict_right_pick"] = False
            out["strict_right_candidates"] = 0
            return current_right_abc, out
        candidates.sort(key=lambda t: t[0], reverse=True)
        best_score, best_line, best_meta = candidates[0]
        out = dict(best_meta or {})
        out["strict_right_pick"] = True
        out["strict_right_candidates"] = int(len(candidates))
        out["strict_right_best_score"] = float(best_score)
        if x_right_curr is not None and math.isfinite(float(x_right_curr)):
            out["strict_right_prev_x_at_y"] = float(x_right_curr)
        return best_line, out
    sideline_fallback_used = False
    if (left_line is None or right_line is None) and (len(lines_left_use) != len(lines_left)):
        sideline_fallback_used = True
        sideline_angle = sideline_angle_raw or sideline_angle
        left_line, left_meta = _pick_rep_line(
            lines_left,
            y_ref=y_ref,
            x_ref=x_ref,
            pick="min_x",
            angle_ref=sideline_angle,
            max_angle_deg=sideline_max_angle_deg,
            min_len=min_len,
            line_mask=line_mask,
            min_support=0.0,
            line_ids=line_ids_left,
        )
        right_line, right_meta = _pick_rep_line(
            lines_left,
            y_ref=y_ref,
            x_ref=x_ref,
            pick="max_x",
            angle_ref=sideline_angle,
            max_angle_deg=sideline_max_angle_deg,
            min_len=min_len,
            line_mask=line_mask,
            min_support=0.0,
            line_ids=line_ids_left,
        )
        if right_line is None and lines_left:
            right_line, right_meta = _pick_rep_line(
                lines_left,
                y_ref=y_ref,
                x_ref=x_ref,
                pick="max_x",
                angle_ref=sideline_angle,
                max_angle_deg=max(sideline_max_angle_deg, 45.0),
                min_len=min_len * 0.6,
                line_mask=line_mask,
                min_support=0.0,
                line_ids=line_ids_left,
            )
    def _pick_bottom_line_disambiguated(
        segs: Sequence[Tuple[float, float, float, float]],
        seg_ids: Sequence[Optional[int]],
    ) -> Tuple[Optional[Tuple[float, float, float]], Dict[str, Any]]:
        cands: list[Dict[str, Any]] = []
        tol_rad = math.radians(float(max(5.0, baseline_pick_angle_deg)))
        for idx, seg in enumerate(segs):
            x1s, y1s, x2s, y2s = [float(v) for v in seg]
            line_i = _line_from_points((x1s, y1s), (x2s, y2s))
            ang_i = float(np.mod(math.atan2(float(y2s - y1s), float(x2s - x1s)), math.pi))
            if baseline_angle is not None and _angle_distance(ang_i, float(baseline_angle)) > tol_rad:
                continue
            seg_len = float(math.hypot(float(x2s - x1s), float(y2s - y1s)))
            if seg_len < (0.55 * min_len):
                continue
            y_at_ref = _y_at_x(line_i, x_ref)
            if y_at_ref is None or (not math.isfinite(float(y_at_ref))):
                y_at_ref = float(0.5 * (y1s + y2s))
            support = _line_support((x1s, y1s, x2s, y2s), line_mask) if line_mask is not None else None
            if support is not None and float(support) < max(0.0, 0.55 * baseline_min_support):
                continue
            line_id_val: Optional[int] = None
            if idx < len(seg_ids):
                raw_id = seg_ids[idx]
                line_id_val = int(raw_id) if raw_id is not None else None
            cands.append(
                {
                    "line": line_i,
                    "line_id": line_id_val,
                    "theta_deg": float(math.degrees(ang_i)),
                    "length": float(seg_len),
                    "support": float(support) if support is not None else None,
                    "y_at_ref": float(y_at_ref),
                }
            )
        if not cands:
            return None, {"reason": "no_filtered_candidates"}

        cands.sort(key=lambda d: float(d["y_at_ref"]), reverse=True)
        chosen = cands[0]
        neighbor_found = False
        for c in cands[1:]:
            dy = float(chosen["y_at_ref"]) - float(c["y_at_ref"])
            if dy < baseline_neighbor_gap_min_px or dy > baseline_neighbor_gap_max_px:
                continue
            if baseline_angle is not None:
                if abs(float(c["theta_deg"]) - float(chosen["theta_deg"])) > max(6.0, baseline_pick_angle_deg):
                    continue
            neighbor_found = True
            break
        out_meta = {
            "line_id": chosen.get("line_id"),
            "theta_deg": float(chosen["theta_deg"]),
            "length": float(chosen["length"]),
            "support": float(chosen["support"]) if chosen.get("support") is not None else None,
            "score": float(chosen["y_at_ref"]),
            "picked_by_completion_baseline_rule": True,
            "neighbor_parallel_found": bool(neighbor_found),
            "num_candidates": int(len(cands)),
        }
        return chosen["line"], out_meta

    line_ids_bottom_seq = list(line_ids_bottom) if line_ids_bottom is not None else [None] * len(lines_bottom)
    bottom_line, bottom_meta = _pick_bottom_line_disambiguated(lines_bottom, line_ids_bottom_seq)
    if bottom_line is None and lines_bottom:
        bottom_line, bottom_meta = _pick_rep_line(
            lines_bottom,
            y_ref=y_ref,
            x_ref=x_ref,
            pick="max_y",
            angle_ref=baseline_angle,
            max_angle_deg=baseline_pick_angle_deg,
            min_len=min_len * 0.65,
            line_mask=line_mask,
            min_support=max(0.0, 0.5 * baseline_min_support),
            line_ids=line_ids_bottom,
        )
    if bottom_line is None and lines_bottom:
        bottom_line, bottom_meta = _pick_rep_line(
            lines_bottom,
            y_ref=y_ref,
            x_ref=x_ref,
            pick="max_y",
            angle_ref=baseline_angle,
            max_angle_deg=max(baseline_max_angle_deg, 35.0),
            min_len=min_len * 0.6,
            line_mask=line_mask,
            min_support=0.0,
            line_ids=line_ids_bottom,
        )

    def _line_span_on_baseline(
        left_abc: Optional[Tuple[float, float, float]],
        right_abc: Optional[Tuple[float, float, float]],
        baseline_abc: Optional[Tuple[float, float, float]],
    ) -> Optional[Tuple[float, np.ndarray, np.ndarray]]:
        if left_abc is None or right_abc is None or baseline_abc is None:
            return None
        p_l = _intersect_lines(left_abc, baseline_abc)
        p_r = _intersect_lines(right_abc, baseline_abc)
        if p_l is None or p_r is None:
            return None
        if (not np.all(np.isfinite(p_l))) or (not np.all(np.isfinite(p_r))):
            return None
        span = float(abs(float(p_r[0]) - float(p_l[0])))
        return span, p_l, p_r

    sideline_left_perp_err_deg: Optional[float] = None
    sideline_right_perp_err_deg: Optional[float] = None
    sideline_lr_nonparallel = False
    sideline_need_repick = False
    if bottom_line is not None and lines_left_use:
        curr_span_info = _line_span_on_baseline(left_line, right_line, bottom_line)
        curr_span = float(curr_span_info[0]) if curr_span_info is not None else 0.0
        span_min_px = float(_env_float("BADC_COMPLETION_SIDELINE_MIN_SPAN_PX", 220.0))
        span_margin_px = float(_env_float("BADC_COMPLETION_SIDELINE_SPAN_MARGIN_PX", 18.0))
        err_l = _perp_error_to_baseline(left_line)
        err_r = _perp_error_to_baseline(right_line)
        sideline_left_perp_err_deg = float(math.degrees(err_l)) if err_l is not None else None
        sideline_right_perp_err_deg = float(math.degrees(err_r)) if err_r is not None else None
        left_bad_perp = (
            sideline_left_perp_err_deg is not None and sideline_left_perp_err_deg > sideline_bad_perp_deg
        )
        right_bad_perp = (
            sideline_right_perp_err_deg is not None and sideline_right_perp_err_deg > sideline_bad_perp_deg
        )
        ang_l = _line_angle_from_abc(left_line)
        ang_r = _line_angle_from_abc(right_line)
        if ang_l is not None and ang_r is not None:
            lr_nonparallel = (
                math.degrees(abs(_angle_distance(float(ang_l), float(ang_r))))
                > sideline_lr_parallel_tol_deg
            )
        sideline_lr_nonparallel = bool(lr_nonparallel)
        need_lr_repick = bool(curr_span < span_min_px or left_bad_perp or right_bad_perp or lr_nonparallel)
        sideline_need_repick = bool(need_lr_repick)
        if left_bad_perp or right_bad_perp or lr_nonparallel:
            strict_right_line, strict_right_meta = _pick_right_sideline_strict(left_line, right_line)
            if strict_right_line is not None:
                right_line = strict_right_line
                right_meta = dict(strict_right_meta or {})
                err_r2 = _perp_error_to_baseline(right_line)
                sideline_right_perp_err_deg = float(math.degrees(err_r2)) if err_r2 is not None else None
                ang_l2 = _line_angle_from_abc(left_line)
                ang_r2 = _line_angle_from_abc(right_line)
                if ang_l2 is not None and ang_r2 is not None:
                    sideline_lr_nonparallel = (
                        math.degrees(abs(_angle_distance(float(ang_l2), float(ang_r2))))
                        > sideline_lr_parallel_tol_deg
                    )
                sideline_need_repick = bool(
                    curr_span < span_min_px
                    or (sideline_right_perp_err_deg is not None and sideline_right_perp_err_deg > sideline_bad_perp_deg)
                    or sideline_lr_nonparallel
                )
        # If current left/right are too close on baseline, repick from baseline intersections.
        if sideline_need_repick:
            side_perp_tol = math.radians(float(max(6.0, sideline_repick_perp_tol_deg)))
            ext_x0 = float(x0) - 0.25 * float(max(1.0, x1 - x0))
            ext_x1 = float(x1) + 0.25 * float(max(1.0, x1 - x0))
            ext_y0 = float(y0) - 0.20 * float(max(1.0, y1 - y0))
            ext_y1 = float(y1) + 0.45 * float(max(1.0, y1 - y0))
            cand_lr: list[Tuple[float, Tuple[float, float, float], Dict[str, Any]]] = []
            for i, seg in enumerate(lines_left_use):
                x1s, y1s, x2s, y2s = seg
                line_i = _line_from_points((x1s, y1s), (x2s, y2s))
                p_i = _intersect_lines(line_i, bottom_line)
                if p_i is None or (not np.all(np.isfinite(p_i))):
                    continue
                px, py = float(p_i[0]), float(p_i[1])
                if px < ext_x0 or px > ext_x1 or py < ext_y0 or py > ext_y1:
                    continue
                ang_i = float(np.mod(math.atan2(float(y2s - y1s), float(x2s - x1s)), math.pi))
                if baseline_angle is not None:
                    d_perp = abs(_angle_distance(ang_i, float(baseline_angle)) - 0.5 * math.pi)
                    if d_perp > side_perp_tol:
                        continue
                len_i = float(math.hypot(float(x2s - x1s), float(y2s - y1s)))
                sup_i = _line_support((x1s, y1s, x2s, y2s), line_mask) if line_mask is not None else None
                score_i = float(len_i * (0.5 + 0.5 * (sup_i if sup_i is not None else 0.0)))
                lid_i = line_ids_left_use[i] if i < len(line_ids_left_use) else None
                cand_lr.append(
                    (
                        px,
                        line_i,
                        {
                            "theta_deg": float(math.degrees(ang_i)),
                            "length": float(len_i),
                            "support": float(sup_i) if sup_i is not None else None,
                            "line_id": lid_i,
                            "score": float(score_i),
                            "picked_by_baseline_span": True,
                            "baseline_cross_x": float(px),
                            "baseline_cross_y": float(py),
                        },
                    )
                )
            if len(cand_lr) >= 2:
                cand_lr.sort(key=lambda t: t[0])
                left_pool = cand_lr[: min(4, len(cand_lr))]
                right_pool = cand_lr[max(0, len(cand_lr) - 4) :]
                best_pair = None
                best_pair_key = None
                for xl, ll, ml in left_pool:
                    for xr, lr, mr in right_pool:
                        if xr <= xl:
                            continue
                        ang_l = _line_angle_from_abc(ll)
                        ang_r = _line_angle_from_abc(lr)
                        if ang_l is not None and ang_r is not None:
                            if (
                                math.degrees(abs(_angle_distance(float(ang_l), float(ang_r))))
                                > sideline_lr_parallel_tol_deg
                            ):
                                continue
                        if baseline_angle is not None:
                            dpl = _perp_error_to_baseline(ll)
                            dpr = _perp_error_to_baseline(lr)
                            if dpl is not None and math.degrees(float(dpl)) > sideline_right_strict_perp_deg:
                                continue
                            if dpr is not None and math.degrees(float(dpr)) > sideline_right_strict_perp_deg:
                                continue
                        span = float(xr - xl)
                        pair_key = float(span + 0.02 * (float(ml.get("score", 0.0)) + float(mr.get("score", 0.0))))
                        if best_pair_key is None or pair_key > best_pair_key:
                            best_pair_key = pair_key
                            best_pair = (ll, ml, lr, mr, span)
                if best_pair is not None:
                    ll, ml, lr, mr, span = best_pair
                    min_req_span = max(span_min_px, curr_span + span_margin_px) if curr_span < span_min_px else span_min_px
                    if span >= min_req_span:
                        left_line = ll
                        left_meta = dict(ml)
                        right_line = lr
                        right_meta = dict(mr)
                        left_meta["span_repicked"] = True
                        right_meta["span_repicked"] = True
                        left_meta["span_prev"] = float(curr_span)
                        right_meta["span_prev"] = float(curr_span)
                        left_meta["span_new"] = float(span)
                        right_meta["span_new"] = float(span)
                        curr_span = float(span)
    top_line, top_meta = _pick_rep_line(
        lines_bottom,
        y_ref=y_ref,
        x_ref=x_ref,
        pick="min_y",
        angle_ref=baseline_angle,
        max_angle_deg=baseline_pick_angle_deg,
        min_len=min_len,
        line_mask=line_mask,
        min_support=baseline_min_support,
        line_ids=line_ids_bottom,
    )
    if top_line is None and lines_bottom:
        top_line, top_meta = _pick_rep_line(
            lines_bottom,
            y_ref=y_ref,
            x_ref=x_ref,
            pick="min_y",
            angle_ref=baseline_angle,
            max_angle_deg=max(baseline_max_angle_deg, 35.0),
            min_len=min_len * 0.6,
            line_mask=line_mask,
            min_support=0.0,
            line_ids=line_ids_bottom,
        )

    def _pick_sideline_pair_projective(
        baseline_abc: Optional[Tuple[float, float, float]],
        top_abc: Optional[Tuple[float, float, float]],
    ) -> Tuple[
        Optional[Tuple[float, float, float]],
        Dict[str, Any],
        Optional[Tuple[float, float, float]],
        Dict[str, Any],
        Dict[str, Any],
    ]:
        if baseline_abc is None or not lines_left_use:
            return None, {}, None, {}, {"used": False, "reason": "no_baseline_or_lines"}

        floor_w = float(max(1.0, x1 - x0))
        floor_h = float(max(1.0, y1 - y0))
        min_len_local = float(max(24.0, min_len * 0.55))
        min_span_ratio = float(max(0.10, min(0.95, _env_float("BADC_COMPLETION_PROJECTIVE_MIN_BASE_SPAN_RATIO", 0.30))))
        min_top_span_ratio = float(max(0.04, min(0.90, _env_float("BADC_COMPLETION_PROJECTIVE_MIN_TOP_SPAN_RATIO", 0.10))))
        min_span_px = float(max(sideline_right_min_sep_px, _env_float("BADC_COMPLETION_PROJECTIVE_MIN_BASE_SPAN_PX", 0.0)))
        min_base_span = float(max(min_span_px, min_span_ratio * floor_w))
        min_top_span = float(max(18.0, min_top_span_ratio * floor_w))
        min_base_y_frac = float(max(0.0, min(0.95, _env_float("BADC_COMPLETION_PROJECTIVE_MIN_BASE_Y_FRAC", 0.35))))
        min_base_y_px = float(y0 + min_base_y_frac * floor_h)
        pair_min_support = float(max(0.0, min(1.0, _env_float("BADC_COMPLETION_PROJECTIVE_PAIR_MIN_SUPPORT", 0.05))))
        pair_min_support_hard = float(
            max(
                0.0,
                min(
                    pair_min_support,
                    _env_float("BADC_COMPLETION_PROJECTIVE_PAIR_MIN_SUPPORT_HARD", 0.0),
                ),
            )
        )
        support_penalty_w = float(
            max(0.0, _env_float("BADC_COMPLETION_PROJECTIVE_SUPPORT_PENALTY_W", 22.0))
        )
        ext_x0 = float(x0) - float(_env_float("BADC_COMPLETION_PROJECTIVE_X_PAD_FRAC", 0.30)) * floor_w
        ext_x1 = float(x1) + float(_env_float("BADC_COMPLETION_PROJECTIVE_X_PAD_FRAC", 0.30)) * floor_w
        ext_y0 = float(y0) - float(_env_float("BADC_COMPLETION_PROJECTIVE_TOP_PAD_FRAC", 0.22)) * floor_h
        ext_y1 = float(y1) + float(_env_float("BADC_COMPLETION_PROJECTIVE_BOTTOM_PAD_FRAC", 0.50)) * floor_h
        pair_top_k = int(max(4, _env_int("BADC_COMPLETION_PROJECTIVE_POOL_TOP_K", 7)))
        min_edge_req = float(
            max(
                8.0,
                _env_float("BADC_COMPLETION_PROJECTIVE_MIN_EDGE_FRAC", 0.018) * float(min(max(1, img_w), max(1, img_h))),
            )
        )

        cand: list[Dict[str, Any]] = []
        for i, seg in enumerate(lines_left_use):
            x1s, y1s, x2s, y2s = seg
            seg_len = float(math.hypot(float(x2s - x1s), float(y2s - y1s)))
            if seg_len < min_len_local:
                continue
            line_i = _line_from_points((x1s, y1s), (x2s, y2s))
            p_base = _intersect_lines(line_i, baseline_abc)
            if p_base is None or (not np.all(np.isfinite(p_base))):
                continue
            xb = float(p_base[0])
            yb = float(p_base[1])
            # Baseline intersections can project outside visible y-range in strong perspective.
            # Keep broad x-only gating to avoid discarding the true far-right sideline.
            if xb < ext_x0 or xb > ext_x1:
                continue
            if yb < ext_y0 or yb > ext_y1:
                continue

            p_top = None
            xt = None
            yt = None
            if top_abc is not None:
                p_top = _intersect_lines(line_i, top_abc)
                if p_top is None or (not np.all(np.isfinite(p_top))):
                    continue
                xt = float(p_top[0])
                yt = float(p_top[1])
                if xt < ext_x0 or xt > ext_x1:
                    continue

            sup = _line_support((x1s, y1s, x2s, y2s), line_mask) if line_mask is not None else None
            lid = line_ids_left_use[i] if i < len(line_ids_left_use) else None
            quality = float(seg_len * (0.4 + 0.6 * (float(sup) if sup is not None else 0.0)))
            cand.append(
                {
                    "line": line_i,
                    "x_base": float(xb),
                    "y_base": float(yb),
                    "x_top": float(xt) if xt is not None else None,
                    "y_top": float(yt) if yt is not None else None,
                    "length": float(seg_len),
                    "support": float(sup) if sup is not None else None,
                    "score": float(quality),
                    "line_id": lid,
                    "theta_deg": float(math.degrees(np.mod(math.atan2(float(y2s - y1s), float(x2s - x1s)), math.pi))),
                }
            )

        meta_local: Dict[str, Any] = {
            "used": False,
            "num_candidates": int(len(cand)),
            "min_base_span_req": float(min_base_span),
            "min_top_span_req": float(min_top_span),
            "min_edge_req": float(min_edge_req),
            "min_base_y_px": float(min_base_y_px),
            "pair_min_support": float(pair_min_support),
            "pair_min_support_hard": float(pair_min_support_hard),
            "support_penalty_w": float(support_penalty_w),
            "parallel_tol_deg": float(sideline_projective_parallel_tol_deg),
            "parallel_hard_deg": float(sideline_projective_parallel_hard_deg),
            "perp_tol_deg": float(sideline_projective_perp_tol_deg),
            "perp_hard_deg": float(sideline_projective_perp_hard_deg),
            "parallel_penalty_w": float(sideline_projective_parallel_penalty_w),
            "perp_penalty_w": float(sideline_projective_perp_penalty_w),
            "max_base_y_spread_px": float(sideline_projective_base_y_spread_frac * floor_h),
        }
        reject_stats = {
            "base_span": 0,
            "base_y": 0,
            "support": 0,
            "support_hard": 0,
            "parallel": 0,
            "perp": 0,
            "base_y_spread": 0,
            "top_span": 0,
            "quad_convex": 0,
            "quad_edge": 0,
        }
        cand_preview = sorted(cand, key=lambda d: float(d["x_base"]))
        meta_local["candidates"] = [
            {
                "line_id": c.get("line_id"),
                "x_base": float(c.get("x_base")),
                "y_base": float(c.get("y_base")),
                "x_top": float(c.get("x_top")) if c.get("x_top") is not None else None,
                "y_top": float(c.get("y_top")) if c.get("y_top") is not None else None,
                "support": float(c.get("support")) if c.get("support") is not None else None,
                "length": float(c.get("length")),
                "theta_deg": float(c.get("theta_deg")),
            }
            for c in cand_preview[: min(24, len(cand_preview))]
        ]
        if len(cand) < 2:
            meta_local["reason"] = "insufficient_candidates"
            return None, {}, None, {}, meta_local

        cand.sort(key=lambda d: float(d["x_base"]))
        left_pool = cand[: min(pair_top_k, len(cand))]
        right_pool = cand[max(0, len(cand) - pair_top_k) :]

        best_tuple = None
        best_key = None
        for cl in left_pool:
            for cr in right_pool:
                if float(cr["x_base"]) <= float(cl["x_base"]):
                    continue
                base_span = float(cr["x_base"] - cl["x_base"])
                if base_span < min_base_span:
                    reject_stats["base_span"] += 1
                    continue
                if max(float(cl["y_base"]), float(cr["y_base"])) < min_base_y_px:
                    reject_stats["base_y"] += 1
                    continue
                sup_l = float(cl["support"]) if cl["support"] is not None else 0.0
                sup_r = float(cr["support"]) if cr["support"] is not None else 0.0
                min_pair_sup = min(sup_l, sup_r)
                if min_pair_sup < pair_min_support_hard:
                    reject_stats["support_hard"] += 1
                    continue
                ang_l = _line_angle_from_abc(cl["line"])
                ang_r = _line_angle_from_abc(cr["line"])
                if ang_l is None or ang_r is None:
                    reject_stats["parallel"] += 1
                    continue
                parallel_err_deg = float(math.degrees(abs(_angle_distance(float(ang_l), float(ang_r)))))
                if parallel_err_deg > float(sideline_projective_parallel_hard_deg):
                    reject_stats["parallel"] += 1
                    continue
                parallel_penalty_deg = max(0.0, parallel_err_deg - float(sideline_projective_parallel_tol_deg))
                if baseline_angle is not None:
                    perp_l_deg = float(
                        math.degrees(abs(_angle_distance(float(ang_l), float(baseline_angle)) - 0.5 * math.pi))
                    )
                    perp_r_deg = float(
                        math.degrees(abs(_angle_distance(float(ang_r), float(baseline_angle)) - 0.5 * math.pi))
                    )
                    if max(perp_l_deg, perp_r_deg) > float(sideline_projective_perp_hard_deg):
                        reject_stats["perp"] += 1
                        continue
                    perp_penalty_deg = max(
                        0.0,
                        max(perp_l_deg, perp_r_deg) - float(sideline_projective_perp_tol_deg),
                    )
                else:
                    perp_l_deg = None
                    perp_r_deg = None
                    perp_penalty_deg = 0.0
                base_y_spread = float(abs(float(cl["y_base"]) - float(cr["y_base"])))
                if base_y_spread > float(sideline_projective_base_y_spread_frac * floor_h):
                    reject_stats["base_y_spread"] += 1
                    continue

                lb = _intersect_lines(cl["line"], baseline_abc)
                rb = _intersect_lines(cr["line"], baseline_abc)
                if lb is None or rb is None or (not np.all(np.isfinite(lb))) or (not np.all(np.isfinite(rb))):
                    continue

                lt = None
                rt = None
                top_span = base_span
                if top_abc is not None:
                    lt = _intersect_lines(cl["line"], top_abc)
                    rt = _intersect_lines(cr["line"], top_abc)
                    if lt is None or rt is None or (not np.all(np.isfinite(lt))) or (not np.all(np.isfinite(rt))):
                        continue
                    top_span = float(rt[0] - lt[0])
                    if top_span < min_top_span:
                        reject_stats["top_span"] += 1
                        continue

                if lt is not None and rt is not None:
                    quad = np.asarray([lb, rb, rt, lt], dtype=np.float32).reshape(4, 2)
                    if not _is_convex_quad(quad):
                        reject_stats["quad_convex"] += 1
                        continue
                    if float(np.min(_quad_edges(quad))) < min_edge_req:
                        reject_stats["quad_edge"] += 1
                        continue

                sep_quality = float(base_span + 0.45 * top_span)
                line_quality = float(0.03 * float(cl["score"]) + 0.03 * float(cr["score"]))
                support_bonus = float(
                    35.0
                    * min(
                        sup_l,
                        sup_r,
                    )
                )
                support_penalty = max(0.0, float(pair_min_support) - float(min_pair_sup))
                pair_key = float(
                    sep_quality
                    + line_quality
                    + support_bonus
                    - float(support_penalty_w) * float(support_penalty)
                    - float(sideline_projective_parallel_penalty_w) * float(parallel_penalty_deg)
                    - float(sideline_projective_perp_penalty_w) * float(perp_penalty_deg)
                )
                if best_key is None or pair_key > best_key:
                    best_key = pair_key
                    best_tuple = (
                        cl,
                        cr,
                        base_span,
                        top_span,
                        pair_key,
                        parallel_err_deg,
                        perp_l_deg,
                        perp_r_deg,
                        parallel_penalty_deg,
                        perp_penalty_deg,
                        base_y_spread,
                    )

        if best_tuple is None:
            meta_local["reason"] = "no_valid_pair"
            meta_local["reject_stats"] = reject_stats
            return None, {}, None, {}, meta_local

        (
            cl,
            cr,
            base_span,
            top_span,
            pair_key,
            parallel_err_deg,
            perp_l_deg,
            perp_r_deg,
            parallel_penalty_deg,
            perp_penalty_deg,
            base_y_spread,
        ) = best_tuple
        left_out = dict(cl)
        right_out = dict(cr)
        left_out["projective_pair_pick"] = True
        right_out["projective_pair_pick"] = True
        left_out["projective_pair_base_span_px"] = float(base_span)
        right_out["projective_pair_base_span_px"] = float(base_span)
        left_out["projective_pair_top_span_px"] = float(top_span)
        right_out["projective_pair_top_span_px"] = float(top_span)
        left_out["projective_pair_score"] = float(pair_key)
        right_out["projective_pair_score"] = float(pair_key)
        left_out["projective_pair_parallel_err_deg"] = float(parallel_err_deg)
        right_out["projective_pair_parallel_err_deg"] = float(parallel_err_deg)
        left_out["projective_pair_parallel_penalty_deg"] = float(parallel_penalty_deg)
        right_out["projective_pair_parallel_penalty_deg"] = float(parallel_penalty_deg)
        left_out["projective_pair_base_y_spread_px"] = float(base_y_spread)
        right_out["projective_pair_base_y_spread_px"] = float(base_y_spread)
        if perp_l_deg is not None:
            left_out["projective_pair_perp_err_deg"] = float(perp_l_deg)
        if perp_r_deg is not None:
            right_out["projective_pair_perp_err_deg"] = float(perp_r_deg)
        left_out["projective_pair_perp_penalty_deg"] = float(perp_penalty_deg)
        right_out["projective_pair_perp_penalty_deg"] = float(perp_penalty_deg)

        meta_local.update(
            {
                "used": True,
                "reason": "ok",
                "pair_score": float(pair_key),
                "pair_base_span_px": float(base_span),
                "pair_top_span_px": float(top_span),
                "left_line_id": cl.get("line_id"),
                "right_line_id": cr.get("line_id"),
                "pair_parallel_err_deg": float(parallel_err_deg),
                "pair_parallel_penalty_deg": float(parallel_penalty_deg),
                "pair_perp_left_err_deg": float(perp_l_deg) if perp_l_deg is not None else None,
                "pair_perp_right_err_deg": float(perp_r_deg) if perp_r_deg is not None else None,
                "pair_perp_penalty_deg": float(perp_penalty_deg),
                "pair_base_y_spread_px": float(base_y_spread),
                "reject_stats": reject_stats,
            }
        )
        return cl["line"], left_out, cr["line"], right_out, meta_local

    projective_pair_meta: Dict[str, Any] = {"used": False}
    projective_pair_applied = False
    projective_pair_rejected_reason = None
    left_line_before_projective = left_line
    right_line_before_projective = right_line
    left_meta_before_projective = dict(left_meta or {})
    right_meta_before_projective = dict(right_meta or {})
    if sideline_projective_pick and bottom_line is not None:
        p_left, p_left_meta, p_right, p_right_meta, p_meta = _pick_sideline_pair_projective(bottom_line, top_line)
        projective_pair_meta = dict(p_meta or {})
        if p_left is not None and p_right is not None:
            ang_l = _line_angle_from_abc(p_left)
            ang_r = _line_angle_from_abc(p_right)
            if ang_l is None or ang_r is None:
                projective_pair_rejected_reason = "pair_angle_invalid"
            else:
                pair_parallel_err = float(math.degrees(abs(_angle_distance(float(ang_l), float(ang_r)))))
                if pair_parallel_err > float(sideline_projective_parallel_hard_deg):
                    projective_pair_rejected_reason = "pair_not_parallel"
                elif baseline_angle is not None:
                    perp_l = float(
                        math.degrees(abs(_angle_distance(float(ang_l), float(baseline_angle)) - 0.5 * math.pi))
                    )
                    perp_r = float(
                        math.degrees(abs(_angle_distance(float(ang_r), float(baseline_angle)) - 0.5 * math.pi))
                    )
                    if max(perp_l, perp_r) > float(sideline_projective_perp_hard_deg):
                        projective_pair_rejected_reason = "pair_not_perpendicular_to_baseline"
            if projective_pair_rejected_reason is None:
                left_line = p_left
                right_line = p_right
                left_meta = dict(p_left_meta or {})
                right_meta = dict(p_right_meta or {})
                projective_pair_applied = True
            else:
                left_line = left_line_before_projective
                right_line = right_line_before_projective
                left_meta = left_meta_before_projective
                right_meta = right_meta_before_projective
                projective_pair_meta["used"] = False
                projective_pair_meta["reason"] = f"rejected_{projective_pair_rejected_reason}"

    def _refine_bottom_line_with_sides(
        baseline_abc: Optional[Tuple[float, float, float]],
        left_abc: Optional[Tuple[float, float, float]],
        right_abc: Optional[Tuple[float, float, float]],
        baseline_meta_in: Dict[str, Any],
    ) -> Tuple[Optional[Tuple[float, float, float]], Dict[str, Any], Dict[str, Any]]:
        meta_refine: Dict[str, Any] = {"enabled": bool(_env_flag("BADC_COMPLETION_BASELINE_REFINE_WITH_SIDES", True))}
        if not meta_refine["enabled"]:
            meta_refine["reason"] = "disabled"
            return baseline_abc, dict(baseline_meta_in or {}), meta_refine
        if baseline_abc is None or left_abc is None or right_abc is None:
            meta_refine["reason"] = "missing_lines"
            return baseline_abc, dict(baseline_meta_in or {}), meta_refine
        if not lines_bottom:
            meta_refine["reason"] = "no_bottom_lines"
            return baseline_abc, dict(baseline_meta_in or {}), meta_refine

        floor_w_local = float(max(1.0, x1 - x0))
        floor_h_local = float(max(1.0, y1 - y0))
        min_seg_len_local = float(max(20.0, min_len * 0.55))
        min_span_local = float(
            max(
                120.0,
                _env_float("BADC_COMPLETION_BASELINE_REFINE_MIN_SPAN_PX", 0.18 * floor_w_local),
            )
        )
        min_cross_deg = float(max(4.0, _env_float("BADC_COMPLETION_BASELINE_REFINE_MIN_CROSS_DEG", 8.0)))
        min_improve_px_local = float(_env_float("BADC_COMPLETION_BASELINE_REFINE_MIN_IMPROVE_PX", 16.0))
        max_bottom_out_frac = float(max(0.6, _env_float("BADC_COMPLETION_BASELINE_REFINE_MAX_BOTTOM_OUT_FRAC", 2.4)))
        max_bottom_y = float(y1 + max_bottom_out_frac * floor_h_local)
        min_bottom_y = float(y0 - 0.25 * floor_h_local)
        support_floor = float(
            max(
                0.0,
                min(
                    1.0,
                    _env_float(
                        "BADC_COMPLETION_BASELINE_REFINE_MIN_SUPPORT",
                        max(0.0, 0.45 * baseline_min_support),
                    ),
                ),
            )
        )
        cand_preview_limit = int(max(4, _env_int("BADC_COMPLETION_BASELINE_REFINE_PREVIEW", 12)))

        line_ids_bottom_seq = list(line_ids_bottom) if line_ids_bottom is not None else [None] * len(lines_bottom)
        left_ang = _line_angle_from_abc(left_abc)
        right_ang = _line_angle_from_abc(right_abc)
        current_lb = _intersect_lines(left_abc, baseline_abc)
        current_rb = _intersect_lines(right_abc, baseline_abc)
        current_y_ref = _y_at_x(baseline_abc, x_ref)
        if current_y_ref is None and current_lb is not None and current_rb is not None:
            current_y_ref = float(0.5 * (float(current_lb[1]) + float(current_rb[1])))
        if current_y_ref is None:
            current_y_ref = float(y_ref)
        current_y_ref = float(current_y_ref)

        cand: list[Dict[str, Any]] = []
        for i, seg in enumerate(lines_bottom):
            x1s, y1s, x2s, y2s = [float(v) for v in seg]
            seg_len = float(math.hypot(float(x2s - x1s), float(y2s - y1s)))
            if seg_len < min_seg_len_local:
                continue
            line_i = _line_from_points((x1s, y1s), (x2s, y2s))
            p_lb = _intersect_lines(left_abc, line_i)
            p_rb = _intersect_lines(right_abc, line_i)
            if p_lb is None or p_rb is None or (not np.all(np.isfinite(p_lb))) or (not np.all(np.isfinite(p_rb))):
                continue
            if float(p_lb[0]) > float(p_rb[0]):
                p_lb, p_rb = p_rb, p_lb
            base_span = float(math.hypot(float(p_rb[0] - p_lb[0]), float(p_rb[1] - p_lb[1])))
            if base_span < min_span_local:
                continue
            y_low = float(max(float(p_lb[1]), float(p_rb[1])))
            y_high = float(min(float(p_lb[1]), float(p_rb[1])))
            if y_low > max_bottom_y or y_high < min_bottom_y:
                continue
            ang_i = _line_angle_from_abc(line_i)
            if ang_i is None:
                continue
            if left_ang is not None:
                cross_left = float(math.degrees(abs(_angle_distance(float(ang_i), float(left_ang)))))
                if cross_left < min_cross_deg:
                    continue
            else:
                cross_left = None
            if right_ang is not None:
                cross_right = float(math.degrees(abs(_angle_distance(float(ang_i), float(right_ang)))))
                if cross_right < min_cross_deg:
                    continue
            else:
                cross_right = None
            sup = _line_support((x1s, y1s, x2s, y2s), line_mask) if line_mask is not None else None
            if sup is not None and float(sup) < support_floor:
                continue
            y_ref_i = _y_at_x(line_i, x_ref)
            if y_ref_i is None or (not math.isfinite(float(y_ref_i))):
                y_ref_i = float(0.5 * (float(p_lb[1]) + float(p_rb[1])))
            y_ref_i = float(y_ref_i)
            lid_val = None
            if i < len(line_ids_bottom_seq):
                raw_id = line_ids_bottom_seq[i]
                lid_val = int(raw_id) if raw_id is not None else None
            cand.append(
                {
                    "line": line_i,
                    "line_id": lid_val,
                    "theta_deg": float(math.degrees(float(ang_i))),
                    "length": float(seg_len),
                    "support": float(sup) if sup is not None else None,
                    "lb": (float(p_lb[0]), float(p_lb[1])),
                    "rb": (float(p_rb[0]), float(p_rb[1])),
                    "span_px": float(base_span),
                    "y_ref": float(y_ref_i),
                    "y_low": float(y_low),
                    "y_high": float(y_high),
                    "cross_left_deg": float(cross_left) if cross_left is not None else None,
                    "cross_right_deg": float(cross_right) if cross_right is not None else None,
                }
            )

        if not cand:
            meta_refine["reason"] = "no_valid_candidates"
            return baseline_abc, dict(baseline_meta_in or {}), meta_refine

        gap_min_px = float(max(4.0, baseline_neighbor_gap_min_px))
        gap_max_px = float(max(gap_min_px + 2.0, baseline_neighbor_gap_max_px))
        neighbor_ang_tol_deg = float(max(8.0, _env_float("BADC_COMPLETION_BASELINE_REFINE_NEIGHBOR_ANG_TOL_DEG", 22.0)))
        neighbor_bonus = float(max(0.0, _env_float("BADC_COMPLETION_BASELINE_REFINE_NEIGHBOR_BONUS", 55.0)))
        best: Optional[Dict[str, Any]] = None
        best_key = float("-inf")
        for ci in cand:
            neighbor_found = False
            for cj in cand:
                if ci is cj:
                    continue
                dy = float(ci["y_ref"]) - float(cj["y_ref"])
                if dy < gap_min_px or dy > gap_max_px:
                    continue
                if abs(float(ci["theta_deg"]) - float(cj["theta_deg"])) > neighbor_ang_tol_deg:
                    continue
                neighbor_found = True
                break
            sup_i = float(ci["support"]) if ci.get("support") is not None else 0.0
            key = float(
                1.00 * float(ci["y_ref"])
                + 0.30 * float(ci["span_px"])
                + 45.0 * sup_i
                + 0.02 * float(ci["length"])
                + (neighbor_bonus if neighbor_found else 0.0)
            )
            ci["neighbor_parallel_found"] = bool(neighbor_found)
            ci["score"] = float(key)
            if key > best_key:
                best_key = float(key)
                best = ci

        preview_sorted = sorted(cand, key=lambda d: float(d["score"]), reverse=True)
        meta_refine["candidates"] = [
            {
                "line_id": c.get("line_id"),
                "y_ref": float(c.get("y_ref")),
                "span_px": float(c.get("span_px")),
                "support": float(c.get("support")) if c.get("support") is not None else None,
                "theta_deg": float(c.get("theta_deg")),
                "score": float(c.get("score", 0.0)),
                "neighbor_parallel_found": bool(c.get("neighbor_parallel_found")),
            }
            for c in preview_sorted[: min(cand_preview_limit, len(preview_sorted))]
        ]
        if best is None:
            meta_refine["reason"] = "no_best"
            return baseline_abc, dict(baseline_meta_in or {}), meta_refine
        improve = float(best["y_ref"]) - float(current_y_ref)
        meta_refine["current_y_ref"] = float(current_y_ref)
        meta_refine["best_y_ref"] = float(best["y_ref"])
        meta_refine["improve_px"] = float(improve)
        if improve < min_improve_px_local:
            meta_refine["reason"] = "insufficient_improve"
            return baseline_abc, dict(baseline_meta_in or {}), meta_refine

        out_meta = {
            "line_id": best.get("line_id"),
            "theta_deg": float(best["theta_deg"]),
            "length": float(best["length"]),
            "support": float(best["support"]) if best.get("support") is not None else None,
            "score": float(best["score"]),
            "picked_by_completion_baseline_refine": True,
            "neighbor_parallel_found": bool(best.get("neighbor_parallel_found")),
            "baseline_refine_improve_px": float(improve),
        }
        meta_refine["reason"] = "refined"
        meta_refine["used"] = True
        meta_refine["line_id"] = best.get("line_id")
        return best["line"], out_meta, meta_refine

    baseline_refine_meta: Dict[str, Any] = {}
    if bottom_line is not None:
        bottom_line_refined, bottom_meta_refined, baseline_refine_meta = _refine_bottom_line_with_sides(
            bottom_line,
            left_line,
            right_line,
            bottom_meta,
        )
        if baseline_refine_meta.get("used"):
            bottom_line = bottom_line_refined
            bottom_meta = bottom_meta_refined
            if sideline_projective_pick and bottom_line is not None:
                p_left2, p_left_meta2, p_right2, p_right_meta2, p_meta2 = _pick_sideline_pair_projective(bottom_line, top_line)
                if p_left2 is not None and p_right2 is not None:
                    left_line = p_left2
                    right_line = p_right2
                    left_meta = dict(p_left_meta2 or {})
                    right_meta = dict(p_right_meta2 or {})
                    projective_pair_applied = True
                    projective_pair_rejected_reason = None
                    projective_pair_meta = dict(p_meta2 or {})
                    projective_pair_meta["second_pass_after_baseline_refine"] = True
    meta: Dict[str, Any] = {
        "baseline_line": bottom_line,
        "left_sideline": left_line,
        "right_sideline": right_line,
        "top_line": top_line,
        "baseline_meta": bottom_meta,
        "left_sideline_meta": left_meta,
        "right_sideline_meta": right_meta,
        "top_line_meta": top_meta,
        "sideline_perp_tol_deg": float(sideline_perp_tol_deg),
        "sideline_perp_filter": bool(sideline_perp_filter),
        "sideline_perp_repick": bool(sideline_perp_repick),
        "sideline_snap_tol_deg": float(sideline_snap_tol_deg),
        "sideline_projective_pick": bool(sideline_projective_pick),
        "sideline_perp_min_keep": int(sideline_perp_min_keep),
        "sideline_rep_top_k": int(sideline_rep_top_k),
        "sideline_bad_perp_deg": float(sideline_bad_perp_deg),
        "sideline_lr_parallel_tol_deg": float(sideline_lr_parallel_tol_deg),
        "sideline_repick_perp_tol_deg": float(sideline_repick_perp_tol_deg),
        "sideline_right_strict_perp_deg": float(sideline_right_strict_perp_deg),
        "sideline_right_strict_parallel_deg": float(sideline_right_strict_parallel_deg),
        "sideline_right_min_sep_px": float(sideline_right_min_sep_px),
        "sideline_left_perp_err_deg": sideline_left_perp_err_deg,
        "sideline_right_perp_err_deg": sideline_right_perp_err_deg,
        "sideline_lr_nonparallel": bool(sideline_lr_nonparallel),
        "sideline_need_repick": bool(sideline_need_repick),
        "sideline_angle_raw_deg": float(math.degrees(sideline_angle_raw))
        if sideline_angle_raw is not None
        else None,
        "baseline_angle_lines_deg": float(math.degrees(baseline_angle_lines))
        if baseline_angle_lines is not None
        else None,
        "baseline_angle_quad_deg": float(math.degrees(baseline_angle_quad))
        if baseline_angle_quad is not None
        else None,
        "baseline_angle_source": str(baseline_angle_source),
        "baseline_ref_disagree_deg": float(baseline_ref_disagree_deg),
        "baseline_pick_angle_deg": float(baseline_pick_angle_deg),
        "sideline_angle_used_deg": float(math.degrees(sideline_angle)) if sideline_angle is not None else None,
        "sideline_angle_perp_deg": float(math.degrees(sideline_angle_perp))
        if sideline_angle_perp is not None
        else None,
        "sideline_snapped_to_perp": bool(sideline_snapped_to_perp),
        "sideline_min_support": float(sideline_min_support),
        "baseline_min_support": float(baseline_min_support),
        "sideline_candidates_before": int(len(lines_left)),
        "sideline_candidates_after": int(len(lines_left_use)),
        "sideline_fallback_used": bool(sideline_fallback_used),
        "sideline_left_right_too_close": bool(_lines_too_close(left_line, right_line)),
        "sideline_projective_pair_meta": projective_pair_meta if isinstance(projective_pair_meta, dict) else None,
        "sideline_projective_pair_applied": bool(projective_pair_applied),
        "sideline_projective_pair_rejected_reason": projective_pair_rejected_reason,
        "baseline_refine_meta": baseline_refine_meta if isinstance(baseline_refine_meta, dict) else None,
    }
    if component_id is not None:
        meta["component_id"] = int(component_id)
    if left_line is None or right_line is None or bottom_line is None:
        return ordered, False, meta
    lb = _intersect_lines(left_line, bottom_line)
    rb = _intersect_lines(right_line, bottom_line)
    if lb is None or rb is None:
        return ordered, False, meta
    if float(lb[0]) > float(rb[0]):
        lb, rb = rb, lb
    completed = ordered.copy()
    completed[0] = lb
    completed[1] = rb
    if top_line is not None:
        top_ok = True
        if top_meta:
            if top_meta.get("length") is not None and float(top_meta["length"]) < min_len:
                top_ok = False
            if top_meta.get("support") is not None and float(top_meta["support"]) < 0.2:
                top_ok = False
        if not top_ok:
            return completed, True, meta
        lt = _intersect_lines(left_line, top_line)
        rt = _intersect_lines(right_line, top_line)
        if lt is not None and rt is not None:
            if float(lt[0]) > float(rt[0]):
                lt, rt = rt, lt
            completed[2] = rt
            completed[3] = lt
            meta["completed_top"] = True
    return completed, True, meta


def _refine_bottom_baseline_postfit(
    ordered_lb_rb_rt_lt: np.ndarray,
    lines_bottom: Sequence[Tuple[float, float, float, float]],
    lines_all: Sequence[Tuple[float, float, float, float]],
    floor_bbox: Tuple[int, int, int, int],
    line_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, bool, Dict[str, Any]]:
    """
    Post-fit baseline correction for "baseline vs doubles long service line" confusion.
    Rules implemented:
    1) bottom line disambiguation with nearby parallel pair
    2) explicit 0.76m spacing prior in model coordinates
    3) if there is a stronger stable line below current bottom, swap to lower one
    """
    enabled = _env_flag("BADC_POSTFIT_BASELINE_VERIFY", True)
    ordered = np.asarray(ordered_lb_rb_rt_lt, dtype=np.float32).reshape(4, 2)
    meta: Dict[str, Any] = {
        "enabled": bool(enabled),
        "used": False,
        "reason": None,
    }
    if not enabled:
        return ordered, False, meta
    if (not lines_bottom and not lines_all) or ordered.shape != (4, 2):
        meta["reason"] = "no_lines"
        return ordered, False, meta

    angle_tol_deg = float(_env_float("BADC_POSTFIT_BASELINE_ANGLE_TOL_DEG", 10.0))
    min_seg_len_m = float(_env_float("BADC_POSTFIT_BASELINE_MIN_SEG_M", 0.65))
    target_gap_m = float(_env_float("BADC_POSTFIT_BASELINE_TARGET_GAP_M", 0.76))
    gap_tol_m = float(_env_float("BADC_POSTFIT_BASELINE_GAP_TOL_M", 0.22))
    below_min_m = float(_env_float("BADC_POSTFIT_BASELINE_BELOW_MIN_M", 0.30))
    below_max_m = float(_env_float("BADC_POSTFIT_BASELINE_BELOW_MAX_M", 1.30))
    min_support = float(_env_float("BADC_POSTFIT_BASELINE_MIN_SUPPORT", 0.18))
    min_improve_px = float(_env_float("BADC_POSTFIT_BASELINE_MIN_IMPROVE_PX", 3.0))
    include_all = bool(_env_flag("BADC_POSTFIT_BASELINE_INCLUDE_ALL", True))
    img_angle_tol_deg = float(_env_float("BADC_POSTFIT_BASELINE_IMG_ANGLE_TOL_DEG", 14.0))
    img_pair_gap_tol_ratio = float(_env_float("BADC_POSTFIT_BASELINE_IMG_GAP_TOL_RATIO", 0.50))
    img_min_span_frac = float(_env_float("BADC_POSTFIT_BASELINE_IMG_MIN_SPAN_FRAC", 0.12))
    img_bottom_support_min = float(_env_float("BADC_POSTFIT_BASELINE_IMG_BOTTOM_SUPPORT_MIN", 0.15))
    img_neighbor_gap_min_ratio = float(_env_float("BADC_POSTFIT_BASELINE_IMG_NEIGHBOR_GAP_MIN_RATIO", 0.20))
    img_neighbor_gap_max_ratio = float(_env_float("BADC_POSTFIT_BASELINE_IMG_NEIGHBOR_GAP_MAX_RATIO", 1.60))

    model_corners = get_bwf_corners().astype(np.float32)
    try:
        H_i2m = cv2.getPerspectiveTransform(ordered.astype(np.float32), model_corners)
    except Exception:
        meta["reason"] = "bad_h_i2m"
        return ordered, False, meta
    if H_i2m is None or not np.all(np.isfinite(H_i2m)):
        meta["reason"] = "bad_h_i2m"
        return ordered, False, meta

    src_lines: list[Tuple[float, float, float, float]] = []
    src_lines.extend([(float(x1), float(y1), float(x2), float(y2)) for x1, y1, x2, y2 in lines_bottom])
    if include_all:
        src_lines.extend([(float(x1), float(y1), float(x2), float(y2)) for x1, y1, x2, y2 in lines_all])
    if not src_lines:
        meta["reason"] = "no_source_lines"
        return ordered, False, meta

    # Deduplicate by rounded endpoints (order-independent).
    dedup: Dict[Tuple[int, int, int, int], Tuple[float, float, float, float]] = {}
    for x1, y1, x2, y2 in src_lines:
        p = (int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)))
        q = (int(round(x2)), int(round(y2)), int(round(x1)), int(round(y1)))
        key = p if p <= q else q
        dedup[key] = (x1, y1, x2, y2)
    src_lines = list(dedup.values())

    tan_tol = math.tan(math.radians(max(1.0, angle_tol_deg)))
    cands: list[Dict[str, Any]] = []
    floor_w = float(max(1.0, float(floor_bbox[2] - floor_bbox[0])))
    x_ref = float(0.5 * (float(floor_bbox[0]) + float(floor_bbox[2])))
    for x1, y1, x2, y2 in src_lines:
        pts = np.array([[float(x1), float(y1)], [float(x2), float(y2)]], dtype=np.float32)
        pts_m = project_points(H_i2m, pts)
        if pts_m.shape != (2, 2) or not np.all(np.isfinite(pts_m)):
            continue
        dx_m = float(pts_m[1, 0] - pts_m[0, 0])
        dy_m = float(pts_m[1, 1] - pts_m[0, 1])
        seg_len_m = float(math.hypot(dx_m, dy_m))
        if seg_len_m < min_seg_len_m:
            continue
        if abs(dy_m) > tan_tol * max(abs(dx_m), 1e-6):
            continue
        support = 0.5
        if line_mask is not None and line_mask.size > 0:
            support = float(_line_support((x1, y1, x2, y2), line_mask, samples=48))
        cands.append(
            {
                "seg": (x1, y1, x2, y2),
                "line": _line_from_points((x1, y1), (x2, y2)),
                "y_m": float(0.5 * (pts_m[0, 1] + pts_m[1, 1])),
                "x_span_m": float(abs(dx_m)),
                "seg_len_m": float(seg_len_m),
                "x_span_img": float(abs(x2 - x1)),
                "seg_len_img": float(math.hypot(float(x2 - x1), float(y2 - y1))),
                "mid_y_img": float(0.5 * (y1 + y2)),
                "support": float(support),
                "score": float(max(0.0, support) * min(1.0, abs(dx_m) / 6.10)),
            }
        )
    if not cands:
        meta["reason"] = "no_horizontal_candidates"
        return ordered, False, meta

    cands.sort(key=lambda d: float(d["y_m"]))
    meta["candidates"] = [
        {
            "y_m": float(c["y_m"]),
            "mid_y_img": float(c["mid_y_img"]),
            "support": float(c["support"]),
            "score": float(c["score"]),
            "len_m": float(c["seg_len_m"]),
        }
        for c in cands[:16]
    ]
    current_bottom_mid = float(0.5 * (ordered[0, 1] + ordered[1, 1]))
    cur = min(cands, key=lambda d: abs(float(d["y_m"])))
    current_ref_mid = float(cur["mid_y_img"])
    cur_line = _line_from_points((float(ordered[0, 0]), float(ordered[0, 1])), (float(ordered[1, 0]), float(ordered[1, 1])))
    cur_ang = float(np.mod(math.atan2(float(ordered[1, 1] - ordered[0, 1]), float(ordered[1, 0] - ordered[0, 0])), math.pi))

    H_m2i = cv2.getPerspectiveTransform(model_corners, ordered.astype(np.float32))
    model_w = float(np.max(model_corners[:, 0]) - np.min(model_corners[:, 0]))
    p_gap = project_points(
        H_m2i,
        np.array([[0.5 * model_w, 0.0], [0.5 * model_w, float(target_gap_m)]], dtype=np.float32),
    )
    expected_gap_px = float(abs(float(p_gap[1, 1]) - float(p_gap[0, 1]))) if p_gap.shape == (2, 2) else 0.0
    expected_gap_px = max(6.0, expected_gap_px)
    # Clamp expected gap to avoid over-trusting a poor current homography.
    max_reasonable_gap_px = float(max(24.0, 0.26 * float(max(1, floor_bbox[3] - floor_bbox[1]))))
    expected_gap_px_eff = float(min(expected_gap_px, max_reasonable_gap_px))
    gap_tol_px = max(6.0, img_pair_gap_tol_ratio * expected_gap_px_eff)
    min_span_img_px = float(max(24.0, img_min_span_frac * floor_w))

    chosen: Optional[Dict[str, Any]] = None
    choose_reason = None

    # Rule 1 (image-space hardening):
    # choose the lowest strong/long baseline-like candidate; if a close parallel
    # neighbor exists above it (service line distance range), force the lower line.
    img_baseline_cands: list[Tuple[float, Dict[str, Any]]] = []
    y_cur_ref = _y_at_x(cur_line, x_ref) if cur_line is not None else None
    if y_cur_ref is None:
        y_cur_ref = current_ref_mid
    for c in cands:
        x1, y1, x2, y2 = c["seg"]
        ang = float(np.mod(math.atan2(float(y2 - y1), float(x2 - x1)), math.pi))
        if _angle_distance(ang, cur_ang) > math.radians(max(6.0, img_angle_tol_deg)):
            continue
        if float(c["support"]) < img_bottom_support_min:
            continue
        if float(c["x_span_img"]) < min_span_img_px:
            continue
        y_ref = _y_at_x(c["line"], x_ref)
        if y_ref is None:
            y_ref = float(c["mid_y_img"])
        if float(y_ref) < float(y_cur_ref) + min_improve_px:
            continue
        img_baseline_cands.append((float(y_ref), c))

    if img_baseline_cands:
        img_baseline_cands.sort(key=lambda t: t[0], reverse=True)
        bottommost_ref_y, bottommost = img_baseline_cands[0]
        neighbor_ok = False
        gap_min_px = float(max(3.0, img_neighbor_gap_min_ratio * expected_gap_px_eff))
        gap_max_px = float(max(gap_min_px + 3.0, img_neighbor_gap_max_ratio * expected_gap_px_eff))
        for y_ref, c in img_baseline_cands[1:]:
            dy_px = float(bottommost_ref_y - y_ref)
            if dy_px < gap_min_px or dy_px > gap_max_px:
                continue
            # keep close parallel check around current baseline direction
            x1, y1, x2, y2 = c["seg"]
            ang = float(np.mod(math.atan2(float(y2 - y1), float(x2 - x1)), math.pi))
            if _angle_distance(ang, cur_ang) <= math.radians(max(6.0, img_angle_tol_deg)):
                neighbor_ok = True
                break
        if neighbor_ok:
            chosen = bottommost
            choose_reason = "img_bottommost_parallel_pair"
        elif float(bottommost_ref_y) >= float(y_cur_ref) + max(min_improve_px, 0.35 * expected_gap_px_eff):
            # Rule 3 extension: if a stable strong line sits clearly below current baseline,
            # accept it even without a clear 0.76m pair.
            chosen = bottommost
            choose_reason = "img_bottommost_strong_below"

    # Rule 2: explicit 0.76m pair near current bottom neighborhood.
    if chosen is None:
        best_pair = None
        best_pair_score = float("-inf")
        for i in range(len(cands)):
            ci = cands[i]
            for j in range(i + 1, len(cands)):
                cj = cands[j]
                dy = abs(float(ci["y_m"]) - float(cj["y_m"]))
                if abs(dy - target_gap_m) > gap_tol_m:
                    continue
                if min(float(ci["support"]), float(cj["support"])) < min_support:
                    continue
                if min(abs(float(ci["y_m"])), abs(float(cj["y_m"]))) > 0.55:
                    continue
                # Baseline should be lower in image (larger y pixel) among the near-parallel pair.
                baseline_cand = ci if float(ci["mid_y_img"]) >= float(cj["mid_y_img"]) else cj
                pair_score = float(
                    min(float(ci["score"]), float(cj["score"]))
                    + 0.25 * math.exp(-abs(dy - target_gap_m) / max(1e-6, 0.18))
                )
                if pair_score > best_pair_score:
                    best_pair_score = pair_score
                    best_pair = baseline_cand
        if best_pair is not None and float(best_pair["mid_y_img"]) >= current_ref_mid - 1.0:
            chosen = best_pair
            choose_reason = "pair_0.76m"

    # Rule 3: a stable strong line exists below current baseline.
    if chosen is None:
        below = [
            c
            for c in cands
            if (-below_max_m <= float(c["y_m"]) <= -below_min_m) and float(c["support"]) >= min_support
        ]
        if below:
            below.sort(
                key=lambda c: (
                    float(c["score"]) + 0.15 * math.exp(-abs(abs(float(c["y_m"])) - target_gap_m) / max(1e-6, 0.18)),
                    float(c["mid_y_img"]),
                ),
                reverse=True,
            )
            cand = below[0]
            if float(cand["mid_y_img"]) >= current_ref_mid + min_improve_px:
                chosen = cand
                choose_reason = "strong_line_below_bottom"

    # Image-space fallback: parallel neighbor below current bottom with expected ~0.76m pixel gap.
    if chosen is None:
        best_img = None
        best_img_score = float("-inf")
        for c in cands:
            x1, y1, x2, y2 = c["seg"]
            ang = float(np.mod(math.atan2(float(y2 - y1), float(x2 - x1)), math.pi))
            if _angle_distance(ang, cur_ang) > math.radians(max(6.0, img_angle_tol_deg)):
                continue
            yc = _y_at_x(c["line"], x_ref)
            if yc is None:
                yc = float(c["mid_y_img"])
            dy_px = float(yc - y_cur_ref)
            if dy_px < min_improve_px:
                continue
            if abs(dy_px - expected_gap_px_eff) > gap_tol_px:
                continue
            sup = float(c["support"])
            if sup < min_support:
                continue
            score_img = float(
                sup + 0.35 * math.exp(-abs(dy_px - expected_gap_px_eff) / max(1e-6, 0.4 * expected_gap_px_eff))
            )
            if score_img > best_img_score:
                best_img_score = score_img
                best_img = c
        if best_img is not None:
            chosen = best_img
            choose_reason = "img_parallel_below_expected_gap"
            meta["expected_gap_px"] = float(expected_gap_px)
            meta["expected_gap_px_eff"] = float(expected_gap_px_eff)
            meta["gap_tol_px"] = float(gap_tol_px)

    if chosen is None:
        meta["reason"] = "no_swap_evidence"
        meta["num_candidates"] = int(len(cands))
        meta["current_bottom_mid_y"] = float(current_bottom_mid)
        meta["current_near_y_model"] = float(cur["y_m"])
        return ordered, False, meta

    left_line = _line_from_points((float(ordered[3, 0]), float(ordered[3, 1])), (float(ordered[0, 0]), float(ordered[0, 1])))
    right_line = _line_from_points((float(ordered[2, 0]), float(ordered[2, 1])), (float(ordered[1, 0]), float(ordered[1, 1])))
    baseline_line = chosen["line"]
    lb = _intersect_lines(left_line, baseline_line)
    rb = _intersect_lines(right_line, baseline_line)
    if lb is None or rb is None or not np.all(np.isfinite(lb)) or not np.all(np.isfinite(rb)):
        meta["reason"] = "line_intersection_failed"
        return ordered, False, meta
    if float(lb[0]) > float(rb[0]):
        lb, rb = rb, lb
    refined = ordered.copy()
    refined[0] = lb
    refined[1] = rb

    if not _is_convex_quad(refined):
        meta["reason"] = "non_convex_after_swap"
        return ordered, False, meta
    ok, gate_reason, _gate_metrics = _passes_geom_gates(refined, floor_bbox)
    if not ok:
        meta["reason"] = f"geom_gate_{gate_reason}"
        return ordered, False, meta

    meta.update(
        {
            "used": True,
            "reason": str(choose_reason),
            "num_candidates": int(len(cands)),
            "chosen_y_model": float(chosen["y_m"]),
            "chosen_mid_y_img": float(chosen["mid_y_img"]),
            "chosen_support": float(chosen["support"]),
            "current_bottom_mid_y": float(current_bottom_mid),
            "current_ref_mid_y": float(current_ref_mid),
            "current_near_y_model": float(cur["y_m"]),
            "target_gap_m": float(target_gap_m),
            "gap_tol_m": float(gap_tol_m),
        }
    )
    return refined, True, meta


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


def _project_named_line_segments(H: np.ndarray) -> Dict[str, Tuple[float, float, float, float]]:
    out: Dict[str, Tuple[float, float, float, float]] = {}
    for ln in get_bwf_lines():
        pts = np.array([ln.p1, ln.p2], dtype=np.float32)
        uv = project_points(H, pts)
        if uv.shape != (2, 2) or not np.all(np.isfinite(uv)):
            continue
        out[str(ln.name)] = (float(uv[0, 0]), float(uv[0, 1]), float(uv[1, 0]), float(uv[1, 1]))
    return out


def _court_role_prior_bonus(
    H: np.ndarray,
    mask: Optional[np.ndarray],
) -> Tuple[float, Dict[str, float]]:
    """
    Optional role prior:
    - near baseline should have at least comparable support to near service lines
    - near baseline should project lower than near short service line in image (y larger)
    """
    if not _env_flag("BADC_ROLE_PRIOR_ENABLE", False):
        return 0.0, {}
    if mask is None or mask.size == 0:
        return 0.0, {}
    named = _project_named_line_segments(H)
    if not named:
        return 0.0, {}
    if "baseline_bottom" not in named or "short_near" not in named or "long_near_d" not in named:
        return 0.0, {}
    base_seg = named["baseline_bottom"]
    short_seg = named["short_near"]
    long_seg = named["long_near_d"]

    s_base = float(_line_support(base_seg, mask, samples=72))
    s_short = float(_line_support(short_seg, mask, samples=56))
    s_long = float(_line_support(long_seg, mask, samples=56))
    s_ref = max(s_short, s_long)
    margin = s_base - s_ref
    min_margin = float(_env_float("BADC_ROLE_PRIOR_MIN_MARGIN", 0.02))
    w_margin = float(_env_float("BADC_ROLE_PRIOR_W", 8.0))
    margin_term = w_margin * (margin - min_margin)

    by = 0.5 * (base_seg[1] + base_seg[3])
    sy = 0.5 * (short_seg[1] + short_seg[3])
    orient_good = float(by > sy)
    w_orient = float(_env_float("BADC_ROLE_PRIOR_ORIENT_W", 2.0))
    orient_term = w_orient if orient_good > 0.5 else (-w_orient)

    bonus = float(margin_term + orient_term)
    cap = float(_env_float("BADC_ROLE_PRIOR_CAP", 10.0))
    bonus = float(np.clip(bonus, -cap, cap))
    meta = {
        "baseline_support": float(s_base),
        "short_near_support": float(s_short),
        "long_near_support": float(s_long),
        "support_margin": float(margin),
        "orient_good": float(orient_good),
        "bonus": float(bonus),
    }
    return bonus, meta


def _unique_axis_targets(
    pairs: Sequence[Tuple[float, float]],
    tol: float = 1e-4,
) -> list[Tuple[float, float]]:
    if not pairs:
        return []
    items = sorted([(float(p), float(max(0.0, s))) for p, s in pairs], key=lambda t: t[0])
    out: list[Tuple[float, float]] = []
    for pos, span in items:
        if not out:
            out.append((pos, span))
            continue
        lp, ls = out[-1]
        if abs(pos - lp) <= float(tol):
            w0 = max(ls, 1e-6)
            w1 = max(span, 1e-6)
            mp = (lp * w0 + pos * w1) / (w0 + w1)
            out[-1] = (float(mp), float(max(ls, span)))
        else:
            out.append((pos, span))
    return out


def _model_axis_targets() -> Tuple[list[Tuple[float, float]], list[Tuple[float, float]], float, float]:
    """
    Returns:
    - horizontal targets: [(y_pos_m, expected_span_m), ...]
    - vertical targets:   [(x_pos_m, expected_span_m), ...]
    - model width (m), model length (m)
    """
    h_targets: list[Tuple[float, float]] = []
    v_targets: list[Tuple[float, float]] = []
    for ln in get_bwf_lines():
        x1, y1 = float(ln.p1[0]), float(ln.p1[1])
        x2, y2 = float(ln.p2[0]), float(ln.p2[1])
        if abs(y1 - y2) <= 1e-6:
            h_targets.append((0.5 * (y1 + y2), abs(x2 - x1)))
        elif abs(x1 - x2) <= 1e-6:
            v_targets.append((0.5 * (x1 + x2), abs(y2 - y1)))
    h_targets = _unique_axis_targets(h_targets)
    v_targets = _unique_axis_targets(v_targets)
    corners = get_bwf_corners()
    model_w = float(np.max(corners[:, 0]) - np.min(corners[:, 0]))
    model_l = float(np.max(corners[:, 1]) - np.min(corners[:, 1]))
    return h_targets, v_targets, model_w, model_l


def _parse_float_csv(text: str, default: Sequence[float]) -> list[float]:
    raw = str(text or "").strip()
    if not raw:
        return [float(v) for v in default]
    vals: list[float] = []
    for part in raw.split(","):
        p = part.strip()
        if not p:
            continue
        try:
            vals.append(float(p))
        except Exception:
            continue
    if not vals:
        return [float(v) for v in default]
    return vals


def _weighted_percentile(values: np.ndarray, weights: np.ndarray, q: float) -> Optional[float]:
    vals = np.asarray(values, dtype=np.float32).reshape(-1)
    wts = np.asarray(weights, dtype=np.float32).reshape(-1)
    if vals.size == 0 or wts.size == 0 or vals.size != wts.size:
        return None
    finite = np.isfinite(vals) & np.isfinite(wts)
    if not np.any(finite):
        return None
    vals = vals[finite]
    wts = wts[finite]
    if vals.size == 0:
        return None
    wts = np.clip(wts, 0.0, None)
    if float(np.sum(wts)) <= 1e-6:
        return float(np.percentile(vals, float(q)))
    order = np.argsort(vals)
    vals = vals[order]
    wts = wts[order]
    cdf = np.cumsum(wts)
    target = float(np.clip(q, 0.0, 100.0)) / 100.0 * float(cdf[-1])
    idx = int(np.searchsorted(cdf, target, side="left"))
    idx = max(0, min(idx, int(vals.size - 1)))
    return float(vals[idx])


def _compress_axis_observations(
    obs: Sequence[Tuple[float, float]],
    tol: float,
) -> list[Tuple[float, float]]:
    if not obs:
        return []
    items = sorted([(float(p), float(max(0.0, ln))) for p, ln in obs], key=lambda t: t[0])
    out: list[Tuple[float, float]] = []
    for pos, ln in items:
        if not out:
            out.append((pos, ln))
            continue
        lp, ll = out[-1]
        if abs(pos - lp) <= float(max(1e-6, tol)):
            out[-1] = (0.5 * (lp + pos), max(ll, ln))
        else:
            out.append((pos, ln))
    return out


def _match_axis_target_support(
    obs: Sequence[Tuple[float, float]],
    target_pos: float,
    target_span: float,
    tol_pos: float,
) -> Tuple[float, Optional[float], float]:
    best_rank = float("-inf")
    best_score = 0.0
    best_pos: Optional[float] = None
    best_raw_support = 0.0
    for pos, ln in obs:
        d = abs(float(pos) - float(target_pos))
        if d > float(max(1e-6, tol_pos)):
            continue
        raw_support = float(ln) / float(max(target_span, 1e-6))
        score = min(1.0, raw_support)
        rank = float(score - 0.25 * (d / float(max(1e-6, tol_pos))))
        if rank > best_rank:
            best_rank = rank
            best_score = float(score)
            best_raw_support = float(raw_support)
            best_pos = float(pos)
    return float(best_score), best_pos, float(best_raw_support)


def _evaluate_court_ratio_structure(
    quad_img_lb_rb_rt_lt: np.ndarray,
    lines_img: Sequence[Tuple[float, float, float, float]],
    img_w: int,
    img_h: int,
    angle_tol_deg: float = 12.0,
    tol_h_m: float = 0.24,
    tol_v_m: float = 0.22,
    min_seg_len_m: float = 0.55,
    model_margin_m: float = 1.2,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "score": 0.0,
        "h_present": 0,
        "v_present": 0,
        "h_expected": 0,
        "v_expected": 0,
        "ratio_est": None,
        "ratio_err": None,
        "ratio_score": 0.0,
        "h_score": 0.0,
        "v_score": 0.0,
        "h_geom_score": 0.0,
        "v_geom_score": 0.0,
        "key_score": 0.0,
    }
    if quad_img_lb_rb_rt_lt is None or quad_img_lb_rb_rt_lt.shape != (4, 2):
        return out
    if not lines_img:
        return out
    model_corners = get_bwf_corners().astype(np.float32)
    quad = quad_img_lb_rb_rt_lt.astype(np.float32)
    if not np.all(np.isfinite(quad)):
        return out
    try:
        H_i2m = cv2.getPerspectiveTransform(quad, model_corners)
    except Exception:
        return out
    if H_i2m is None or not np.all(np.isfinite(H_i2m)):
        return out

    h_targets, v_targets, model_w, model_l = _model_axis_targets()
    if not h_targets or not v_targets:
        return out
    out["h_expected"] = int(len(h_targets))
    out["v_expected"] = int(len(v_targets))

    tan_tol = math.tan(math.radians(float(max(1.0, angle_tol_deg))))
    h_obs: list[Tuple[float, float]] = []
    v_obs: list[Tuple[float, float]] = []
    m_margin = float(max(0.0, model_margin_m))
    for x1, y1, x2, y2 in lines_img:
        pts = np.array([[float(x1), float(y1)], [float(x2), float(y2)]], dtype=np.float32)
        pts_m = project_points(H_i2m, pts)
        if pts_m.shape != (2, 2) or not np.all(np.isfinite(pts_m)):
            continue
        mx = float(0.5 * (pts_m[0, 0] + pts_m[1, 0]))
        my = float(0.5 * (pts_m[0, 1] + pts_m[1, 1]))
        if (
            mx < -m_margin
            or mx > float(model_w + m_margin)
            or my < -m_margin
            or my > float(model_l + m_margin)
        ):
            continue
        dx = float(pts_m[1, 0] - pts_m[0, 0])
        dy = float(pts_m[1, 1] - pts_m[0, 1])
        seg_len = float(math.hypot(dx, dy))
        if seg_len < float(max(0.05, min_seg_len_m)):
            continue
        adx = abs(dx)
        ady = abs(dy)
        if ady <= tan_tol * max(adx, 1e-6):
            h_obs.append((my, seg_len))
        elif adx <= tan_tol * max(ady, 1e-6):
            v_obs.append((mx, seg_len))

    h_obs = _compress_axis_observations(h_obs, tol=max(0.05, tol_h_m * 0.5))
    v_obs = _compress_axis_observations(v_obs, tol=max(0.05, tol_v_m * 0.5))
    if not h_obs and not v_obs:
        return out

    h_scores: list[float] = []
    h_pos_found: list[Optional[float]] = []
    h_support_raw: list[float] = []
    for y_exp, span_exp in h_targets:
        s, pos, raw_s = _match_axis_target_support(h_obs, y_exp, span_exp, tol_h_m)
        h_scores.append(float(s))
        h_pos_found.append(pos)
        h_support_raw.append(float(raw_s))
    v_scores: list[float] = []
    v_pos_found: list[Optional[float]] = []
    v_support_raw: list[float] = []
    for x_exp, span_exp in v_targets:
        s, pos, raw_s = _match_axis_target_support(v_obs, x_exp, span_exp, tol_v_m)
        v_scores.append(float(s))
        v_pos_found.append(pos)
        v_support_raw.append(float(raw_s))

    present_thr = 0.22
    h_present = int(sum(1 for s in h_scores if s >= present_thr))
    v_present = int(sum(1 for s in v_scores if s >= present_thr))
    h_score = float(np.mean(h_scores)) if h_scores else 0.0
    v_score = float(np.mean(v_scores)) if v_scores else 0.0

    h_errs: list[float] = []
    for (y_exp, _), pos in zip(h_targets, h_pos_found):
        if pos is None:
            h_errs.append(1.0)
        else:
            h_errs.append(float(abs(pos - y_exp) / max(model_l, 1e-6)))
    v_errs: list[float] = []
    for (x_exp, _), pos in zip(v_targets, v_pos_found):
        if pos is None:
            v_errs.append(1.0)
        else:
            v_errs.append(float(abs(pos - x_exp) / max(model_w, 1e-6)))
    h_err_mean = float(np.mean(h_errs)) if h_errs else 1.0
    v_err_mean = float(np.mean(v_errs)) if v_errs else 1.0
    h_geom_score = float(math.exp(-h_err_mean / 0.07))
    v_geom_score = float(math.exp(-v_err_mean / 0.10))

    ratio_est: Optional[float] = None
    ratio_err: Optional[float] = None
    ratio_score = 0.0
    y0 = h_pos_found[0] if h_pos_found else None
    y1 = h_pos_found[-1] if h_pos_found else None
    x0 = v_pos_found[0] if v_pos_found else None
    x1 = v_pos_found[-1] if v_pos_found else None
    if y0 is not None and y1 is not None and x0 is not None and x1 is not None:
        len_est = float(abs(y1 - y0))
        wid_est = float(abs(x1 - x0))
        if len_est > 1e-6 and wid_est > 1e-6:
            ratio_est = float(len_est / wid_est)
            ratio_target = float(model_l / max(model_w, 1e-6))
            ratio_err = float(abs(ratio_est - ratio_target) / max(ratio_target, 1e-6))
            ratio_score = float(math.exp(-ratio_err / 0.12))

    key_hits = 0
    if h_scores:
        key_hits += int(h_scores[0] >= present_thr)
        key_hits += int(h_scores[-1] >= present_thr)
    if v_scores:
        key_hits += int(v_scores[0] >= present_thr)
        key_hits += int(v_scores[-1] >= present_thr)
    key_score = float(key_hits / 4.0)

    coverage_score = float(0.5 * h_score + 0.5 * v_score)
    geom_score = float(0.5 * h_geom_score + 0.5 * v_geom_score)
    score = float(0.55 * coverage_score + 0.20 * geom_score + 0.15 * ratio_score + 0.10 * key_score)
    score = float(np.clip(score, 0.0, 1.0))

    out.update(
        {
            "score": float(score),
            "h_present": int(h_present),
            "v_present": int(v_present),
            "h_score": float(h_score),
            "v_score": float(v_score),
            "h_geom_score": float(h_geom_score),
            "v_geom_score": float(v_geom_score),
            "ratio_est": float(ratio_est) if ratio_est is not None else None,
            "ratio_err": float(ratio_err) if ratio_err is not None else None,
            "ratio_score": float(ratio_score),
            "key_score": float(key_score),
            "h_obs_count": int(len(h_obs)),
            "v_obs_count": int(len(v_obs)),
            "h_support_raw": [float(v) for v in h_support_raw],
            "v_support_raw": [float(v) for v in v_support_raw],
        }
    )
    return out


def _scale_quad_about_center(quad_xy: np.ndarray, scale: float) -> np.ndarray:
    c = np.mean(quad_xy, axis=0, keepdims=True)
    out = c + (quad_xy - c) * float(scale)
    return out.astype(np.float32)


def _refine_quad_by_structure_scale(
    ordered: np.ndarray,
    lines_all: Sequence[Tuple[float, float, float, float]],
    floor_bbox: Tuple[int, int, int, int],
    img_w: int,
    img_h: int,
    scales: Sequence[float],
    min_gain: float,
    angle_tol_deg: float,
    tol_h_m: float,
    tol_v_m: float,
    min_seg_len_m: float,
    min_area_ratio_img: float,
    aspect_min: float,
    aspect_max: float,
    min_bbox_w: float,
    min_bbox_h: float,
    min_edge: float,
) -> Tuple[np.ndarray, Dict[str, Any], bool]:
    base_meta = _evaluate_court_ratio_structure(
        ordered,
        lines_all,
        img_w=img_w,
        img_h=img_h,
        angle_tol_deg=angle_tol_deg,
        tol_h_m=tol_h_m,
        tol_v_m=tol_v_m,
        min_seg_len_m=min_seg_len_m,
    )
    best_quad = ordered.astype(np.float32).copy()
    best_meta = dict(base_meta)
    best_score = float(base_meta.get("score", 0.0))
    improved = False
    for s in scales:
        ss = float(s)
        if not np.isfinite(ss) or ss <= 0.0 or abs(ss - 1.0) < 1e-4:
            continue
        cand = _scale_quad_about_center(ordered, ss)
        if not _is_convex_quad(cand):
            continue
        if float(_quad_area(cand)) < 1.0:
            continue
        ok, _gate_reason, _gate_metrics = _passes_geom_gates(cand, floor_bbox)
        if not ok:
            continue
        bbox_w = float(np.max(cand[:, 0]) - np.min(cand[:, 0]))
        bbox_h = float(np.max(cand[:, 1]) - np.min(cand[:, 1]))
        if bbox_w < float(min_bbox_w) or bbox_h < float(min_bbox_h):
            continue
        cand_clip = cand.copy()
        cand_clip[:, 0] = np.clip(cand_clip[:, 0], 0.0, float(max(0, img_w - 1)))
        cand_clip[:, 1] = np.clip(cand_clip[:, 1], 0.0, float(max(0, img_h - 1)))
        area_ratio_img = float(_quad_area(cand_clip)) / float(max(1, img_w * img_h))
        if area_ratio_img < float(min_area_ratio_img):
            continue
        edge_min = float(np.min(_quad_edges(cand)))
        if edge_min < float(min_edge):
            continue
        w1 = float(np.linalg.norm(cand[1] - cand[0]))
        w2 = float(np.linalg.norm(cand[2] - cand[3]))
        h1 = float(np.linalg.norm(cand[3] - cand[0]))
        h2 = float(np.linalg.norm(cand[2] - cand[1]))
        mean_w = 0.5 * (w1 + w2)
        mean_h = 0.5 * (h1 + h2)
        aspect_ratio = max(mean_w, mean_h) / max(min(mean_w, mean_h), 1e-6)
        if aspect_ratio < float(aspect_min) or aspect_ratio > float(aspect_max):
            continue
        meta = _evaluate_court_ratio_structure(
            cand,
            lines_all,
            img_w=img_w,
            img_h=img_h,
            angle_tol_deg=angle_tol_deg,
            tol_h_m=tol_h_m,
            tol_v_m=tol_v_m,
            min_seg_len_m=min_seg_len_m,
        )
        sc = float(meta.get("score", 0.0))
        if sc > best_score + float(max(0.0, min_gain)):
            best_score = sc
            best_quad = cand
            best_meta = dict(meta)
            best_meta["scaled_from"] = 1.0
            best_meta["scaled_to"] = float(ss)
            improved = True
    return best_quad, best_meta, improved


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
    min_ratio: float = 0.08,
    max_ratio: float = 0.65,
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
    dt_oob: float = 255.0,
    clamp_max: float = 15.0,
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
    r = np.full((uv.shape[0],), float(dt_oob), dtype=np.float64)
    if np.any(valid):
        r[valid] = dt[v[valid], u[valid]].astype(np.float64)
    r = np.clip(r, 0.0, float(clamp_max))
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
        penalty_area *= 20.0
        penalty_inside *= 5.0
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


def hough_lines_from_edges(
    edges: np.ndarray,
    floor_mask: np.ndarray,
    min_length_ratio: float = 0.08,
    threshold: int = 80,
    max_gap_ratio: float = 0.02,
) -> list[Tuple[float, float, float, float]]:
    h, w = edges.shape[:2]
    # Allow runtime tuning for thin/far lines.
    thr = int(os.getenv("BADC_HOUGH_THRESHOLD", str(threshold)) or threshold)
    minlen_rt = float(os.getenv("BADC_HOUGH_MINLEN_RATIO", str(min_length_ratio)) or min_length_ratio)
    maxgap_rt = float(os.getenv("BADC_HOUGH_MAXGAP_RATIO", str(max_gap_ratio)) or max_gap_ratio)
    minlen_px = int(os.getenv("BADC_HOUGH_MINLEN_PX", "20") or 20)
    maxgap_px = int(os.getenv("BADC_HOUGH_MAXGAP_PX", "5") or 5)

    min_len = int(max(minlen_px, round(float(min(h, w)) * float(minlen_rt))))
    max_gap = int(max(maxgap_px, round(float(min(h, w)) * float(maxgap_rt))))

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180.0, thr, minLineLength=min_len, maxLineGap=max_gap)
    if lines is None:
        return []
    out = []
    y_excl = None
    if floor_mask is not None:
        ys = np.where(floor_mask > 0)[0]
        if ys.size > 0:
            y_top = int(ys.min())
            y_bot = int(ys.max())
            y_excl = int(y_top + 0.10 * max(1, (y_bot - y_top)))
    for x1, y1, x2, y2 in lines[:, 0]:
        mx = int(round((int(x1) + int(x2)) * 0.5))
        my = int(round((int(y1) + int(y2)) * 0.5))
        if mx < 0 or mx >= w or my < 0 or my >= h:
            continue
        if floor_mask[my, mx] == 0:
            continue
        if y_excl is not None:
            y_mid = 0.5 * (float(y1) + float(y2))
            if y_mid < float(y_excl):
                continue
        out.append((float(x1), float(y1), float(x2), float(y2)))
    return out


def _order_corners_tl_tr_br_bl(pts_xy: np.ndarray) -> np.ndarray:
    pts = np.array(pts_xy, dtype=np.float32).reshape(4, 2)
    sums = pts[:, 0] + pts[:, 1]
    diffs = pts[:, 0] - pts[:, 1]
    tl = pts[np.argmin(sums)]
    br = pts[np.argmax(sums)]
    tr = pts[np.argmin(diffs)]
    bl = pts[np.argmax(diffs)]
    return np.stack([tl, tr, br, bl], axis=0)


def _draw_model_lines(image: np.ndarray, H: np.ndarray, color: Tuple[int, int, int]) -> np.ndarray:
    out = image.copy()
    for ln in get_bwf_lines():
        pts = np.array([ln.p1, ln.p2], dtype=np.float32)
        uv = project_points(H, pts)
        if not np.all(np.isfinite(uv)):
            continue
        p1 = (int(round(float(uv[0, 0]))), int(round(float(uv[0, 1]))))
        p2 = (int(round(float(uv[1, 0]))), int(round(float(uv[1, 1]))))
        cv2.line(out, p1, p2, color, 2)
    return out


def _merge_lines_by_rho(
    segments: list[Tuple[float, float, float, float]],
    support_mask: np.ndarray,
    rho_bin: float = 20.0,
    top_n: int = 10,
) -> list[Tuple[Tuple[float, float, float], float]]:
    if not segments:
        return []
    if rho_bin <= 0:
        rho_bin = 1

    # bin_id -> list[(line_abc, weight)]
    bins: Dict[int, list[Tuple[Tuple[float, float, float], float]]] = {}
    for seg in segments:
        x1, y1, x2, y2 = seg
        length = float(math.hypot(x2 - x1, y2 - y1))
        if length <= 1e-3:
            continue
        line = _line_from_points((x1, y1), (x2, y2))  # normalized (a,b,c)
        a, b, c = _canonicalize_line_abc(float(line[0]), float(line[1]), float(line[2]))
        # Hough form: n·p = rho, with n=(a,b) unit normal and rho = -c
        rho = float(-c)
        key = int(round(rho / float(rho_bin)))
        w = float(_line_support(seg, support_mask, samples=24))
        if w <= 1e-6:
            w = float(max(1.0, length))
        bins.setdefault(key, []).append(((a, b, c), w))

    merged: list[Tuple[Tuple[float, float, float], float]] = []
    for _, items in bins.items():
        if not items:
            continue
        # Use first item as reference to align normals inside this bin
        a0, b0, c0 = items[0][0]
        acc = np.zeros((3,), dtype=np.float64)
        acc_w = 0.0
        for (a, b, c), w in items:
            # If normal points opposite to reference, flip to avoid cancellation
            if (a * a0 + b * b0) < 0.0:
                a, b, c = -a, -b, -c
            acc += np.array([a, b, c], dtype=np.float64) * float(w)
            acc_w += float(w)
        if acc_w <= 1e-9:
            continue
        line_avg = acc / float(acc_w)
        na, nb = float(line_avg[0]), float(line_avg[1])
        norm = float(math.hypot(na, nb))
        if norm <= 1e-9:
            continue
        line_avg = line_avg / norm
        merged.append(((float(line_avg[0]), float(line_avg[1]), float(line_avg[2])), float(acc_w)))
    merged.sort(key=lambda item: item[1], reverse=True)
    return merged[: max(2, int(top_n))]


def _fit_court_homography_from_raw_floor_debug(
    white_mask_raw_floor: np.ndarray,
    frame_shape: Tuple[int, int, int] | Tuple[int, int],
    floor_roi_mask: Optional[np.ndarray] = None,
    white_mask_raw_floor_noblob: Optional[np.ndarray] = None,
) -> Tuple[Optional[np.ndarray], Dict[str, Any], Dict[str, Optional[np.ndarray]]]:
    h, w = int(frame_shape[0]), int(frame_shape[1])
    rng = np.random.default_rng(0)

    # ---- raw_floor Hough tuning (env-configurable) ----
    raw_hough_threshold = _RAW_HOUGH_THRESHOLD
    raw_hough_minlen_ratio = _RAW_HOUGH_MINLEN_RT
    raw_seg_minlen_px = _RAW_SEG_MINLEN_PX
    raw_seg_max_keep = _RAW_SEG_MAX_KEEP

    def _run_hough_and_filter(edge_map_: np.ndarray, floor_roi_mask_: np.ndarray):
        # Tighten ROI for raw-floor Hough to avoid wall/ceiling/border edges dominating.
        hough_roi_mask = floor_roi_mask_.copy()
        top_cut = int(h * float(os.getenv("BADC_RAW_FLOOR_HOUGH_TOP_CUT", "0.18")))
        border = int(w * float(os.getenv("BADC_RAW_FLOOR_HOUGH_BORDER", "0.03")))
        if top_cut > 0:
            hough_roi_mask[:top_cut, :] = 0
        if border > 0:
            hough_roi_mask[:, :border] = 0
            hough_roi_mask[:, (w - border):] = 0
        erode_k = int(os.getenv("BADC_RAW_FLOOR_HOUGH_ERODE", "5"))
        if erode_k > 0:
            k = np.ones((erode_k, erode_k), np.uint8)
            hough_roi_mask = cv2.erode(hough_roi_mask, k, iterations=1)

        # Directional closing inside ROI to bridge occluded boundaries
        close_v = int(os.getenv("BADC_RAW_FLOOR_HOUGH_CLOSE_V", "0") or 0)
        close_h = int(os.getenv("BADC_RAW_FLOOR_HOUGH_CLOSE_H", "0") or 0)
        edge_work = edge_map_.copy()
        if close_v > 0:
            kv = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(1, close_v)))
            edge_work = cv2.morphologyEx(edge_work, cv2.MORPH_CLOSE, kv)
        if close_h > 0:
            kh = cv2.getStructuringElement(cv2.MORPH_RECT, (max(1, close_h), 1))
            edge_work = cv2.morphologyEx(edge_work, cv2.MORPH_CLOSE, kh)

        segs = hough_lines_from_edges(
            edge_work,
            hough_roi_mask,
            min_length_ratio=raw_hough_minlen_ratio,
            threshold=raw_hough_threshold,
        )
        before = len(segs)

        # Boundary-rescue: keep moderately-short vertical segments near left/right floor edges (occlusion-safe).
        if int(os.getenv("BADC_RAW_FLOOR_RESQ_BOUNDARY", "1") or "0") == 1:
            ys, xs = np.nonzero(hough_roi_mask > 0)
            if len(xs) > 0:
                bb = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
            else:
                bb = None
            segs = _filter_segments_minlen_with_boundary_rescue(
                segs,
                raw_seg_minlen_px,
                floor_bbox_xyxy=bb,
                max_keep=raw_seg_max_keep,
                resq_enable=True,
                resq_margin_frac=float(os.getenv("BADC_RAW_FLOOR_RESQ_MARGIN", "0.08")),
                resq_minlen_ratio=float(os.getenv("BADC_RAW_FLOOR_RESQ_MINLEN_RATIO", "0.35")),
                resq_vert_cos_max=float(os.getenv("BADC_RAW_FLOOR_RESQ_VERT_COS_MAX", "0.35")),
            )
        else:
            segs = _filter_segments_minlen(segs, raw_seg_minlen_px, max_keep=raw_seg_max_keep)
        after = len(segs)
        return segs, before, after
    # -----------------------------------------------

    if floor_roi_mask is None:
        floor_roi_mask = np.ones((h, w), dtype=np.uint8) * 255
    mask_input = white_mask_raw_floor
    mask_thin, edge_map, band, blob_mask = preprocess_raw_floor(mask_input, floor_roi_mask)
    obs = (mask_thin > 0).astype(np.uint8)
    dt = cv2.distanceTransform(1 - obs, cv2.DIST_L2, 3)
    dt_vis = cv2.normalize(dt, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    dt_p90 = float(np.percentile(dt, 90)) if dt.size > 0 else 0.0
    dt_oob = float(max(dt_p90, 15.0))

    # Optional RANSAC merge to rescue thin far-court lines
    use_ransac_merge = int(os.getenv("BADC_RAW_FLOOR_MERGE_RANSAC", "1")) != 0
    ransac_segments: list[Tuple[float, float, float, float]] = []
    ransac_tls_stats: Dict[str, int] = {"tls_num_input_pts": 0, "tls_num_finite_pts": 0, "tls_skipped_degenerate": 0}
    if use_ransac_merge:
        ransac_mask = white_mask_raw_floor_noblob if white_mask_raw_floor_noblob is not None else mask_thin
        try:
            r_max_lines = int(os.getenv("BADC_RAW_FLOOR_RANSAC_MAX_LINES", "12"))
            r_iters = int(os.getenv("BADC_RAW_FLOOR_RANSAC_ITERS", "600"))
            r_inlier_thr = float(os.getenv("BADC_RAW_FLOOR_RANSAC_INLIER_THR", "2.0"))
            r_min_inliers = int(os.getenv("BADC_RAW_FLOOR_RANSAC_MIN_INLIERS", "90"))
            r_min_len_ratio = float(os.getenv("BADC_RAW_FLOOR_RANSAC_MINLEN_RT", "0.10"))
            r_seed = int(os.getenv("BADC_RAW_FLOOR_RANSAC_SEED", "0"))
            rsegs, rstats = ransac_line_segments_from_mask(
                ransac_mask.astype(np.uint8),
                floor_roi_mask.astype(np.uint8),
                max_lines=r_max_lines,
                iters=r_iters,
                inlier_thr=r_inlier_thr,
                min_inliers=r_min_inliers,
                min_length_ratio=r_min_len_ratio,
                seed=r_seed,
            )
            ransac_segments = [(s.x1, s.y1, s.x2, s.y2) for s in rsegs]
            ransac_tls_stats = dict(rstats) if isinstance(rstats, dict) else ransac_tls_stats
        except Exception:
            ransac_segments = []

    def _merge_and_dedup_segments(
        hough_segs: list[tuple[int, int, int, int]],
        extra_segs: list[Tuple[float, float, float, float]],
        *,
        tol_px: int = 3,
        max_extra: int = 50,
    ) -> list[Tuple[float, float, float, float]]:
        if not extra_segs:
            return [(float(x1), float(y1), float(x2), float(y2)) for (x1, y1, x2, y2) in hough_segs]
        extra_sorted = sorted(
            extra_segs, key=lambda t: float(math.hypot(t[2] - t[0], t[3] - t[1])), reverse=True
        )[: int(max_extra)]
        out: list[Tuple[float, float, float, float]] = []
        seen: set[Tuple[int, int, int, int]] = set()

        def _key(x1: float, y1: float, x2: float, y2: float) -> Tuple[int, int, int, int]:
            a = (x1, y1)
            b = (x2, y2)
            if (b[0] < a[0]) or (b[0] == a[0] and b[1] < a[1]):
                a, b = b, a
            return (
                int(round(a[0] / tol_px)),
                int(round(a[1] / tol_px)),
                int(round(b[0] / tol_px)),
                int(round(b[1] / tol_px)),
            )

        for x1, y1, x2, y2 in hough_segs:
            k = _key(float(x1), float(y1), float(x2), float(y2))
            if k in seen:
                continue
            seen.add(k)
            out.append((float(x1), float(y1), float(x2), float(y2)))

        for x1, y1, x2, y2 in extra_sorted:
            x1 = float(np.clip(x1, 0.0, float(w - 1)))
            y1 = float(np.clip(y1, 0.0, float(h - 1)))
            x2 = float(np.clip(x2, 0.0, float(w - 1)))
            y2 = float(np.clip(y2, 0.0, float(h - 1)))
            k = _key(x1, y1, x2, y2)
            if k in seen:
                continue
            seen.add(k)
            out.append((x1, y1, x2, y2))
        return out

    # Optionally bypass Hough entirely and use RANSAC segments as the candidate set.
    if _env_flag("BADC_RAW_FLOOR_USE_RANSAC_ONLY", False):
        segments = [(float(s[0]), float(s[1]), float(s[2]), float(s[3])) for s in ransac_segments]
        seg_before0 = seg_after0 = int(len(segments))
    else:
        segments, seg_before0, seg_after0 = _run_hough_and_filter(edge_map, floor_roi_mask)
    _, _, floor_bbox = _white_mask_stats(floor_roi_mask)
    floor_y_cut = int(floor_bbox[1])
    if floor_y_cut is not None:
        y0 = max(int(floor_y_cut) - 2, 0)
        y1 = min(int(floor_y_cut) + 18, edge_map.shape[0])
        edge_map[y0:y1, :] = 0
        segments, seg_before1, seg_after1 = _run_hough_and_filter(edge_map, floor_roi_mask)

    # merge Hough + RANSAC segments (dedup + bound clip)
    seg_before_merge = int(len(segments))
    segments = _merge_and_dedup_segments(segments, ransac_segments)
    seg_after_merge = int(len(segments))

    angles_pair = _dominant_angles_from_segments(segments)
    if blob_mask is not None and angles_pair is not None:
        ang_a, ang_b = angles_pair
        edge_map = _bridge_edges_across_hole(
            edge_map,
            blob_mask,
            [math.degrees(ang_a), math.degrees(ang_b)],
            length=41,
            thickness=3,
        )
        segments, seg_before2, seg_after2 = _run_hough_and_filter(edge_map, floor_roi_mask)
    angles = []
    lengths = []
    centers_y = []
    segments_filtered = []
    for x1, y1, x2, y2 in segments:
        length = float(math.hypot(x2 - x1, y2 - y1))
        segments_filtered.append((x1, y1, x2, y2))
        ang = math.atan2(y2 - y1, x2 - x1)
        ang = float(np.mod(ang, math.pi))
        angles.append(ang)
        lengths.append(length)
        centers_y.append(0.5 * (float(y1) + float(y2)))
    segments = segments_filtered
    # init counters for metrics before potential early return
    kept_ransac_a = 0
    kept_ransac_b = 0

    metrics: Dict[str, Any] = {
        "raw_floor_merge_ransac": int(1 if use_ransac_merge else 0),
        "raw_floor_num_segments_hough": int(seg_before_merge),
        "raw_floor_num_segments_merged": int(seg_after_merge),
        "raw_floor_ransac_num_segments": int(len(ransac_segments)),
        "raw_floor_ransac_added": int(max(seg_after_merge - seg_before_merge, 0)),
        "raw_floor_ransac_kept_a": int(kept_ransac_a),
        "raw_floor_ransac_kept_b": int(kept_ransac_b),
        "raw_floor_ransac_tls_num_input_pts": int(ransac_tls_stats.get("tls_num_input_pts", 0)),
        "raw_floor_ransac_tls_num_finite_pts": int(ransac_tls_stats.get("tls_num_finite_pts", 0)),
        "raw_floor_ransac_tls_skipped_degenerate": int(ransac_tls_stats.get("tls_skipped_degenerate", 0)),
        "raw_floor_num_segments": int(len(segments)),
        "raw_floor_hough_threshold": int(raw_hough_threshold),
        "raw_floor_hough_minlen_ratio": float(raw_hough_minlen_ratio),
        "raw_floor_seg_minlen_px": float(raw_seg_minlen_px),
        "raw_floor_seg_max_keep": int(raw_seg_max_keep),
        "raw_floor_num_segments_before_minlen_0": int(locals().get("seg_before0", -1)),
        "raw_floor_num_segments_after_minlen_0": int(locals().get("seg_after0", -1)),
        "raw_floor_num_segments_before_minlen_1": int(locals().get("seg_before1", -1)),
        "raw_floor_num_segments_after_minlen_1": int(locals().get("seg_after1", -1)),
        "raw_floor_num_segments_before_minlen_2": int(locals().get("seg_before2", -1)),
        "raw_floor_num_segments_after_minlen_2": int(locals().get("seg_after2", -1)),
        "raw_floor_bright_band_removed": 1 if band is not None else 0,
        "raw_floor_bright_band_y1": int(band[0]) if band is not None else None,
        "raw_floor_bright_band_y2": int(band[1]) if band is not None else None,
        "raw_floor_bridge_used": 1 if blob_mask is not None and angles_pair is not None else 0,
    }
    if not angles:
        return (
            None,
            metrics,
            {
                "dt": dt_vis,
                "hough_lines_a": None,
                "hough_lines_b": None,
                "overlay_raw": None,
                "overlay_top5": None,
                "preprocessed": mask_thin,
                "edges": edge_map,
            },
        )
    angles_arr = np.array(angles, dtype=np.float32)
    weights = np.array(lengths, dtype=np.float32)
    feats = np.stack([np.cos(2.0 * angles_arr), np.sin(2.0 * angles_arr)], axis=1)
    idx0 = int(np.argmax(weights))
    c1 = feats[idx0]
    dists = np.sum((feats - c1) ** 2, axis=1)
    idx1 = int(np.argmax(dists))
    c2 = feats[idx1]
    for _ in range(10):
        d1 = np.sum((feats - c1) ** 2, axis=1)
        d2 = np.sum((feats - c2) ** 2, axis=1)
        assign_a = d1 <= d2
        if not np.any(assign_a) or np.all(assign_a):
            break
        w_a = weights[assign_a][:, None]
        w_b = weights[~assign_a][:, None]
        c1 = np.sum(feats[assign_a] * w_a, axis=0) / max(np.sum(w_a), 1e-6)
        c2 = np.sum(feats[~assign_a] * w_b, axis=0) / max(np.sum(w_b), 1e-6)

    # --- PATCH START: stabilize A/B split + FIX top_k truncation bug ---
    top_k = 40

    # Pre-merge Hough segments inside each orientation bucket (A/B) before Top-K.
    # This reduces duplicate/fragmented segments that stack on the same physical line.
    merge_segs = _env_flag("BADC_RAW_FLOOR_MERGE_SEGS", False)
    merge_ang_tol_deg = _env_float("BADC_RAW_FLOOR_MERGE_ANG_TOL_DEG", 5.0)
    merge_rho_bin_px = _env_float("BADC_RAW_FLOOR_MERGE_RHO_BIN_PX", 8.0)
    merge_gap_px = _env_float("BADC_RAW_FLOOR_MERGE_GAP_PX", 25.0)
    merge_filter_ang = _env_flag("BADC_RAW_FLOOR_MERGE_FILTER_ANG", False)
    merge_min_keep = int(_env_int("BADC_RAW_FLOOR_MERGE_MIN_KEEP", 12))
    merge_min_keep_bucket = int(_env_int("BADC_RAW_FLOOR_MERGE_MIN_KEEP_BUCKET", 8))

    def _merge_collinear_segments(segs_in, scores_in):
        """
        Merge nearly-collinear segments that are close along the line direction.
        - Bin by rho (perpendicular distance to the representative line normal)
        - Within each rho-bin, merge 1D projected intervals with a gap threshold
        Returns: (merged_segs_list, merged_scores_np)
        """
        if segs_in is None or len(segs_in) == 0:
            return [], np.asarray([], dtype=np.float32)

        segs = np.asarray(segs_in, dtype=np.float32).reshape(-1, 4)
        scores = np.asarray(scores_in, dtype=np.float32).reshape(-1)
        dx = segs[:, 2] - segs[:, 0]
        dy = segs[:, 3] - segs[:, 1]
        lens = np.maximum(np.hypot(dx, dy), 1e-3).astype(np.float32)

        # Representative angle for this bucket (directionless): double-angle mean (theta == theta+pi).
        ang = np.mod(np.arctan2(dy, dx), np.pi).astype(np.float32)
        c2 = float(np.sum(np.cos(2.0 * ang) * lens))
        s2 = float(np.sum(np.sin(2.0 * ang) * lens))
        a0 = float(0.5 * math.atan2(s2, c2))
        a0 = float(np.mod(a0, math.pi))

        # Optional: filter segments far from a0, but NEVER empty the bucket.
        if merge_filter_ang:
            ang_tol = math.radians(float(merge_ang_tol_deg))
            da = np.abs(((ang - a0 + np.pi / 2) % np.pi) - np.pi / 2)
            keep = da <= float(ang_tol)
            if int(np.sum(keep)) >= int(merge_min_keep):
                segs = segs[keep]
                scores = scores[keep]
                lens = lens[keep]
                ang = ang[keep]

        if segs.shape[0] == 0:
            return segs_in, np.asarray(scores_in, dtype=np.float32)

        ca, sa = math.cos(a0), math.sin(a0)
        d = np.asarray([ca, sa], dtype=np.float32)        # direction
        n = np.asarray([-sa, ca], dtype=np.float32)       # normal

        mids = 0.5 * (segs[:, 0:2] + segs[:, 2:4])
        rho = (mids @ n).astype(np.float32)

        bin_size = float(max(1.0, float(merge_rho_bin_px)))
        keys = np.round(rho / bin_size).astype(np.int32)

        merged_segs = []
        merged_scores = []
        for k in np.unique(keys):
            idx = np.where(keys == k)[0]
            if idx.size == 0:
                continue

            p1 = segs[idx, 0:2]
            p2 = segs[idx, 2:4]
            t1 = (p1 @ d).astype(np.float32)
            t2 = (p2 @ d).astype(np.float32)
            tmin = np.minimum(t1, t2)
            tmax = np.maximum(t1, t2)

            order = np.argsort(tmin)
            idx = idx[order]
            tmin = tmin[order]
            tmax = tmax[order]

            rho_bin = float(np.average(rho[idx], weights=lens[idx]))

            cur_t0 = float(tmin[0])
            cur_t1 = float(tmax[0])
            cur_score = float(scores[idx[0]])
            for j in range(1, int(idx.size)):
                if float(tmin[j]) <= cur_t1 + float(merge_gap_px):
                    cur_t0 = min(cur_t0, float(tmin[j]))
                    cur_t1 = max(cur_t1, float(tmax[j]))
                    cur_score += float(scores[idx[j]])
                else:
                    p0 = n * rho_bin + d * cur_t0
                    p1e = n * rho_bin + d * cur_t1
                    merged_segs.append([float(p0[0]), float(p0[1]), float(p1e[0]), float(p1e[1])])
                    merged_scores.append(cur_score)
                    cur_t0 = float(tmin[j])
                    cur_t1 = float(tmax[j])
                    cur_score = float(scores[idx[j]])

            p0 = n * rho_bin + d * cur_t0
            p1e = n * rho_bin + d * cur_t1
            merged_segs.append([float(p0[0]), float(p0[1]), float(p1e[0]), float(p1e[1])])
            merged_scores.append(cur_score)

        if len(merged_segs) == 0:
            return segs_in, np.asarray(scores_in, dtype=np.float32)
        return merged_segs, np.asarray(merged_scores, dtype=np.float32)

    def _take_topk(segs_in, scores_in, k):
        # Allow disabling top-k truncation (useful for debugging missing far/left lines)
        if _env_flag("BADC_RAW_FLOOR_TAKE_TOPK_DISABLE", False):
            return segs_in

        if int(k) <= 0 or len(segs_in) <= int(k):
            return segs_in

        scores = np.array(scores_in, dtype=np.float32)
        order = np.argsort(-scores)  # descending

        # Simple top-k by score (deterministic)
        picked = order[: int(k)].tolist()
        return [segs_in[i] for i in picked]

    # Prefer dominant-angle split when angles are well separated
    use_dom_split = False
    if angles_pair is not None:
        ang0, ang1 = float(angles_pair[0]), float(angles_pair[1])
        if _angle_distance(ang0, ang1) > math.radians(35.0):
            use_dom_split = True

    if not use_dom_split:
        assign_a = np.sum((feats - c1) ** 2, axis=1) <= np.sum((feats - c2) ** 2, axis=1)
    else:
        ang0, ang1 = float(angles_pair[0]), float(angles_pair[1])
        d0 = np.array([_angle_distance(float(a), ang0) for a in angles], dtype=np.float32)
        d1 = np.array([_angle_distance(float(a), ang1) for a in angles], dtype=np.float32)
        assign_a = d0 <= d1

    segs_arr = np.array(segments, dtype=np.float32)
    assign_a = np.array(assign_a, dtype=bool)
    lengths_arr = np.array(lengths, dtype=np.float32)
    centers_y_arr = np.array(centers_y, dtype=np.float32)
    floor_y0 = int(floor_bbox[1])
    pos_weight = np.array([_segment_y_weight(cy, floor_y0, h, gamma=2.5) for cy in centers_y_arr], dtype=np.float32)
    score_all = lengths_arr * (0.7 + 0.3 * pos_weight)

    seg_a = segs_arr[assign_a].tolist()
    seg_b = segs_arr[~assign_a].tolist()
    score_a = score_all[assign_a]
    score_b = score_all[~assign_a]

    # Merge adjacent/collinear segments inside each bucket BEFORE top-k truncation.
    if merge_segs:
        seg_a0, score_a0 = seg_a, score_a
        seg_b0, score_b0 = seg_b, score_b
        metrics["raw_floor_num_segments_a_split"] = int(len(seg_a0))
        metrics["raw_floor_num_segments_b_split"] = int(len(seg_b0))
        seg_a, score_a = _merge_collinear_segments(seg_a0, score_a0)
        seg_b, score_b = _merge_collinear_segments(seg_b0, score_b0)
        metrics["raw_floor_num_segments_a_merged"] = int(len(seg_a))
        metrics["raw_floor_num_segments_b_merged"] = int(len(seg_b))
        # Safety: if merging wipes out a bucket, revert to pre-merge.
        if int(len(seg_a)) < int(merge_min_keep_bucket):
            metrics["raw_floor_merge_fallback_a"] = 1
            seg_a, score_a = seg_a0, score_a0
        if int(len(seg_b)) < int(merge_min_keep_bucket):
            metrics["raw_floor_merge_fallback_b"] = 1
            seg_b, score_b = seg_b0, score_b0

    # --- PATCH: angle-peak re-split to prevent A/B role mix-ups ---
    if int(os.getenv("BADC_RAW_FLOOR_ANGLE_RESPLIT", "1") or "0") == 1:
        seg_a2, score_a2, seg_b2, score_b2, meta_resplit = _resplit_segments_by_angle_peaks(
            seg_a,
            score_a,
            seg_b,
            score_b,
            min_keep_each=int(os.getenv("BADC_RAW_FLOOR_ANGLE_RESPLIT_MIN_KEEP", "8")),
            min_sep_deg=float(os.getenv("BADC_RAW_FLOOR_ANGLE_RESPLIT_MIN_SEP_DEG", "55")),
        )
        metrics["raw_floor_angle_resplit"] = meta_resplit
        if meta_resplit.get("used"):
            seg_a, score_a = seg_a2, score_a2
            seg_b, score_b = seg_b2, score_b2
            metrics["raw_floor_num_segments_a_resplit"] = int(len(seg_a))
            metrics["raw_floor_num_segments_b_resplit"] = int(len(seg_b))
    # --- PATCH END ---

    # Top-K truncation is applied AFTER the optional merge stage (see above)
    seg_a = _take_topk(seg_a, score_a, top_k)
    seg_b = _take_topk(seg_b, score_b, top_k)
    metrics["raw_floor_num_segments_a_post_topk"] = int(len(seg_a))
    metrics["raw_floor_num_segments_b_post_topk"] = int(len(seg_b))

    # Optionally force-keep RANSAC segments even if they drop out of top-K (they can be thin/faint).
    keep_ransac_always = _env_flag("BADC_RAW_FLOOR_KEEP_RANSAC_ALWAYS", True)
    if keep_ransac_always and ransac_segments:
        def _seg_key(seg):
            x1, y1, x2, y2 = [int(round(float(v))) for v in seg]
            if (x1, y1) <= (x2, y2):
                return (x1, y1, x2, y2)
            return (x2, y2, x1, y1)

        ransac_keys = set(_seg_key(s) for s in ransac_segments)
        keys_a = set(_seg_key(s) for s in seg_a)
        keys_b = set(_seg_key(s) for s in seg_b)

        for i in range(len(segments)):
            k = _seg_key(segments[i])
            if k not in ransac_keys:
                continue
            if bool(assign_a[i]):
                if k not in keys_a:
                    seg_a.append(segments[i])
                    keys_a.add(k)
                    kept_ransac_a += 1
            else:
                if k not in keys_b:
                    seg_b.append(segments[i])
                    keys_b.add(k)
                    kept_ransac_b += 1

    # Update metrics with final kept counts after optional RANSAC reinsertion.
    metrics["raw_floor_ransac_kept_a"] = int(kept_ransac_a)
    metrics["raw_floor_ransac_kept_b"] = int(kept_ransac_b)

    metrics["raw_floor_num_segments_a"] = int(len(seg_a))
    metrics["raw_floor_num_segments_b"] = int(len(seg_b))

    # Optional *post-topk* segment merge.
    # NOTE: pre-topk merge is already controlled by BADC_RAW_FLOOR_MERGE_SEGS (recommended).
    # Post-topk merge can over-merge; keep it OFF by default.
    merge_segs = _env_flag("BADC_RAW_FLOOR_POSTMERGE_SEGS", False)
    merge_filter_ang = _env_flag("BADC_RAW_FLOOR_MERGE_FILTER_ANG", False)
    merge_ang_tol_deg = float(os.getenv("BADC_RAW_FLOOR_MERGE_ANG_TOL_DEG", "5"))
    merge_rho_bin_px = float(os.getenv("BADC_RAW_FLOOR_MERGE_RHO_BIN_PX", "12"))
    merge_gap_px = float(os.getenv("BADC_RAW_FLOOR_MERGE_GAP_PX", "25"))
    merge_min_len_px = float(os.getenv("BADC_RAW_FLOOR_MERGE_MIN_LEN_PX", "30"))
    merge_min_keep = int(os.getenv("BADC_RAW_FLOOR_MERGE_MIN_KEEP", "12"))
    merge_min_keep_bucket = int(os.getenv("BADC_RAW_FLOOR_MERGE_MIN_KEEP_BUCKET", "8"))

    def _as_int_segs(_segs):
        out = []
        for s in _segs:
            x1, y1, x2, y2 = s
            out.append((int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))))
        return out

    seg_a_i = _as_int_segs(seg_a)
    seg_b_i = _as_int_segs(seg_b)
    metrics["raw_floor_segA_before"] = int(len(seg_a_i))
    metrics["raw_floor_segB_before"] = int(len(seg_b_i))

    if merge_filter_ang:
        seg_a_i, metaA = _filter_segments_by_angle_keep_min(seg_a_i, merge_ang_tol_deg, merge_min_keep_bucket)
        seg_b_i, metaB = _filter_segments_by_angle_keep_min(seg_b_i, merge_ang_tol_deg, merge_min_keep_bucket)
        metrics["raw_floor_merge_filter_ang_A"] = metaA
        metrics["raw_floor_merge_filter_ang_B"] = metaB

    if merge_segs:
        seg_a_i, metaMA = _merge_collinear_segments_by_rho(seg_a_i, merge_rho_bin_px, merge_gap_px, merge_min_len_px)
        seg_b_i, metaMB = _merge_collinear_segments_by_rho(seg_b_i, merge_rho_bin_px, merge_gap_px, merge_min_len_px)
        metrics["raw_floor_merge_segs_A"] = metaMA
        metrics["raw_floor_merge_segs_B"] = metaMB

        # If merging becomes too aggressive, fall back to unmerged but filtered segments.
        if len(seg_a_i) < merge_min_keep:
            metrics["raw_floor_merge_fallback_A"] = True
            seg_a_i = _as_int_segs(seg_a)
            if merge_filter_ang:
                seg_a_i, metaA2 = _filter_segments_by_angle_keep_min(seg_a_i, merge_ang_tol_deg, merge_min_keep)
                metrics["raw_floor_merge_filter_ang_A_fallback"] = metaA2
        if len(seg_b_i) < merge_min_keep:
            metrics["raw_floor_merge_fallback_B"] = True
            seg_b_i = _as_int_segs(seg_b)
            if merge_filter_ang:
                seg_b_i, metaB2 = _filter_segments_by_angle_keep_min(seg_b_i, merge_ang_tol_deg, merge_min_keep)
                metrics["raw_floor_merge_filter_ang_B_fallback"] = metaB2

    # Use the (optionally) merged segments downstream
    seg_a = seg_a_i
    seg_b = seg_b_i
    metrics["raw_floor_segA_after"] = int(len(seg_a))
    metrics["raw_floor_segB_after"] = int(len(seg_b))
    # --- PATCH END ---

    hough_base = cv2.cvtColor(white_mask_raw_floor, cv2.COLOR_GRAY2BGR)
    hough_a = hough_base.copy()
    hough_b = hough_base.copy()
    hough_lines = hough_base.copy()
    for x1, y1, x2, y2 in seg_a:
        cv2.line(hough_a, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        cv2.line(hough_lines, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
    for x1, y1, x2, y2 in seg_b:
        cv2.line(hough_b, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
        cv2.line(hough_lines, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)

    lines_a = _merge_lines_by_rho(seg_a, white_mask_raw_floor, rho_bin=20.0, top_n=12)
    lines_b = _merge_lines_by_rho(seg_b, white_mask_raw_floor, rho_bin=20.0, top_n=12)
    metrics["raw_floor_num_lines_a"] = int(len(lines_a))
    metrics["raw_floor_num_lines_b"] = int(len(lines_b))
    if len(lines_a) < 2 or len(lines_b) < 2:
        return (
            None,
            metrics,
            {
                "dt": dt_vis,
                "hough_lines": hough_lines,
                "hough_lines_a": hough_a,
                "hough_lines_b": hough_b,
                "overlay_raw": None,
                "overlay_top5": None,
                "preprocessed": mask_thin,
                "edges": edge_map,
            },
        )

    model_cfgs = [
        ("outer", np.array([[0.0, 13.40], [6.10, 13.40], [6.10, 0.0], [0.0, 0.0]], dtype=np.float32)),
        ("singles", np.array([[0.46, 13.40], [5.64, 13.40], [5.64, 0.0], [0.46, 0.0]], dtype=np.float32)),
        ("service_near", np.array([[0.0, 4.72], [6.10, 4.72], [6.10, 0.76], [0.0, 0.76]], dtype=np.float32)),
    ]

    # Prefer full-court fits by default; service-box configs can be noisy. Re-enable via env if needed.
    if int(os.getenv("BADC_RAW_FLOOR_DISABLE_SERVICE_CFG", "1") or "0") == 1:
        model_cfgs = [cfg for cfg in model_cfgs if not str(cfg[0]).startswith("service")]
    if int(os.getenv("BADC_RAW_FLOOR_DISABLE_SINGLES_CFG", "0") or "0") == 1:
        model_cfgs = [cfg for cfg in model_cfgs if str(cfg[0]) != "singles"]

    Xw_full, w_full, names_full = sample_model_points(points_per_meter=25.0, min_weight=0.3)
    w_full, role_w_cfg = _apply_model_role_weights(names_full, w_full)
    metrics["raw_floor_model_role_weights"] = role_w_cfg
    best_H = None
    best_score = float("-inf")
    best_cfg = None
    best_oob = 0
    best_mixer_parts: Optional[Dict[str, float]] = None
    best_area_ratio_img: Optional[float] = None
    best_ymax_ratio: Optional[float] = None
    best_top_edge_floor_ratio: Optional[float] = None
    best_bottom_support: Optional[float] = None
    best_role_prior_bonus: Optional[float] = None
    best_role_prior_meta: Optional[Dict[str, float]] = None
    best_cfg_prior: Optional[float] = None
    num_quads = 0
    num_scored = 0
    rejects = {
        "parallel": 0,
        "area": 0,
        "convex": 0,
        "inside": 0,
        "height": 0,
        "bottom": 0,
        "roi_bottom": 0,
        "thin_penalty": 0,
        "area_too_small": 0,
        "bbox_w_too_small": 0,
        "bbox_h_too_small": 0,
        "min_edge_too_small": 0,
        "bottom_y_too_high": 0,
        "bottom_endpoints_outside_floor": 0,
        "bottom_endpoints_too_high": 0,
        "top_endpoints_too_low": 0,
        "area_ratio_low": 0,
        "ymax_ratio_low": 0,
        "degenerate_bbox": 0,
        "degenerate_min_edge": 0,
        "raw_floor_reject_small_bbox": 0,
        "raw_floor_reject_vp_close": 0,
        "corners_in_image_low": 0,
        "bottom_corners_oob": 0,
        "bottom_corners_outside_floor_bbox": 0,
        "inlier_ratio_low": 0,
        "cover_ratio_low": 0,
        "oob_frac_high": 0,
        "bottom_span_floor_ratio_low": 0,
    }
    img_area = float(h * w)
    roi_area = float(np.count_nonzero(floor_roi_mask)) or img_area
    min_thin_ratio = 0.0
    top5: list[Tuple[float, np.ndarray, str]] = []
    top3_main: list[Tuple[float, float, float, float, int, float]] = []
    relaxed_pass = 0
    vp_a = _estimate_vanishing_point([ln[0] for ln in lines_a], rng) if lines_a else None
    vp_b = _estimate_vanishing_point([ln[0] for ln in lines_b], rng) if lines_b else None
    # --- Hard constraints (configurable) ---
    min_area_ratio_hard = float(os.getenv("BADC_FIT_MIN_AREA_RATIO", "0.05"))
    min_ymax_ratio_hard = float(os.getenv("BADC_FIT_MIN_YMAX_RATIO", "0.70"))
    min_quad_w_hard = float(os.getenv("BADC_FIT_MIN_QUAD_W", "220"))
    min_quad_h_hard = float(os.getenv("BADC_FIT_MIN_QUAD_H", "160"))
    min_corners_in_image_hard = int(os.getenv("BADC_FIT_MIN_CORNERS_IN_IMAGE", "3"))
    min_bottom_corners_in_image = int(os.getenv("BADC_FIT_MIN_BOTTOM_CORNERS_IN_IMAGE", "2"))
    require_bottom_in_floor = int(os.getenv("BADC_FIT_REQUIRE_BOTTOM_IN_FLOOR", "1")) == 1
    bottom_in_floor_margin = int(os.getenv("BADC_FIT_BOTTOM_IN_FLOOR_MARGIN", "8"))
    max_top_edge_floor_ratio_hard = float(os.getenv("BADC_FIT_MAX_TOP_EDGE_FLOOR_RATIO", "0.30"))
    min_inlier_ratio_hard = float(os.getenv("BADC_FIT_MIN_INLIER_RATIO", "0.25"))
    min_cover_ratio_hard = float(os.getenv("BADC_FIT_MIN_COVER_RATIO", "0.22"))
    max_oob_frac_hard = float(os.getenv("BADC_FIT_MAX_OOB_FRAC", "0.70"))
    allow_small_quads = int(os.getenv("BADC_FIT_ALLOW_SMALL_QUADS", "0")) == 1
    floor_bbox_w_px = float(max(1.0, float(floor_bbox[2] - floor_bbox[0])))
    min_bottom_span_floor_ratio_hard = float(
        os.getenv("BADC_FIT_MIN_BOTTOM_SPAN_FLOOR_RATIO", "0.45")
    )
    cfg_prior_outer = float(_env_float("BADC_RAW_FLOOR_CFG_PRIOR_OUTER", 1.2))
    cfg_prior_singles = float(_env_float("BADC_RAW_FLOOR_CFG_PRIOR_SINGLES", -0.8))
    cfg_prior_service = float(_env_float("BADC_RAW_FLOOR_CFG_PRIOR_SERVICE", -2.0))
    vp_min_dist = float(os.getenv("BADC_FIT_MIN_VP_DIST", "0.0"))  # 0 disables
    role_prior_enabled = bool(_env_flag("BADC_ROLE_PRIOR_ENABLE", False))
    role_prior_mask = white_mask_raw_floor_noblob if white_mask_raw_floor_noblob is not None else white_mask_raw_floor
    metrics["raw_floor_role_prior_enabled"] = bool(role_prior_enabled)
    metrics["raw_floor_role_prior_mask"] = "raw_floor_noblob"

    def _vp_dist_ok(vp_xy: Optional[np.ndarray], w_: int, h_: int) -> bool:
        if vp_xy is None:
            return True
        if not np.isfinite(vp_xy).all():
            return False
        cx, cy = 0.5 * float(w_), 0.5 * float(h_)
        d = float(math.hypot(float(vp_xy[0]) - cx, float(vp_xy[1]) - cy))
        return d >= vp_min_dist * float(max(w_, h_))
    qrt_min_bbox_w_frac = float(_env_float("BADC_QRT_MIN_BBOX_W_FRAC", 0.08))
    qrt_min_bbox_h_frac = float(_env_float("BADC_QRT_MIN_BBOX_H_FRAC", 0.06))
    qrt_min_edge_frac = float(_env_float("BADC_QRT_MIN_EDGE_FRAC", 0.05))
    for relax_scale in (1.0, 0.7):
        if relax_scale < 1.0:
            relaxed_pass = 1
        min_bottom_span_floor_ratio_loop = float(max(0.28, min_bottom_span_floor_ratio_hard * float(relax_scale)))
        best_H = None
        best_score = float("-inf")
        best_cfg = None
        best_oob = 0
        best_mixer_parts = None
        best_area_ratio_img = None
        best_ymax_ratio = None
        best_top_edge_floor_ratio = None
        best_bottom_support = None
        best_role_prior_bonus = None
        best_role_prior_meta = None
        best_cfg_prior = None
        num_scored = 0
        top5 = []
        top3_main = []
        deg_bbox_w = float(max(1.0, qrt_min_bbox_w_frac * float(w) * float(relax_scale)))
        deg_bbox_h = float(max(1.0, qrt_min_bbox_h_frac * float(h) * float(relax_scale)))
        deg_edge = float(max(1.0, qrt_min_edge_frac * float(min(h, w)) * float(relax_scale)))
        for i in range(len(lines_a)):
            for j in range(i + 1, len(lines_a)):
                la1 = lines_a[i][0]
                la2 = lines_a[j][0]
                for m in range(len(lines_b)):
                    for n in range(m + 1, len(lines_b)):
                        lb1 = lines_b[m][0]
                        lb2 = lines_b[n][0]
                        num_quads += 1
                        quad = _intersections_from_pairs((la1, la2), (lb1, lb2))
                        if quad is None:
                            rejects["parallel"] += 1
                            continue
                        img_pts = _order_corners_tl_tr_br_bl(quad)
                        if not _is_convex_quad(img_pts):
                            rejects["convex"] += 1
                            continue
                        if _quad_area(img_pts) < 1.0:
                            rejects["area"] += 1
                            continue
                        ordered = _order_corners_lb_rb_rt_lt(img_pts)
                        # --- Visibility gates (apply to both strict and relaxed) ---
                        corners_in_img = int(
                            np.sum(
                                (ordered[:, 0] >= 0.0)
                                & (ordered[:, 0] <= float(w - 1))
                                & (ordered[:, 1] >= 0.0)
                                & (ordered[:, 1] <= float(h - 1))
                            )
                        )
                        if corners_in_img < min_corners_in_image_hard:
                            rejects["corners_in_image_low"] += 1
                            continue
                        bottom = ordered[:2, :]
                        bottom_in_img = int(
                            np.sum(
                                (bottom[:, 0] >= 0.0)
                                & (bottom[:, 0] <= float(w - 1))
                                & (bottom[:, 1] >= 0.0)
                                & (bottom[:, 1] <= float(h - 1))
                            )
                        )
                        if bottom_in_img < min_bottom_corners_in_image:
                            rejects["bottom_corners_oob"] += 1
                            continue
                        if require_bottom_in_floor:
                            fx1, fy1, fx2, fy2 = floor_bbox
                            mrg = float(bottom_in_floor_margin)
                            in_floor = (
                                (bottom[:, 0] >= float(fx1) - mrg)
                                & (bottom[:, 0] <= float(fx2) + mrg)
                                & (bottom[:, 1] >= float(fy1) - mrg)
                                & (bottom[:, 1] <= float(fy2) + mrg)
                            )
                            if int(np.sum(in_floor)) < 2:
                                rejects["bottom_corners_outside_floor_bbox"] += 1
                                continue
                        bottom_span_px = float(np.linalg.norm(bottom[1] - bottom[0]))
                        bottom_span_floor_ratio = float(bottom_span_px / max(1.0, floor_bbox_w_px))
                        if bottom_span_floor_ratio < min_bottom_span_floor_ratio_loop:
                            rejects["bottom_span_floor_ratio_low"] += 1
                            if not allow_small_quads:
                                continue
                        top_edge_floor_ratio = _top_edge_floor_ratio(ordered, floor_bbox)
                        if top_edge_floor_ratio > max_top_edge_floor_ratio_hard:
                            rejects["top_endpoints_too_low"] += 1
                            if not allow_small_quads:
                                continue
                        ok, why_gate, _gate_metrics = _passes_geom_gates(ordered, floor_bbox)
                        if not ok:
                            rejects[why_gate] = rejects.get(why_gate, 0) + 1
                            continue
                        # Optional VP sanity (soft unless explicitly hardened)
                        if (not _vp_dist_ok(vp_a, w, h)) or (not _vp_dist_ok(vp_b, w, h)):
                            rejects["raw_floor_reject_vp_close"] += 1
                            if relax_scale >= 1.0 and (not allow_small_quads) and vp_min_dist > 0.0:
                                continue
                        bbox_w = float(np.max(ordered[:, 0]) - np.min(ordered[:, 0]))
                        bbox_h = float(np.max(ordered[:, 1]) - np.min(ordered[:, 1]))
                        min_edge = float(np.min(_quad_edges(ordered)))
                        if bbox_w < deg_bbox_w or bbox_h < deg_bbox_h:
                            rejects["degenerate_bbox"] += 1
                            continue
                        if min_edge < deg_edge:
                            rejects["degenerate_min_edge"] += 1
                            continue
                        quad_clip = ordered.copy()
                        quad_clip[:, 0] = np.clip(quad_clip[:, 0], 0.0, float(w - 1))
                        quad_clip[:, 1] = np.clip(quad_clip[:, 1], 0.0, float(h - 1))
                        quad_area = float(_quad_area(quad_clip))
                        area_ratio_img = quad_area / float(max(img_area, 1.0))
                        y_vals = ordered[:, 1]
                        ymax_ratio = max(0.0, min(1.0, float(np.max(y_vals)) / float(max(h - 1, 1))))
                        ymean_ratio = max(0.0, min(1.0, float(np.mean(y_vals)) / float(max(h - 1, 1))))
                        quad_w = float(np.max(ordered[:, 0]) - np.min(ordered[:, 0]))
                        quad_h = float(np.max(ordered[:, 1]) - np.min(ordered[:, 1]))

                        # --- HARD REJECT in strict pass ---
                        if relax_scale >= 1.0 and (not allow_small_quads):
                            if area_ratio_img < min_area_ratio_hard:
                                rejects["area_ratio_low"] += 1
                                continue
                            if ymax_ratio < min_ymax_ratio_hard:
                                rejects["ymax_ratio_low"] += 1
                                continue
                            if quad_w < min_quad_w_hard or quad_h < min_quad_h_hard:
                                rejects["raw_floor_reject_small_bbox"] += 1
                                continue
                        else:
                            # relaxed pass: keep soft counters for area/ymax,
                            # BUT still hard-reject tiny quads unless explicitly allowed
                            if area_ratio_img < min_area_ratio_hard:
                                rejects["area_ratio_low"] += 1
                            if ymax_ratio < min_ymax_ratio_hard:
                                rejects["ymax_ratio_low"] += 1
                            if quad_w < min_quad_w_hard or quad_h < min_quad_h_hard:
                                rejects["raw_floor_reject_small_bbox"] += 1
                                if not allow_small_quads:
                                    continue
                        thin_penalty = 0.0
                        penalty = 0.0
                        for cfg_name, model_pts in model_cfgs:
                            H = cv2.getPerspectiveTransform(model_pts, img_pts.astype(np.float32))
                            if not np.all(np.isfinite(H)):
                                rejects["parallel"] += 1
                                continue
                            loss_info = _compute_loss_terms(
                                H,
                                dt=dt,
                                Xw=Xw_full,
                                weights=w_full,
                                cover_points=None,
                                floor_bbox=floor_bbox,
                                tau_px=5.0,
                                dt_oob=dt_oob,
                            )
                            # Fit-quality gates (strict pass only; relaxed relies on score penalties)
                            if relax_scale >= 1.0 and (not allow_small_quads):
                                inlier_ratio = float(loss_info.get("inlier_ratio", 0.0))
                                cover_ratio = float(loss_info.get("cover_ratio", 0.0))
                                dt_oob = int(loss_info.get("dt_oob_count", 0))
                                nsamp = int(loss_info.get("num_samples", 0))
                                oob_frac = float(dt_oob) / float(max(nsamp, 1))
                                if inlier_ratio < min_inlier_ratio_hard:
                                    rejects["inlier_ratio_low"] += 1
                                    continue
                                if cover_ratio < min_cover_ratio_hard:
                                    rejects["cover_ratio_low"] += 1
                                    continue
                                if oob_frac > max_oob_frac_hard:
                                    rejects["oob_frac_high"] += 1
                                    continue
                            # Fit-quality gates (strict pass only; relaxed relies on score penalties)
                            if relax_scale >= 1.0 and (not allow_small_quads):
                                inlier_ratio = float(loss_info.get("inlier_ratio", 0.0))
                                cover_ratio = float(loss_info.get("cover_ratio", 0.0))
                                dt_oob = int(loss_info.get("dt_oob_count", 0))
                                nsamp = int(loss_info.get("num_samples", 0))
                                oob_frac = float(dt_oob) / float(max(nsamp, 1))
                                if inlier_ratio < min_inlier_ratio_hard:
                                    rejects["inlier_ratio_low"] += 1
                                    continue
                                if cover_ratio < min_cover_ratio_hard:
                                    rejects["cover_ratio_low"] += 1
                                    continue
                                if oob_frac > max_oob_frac_hard:
                                    rejects["oob_frac_high"] += 1
                                    continue
                            score_val, mixer_parts = _mixer_score_candidate(
                                ordered,
                                loss_info,
                                floor_bbox,
                                vp_long=vp_a,
                                vp_short=vp_b,
                            )
                            score_val = float(score_val + penalty + thin_penalty)
                            bottom_support = _bottom_support_from_loss(loss_info, ymax_ratio, h, tau_px=5.0)
                            role_bonus = 0.0
                            role_meta: Dict[str, float] = {}
                            if role_prior_enabled:
                                role_bonus, role_meta = _court_role_prior_bonus(H, role_prior_mask)
                            score_final = float(
                                score_val
                                + 6.0 * math.log(max(area_ratio_img, 1e-6))
                                + 3.0 * ymean_ratio
                                + 2.0 * ymax_ratio
                                + 4.0 * bottom_support
                                + float(role_bonus)
                            )
                            cfg_prior = 0.0
                            if str(cfg_name) == "outer":
                                cfg_prior = float(cfg_prior_outer)
                            elif str(cfg_name) == "singles":
                                cfg_prior = float(cfg_prior_singles)
                            elif str(cfg_name).startswith("service"):
                                cfg_prior = float(cfg_prior_service)
                            score_final = float(score_final + cfg_prior)
                            num_scored += 1
                            if score_final > best_score:
                                best_score = score_final
                                best_H = H
                                best_cfg = cfg_name
                                best_oob = int(loss_info.get("dt_oob_count", 0))
                                best_mixer_parts = mixer_parts
                                best_area_ratio_img = float(area_ratio_img)
                                best_ymax_ratio = float(ymax_ratio)
                                best_bottom_support = float(bottom_support)
                                best_role_prior_bonus = float(role_bonus)
                                best_role_prior_meta = dict(role_meta) if role_meta else None
                                best_cfg_prior = float(cfg_prior)
                            top5.append((score_final, img_pts.copy(), cfg_name))
                            top3_main.append((score_final, area_ratio_img, ymax_ratio, bottom_support))
        if best_H is not None:
            break
    top5.sort(key=lambda item: item[0], reverse=True)
    top5 = top5[:5]
    top3_main.sort(key=lambda item: item[0], reverse=True)
    top3_main = top3_main[:3]

    metrics.update(
        {
            "raw_floor_num_quads_generated": int(num_quads),
            "raw_floor_num_quads_scored": int(num_scored),
            "raw_floor_reject_parallel": int(rejects["parallel"]),
            "raw_floor_reject_area": int(rejects["area"]),
            "raw_floor_reject_convex": int(rejects["convex"]),
            "raw_floor_reject_inside": int(rejects["inside"]),
            "raw_floor_reject_height": int(rejects["height"]),
            "raw_floor_reject_bottom": int(rejects["bottom"]),
            "raw_floor_reject_bottom_corners": int(rejects["roi_bottom"]),
            "raw_floor_thin_penalty_count": int(rejects["thin_penalty"]),
            "raw_floor_reject_thin": int(rejects["thin_penalty"]),
            "raw_floor_reject_area_too_small": int(rejects.get("area_too_small", 0)),
            "raw_floor_reject_bbox_w_too_small": int(rejects.get("bbox_w_too_small", 0)),
            "raw_floor_reject_bbox_h_too_small": int(rejects.get("bbox_h_too_small", 0)),
            "raw_floor_reject_min_edge_too_small": int(rejects.get("min_edge_too_small", 0)),
            "raw_floor_reject_bottom_y_too_high": int(rejects.get("bottom_y_too_high", 0)),
            "raw_floor_reject_bottom_endpoints_outside_floor": int(rejects.get("bottom_endpoints_outside_floor", 0)),
            "raw_floor_reject_bottom_endpoints_too_high": int(rejects.get("bottom_endpoints_too_high", 0)),
            "raw_floor_reject_top_endpoints_too_low": int(rejects.get("top_endpoints_too_low", 0)),
            "raw_floor_reject_area_ratio_low": int(rejects.get("area_ratio_low", 0)),
            "raw_floor_reject_ymax_ratio_low": int(rejects.get("ymax_ratio_low", 0)),
            "raw_floor_reject_degenerate_bbox": int(rejects.get("degenerate_bbox", 0)),
            "raw_floor_reject_degenerate_min_edge": int(rejects.get("degenerate_min_edge", 0)),
            "raw_floor_reject_corners_in_image_low": int(rejects.get("corners_in_image_low", 0)),
            "raw_floor_reject_bottom_corners_oob": int(rejects.get("bottom_corners_oob", 0)),
            "raw_floor_reject_bottom_corners_outside_floor_bbox": int(
                rejects.get("bottom_corners_outside_floor_bbox", 0)
            ),
            "raw_floor_reject_inlier_ratio_low": int(rejects.get("inlier_ratio_low", 0)),
            "raw_floor_reject_cover_ratio_low": int(rejects.get("cover_ratio_low", 0)),
            "raw_floor_reject_oob_frac_high": int(rejects.get("oob_frac_high", 0)),
            "raw_floor_reject_bottom_span_floor_ratio_low": int(rejects.get("bottom_span_floor_ratio_low", 0)),
            "raw_floor_relaxed_pass": int(relaxed_pass),
            "raw_floor_gate_min_corners_in_image": int(min_corners_in_image_hard),
            "raw_floor_gate_min_bottom_corners_in_image": int(min_bottom_corners_in_image),
            "raw_floor_gate_require_bottom_in_floor": bool(require_bottom_in_floor),
            "raw_floor_gate_bottom_in_floor_margin": int(bottom_in_floor_margin),
            "raw_floor_gate_max_top_edge_floor_ratio": float(max_top_edge_floor_ratio_hard),
            "raw_floor_gate_min_inlier_ratio": float(min_inlier_ratio_hard),
            "raw_floor_gate_min_cover_ratio": float(min_cover_ratio_hard),
            "raw_floor_gate_max_oob_frac": float(max_oob_frac_hard),
            "raw_floor_gate_min_bottom_span_floor_ratio": float(min_bottom_span_floor_ratio_hard),
            "raw_floor_cfg_prior_outer": float(cfg_prior_outer),
            "raw_floor_cfg_prior_singles": float(cfg_prior_singles),
            "raw_floor_cfg_prior_service": float(cfg_prior_service),
            "raw_floor_cfg_disable_service": bool(
                int(os.getenv("BADC_RAW_FLOOR_DISABLE_SERVICE_CFG", "1") or "0") == 1
            ),
            "raw_floor_cfg_disable_singles": bool(
                int(os.getenv("BADC_RAW_FLOOR_DISABLE_SINGLES_CFG", "0") or "0") == 1
            ),
            "raw_floor_best_score": float(best_score) if np.isfinite(best_score) else None,
            "raw_floor_best_model_config": best_cfg,
            "raw_floor_best_cfg_prior": float(best_cfg_prior) if best_cfg_prior is not None else None,
            "raw_floor_oob_points": int(best_oob),
            "raw_floor_best_mixer": best_mixer_parts if isinstance(best_mixer_parts, dict) else None,
            "raw_floor_best_role_prior_bonus": float(best_role_prior_bonus)
            if best_role_prior_bonus is not None
            else None,
            "raw_floor_best_role_prior_meta": best_role_prior_meta if isinstance(best_role_prior_meta, dict) else None,
            "raw_floor_selected_area_ratio": float(best_area_ratio_img)
            if best_area_ratio_img is not None
            else None,
            "raw_floor_selected_ymax_ratio": float(best_ymax_ratio)
            if best_ymax_ratio is not None
            else None,
            "raw_floor_selected_bottom_support": float(best_bottom_support)
            if best_bottom_support is not None
            else None,
            "raw_floor_top5": [
                {"score": float(score), "model_config": cfg} for score, _, cfg in top5
            ],
            "raw_floor_top3_mainfield": [
                {
                    "score_final": float(s),
                    "area_ratio": float(a),
                    "ymax_ratio": float(y),
                    "bottom_support": float(b),
                }
                for s, a, y, b in top3_main
            ],
        }
    )
    if best_H is None:
        return (
            None,
            metrics,
            {
                "dt": dt_vis,
                "hough_lines": hough_lines,
                "hough_lines_a": hough_a,
                "hough_lines_b": hough_b,
                "overlay_raw": None,
                "overlay_top5": None,
                "preprocessed": mask_thin,
                "edges": edge_map,
            },
        )

    overlay_raw = _draw_model_lines(cv2.cvtColor(white_mask_raw_floor, cv2.COLOR_GRAY2BGR), best_H, (0, 255, 0))
    overlay_top5 = cv2.cvtColor(white_mask_raw_floor, cv2.COLOR_GRAY2BGR)
    colors = [(0, 255, 0), (0, 200, 255), (255, 200, 0), (255, 0, 255), (0, 255, 255)]
    for idx, (score_val, quad_pts, cfg) in enumerate(top5):
        color = colors[idx % len(colors)]
        pts = quad_pts.reshape(-1, 1, 2).astype(np.int32)
        cv2.polylines(overlay_top5, [pts], True, color, 2)
        label = f"{idx+1}:{score_val:.2f}"
        cv2.putText(overlay_top5, label, (int(pts[0][0][0]), int(pts[0][0][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return (
        best_H,
        metrics,
        {
            "dt": dt_vis,
            "hough_lines": hough_lines,
            "hough_lines_a": hough_a,
            "hough_lines_b": hough_b,
            "overlay_raw": overlay_raw,
            "overlay_top5": overlay_top5,
            "preprocessed": mask_thin,
            "edges": edge_map,
        },
    )


def fit_court_homography_from_raw_floor(
    white_mask_raw_floor: np.ndarray,
    frame_shape: Tuple[int, int, int] | Tuple[int, int],
    debug_dir: Optional[str] = None,
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    H, metrics, _ = _fit_court_homography_from_raw_floor_debug(white_mask_raw_floor, frame_shape)
    return H, metrics


def _confidence_terms(
    inlier_ratio: float,
    cover_ratio: float,
    p90_dist_px: float,
    area_ratio: float,
    a: float = 12.0,
    b: float = 10.0,
    c: float = 0.35,
    d: float = 12.0,
) -> Dict[str, float]:
    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-float(x)))

    s_inlier = _sigmoid(float(a) * (float(inlier_ratio) - 0.55))
    s_cover = _sigmoid(float(b) * (float(cover_ratio) - 0.50))
    s_p90 = _sigmoid(float(c) * (12.0 - float(p90_dist_px)))
    s_area = _sigmoid(float(d) * (float(area_ratio) - 0.10)) * _sigmoid(
        float(d) * (0.95 - float(area_ratio))
    )
    conf = float(np.clip(s_inlier * s_cover * s_p90 * s_area, 0.0, 1.0))
    return {
        "confidence": conf,
        "s_inlier": float(s_inlier),
        "s_cover": float(s_cover),
        "s_p90": float(s_p90),
        "s_area": float(s_area),
    }


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
    dt_oob: float = 255.0,
    clamp_max: float = 15.0,
) -> Dict[str, Any]:
    uv = project_points(H, Xw)
    h, w = dt.shape[:2]
    u = np.rint(uv[:, 0]).astype(np.int32)
    v = np.rint(uv[:, 1]).astype(np.int32)
    valid = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    r = np.zeros((uv.shape[0],), dtype=np.float32)
    if np.any(valid):
        r[valid] = dt[v[valid], u[valid]].astype(np.float32)
    r_raw = r.copy()
    r = np.clip(r, 0.0, float(clamp_max))
    num_valid = int(np.count_nonzero(valid))
    num_oob = int(uv.shape[0] - num_valid)
    w_valid = None
    if num_valid > 0:
        w_valid = weights[valid].astype(np.float32)
        huber_vals = _huber(r[valid], float(huber_delta))
        l_dist = float(np.average(huber_vals, weights=w_valid))
    else:
        l_dist = float(clamp_max)

    inlier_mask = (r < float(tau_px)) & valid
    num_inliers = int(np.count_nonzero(inlier_mask))
    inlier_ratio_unweighted = float(num_inliers) / float(max(num_valid, 1))
    if w_valid is not None and w_valid.size > 0:
        w_valid_sum = float(np.sum(w_valid))
    else:
        w_valid_sum = 0.0
    w_inlier = weights[inlier_mask].astype(np.float32) if np.any(inlier_mask) else np.zeros((0,), dtype=np.float32)
    w_inlier_sum = float(np.sum(w_inlier)) if w_inlier.size > 0 else 0.0
    if w_valid_sum > 1e-6:
        inlier_ratio = float(w_inlier_sum / w_valid_sum)
    else:
        inlier_ratio = float(inlier_ratio_unweighted)
    cover_ratio = float(inlier_ratio)
    inlier_dists = r[inlier_mask]
    inlier_dist_p90 = float(np.percentile(inlier_dists, 90)) if inlier_dists.size > 0 else None

    sample_dist_mean = float(np.mean(r[valid])) if np.any(valid) else None
    sample_dist_p50 = float(np.percentile(r[valid], 50)) if np.any(valid) else None
    sample_dist_p90 = float(np.percentile(r[valid], 90)) if np.any(valid) else None
    sample_dist_max = float(np.max(r[valid])) if np.any(valid) else None
    sample_dist_p90_raw = float(np.percentile(r_raw[valid], 90)) if np.any(valid) else None
    sample_dist_p90_weighted = _weighted_percentile(r[valid], w_valid, 90.0) if np.any(valid) and w_valid is not None else None
    sample_dist_p90_raw_weighted = (
        _weighted_percentile(r_raw[valid], w_valid, 90.0) if np.any(valid) and w_valid is not None else None
    )

    l_cover = 0.0
    if cover_points is not None and cover_points.size > 0:
        segs = _project_line_segments(H)
        cover_dist = _dist_points_to_segments(cover_points, segs)
        cover_dist = np.clip(cover_dist, 0.0, float(clamp_max))
        l_cover = float(np.mean(_huber(cover_dist, float(huber_delta))))

    corners_uv = project_points(H, get_bwf_corners())
    area_ratio, penalty_area = _area_ratio_and_penalty(corners_uv, floor_bbox)
    diag = float(np.hypot(w, h))
    penalty_inside = _outside_bbox_penalty(corners_uv, floor_bbox) / float(max(diag, 1.0))
    penalty_area *= 20.0
    penalty_inside *= 5.0
    l_reg = float(penalty_area + penalty_inside)
    cover_target = 0.55
    cover_penalty = max(0.0, float(cover_target) - cover_ratio)
    cover_penalty_weight = 50.0

    valid_ratio = float(num_valid) / float(max(uv.shape[0], 1))
    valid_ratio_penalty = float(1.0 - valid_ratio)
    valid_ratio_weight = 2.0
    total_loss = float(
        w_dist * l_dist
        + w_cover * l_cover
        + w_reg * l_reg
        + cover_penalty_weight * cover_penalty
        + valid_ratio_weight * valid_ratio_penalty
    )
    return {
        "loss": total_loss,
        "l_dist": float(l_dist),
        "l_cover": float(l_cover),
        "l_reg": float(l_reg),
        "valid_ratio": float(valid_ratio),
        "valid_ratio_penalty": float(valid_ratio_penalty),
        "area_ratio": float(area_ratio),
        "cover_ratio": float(cover_ratio),
        "cover_ratio_unweighted": float(inlier_ratio_unweighted),
        "cover_penalty": float(cover_penalty),
        "inlier_ratio": float(inlier_ratio),
        "inlier_ratio_unweighted": float(inlier_ratio_unweighted),
        "inlier_ratio_weighted": float(inlier_ratio),
        "weighted_valid_sum": float(w_valid_sum),
        "weighted_inlier_sum": float(w_inlier_sum),
        "inlier_dist_p90": inlier_dist_p90,
        "num_inliers": num_inliers,
        "num_valid_samples": num_valid,
        "dt_oob_count": num_oob,
        "num_samples": int(uv.shape[0]),
        "sample_dist_mean": sample_dist_mean,
        "sample_dist_p50": sample_dist_p50,
        "sample_dist_p90": sample_dist_p90,
        "sample_dist_max": sample_dist_max,
        "sample_dist_p90_raw": sample_dist_p90_raw,
        "sample_dist_p90_weighted": sample_dist_p90_weighted,
        "sample_dist_p90_raw_weighted": sample_dist_p90_raw_weighted,
        "sample_dists": r,
        "sample_uv": uv,
    }


def _init_from_line_clusters(
    horiz_lines: list[LineSeg],
    vert_lines: list[LineSeg],
    dt: np.ndarray,
    Xw: np.ndarray,
    weights: np.ndarray,
    cover_points: np.ndarray,
    floor_bbox: Tuple[int, int, int, int],
    world_corners: np.ndarray,
    ratio_ref: float = 13.40 / 6.10,
    ratio_tol: float = 0.40,
    top_k: int = 3,
    min_area_ratio: float = 0.01,
    dt_oob: float = 255.0,
    vp_long: Optional[np.ndarray] = None,
    vp_short: Optional[np.ndarray] = None,
) -> Tuple[Optional[np.ndarray], Optional[Dict[str, Any]]]:
    if len(horiz_lines) < 2 or len(vert_lines) < 2:
        return None, None
    h, w = dt.shape[:2]
    # Use floor_bbox center (not full-frame center) to decide outer/inner lines.
    # This avoids picking diagonals as extremes when ROI is bottom-biased.
    if floor_bbox is not None and len(floor_bbox) == 4:
        bx0, by0, bx1, by1 = [float(v) for v in floor_bbox]
        center = (0.5 * (bx0 + bx1), 0.5 * (by0 + by1))
        min_sep_frac = _env_float("BADC_RAW_FLOOR_MIN_SEP_FRAC", 0.0)
        min_sep = min_sep_frac * float(min(max(1.0, bx1 - bx0), max(1.0, by1 - by0)))
    else:
        center = (float(w) * 0.5, float(h) * 0.5)
        min_sep_frac = _env_float("BADC_RAW_FLOOR_MIN_SEP_FRAC", 0.0)
        min_sep = min_sep_frac * float(min(h, w))

    def _select_outer(lines: list[LineSeg]) -> Tuple[list[LineSeg], list[LineSeg]]:
        # Keep extremes by signed distance, even if weak, then fill by weight.
        scored = [(ln, _line_signed_distance(ln.line, center)) for ln in lines]
        scored.sort(key=lambda item: item[1])

        span = max(2, int(top_k) * 2)
        k = max(1, int(top_k))

        low_scored = scored[:span]
        high_scored = scored[-span:]

        extreme_low = low_scored[0][0]
        extreme_high = high_scored[-1][0]

        def _fill_by_weight(scored_list, extreme_ln):
            rest = [ln for ln, _ in scored_list if ln is not extreme_ln]
            rest.sort(key=lambda ln: float(ln.weight), reverse=True)
            out = [extreme_ln]
            for ln in rest:
                if ln not in out:
                    out.append(ln)
                if len(out) >= k:
                    break
            return out

        low = _fill_by_weight(low_scored, extreme_low)
        high = _fill_by_weight(high_scored, extreme_high)
        return low, high

    h_low, h_high = _select_outer(horiz_lines)
    v_low, v_high = _select_outer(vert_lines)
    if not h_low or not h_high or not v_low or not v_high:
        return None, None

    min_area = 1.0
    best_H = None
    best_info: Optional[Dict[str, Any]] = None
    best_score = float("inf")

    for h1 in h_low:
        for h2 in h_high:
            if h1 is h2:
                continue
            d_h1 = _line_signed_distance(h1.line, center)
            d_h2 = _line_signed_distance(h2.line, center)
            if abs(d_h1 - d_h2) < min_sep:
                continue
            for v1 in v_low:
                for v2 in v_high:
                    if v1 is v2:
                        continue
                    d_v1 = _line_signed_distance(v1.line, center)
                    d_v2 = _line_signed_distance(v2.line, center)
                    if abs(d_v1 - d_v2) < min_sep:
                        continue
                    p1 = _intersect_lines(h1.line, v1.line)
                    p2 = _intersect_lines(h1.line, v2.line)
                    p3 = _intersect_lines(h2.line, v2.line)
                    p4 = _intersect_lines(h2.line, v1.line)
                    if p1 is None or p2 is None or p3 is None or p4 is None:
                        continue
                    quad = np.stack([p1, p2, p3, p4], axis=0)
                    if not np.all(np.isfinite(quad)):
                        continue
                    if not _is_convex_quad(quad):
                        continue
                    area = _quad_area(quad)
                    if area < min_area:
                        continue
                    ordered = _order_corners_lb_rb_rt_lt(quad)
                    ok, _gate_reason, gate_metrics = _passes_geom_gates(ordered, floor_bbox)
                    if not ok:
                        continue
                    H = cv2.getPerspectiveTransform(world_corners.astype(np.float32), ordered.astype(np.float32))
                    if not np.all(np.isfinite(H)):
                        continue
                    info = _compute_loss_terms(
                        H,
                        dt=dt,
                        Xw=Xw,
                        weights=weights,
                        cover_points=cover_points,
                        floor_bbox=floor_bbox,
                        dt_oob=dt_oob,
                    )
                    score, mixer_parts = _mixer_score_candidate(
                        ordered,
                        info,
                        floor_bbox,
                        vp_long=vp_long,
                        vp_short=vp_short,
                    )
                    if score < best_score:
                        best_score = float(score)
                        best_H = H
                        best_info = info
                        best_info.update(gate_metrics)
                        best_info.update(mixer_parts)
    return best_H, best_info

def ransac_init_homography(
    white_mask: np.ndarray,
    white_mask_raw: np.ndarray,
    dt: np.ndarray,
    world_corners: np.ndarray,
    Xw: np.ndarray,
    weights: np.ndarray,
    cover_points: np.ndarray,
    floor_bbox: Tuple[int, int, int, int],
    iters: int = 2000,
    tau_px: float = 3.0,
    min_area_ratio: float = 0.01,
    dt_oob: float = 255.0,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[Optional[np.ndarray], Optional[Dict[str, Any]]]:
    ys, xs = np.where(white_mask > 0)
    if xs.size < 4:
        return None, None
    coords = np.stack([xs, ys], axis=1).astype(np.float32)
    rng = rng or np.random.default_rng()
    h, w = white_mask.shape[:2]
    min_area = 1.0
    best_H: Optional[np.ndarray] = None
    best_info: Optional[Dict[str, Any]] = None
    best_score: float = float("inf")
    world = world_corners.astype(np.float32)

    x1, y1, x2, y2 = floor_bbox
    support_mask = np.zeros_like(white_mask_raw)
    support_mask[y1 : y2 + 1, x1 : x2 + 1] = white_mask_raw[y1 : y2 + 1, x1 : x2 + 1]
    lsd_lines, num_lsd_raw = _detect_lsd_lines(
        white_mask,
        support_mask,
        floor_bbox,
        min_length=20.0,
    )
    cluster_a, cluster_b, cluster_info = _cluster_lines_by_angle(lsd_lines)
    if cluster_info.get("cluster_ratio", 0.0) < 0.25:
        sum_a = float(cluster_info.get("sum_len_a", 0.0))
        sum_b = float(cluster_info.get("sum_len_b", 0.0))
        main = cluster_a if sum_a >= sum_b else cluster_b
        missing_is_a = sum_a < sum_b
        vp_main = _estimate_vanishing_point([ln.line for ln in main], rng)
        if isinstance(vp_main, np.ndarray):
            center = (float(w) * 0.5, float(h) * 0.5)
            base_angle = math.atan2(center[1] - float(vp_main[1]), center[0] - float(vp_main[0]))
            base_angle = float(base_angle % math.pi)
            perp = float((base_angle + math.pi * 0.5) % math.pi)
            alt = [ln for ln in lsd_lines if _angle_distance(ln.theta, perp) < math.radians(15.0)]
            if alt:
                if missing_is_a:
                    cluster_a = alt
                else:
                    cluster_b = alt
                sum_a = float(np.sum([ln.weight for ln in cluster_a])) if cluster_a else 0.0
                sum_b = float(np.sum([ln.weight for ln in cluster_b])) if cluster_b else 0.0
                denom = max(sum_a, sum_b, 1e-6)
                cluster_info = {
                    "mean_angle_a": _mean_angle(cluster_a),
                    "mean_angle_b": _mean_angle(cluster_b),
                    "sum_len_a": sum_a,
                    "sum_len_b": sum_b,
                    "cluster_ratio": float(min(sum_a, sum_b) / denom),
                }
    horiz, vert, role_info = _assign_cluster_roles(cluster_a, cluster_b)

    vp_a = _estimate_vanishing_point([ln.line for ln in cluster_a], rng)
    vp_b = _estimate_vanishing_point([ln.line for ln in cluster_b], rng)
    horiz_is_a = horiz is cluster_a
    vp_short = vp_a if horiz_is_a else vp_b
    vp_long = vp_b if horiz_is_a else vp_a

    if horiz is not None and vert is not None:
        H_line, info_line = _init_from_line_clusters(
            horiz,
            vert,
            dt=dt,
            Xw=Xw,
            weights=weights,
            cover_points=cover_points,
            floor_bbox=floor_bbox,
            world_corners=world,
            ratio_ref=13.40 / 6.10,
            dt_oob=dt_oob,
            vp_long=vp_long,
            vp_short=vp_short,
        )
        if H_line is not None and info_line is not None:
            best_H = H_line
            best_info = info_line
            best_score = float(info_line.get("mixer_score", info_line.get("loss", float("inf"))))

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

    if best_H is None:
        for _ in range(int(iters)):
            pts = _sample_from_quadrants()
            if pts is None:
                idx = rng.choice(coords.shape[0], size=4, replace=False)
                pts = coords[idx]
            ordered = _order_corners_lb_rb_rt_lt(pts)
            if not _is_convex_quad(ordered):
                continue
            ok, _gate_reason, gate_metrics = _passes_geom_gates(ordered, floor_bbox)
            if not ok:
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
                dt_oob=dt_oob,
            )
            score, mixer_parts = _mixer_score_candidate(
                ordered,
                info,
                floor_bbox,
                vp_long=vp_long,
                vp_short=vp_short,
            )
            area_ratio_img = float(info.get("area_ratio", 0.0))

            # For service-box configs, penalize tiny quads and add a base penalty so full-court wins when plausible.
            if str(cfg_name).startswith("service"):
                service_min_area = float(os.getenv("BADC_RAW_FLOOR_SERVICE_MIN_AREA_RATIO", "0.02"))
                service_pen = float(os.getenv("BADC_RAW_FLOOR_SERVICE_PEN", "250.0"))
                if area_ratio_img < service_min_area:
                    rejects["area"] += 1
                    continue
                score += service_pen
            if score < best_score:
                best_score = float(score)
                best_H = H
                best_info = info
                best_info.update(gate_metrics)
                best_info.update(mixer_parts)

    line_info = {
        "vp_long": vp_long.tolist() if isinstance(vp_long, np.ndarray) else None,
        "vp_short": vp_short.tolist() if isinstance(vp_short, np.ndarray) else None,
        "num_lsd_raw": int(num_lsd_raw),
        "num_lsd_kept": int(len(lsd_lines)),
        "dirA_count": int(len(cluster_a)),
        "dirB_count": int(len(cluster_b)),
        "sum_len_a": float(cluster_info.get("sum_len_a", 0.0)),
        "sum_len_b": float(cluster_info.get("sum_len_b", 0.0)),
        "cluster_ratio": float(cluster_info.get("cluster_ratio", 0.0)),
        "mean_angle_a": role_info.get("mean_angle_a"),
        "mean_angle_b": role_info.get("mean_angle_b"),
        "lsd_lines_a": [
            [float(ln.p1[0]), float(ln.p1[1]), float(ln.p2[0]), float(ln.p2[1])] for ln in cluster_a
        ],
        "lsd_lines_b": [
            [float(ln.p1[0]), float(ln.p1[1]), float(ln.p2[0]), float(ln.p2[1])] for ln in cluster_b
        ],
    }
    if best_info is None:
        return None, line_info
    best_info.update(line_info)
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
    dt_oob: float = 255.0,
) -> Tuple[np.ndarray, float]:
    p = p0.astype(np.float64).copy()
    for _ in range(int(max_iter)):
        r = _residual_vector(p, dt, Xw, weights, cover_points, floor_bbox, dt_oob=dt_oob)
        cost = 0.5 * float(np.dot(r, r))
        J = np.zeros((r.shape[0], 8), dtype=np.float64)
        for i in range(8):
            p_eps = p.copy()
            p_eps[i] += float(eps)
            r_eps = _residual_vector(p_eps, dt, Xw, weights, cover_points, floor_bbox, dt_oob=dt_oob)
            J[:, i] = (r_eps - r) / float(eps)
        A = J.T @ J + float(lam) * np.eye(8, dtype=np.float64)
        g = J.T @ r
        try:
            delta = -np.linalg.solve(A, g)
        except np.linalg.LinAlgError:
            break
        p_new = p + delta
        r_new = _residual_vector(p_new, dt, Xw, weights, cover_points, floor_bbox, dt_oob=dt_oob)
        new_cost = 0.5 * float(np.dot(r_new, r_new))
        if new_cost < cost:
            p = p_new
            lam *= 0.7
        else:
            lam *= 2.0
    final_r = _residual_vector(p, dt, Xw, weights, cover_points, floor_bbox, dt_oob=dt_oob)
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
    dt_oob: float = 255.0,
) -> Tuple[np.ndarray, float]:
    p0 = _H_to_p(H_init)
    if _HAS_SCIPY and least_squares is not None:
        res = least_squares(
            lambda p: _residual_vector(p, dt, Xw, weights, cover_points, floor_bbox, dt_oob=dt_oob),
            p0,
            loss="huber",
            f_scale=3.0,
            max_nfev=int(max_nfev),
        )
        return _p_to_H(res.x), float(res.cost)
    p_ref, cost = _lm_numeric(
        p0, dt, Xw, weights, cover_points, floor_bbox, max_iter=int(max_nfev), dt_oob=dt_oob
    )
    return _p_to_H(p_ref), float(cost)


def draw_debug_overlay(
    frame_bgr: np.ndarray,
    white_mask: np.ndarray,
    H: np.ndarray,
    conf: float,
    cost_total: float,
    s_inlier: float,
    s_cover: float,
    s_p90: float,
    s_area: float,
    inlier_ratio: float,
    mean_dist_px: float,
    p90_dist_px: float,
    white_mask_ratio: float,
    cover_ratio: float,
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

    title = (
        f"conf={conf:.2f} cost={cost_total:.2f} inlier={inlier_ratio:.2f} "
        f"mean={mean_dist_px:.2f}px p90={p90_dist_px:.2f}px"
    )
    cv2.putText(out, title, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
    cv2.putText(out, title, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    row2 = f"s_inlier={s_inlier:.2f} s_cover={s_cover:.2f} s_p90={s_p90:.2f} s_area={s_area:.2f}"
    cv2.putText(out, row2, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
    cv2.putText(out, row2, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    row3 = (
        f"mask={white_mask_ratio:.3f} cover={cover_ratio:.2f} "
        f"Ld={l_dist:.3f} Lc={l_cover:.3f} Lr={l_reg:.3f} area={area_ratio:.3f}"
    )
    cv2.putText(out, row3, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
    cv2.putText(out, row3, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    row3 = f"dt_min={dt_min:.2f} dt_mean={dt_mean:.2f} dt_p90={dt_p90:.2f}" if dt_min is not None else "dt_min=NA dt_mean=NA dt_p90=NA"
    cv2.putText(out, row3, (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
    cv2.putText(out, row3, (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    if sample_dist_mean is not None:
        row4 = (
            f"samp_mean={sample_dist_mean:.2f} p50={sample_dist_p50:.2f} "
            f"p90={sample_dist_p90:.2f} max={sample_dist_max:.2f}"
        )
    else:
        row4 = "samp_mean=NA p50=NA p90=NA max=NA"
    cv2.putText(out, row4, (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
    cv2.putText(out, row4, (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return out


def _remove_big_blobs(
    mask: np.ndarray,
    roi_mask: np.ndarray,
    area_ratio: float = 0.01,
    extent_thresh: float = 0.25,
    huge_ratio: float = 0.02,
    min_aspect: float = 6.0,
    thin_thresh: float = 6.0,
    huge_extent_thresh: float = 0.14,
) -> np.ndarray:
    h, w = mask.shape[:2]
    roi_area = int(np.count_nonzero(roi_mask))
    if roi_area <= 0:
        return mask.copy()
    max_area = float(roi_area) * float(area_ratio)
    huge_area = float(roi_area) * float(huge_ratio)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros_like(mask)
    for idx in range(1, num):
        area = float(stats[idx, cv2.CC_STAT_AREA])
        if area <= 0.0:
            continue
        x, y, bw, bh, _ = stats[idx]
        bw = max(1, int(bw))
        bh = max(1, int(bh))
        aspect = float(max(bw, bh)) / float(max(1, min(bw, bh)))
        extent = float(area) / float(max(bw * bh, 1))
        is_line_like = aspect >= float(min_aspect)
        thickness = float(area) / float(bw + bh + 1.0)
        is_thin = thickness <= float(thin_thresh)
        drop_blob = (
            (area > huge_area and extent > float(huge_extent_thresh))
            or (area > max_area and extent > float(extent_thresh))
        )
        if is_line_like or is_thin or not drop_blob:
            keep[labels == idx] = 255
    return keep


def _remove_solid_mid_blobs(
    mask: np.ndarray,
    roi_mask: Optional[np.ndarray],
    area_min: int = 120,
    area_max: int = 2500,
    extent_min: float = 0.38,
    aspect_max: float = 3.0,
) -> np.ndarray:
    """Remove medium, solid, non-elongated blobs (e.g., shoes) without killing sparse line nets."""
    if roi_mask is None:
        roi_mask = np.ones_like(mask)
    work = cv2.bitwise_and(mask, roi_mask)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(work, connectivity=8)
    out = mask.copy()
    for idx in range(1, num):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area < int(area_min) or area > int(area_max):
            continue
        bw = max(1, int(stats[idx, cv2.CC_STAT_WIDTH]))
        bh = max(1, int(stats[idx, cv2.CC_STAT_HEIGHT]))
        extent = float(area) / float(max(bw * bh, 1))
        aspect = float(max(bw, bh)) / float(max(1, min(bw, bh)))
        if extent >= float(extent_min) and aspect <= float(aspect_max):
            out[labels == idx] = 0
    return out


def _remove_thick_solid_components_by_dt(
    mask: np.ndarray,
    roi_mask: Optional[np.ndarray],
    max_r: float = 3.2,
    extent_min: float = 0.35,
    aspect_max: float = 4.0,
    area_min: int = 120,
) -> np.ndarray:
    """
    Remove whole connected components that are thick (DT max > max_r) and solid-ish,
    avoiding hollow-line artifacts from pixelwise thinning.
    """
    if mask.ndim == 3:
        m = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    else:
        m = mask.copy()
    _, m = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)
    if roi_mask is None:
        roi_mask = np.ones_like(m)
    work = cv2.bitwise_and(m, roi_mask)
    dist = cv2.distanceTransform(work, cv2.DIST_L2, 3)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(work, connectivity=8)
    out = m.copy()
    for idx in range(1, num):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area < int(area_min):
            continue
        bw = max(1, int(stats[idx, cv2.CC_STAT_WIDTH]))
        bh = max(1, int(stats[idx, cv2.CC_STAT_HEIGHT]))
        extent = float(area) / float(max(bw * bh, 1))
        aspect = float(max(bw, bh)) / float(max(1, min(bw, bh)))
        cc_mask = labels == idx
        max_dt = float(dist[cc_mask].max()) if np.any(cc_mask) else 0.0
        if (max_dt > float(max_r)) and (extent >= float(extent_min)) and (aspect <= float(aspect_max)):
            out[cc_mask] = 0
    return out


def _clamp_by_thickness_dt(mask: np.ndarray, max_r: float = 3.2) -> np.ndarray:
    """
    Remove thick white regions while keeping thin line strokes using distance transform.
    max_r ~ maximum allowed distance-to-background in pixels.
    """
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    _, m = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    dist = cv2.distanceTransform(m, cv2.DIST_L2, 3)
    keep = (dist <= float(max_r))
    out = np.zeros_like(m)
    out[keep & (m > 0)] = 255
    return out


def _remove_mid_solid_blobs_near_bottom(
    mask: np.ndarray,
    roi_mask: np.ndarray,
    area_min: int = 120,
    area_max: int = 12000,
    extent_min: float = 0.35,
    aspect_max: float = 3.2,
    y_center_min_ratio: float = 0.62,
) -> np.ndarray:
    """
    Remove mid-size solid-ish blobs near the bottom (e.g., shoes), while keeping lines.
    """
    if mask.ndim == 3:
        m = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    else:
        m = mask.copy()
    _, m = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)
    if roi_mask is None:
        roi_mask = np.ones_like(m)
    work = cv2.bitwise_and(m, roi_mask)
    h, w = work.shape[:2]
    num, labels, stats, _ = cv2.connectedComponentsWithStats(work, connectivity=8)
    out = m.copy()
    for idx in range(1, num):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area < int(area_min) or area > int(area_max):
            continue
        bw = max(1, int(stats[idx, cv2.CC_STAT_WIDTH]))
        bh = max(1, int(stats[idx, cv2.CC_STAT_HEIGHT]))
        aspect = float(max(bw, bh)) / float(max(1, min(bw, bh)))
        extent = float(area) / float(max(bw * bh, 1))
        cy = float(stats[idx, cv2.CC_STAT_TOP] + 0.5 * bh)
        if (cy / float(max(h, 1))) < float(y_center_min_ratio):
            continue
        if extent >= float(extent_min) and aspect <= float(aspect_max):
            out[labels == idx] = 0
    return out


def preprocess_raw_floor(
    mask: np.ndarray,
    roi_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[Tuple[int, int]], Optional[np.ndarray]]:
    if roi_mask is None:
        roi_mask = np.ones_like(mask)
    work = mask.copy()
    h, w = work.shape[:2]
    y_min = int(round(0.35 * float(h)))
    if y_min > 0:
        work[:y_min, :] = 0
    if work.ndim == 3:
        work = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    _, work = cv2.threshold(work, 127, 255, cv2.THRESH_BINARY)
    work = cv2.bitwise_and(work, roi_mask)
    blob_mask = _detect_blob_mask(
        work,
        roi_mask=roi_mask,
        dt_thr=5.5,
        min_core_area=800,
        dilate_ksize=17,
    )
    keep_lines = _keep_thin_structures_in_blob(work, blob_mask)
    base_no_blob = cv2.bitwise_and(work, cv2.bitwise_not(blob_mask))
    mask_for_edges = cv2.bitwise_or(base_no_blob, keep_lines)
    edges = cv2.Canny(mask_for_edges, 30, 110)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    ys, xs = np.where(roi_mask > 0)
    floor_y0 = int(ys.min()) if xs.size > 0 else 0
    edges, band = _suppress_bright_band_on_edges(
        edges,
        floor_y0=floor_y0,
        search_height_frac=0.30,
        row_white_ratio_thr=0.20,
        min_run=10,
        pad=8,
    )
    return mask_for_edges, edges, band, blob_mask


def _fit_court_homography_hough(
    frame_bgr: np.ndarray,
    person_boxes: Optional[Sequence[Sequence[float]]] = None,
    linepix_mask_override: Optional[np.ndarray] = None,
    skip_component_split: bool = False,
    component_id: Optional[int] = None,
    component_bbox: Optional[Tuple[int, int, int, int]] = None,
    component_area: Optional[int] = None,
) -> CourtFitResult:
    rng = np.random.default_rng(0)
    save_raw_debug = _env_flag("BADC_SAVE_RAW_FLOOR_DEBUG", True)
    raw_debug_save: Optional[Dict[str, Any]] = None
    debug_dir: Optional[str] = None
    debug_prefix: str = "court_fit"
    floor_mask, floor_debug = get_floor_roi_mask_debug(frame_bgr)
    # Lock to legacy HV split to keep oriA/oriB behavior aligned with backup logic.
    legacy_ori_split = True
    # Keep these defined for early-failure returns / debug.
    linepix_mask_pre: Optional[np.ndarray] = None
    floor_gate_mask: Optional[np.ndarray] = None
    disable_anchor_prior = os.getenv("BADC_DISABLE_ANCHOR_PRIOR", "").strip() == "1"
    enable_anchor_prior = os.getenv("BADC_ENABLE_ANCHOR_PRIOR", "").strip() == "1" and not disable_anchor_prior
    manual_path = os.path.join("reports", "demo_friend", "manual_vp_corners.json")
    manual = _load_manual_vp_corners(manual_path) if enable_anchor_prior else None
    manual_vp_left = None
    manual_vp_bottom = None
    manual_lt = None
    manual_rt = None
    manual_rb = None
    manual_lb = None
    manual_vis = {}
    manual_quad = None
    manual_metrics: Dict[str, Any] = {}
    if disable_anchor_prior:
        manual_metrics["manual_present"] = False
        # Widen ROI if lab-seed mask is too narrow without manual anchors.
        x0, y0, x1, y1 = _white_mask_stats(floor_mask)[2]
        h, w = frame_bgr.shape[:2]
        if (x1 - x0) < int(0.6 * w):
            floor_mask = floor_mask.copy()
            floor_mask[y0 : y1 + 1, :] = 255
            floor_debug["fallback_mode"] = "lab_seed_widen"
    if isinstance(manual, dict):
        try:
            corners = manual.get("court_corners", {})
            manual_lt = corners.get("LT")
            manual_rt = corners.get("RT")
            manual_rb = corners.get("RB")
            manual_lb = corners.get("LB")
            manual_vis = {
                "LT": bool(manual_lt and manual_lt.get("visible", True)),
                "RT": bool(manual_rt and manual_rt.get("visible", True)),
                "RB": bool(manual_rb and manual_rb.get("visible", True)),
            }
            vps = manual.get("vanishing_points", {})
            manual_vp_left = vps.get("VP_left")
            manual_vp_bottom = vps.get("VP_bottom")
            if manual_lb and manual_rb and manual_rt and manual_lt:
                manual_quad = np.array(
                    [
                        [float(manual_lb.get("x")), float(manual_lb.get("y"))],
                        [float(manual_rb.get("x")), float(manual_rb.get("y"))],
                        [float(manual_rt.get("x")), float(manual_rt.get("y"))],
                        [float(manual_lt.get("x")), float(manual_lt.get("y"))],
                    ],
                    dtype=np.float32,
                )
            manual_metrics["manual_present"] = True
        except Exception:
            manual = None
            manual_metrics["manual_present"] = False
    manual_x_range = None
    if manual_lt and manual_rt and manual_rb:
        h, w = frame_bgr.shape[:2]
        x0, y0, x1, y1 = _white_mask_stats(floor_mask)[2]
        pad = max(24, int(round(0.05 * float(min(h, w)))))
        xs = [manual_lt["x"], manual_rt["x"], manual_rb["x"]]
        ys = [manual_lt["y"], manual_rt["y"], manual_rb["y"]]
        x0 = int(min(x0, min(xs) - pad))
        x1 = int(max(x1, max(xs) + pad))
        y0 = int(min(y0, min(ys) - pad))
        y1 = int(max(y1, max(ys) + pad))
        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(w - 1, x1)
        y1 = min(h - 1, y1)
        if x1 > x0 and y1 > y0:
            floor_mask = floor_mask.copy()
            floor_mask[y0 : y1 + 1, x0 : x1 + 1] = 255
            floor_debug["manual_floor_bbox"] = [int(x0), int(y0), int(x1), int(y1)]
            floor_debug["manual_present"] = True
        manual_x0 = int(max(0, min(xs) - pad))
        manual_x1 = int(min(w - 1, max(xs) + pad))
        floor_mask[:, :manual_x0] = 0
        floor_mask[:, manual_x1 + 1 :] = 0
        floor_debug["floor_x_tightened"] = True
        floor_debug["floor_x_range"] = [int(manual_x0), int(manual_x1)]
        manual_x_range = (manual_x0, manual_x1)
    floor_roi_mask = floor_mask
    floor_roi_source = "floor_mask"
    floor_y_cut = floor_debug.get("floor_y_cut")
    top_guard_meta = floor_debug.get("floor_roi_top_guard")
    top_guard_applied = bool(isinstance(top_guard_meta, dict) and top_guard_meta.get("applied", False))
    x0, y0, x1, y1 = _white_mask_stats(floor_mask)[2]

    # Ensure dimensions are defined before padding
    h, w = floor_mask.shape[:2]

    # Expand floor ROI bbox via env knobs to avoid missing near-border lines.
    pad_px = int(os.getenv("BADC_FLOOR_ROI_BBOX_PAD_PX", "0") or 0)
    pad_frac = float(os.getenv("BADC_FLOOR_ROI_BBOX_PAD_FRAC", "0.0") or 0.0)
    pad = max(pad_px, int(round(pad_frac * float(min(h, w)))))
    if pad > 0:
        floor_debug["floor_roi_bbox_pad"] = int(pad)

    x0p = max(0, int(x0) - pad)
    y0p = max(0, int(y0) - pad)
    x1p = min(w - 1, int(x1) + pad)
    y1p = min(h - 1, int(y1) + pad)
    bbox_mask = np.zeros_like(floor_mask)
    orig_floor_mask = floor_mask.copy()
    if x1p > x0p and y1p > y0p:
        bbox_mask[y0p : y1p + 1, x0p : x1p + 1] = 255
    if floor_y_cut is not None:
        bbox_mask[: int(floor_y_cut), :] = 0
    green_ratio = None
    bottom_only = 0
    y_split_for_floor = None
    green_in_bbox = np.zeros_like(bbox_mask)
    green_mask = floor_debug.get("green_mask")
    green_direct_apply = bool(isinstance(green_mask, np.ndarray) and green_mask.size > 0) and _env_flag(
        "BADC_FLOOR_ROI_DIRECT_USE_GREEN_MASK", True
    )
    if legacy_ori_split and ("BADC_FLOOR_ROI_DIRECT_USE_GREEN_MASK" not in os.environ):
        green_direct_apply = False
        floor_debug["floor_roi_green_direct_disabled_by_legacy_ori"] = 1
    green_direct_strict = bool(green_direct_apply and _env_flag("BADC_FLOOR_ROI_DIRECT_STRICT", True))
    if isinstance(green_mask, np.ndarray) and green_mask.size > 0:
        ys, xs = np.where(bbox_mask > 0)
        if xs.size > 0:
            green_ratio = float(np.mean(green_mask[ys, xs] > 0))
        if green_direct_apply:
            green_bin = ((green_mask > 0).astype(np.uint8) * 255)
            green_in_bbox = cv2.bitwise_and(green_bin, bbox_mask)
            if _env_flag("BADC_FLOOR_ROI_DIRECT_CLIP_TO_BBOX", False):
                floor_roi_mask = green_in_bbox
                floor_roi_source = "green_direct_mask_bbox"
            else:
                floor_roi_mask = green_bin
                floor_roi_source = "green_direct_mask"
            floor_debug["floor_roi_green_direct_applied"] = 1
            floor_debug["floor_roi_direct_strict"] = int(green_direct_strict)
        elif green_ratio is None or green_ratio < 0.15 or green_ratio > 0.95:
            floor_roi_mask = bbox_mask
            floor_roi_source = "bbox_fallback"
        elif top_guard_applied and _env_flag("BADC_FLOOR_ROI_TOP_GUARD_SKIP_GREEN_FLOOD", False):
            # Top-guard means the initial green coverage was unreliable; re-running
            # flood fill on green often collapses ROI back to near-half only.
            floor_roi_mask = bbox_mask
            floor_roi_source = "bbox_top_guard"
            floor_debug["floor_roi_green_flood_skipped_by_top_guard"] = 1
        else:
            # --- Flood-fill based pick of main floor region (bottom seeds) ---
            green_bin = ((green_mask > 0).astype(np.uint8) * 255)
            green_in_bbox = cv2.bitwise_and(green_bin, bbox_mask)

            h, w = green_in_bbox.shape[:2]
            seed_y_frac = float(os.getenv("BADC_FLOOR_ROI_SEED_Y_FRAC", "0.93") or 0.93)
            seed_y = int(round(seed_y_frac * float(max(h - 1, 0))))
            seed_xs_str = os.getenv("BADC_FLOOR_ROI_SEED_XS", "0.20,0.40,0.60,0.80")
            seed_x_fracs = []
            for s in seed_xs_str.split(","):
                s = s.strip()
                if not s:
                    continue
                try:
                    seed_x_fracs.append(float(s))
                except Exception:
                    pass
            if not seed_x_fracs:
                seed_x_fracs = [0.2, 0.4, 0.6, 0.8]
            radius = int(os.getenv("BADC_FLOOR_ROI_SEED_RADIUS", "60") or 60)
            radius = max(5, radius)

            picked = np.zeros_like(green_in_bbox)

            def _find_nearest_green(xc: int, yc: int, r: int):
                x0p = max(0, xc - r)
                x1p = min(w - 1, xc + r)
                y0p = max(0, yc - r)
                y1p = min(h - 1, yc + r)
                win = green_in_bbox[y0p : y1p + 1, x0p : x1p + 1]
                ys, xs = np.where(win > 0)
                if len(xs) == 0:
                    return None
                ix = int(np.median(xs)) + x0p
                iy = int(np.median(ys)) + y0p
                return (ix, iy)

            for fx in seed_x_fracs:
                sx = int(round(fx * float(max(w - 1, 0))))
                pt = _find_nearest_green(sx, seed_y, radius)
                if pt is None:
                    continue
                ff_img = green_in_bbox.copy()
                ff_mask = np.zeros((h + 2, w + 2), np.uint8)
                cv2.floodFill(ff_img, ff_mask, seedPoint=pt, newVal=255)
                comp = (ff_mask[1:-1, 1:-1] > 0).astype(np.uint8) * 255
                picked = cv2.bitwise_or(picked, comp)

            picked_area = int(np.count_nonzero(picked))
            min_area_frac = float(os.getenv("BADC_FLOOR_ROI_PICK_MIN_AREA_FRAC", "0.01") or 0.01)
            if picked_area < int(min_area_frac * h * w):
                picked = green_in_bbox
                floor_roi_source = "green_flood_fallback"
            else:
                floor_roi_source = "green_flood"
            floor_debug["floor_roi_picked_area"] = int(np.count_nonzero(picked))

            # gentle close to fill small holes / gaps
            close_k = int(os.getenv("BADC_FLOOR_ROI_GREEN_CLOSE", "19") or 19)
            close_k = max(3, close_k)
            if close_k % 2 == 0:
                close_k += 1
            k_close = np.ones((close_k, close_k), np.uint8)
            picked = cv2.morphologyEx(picked, cv2.MORPH_CLOSE, k_close, iterations=1)
            floor_debug["floor_roi_green_close"] = int(close_k)

            # small dilation (optionally only bottom part to avoid expanding upwards)
            d = int(os.getenv("BADC_FLOOR_ROI_GREEN_DILATE", "15") or 15)
            d = max(3, d)
            if d % 2 == 0:
                d += 1
            iters = int(os.getenv("BADC_FLOOR_ROI_GREEN_DILATE_ITERS", "1") or 1)
            iters = max(1, int(iters))
            h_dil, w_dil = picked.shape[:2]
            bottom_only = int(os.getenv("BADC_FLOOR_ROI_DILATE_BOTTOM_ONLY", "0") or 0)
            y_frac = float(os.getenv("BADC_FLOOR_ROI_DILATE_Y_FRAC", "0.55") or 0.55)
            # Split based on original floor mask (more stable than picked-only)
            y_split_for_floor = None
            if bottom_only:
                ys0 = np.where(orig_floor_mask > 0)[0]
                if ys0.size > 0:
                    y0g = int(ys0.min())
                    y1g = int(ys0.max())
                    y_split = int(round(float(y0g) + y_frac * float(max(y1g - y0g, 1))))
                else:
                    y_split = int(round(y_frac * float(h_dil)))
                y_split = int(max(0, min(h_dil - 1, y_split)))
                y_split_for_floor = int(y_split)
                floor_debug["floor_roi_y_split"] = int(y_split)
                try:
                    last_metrics["floor_roi_dilate_y_split"] = int(y_split)
                except Exception:
                    pass
            floor_debug["floor_roi_dilate_bottom_only"] = int(bottom_only)

            k = np.ones((d, d), np.uint8)

            if bottom_only:
                bottom_mask = np.zeros_like(picked)
                bottom_mask[y_split:, :] = 255
                picked_top = cv2.bitwise_and(picked, cv2.bitwise_not(bottom_mask))
                picked_bot = cv2.bitwise_and(picked, bottom_mask)
                picked_bot = cv2.dilate(picked_bot, k, iterations=iters)
                picked_bot = cv2.bitwise_and(picked_bot, bottom_mask)
                picked = cv2.bitwise_or(picked_top, picked_bot)
            else:
                picked = cv2.dilate(picked, k, iterations=iters)
            floor_debug["floor_roi_green_dilate"] = int(d)
            floor_debug["floor_roi_green_dilate_iters"] = int(iters)

            floor_roi_mask = picked
            floor_roi_source = floor_roi_source or "green_flood"

            # Optional: expand into a thin band of non-green floor near picked region
            band_px = int(os.getenv("BADC_FLOOR_ROI_FLOOR_BAND_PX", "0") or 0)
            if band_px > 0:
                band_px = max(1, int(band_px))
                src = np.where(picked > 0, 0, 255).astype(np.uint8)
                dist = cv2.distanceTransform(src, cv2.DIST_L2, 3)
                band = (dist <= float(band_px)).astype(np.uint8) * 255
                band = cv2.bitwise_and(band, bbox_mask)
                # Top part stays clipped to original floor; bottom may extend outside if allowed.
                allow_bottom_outside = int(os.getenv("BADC_FLOOR_ROI_BAND_ALLOW_OUTSIDE_ORIG_BOTTOM", "1") or 1)
                if allow_bottom_outside:
                    band_top = band.copy()
                    band_bot = band.copy()
                    band_top[y_split_for_floor:, :] = 0
                    band_bot[:y_split_for_floor, :] = 0
                    band_top = cv2.bitwise_and(band_top, orig_floor_mask)
                    band = cv2.bitwise_or(band_top, band_bot)
                    floor_debug["floor_roi_band_allow_outside_orig_bottom"] = 1
                else:
                    band = cv2.bitwise_and(band, orig_floor_mask)
                    floor_debug["floor_roi_band_allow_outside_orig_bottom"] = 0
                if int(os.getenv("BADC_FLOOR_ROI_BAND_BOTTOM_ONLY", "0") or 0) and y_split_for_floor is not None:
                    band[:y_split_for_floor, :] = 0
                    floor_debug["floor_roi_band_bottom_only"] = 1
                floor_roi_mask = cv2.bitwise_or(floor_roi_mask, band)
                floor_debug["floor_roi_floor_band_px"] = int(band_px)
                floor_debug["floor_roi_floor_band_area"] = int(np.count_nonzero(band))
    else:
        floor_roi_mask = bbox_mask
        floor_roi_source = "bbox_fallback"
    # Optional: post dilate final ROI
    post_d = int(os.getenv("BADC_FLOOR_ROI_POST_DILATE", "0") or 0)
    if green_direct_strict:
        post_d = 0
    if post_d > 0:
        post_d = max(3, int(post_d))
        if post_d % 2 == 0:
            post_d += 1
        k = np.ones((post_d, post_d), np.uint8)
        floor_roi_mask = cv2.dilate(floor_roi_mask, k, iterations=1)
        floor_debug["floor_roi_post_dilate"] = int(post_d)

    # keep_true_floor settings
    keep_true_floor = int(os.getenv("BADC_FLOOR_ROI_KEEP_TRUE_FLOOR", "1") or 0) == 1
    if green_direct_strict:
        keep_true_floor = False
    keep_true_floor_top_only = int(os.getenv("BADC_FLOOR_ROI_KEEP_TRUE_FLOOR_TOP_ONLY", "1") or 0) == 1
    keep_true_floor_top_from_orig = int(os.getenv("BADC_FLOOR_ROI_KEEP_TRUE_FLOOR_TOP_FROM_ORIG", "1") or 0) == 1

    apply_direct_y_cut = _env_flag("BADC_FLOOR_ROI_DIRECT_APPLY_FLOOR_Y_CUT", False)
    if floor_y_cut is not None and (not green_direct_strict or apply_direct_y_cut):
        floor_roi_mask[: int(floor_y_cut), :] = 0

    # Add back any original green pixels in the bottom part that flood/pick/dilate may have missed
    if bottom_only and (not green_direct_strict):
        mask_low = np.zeros((h, w), dtype=np.uint8)
        mask_low[y_split_for_floor:, :] = 255
        add_back = cv2.bitwise_and(green_in_bbox, mask_low)
        floor_roi_mask = cv2.bitwise_or(floor_roi_mask, add_back)

    # Apply true floor constraint: either keep strict OR top-only strict when bottom dilation is used
    if keep_true_floor:
        if keep_true_floor_top_only and (y_split_for_floor is not None) and bottom_only:
            if keep_true_floor_top_from_orig:
                top = orig_floor_mask.copy()
                floor_debug["floor_roi_keep_true_floor_top_from_orig"] = 1
                try:
                    last_metrics["floor_roi_keep_true_floor_top_from_orig"] = 1
                except Exception:
                    pass
            else:
                top = cv2.bitwise_and(floor_roi_mask, orig_floor_mask)
            top[y_split_for_floor:, :] = 0
            bot = floor_roi_mask.copy()
            bot[:y_split_for_floor, :] = 0
            floor_roi_mask = cv2.bitwise_or(top, bot)
            floor_debug["floor_roi_keep_true_floor_top_only"] = 1
            try:
                last_metrics["floor_roi_keep_true_floor_top_only"] = 1
            except Exception:
                pass
        else:
            floor_roi_mask = cv2.bitwise_and(floor_roi_mask, orig_floor_mask)

    # Reinjection pass: ensure green court pixels are not lost by top clipping/strict intersection.
    reinject_green = int(os.getenv("BADC_FLOOR_ROI_REINJECT_GREEN", "1") or 0) == 1
    if green_direct_strict:
        reinject_green = False
    if green_direct_apply and _env_flag("BADC_FLOOR_ROI_GREEN_DIRECT_DISABLE_REINJECT", True):
        reinject_green = False
        floor_debug["floor_roi_reinject_green_skipped_green_direct"] = 1
    if reinject_green and isinstance(green_mask, np.ndarray) and green_mask.size > 0:
        green_reinject = cv2.bitwise_and(((green_mask > 0).astype(np.uint8) * 255), bbox_mask)
        if floor_y_cut is not None:
            green_reinject[: int(floor_y_cut), :] = 0
        if keep_true_floor and keep_true_floor_top_only and (y_split_for_floor is not None) and bottom_only:
            rein_top = cv2.bitwise_and(green_reinject, orig_floor_mask)
            rein_top[y_split_for_floor:, :] = 0
            rein_bot = green_reinject.copy()
            rein_bot[:y_split_for_floor, :] = 0
            green_reinject = cv2.bitwise_or(rein_top, rein_bot)
        floor_roi_mask = cv2.bitwise_or(floor_roi_mask, green_reinject)
        reinject_close_k = int(os.getenv("BADC_FLOOR_ROI_REINJECT_CLOSE_K", "9") or 9)
        reinject_close_k = max(3, reinject_close_k)
        if reinject_close_k % 2 == 0:
            reinject_close_k += 1
        floor_roi_mask = cv2.morphologyEx(
            floor_roi_mask,
            cv2.MORPH_CLOSE,
            np.ones((reinject_close_k, reinject_close_k), np.uint8),
            iterations=1,
        )
        floor_debug["floor_roi_reinject_green"] = 1
        floor_debug["floor_roi_reinject_area"] = int(np.count_nonzero(green_reinject))
        floor_debug["floor_roi_reinject_close_k"] = int(reinject_close_k)

    # Final regularization: fill internal holes while preserving plausible far-end
    # components (largest-CC-only can wrongly drop disconnected top-half floor).
    fill_holes = int(os.getenv("BADC_FLOOR_ROI_FILL_HOLES", "0") or 0) == 1
    if green_direct_strict:
        fill_holes = False
    if fill_holes and int(np.count_nonzero(floor_roi_mask)) > 0:
        bin_mask = ((floor_roi_mask > 0).astype(np.uint8) * 255)
        num_cc, labels_cc, stats_cc, _ = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
        kept_cc_ids: list[int] = []
        dropped_cc_ids: list[int] = []
        if num_cc > 1:
            best_idx = 1 + int(np.argmax(stats_cc[1:, cv2.CC_STAT_AREA]))
            main_cc = (labels_cc == best_idx).astype(np.uint8) * 255
            kept_cc_ids.append(int(best_idx))
            keep_top_cc = _env_flag("BADC_FLOOR_ROI_FILL_HOLES_KEEP_TOP_CC", True)
            if keep_top_cc:
                ys_m, xs_m = np.where(main_cc > 0)
                if xs_m.size > 0:
                    mx1 = int(xs_m.min())
                    my1 = int(ys_m.min())
                    mx2 = int(xs_m.max())
                    my2 = int(ys_m.max())
                    min_area_frac = float(
                        np.clip(_env_float("BADC_FLOOR_ROI_FILL_HOLES_KEEP_TOP_MIN_AREA_FRAC", 0.0012), 0.0, 0.20)
                    )
                    min_area_px = int(max(64, round(min_area_frac * float(max(h * w, 1)))))
                    max_gap_frac = float(
                        np.clip(_env_float("BADC_FLOOR_ROI_FILL_HOLES_KEEP_TOP_MAX_GAP_FRAC", 0.12), 0.0, 0.70)
                    )
                    max_gap_px = int(max(2, round(max_gap_frac * float(max(h, 1)))))
                    max_start_below_frac = float(
                        np.clip(_env_float("BADC_FLOOR_ROI_FILL_HOLES_KEEP_TOP_MAX_START_BELOW_FRAC", 0.08), 0.0, 0.50)
                    )
                    max_start_below_px = int(max(0, round(max_start_below_frac * float(max(h, 1)))))
                    min_overlap = float(
                        np.clip(_env_float("BADC_FLOOR_ROI_FILL_HOLES_KEEP_TOP_MIN_X_OVERLAP", 0.55), 0.0, 1.0)
                    )
                    min_w_frac = float(
                        np.clip(_env_float("BADC_FLOOR_ROI_FILL_HOLES_KEEP_TOP_MIN_W_FRAC", 0.06), 0.0, 1.0)
                    )
                    min_w_px = int(max(6, round(min_w_frac * float(max(w, 1)))))
                    for idx in range(1, int(num_cc)):
                        if idx == best_idx:
                            continue
                        x = int(stats_cc[idx, cv2.CC_STAT_LEFT])
                        y = int(stats_cc[idx, cv2.CC_STAT_TOP])
                        ww = int(stats_cc[idx, cv2.CC_STAT_WIDTH])
                        hh = int(stats_cc[idx, cv2.CC_STAT_HEIGHT])
                        area = int(stats_cc[idx, cv2.CC_STAT_AREA])
                        cx1 = int(x)
                        cy1 = int(y)
                        cx2 = int(x + ww - 1)
                        cy2 = int(y + hh - 1)
                        overlap = int(max(0, min(mx2, cx2) - max(mx1, cx1) + 1))
                        overlap_frac = float(overlap) / float(max(1, ww))
                        gap_to_main_top = int(max(0, my1 - cy2))
                        keep_this = (
                            area >= min_area_px
                            and ww >= min_w_px
                            and cy1 <= (my1 + max_start_below_px)
                            and gap_to_main_top <= max_gap_px
                            and overlap_frac >= min_overlap
                        )
                        if keep_this:
                            main_cc[labels_cc == idx] = 255
                            kept_cc_ids.append(int(idx))
                        else:
                            dropped_cc_ids.append(int(idx))
                    floor_debug["floor_roi_fill_holes_keep_top_cc"] = 1
                    floor_debug["floor_roi_fill_holes_keep_top_min_area_px"] = int(min_area_px)
                    floor_debug["floor_roi_fill_holes_keep_top_max_gap_px"] = int(max_gap_px)
                    floor_debug["floor_roi_fill_holes_keep_top_max_start_below_px"] = int(max_start_below_px)
                    floor_debug["floor_roi_fill_holes_keep_top_min_x_overlap"] = float(min_overlap)
                    floor_debug["floor_roi_fill_holes_keep_top_min_w_px"] = int(min_w_px)
                    floor_debug["floor_roi_fill_holes_keep_top_main_bbox"] = [int(mx1), int(my1), int(mx2), int(my2)]
        else:
            main_cc = bin_mask
            kept_cc_ids.append(1)
        h_fill, w_fill = main_cc.shape[:2]
        flood = main_cc.copy()
        flood_mask = np.zeros((h_fill + 2, w_fill + 2), np.uint8)
        seed_pt = None
        border_pts = []
        for xx in range(w_fill):
            border_pts.append((xx, 0))
            border_pts.append((xx, h_fill - 1))
        for yy in range(h_fill):
            border_pts.append((0, yy))
            border_pts.append((w_fill - 1, yy))
        for sx, sy in border_pts:
            if flood[sy, sx] == 0:
                seed_pt = (int(sx), int(sy))
                break
        if seed_pt is not None:
            cv2.floodFill(flood, flood_mask, seedPoint=seed_pt, newVal=255)
            holes = cv2.bitwise_not(flood)
            filled = cv2.bitwise_or(main_cc, holes)
        else:
            filled = main_cc
        hole_close_k = int(os.getenv("BADC_FLOOR_ROI_HOLE_CLOSE_K", "7") or 7)
        hole_close_k = max(3, hole_close_k)
        if hole_close_k % 2 == 0:
            hole_close_k += 1
        filled = cv2.morphologyEx(
            filled,
            cv2.MORPH_CLOSE,
            np.ones((hole_close_k, hole_close_k), np.uint8),
            iterations=1,
        )
        floor_roi_mask = filled
        floor_debug["floor_roi_fill_holes"] = 1
        floor_debug["floor_roi_fill_holes_close_k"] = int(hole_close_k)
        floor_debug["floor_roi_main_cc_area"] = int(np.count_nonzero(main_cc))
        floor_debug["floor_roi_filled_area"] = int(np.count_nonzero(filled))
        floor_debug["floor_roi_fill_holes_kept_cc"] = [int(i) for i in kept_cc_ids]
        floor_debug["floor_roi_fill_holes_dropped_cc"] = [int(i) for i in dropped_cc_ids]
        floor_debug["floor_roi_fill_holes_kept_cc_count"] = int(len(kept_cc_ids))
        floor_debug["floor_roi_fill_holes_dropped_cc_count"] = int(len(dropped_cc_ids))

    # Top-band row completion:
    # fill staircase-like gaps near far court caused by players/shadows/strict color mask.
    row_fill_top = int(os.getenv("BADC_FLOOR_ROI_ROW_FILL_TOP", "0") or 0) == 1
    if green_direct_strict:
        row_fill_top = False
    if row_fill_top and int(np.count_nonzero(floor_roi_mask)) > 0:
        bin_mask = ((floor_roi_mask > 0).astype(np.uint8) * 255)
        ys_any = np.where(np.any(bin_mask > 0, axis=1))[0]
        if ys_any.size > 0:
            y_top = int(ys_any.min())
            y_bot = int(ys_any.max())
            roi_h = int(max(1, y_bot - y_top + 1))
            top_frac = float(max(0.15, min(0.95, _env_float("BADC_FLOOR_ROI_ROW_FILL_TOP_FRAC", 0.85))))
            y_end = int(min(y_bot, y_top + int(round(top_frac * float(roi_h)))))
            neighbor = int(max(1, _env_int("BADC_FLOOR_ROI_ROW_FILL_NEIGHBOR", 3)))
            widths_ref: list[float] = []
            y_ref0 = int(max(y_top, y_bot - int(round(0.45 * float(roi_h)))))
            for yy in range(y_ref0, y_bot + 1):
                xs = np.where(bin_mask[yy] > 0)[0]
                if xs.size >= 2:
                    widths_ref.append(float(xs[-1] - xs[0] + 1))
            if widths_ref:
                ref_width = float(np.median(np.asarray(widths_ref, dtype=np.float32)))
            else:
                row_counts = np.count_nonzero(bin_mask > 0, axis=1)
                ref_width = float(np.max(row_counts)) if row_counts.size > 0 else 0.0
            min_w_frac = float(max(0.08, min(0.95, _env_float("BADC_FLOOR_ROI_ROW_FILL_MIN_WIDTH_FRAC", 0.15))))
            min_w_px_cfg = float(max(0.0, _env_float("BADC_FLOOR_ROI_ROW_FILL_MIN_WIDTH_PX", 0.0)))
            min_w_req = float(max(20.0, min_w_px_cfg, min_w_frac * max(1.0, ref_width)))
            fill = np.zeros_like(bin_mask)
            filled_rows = 0
            for yy in range(y_top, y_end + 1):
                yy0 = int(max(y_top, yy - neighbor))
                yy1 = int(min(y_bot, yy + neighbor))
                row_union = np.any(bin_mask[yy0 : yy1 + 1] > 0, axis=0)
                xs = np.where(row_union)[0]
                if xs.size < 2:
                    continue
                xl = int(xs[0])
                xr = int(xs[-1])
                if float(xr - xl + 1) < min_w_req:
                    continue
                fill[yy, xl : xr + 1] = 255
                filled_rows += 1
            floor_debug["floor_roi_row_fill_top"] = 1
            floor_debug["floor_roi_row_fill_top_frac"] = float(top_frac)
            floor_debug["floor_roi_row_fill_neighbor"] = int(neighbor)
            floor_debug["floor_roi_row_fill_min_width_px"] = float(min_w_req)
            floor_debug["floor_roi_row_fill_rows"] = int(filled_rows)
            if filled_rows > 0:
                bin_mask = cv2.bitwise_or(bin_mask, fill)
                bin_mask = cv2.bitwise_and(bin_mask, bbox_mask)
                if floor_y_cut is not None:
                    bin_mask[: int(floor_y_cut), :] = 0
                floor_roi_mask = bin_mask

    # If far-end rows are still missing, extrapolate the top span upward by a short distance.
    top_extend_enable = bool(_env_flag("BADC_FLOOR_ROI_TOP_EXTEND_ENABLE", False))
    if green_direct_strict:
        top_extend_enable = False
    if top_extend_enable and int(np.count_nonzero(floor_roi_mask)) > 0:
        floor_bbox_top = _white_mask_stats(floor_roi_mask)[2]
        bbox_rows = np.where(np.any((bbox_mask > 0), axis=1))[0]
        y_bbox_min = int(bbox_rows.min()) if bbox_rows.size > 0 else 0
        bin_mask = ((floor_roi_mask > 0).astype(np.uint8) * 255)
        ys_any = np.where(np.any(bin_mask > 0, axis=1))[0]
        if ys_any.size > 0:
            y_top = int(ys_any.min())
            y_bot = int(ys_any.max())
            roi_h = int(max(1, y_bot - y_top + 1))
            ext_frac = float(max(0.02, min(0.45, _env_float("BADC_FLOOR_ROI_TOP_EXTEND_FRAC", 0.14))))
            ext_rows = int(max(0, round(ext_frac * float(roi_h))))
            if ext_rows > 0:
                max_up = int(max(0, y_top - y_bbox_min))
                if max_up > 0:
                    fill_missing_frac = float(
                        max(0.0, min(1.0, _env_float("BADC_FLOOR_ROI_TOP_EXTEND_MISS_FILL_FRAC", 0.75)))
                    )
                    ext_rows = int(max(ext_rows, round(fill_missing_frac * float(max_up))))
                    ext_rows = int(min(ext_rows, max_up))
                ref_span_rows = int(max(2, _env_int("BADC_FLOOR_ROI_TOP_EXTEND_REF_ROWS", 6)))
                y_ref1 = int(min(y_bot, y_top + ref_span_rows))
                row_union = np.any(bin_mask[y_top : y_ref1 + 1] > 0, axis=0)
                xs = np.where(row_union)[0]
                if xs.size >= 2:
                    xl = int(xs[0])
                    xr = int(xs[-1])
                    span_w = float(xr - xl + 1)
                    floor_w = float(max(1, int(floor_bbox_top[2]) - int(floor_bbox_top[0]) + 1))
                    min_span_frac = float(max(0.20, min(0.95, _env_float("BADC_FLOOR_ROI_TOP_EXTEND_MIN_SPAN_FRAC", 0.45))))
                    if span_w >= min_span_frac * floor_w:
                        y_from = int(max(int(y_bbox_min), y_top - ext_rows))
                        if y_from < y_top:
                            fill = np.zeros_like(bin_mask)
                            fill[y_from:y_top, xl : xr + 1] = 255
                            bin_mask = cv2.bitwise_or(bin_mask, fill)
                            bin_mask = cv2.bitwise_and(bin_mask, bbox_mask)
                            if floor_y_cut is not None:
                                bin_mask[: int(floor_y_cut), :] = 0
                            floor_roi_mask = bin_mask
                            floor_debug["floor_roi_top_extend"] = 1
                            floor_debug["floor_roi_top_extend_rows"] = int(y_top - y_from)
                            floor_debug["floor_roi_top_extend_span"] = [int(xl), int(xr)]
                            floor_debug["floor_roi_top_extend_frac"] = float(ext_frac)
                            floor_debug["floor_roi_top_extend_bbox_y_min"] = int(y_bbox_min)
                            floor_debug["floor_roi_top_extend_min_span_frac"] = float(min_span_frac)

    floor_mask = floor_roi_mask
    floor_debug["floor_roi_source"] = floor_roi_source
    if green_ratio is not None:
        floor_debug["floor_roi_green_ratio"] = float(green_ratio)

    floor_roi_overlay = frame_bgr.copy()
    if np.any(floor_mask > 0):
        mask = floor_mask > 0
        overlay_color = np.zeros_like(frame_bgr)
        overlay_color[:, :] = (0, 255, 0)
        floor_roi_overlay[mask] = (
            0.7 * floor_roi_overlay[mask].astype(np.float32)
            + 0.3 * overlay_color[mask].astype(np.float32)
        ).astype(np.uint8)
    white_mask_raw_full = build_white_mask_raw(
        frame_bgr,
        white_s_max=140,
        white_v_min=155,
        l_min=170,
        chroma_max=55,
        roi_mask_u8=floor_mask,
    )
    h, w = frame_bgr.shape[:2]
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    k = int(max(21, round(min(h, w) * 0.03)))
    if k % 2 == 0:
        k += 1
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1))
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, k))
    tophat_h = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel_h)
    tophat_v = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel_v)
    tophat = cv2.max(tophat_h, tophat_v)
    roi_vals = tophat[floor_mask > 0]
    thin_line_mask = None
    if roi_vals.size > 0:
        floor_bbox_preblob = _white_mask_stats(floor_mask)[2]
        base_pct = float(max(40.0, min(95.0, _env_float("BADC_PREBLOB_TOPHAT_PCT", 75.0))))
        thresh = max(float(np.percentile(roi_vals, base_pct)), 10.0)
        thin_line_mask = (tophat >= thresh).astype(np.uint8) * 255
        thin_line_mask = cv2.bitwise_and(thin_line_mask, floor_mask)
        floor_debug["preblob_tophat_pct"] = float(base_pct)
        floor_debug["preblob_tophat_thr"] = float(thresh)

        # Recover weak far-end lines: in top court band use a looser tophat gate,
        # then keep only elongated white structures to avoid pulling in logos/blobs.
        top_relax_enable = bool(_env_flag("BADC_PREBLOB_TOP_RELAX_ENABLE", True))
        if legacy_ori_split and ("BADC_PREBLOB_TOP_RELAX_ENABLE" not in os.environ):
            top_relax_enable = False
        if top_relax_enable:
            y0f, y1f = int(floor_bbox_preblob[1]), int(floor_bbox_preblob[3])
            roi_h = int(max(1, y1f - y0f + 1))
            top_frac = float(max(0.10, min(0.75, _env_float("BADC_PREBLOB_TOP_RELAX_FRAC", 0.54))))
            y_top_end = int(max(y0f, min(y1f, y0f + int(round(top_frac * float(roi_h))))))
            top_band = np.zeros_like(floor_mask)
            if y_top_end >= y0f:
                top_band[y0f : y_top_end + 1, :] = 255
            top_vals = tophat[np.logical_and(floor_mask > 0, top_band > 0)]
            top_pct = float(max(25.0, min(base_pct, _env_float("BADC_PREBLOB_TOP_RELAX_PCT", 50.0))))
            if top_vals.size > 0:
                top_thr = max(float(np.percentile(top_vals, top_pct)), 8.0)
            else:
                top_thr = max(8.0, 0.88 * float(thresh))
            thin_top = (tophat >= top_thr).astype(np.uint8) * 255
            thin_top = cv2.bitwise_and(thin_top, floor_mask)
            thin_top = cv2.bitwise_and(thin_top, top_band)
            thin_line_mask = cv2.bitwise_or(thin_line_mask, thin_top)
            floor_debug["preblob_top_relax_enabled"] = 1
            floor_debug["preblob_top_relax_frac"] = float(top_frac)
            floor_debug["preblob_top_relax_pct"] = float(top_pct)
            floor_debug["preblob_top_relax_thr"] = float(top_thr)
            floor_debug["preblob_top_relax_rows"] = int(max(0, y_top_end - y0f + 1))

            top_raw_recover = bool(_env_flag("BADC_PREBLOB_TOP_RAW_RECOVER_ENABLE", True))
            if legacy_ori_split and ("BADC_PREBLOB_TOP_RAW_RECOVER_ENABLE" not in os.environ):
                top_raw_recover = False
            if top_raw_recover and y_top_end >= y0f:
                top_raw = cv2.bitwise_and(white_mask_raw_full, top_band)

                k_h = int(max(5, _env_int("BADC_PREBLOB_TOP_RAW_OPEN_H", 9)))
                if k_h % 2 == 0:
                    k_h += 1
                k_v = int(max(5, _env_int("BADC_PREBLOB_TOP_RAW_OPEN_V", 9)))
                if k_v % 2 == 0:
                    k_v += 1

                top_keep_h = cv2.morphologyEx(
                    top_raw,
                    cv2.MORPH_OPEN,
                    cv2.getStructuringElement(cv2.MORPH_RECT, (k_h, 1)),
                )
                top_keep_v = cv2.morphologyEx(
                    top_raw,
                    cv2.MORPH_OPEN,
                    cv2.getStructuringElement(cv2.MORPH_RECT, (1, k_v)),
                )
                top_keep = cv2.bitwise_or(top_keep_h, top_keep_v)
                bridge_h = int(max(3, _env_int("BADC_PREBLOB_TOP_RAW_BRIDGE_H", 11)))
                if bridge_h % 2 == 0:
                    bridge_h += 1
                bridge_v = int(max(3, _env_int("BADC_PREBLOB_TOP_RAW_BRIDGE_V", 5)))
                if bridge_v % 2 == 0:
                    bridge_v += 1
                top_keep = cv2.morphologyEx(
                    top_keep,
                    cv2.MORPH_CLOSE,
                    cv2.getStructuringElement(cv2.MORPH_RECT, (bridge_h, 1)),
                    iterations=1,
                )
                top_keep = cv2.morphologyEx(
                    top_keep,
                    cv2.MORPH_CLOSE,
                    cv2.getStructuringElement(cv2.MORPH_RECT, (1, bridge_v)),
                    iterations=1,
                )
                top_keep = cv2.bitwise_and(top_keep, floor_mask)
                top_keep = cv2.bitwise_and(top_keep, top_band)
                thin_line_mask = cv2.bitwise_or(thin_line_mask, top_keep)
                floor_debug["preblob_top_raw_recover"] = 1
                floor_debug["preblob_top_raw_open_h"] = int(k_h)
                floor_debug["preblob_top_raw_open_v"] = int(k_v)
                floor_debug["preblob_top_raw_bridge_h"] = int(bridge_h)
                floor_debug["preblob_top_raw_bridge_v"] = int(bridge_v)
                floor_debug["preblob_top_raw_recover_area"] = int(np.count_nonzero(top_keep))
    if thin_line_mask is not None:
        white_mask_raw_floor_preblob = cv2.bitwise_and(white_mask_raw_full, thin_line_mask)
    else:
        white_mask_raw_floor_preblob = cv2.bitwise_and(white_mask_raw_full, floor_mask)
    white_mask_raw_floor_postblob, blob_stats = filter_blobs_keep_lines(
        white_mask_raw_floor_preblob,
        close_kernel_lens=(),
        close_thickness=1,
        open_kernel_lens=(7, 9, 15, 21, 31),
        line_kernel_thickness=1,
        line_thickness_max=4.0,
        speckle_area=20,
        min_blob_area_ratio=0.001,
        max_aspect_for_blob=2.5,
        min_fill_ratio=0.55,
        return_stats=True,
    )

    white_mask_raw_floor_postblob = _filter_postblob_non_line(
        white_mask_raw_floor_postblob,
        line_lengths=(15, 21, 31),
        thickness_thr=999.0,
        speckle_area=25,
        min_aspect=4.0,
        min_long=20,
        density_win=11,
        dense_soft=0.25,
        dense_hard=0.55,
        line_density_max=0.30,
    )
    white_mask_raw_floor = white_mask_raw_floor_preblob
    white_mask_raw_floor_noblob = _remove_big_blobs(white_mask_raw_floor_postblob, floor_mask)
    white_mask_raw_floor_noblob = _remove_thick_solid_components_by_dt(
        white_mask_raw_floor_noblob,
        floor_mask,
        max_r=3.2,
        extent_min=0.35,
        aspect_max=4.0,
        area_min=120,
    )
    white_mask_raw_floor_noblob = _remove_solid_mid_blobs(
        white_mask_raw_floor_noblob,
        floor_mask,
        area_min=120,
        area_max=3000,
        extent_min=0.40,
        aspect_max=3.0,
    )
    white_mask_raw_floor_noblob = _remove_mid_solid_blobs_near_bottom(
        white_mask_raw_floor_noblob,
        floor_mask,
        area_min=120,
        area_max=12000,
        extent_min=0.35,
        aspect_max=3.2,
        y_center_min_ratio=0.62,
    )

    if roi_vals.size == 0:
        mask_floor_lines = white_mask_raw_floor_noblob.copy()
    else:
        thresh = max(float(np.percentile(roi_vals, 75.0)), 10.0)
        mask_floor_lines = (tophat >= thresh).astype(np.uint8) * 255
        mask_floor_lines = cv2.bitwise_and(mask_floor_lines, floor_mask)
        mask_floor_lines = cv2.bitwise_and(mask_floor_lines, white_mask_raw_floor_noblob)
    # Keep the raw line structure; avoid extending/bridging lines.
    mask_floor_lines = cv2.morphologyEx(mask_floor_lines, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    white_mask_clean = mask_floor_lines

    raw_full_sum, raw_full_ratio, raw_full_bbox = _white_mask_stats(white_mask_raw_full)
    raw_floor_pre_sum, raw_floor_pre_ratio, _ = _white_mask_stats(white_mask_raw_floor_preblob)
    raw_floor_post_sum, raw_floor_post_ratio, _ = _white_mask_stats(white_mask_raw_floor_postblob)
    raw_floor_noblob_sum, raw_floor_noblob_ratio, _ = _white_mask_stats(white_mask_raw_floor_noblob)
    clean_sum, clean_ratio, _ = _white_mask_stats(white_mask_clean)
    _, floor_ratio, floor_bbox = _white_mask_stats(floor_mask)

    max_cc_area = 0
    if clean_sum > 0:
        num, _, stats, _ = cv2.connectedComponentsWithStats(white_mask_clean, connectivity=8)
        for idx in range(1, num):
            area = int(stats[idx, cv2.CC_STAT_AREA])
            max_cc_area = max(max_cc_area, area)

    dt_from_label = "white_mask_clean"
    dt_source = white_mask_raw_floor_postblob if raw_floor_post_sum > 0 else white_mask_clean
    if raw_floor_post_sum > 0:
        dt_from_label = "white_mask_raw_floor"
    dt = build_distance_transform(dt_source)
    dt_min = float(np.min(dt)) if dt.size > 0 else None
    dt_mean = float(np.mean(dt)) if dt.size > 0 else None
    dt_p90 = float(np.percentile(dt, 90)) if dt.size > 0 else None
    dt_oob = float(max(dt_p90 or 0.0, 15.0))
    x1, y1, x2, y2 = floor_bbox
    x1 = int(np.clip(x1, 0, dt.shape[1] - 1))
    x2 = int(np.clip(x2, x1 + 1, dt.shape[1]))
    y1 = int(np.clip(y1, 0, dt.shape[0] - 1))
    y2 = int(np.clip(y2, y1 + 1, dt.shape[0]))
    roi_h = max(1, y2 - y1)
    roi_w = max(1, x2 - x1)
    rand_n = 1000
    xs = rng.integers(x1, x1 + roi_w, size=rand_n)
    ys = rng.integers(y1, y1 + roi_h, size=rand_n)
    dt_rand = dt[ys, xs].astype(np.float32) if dt.size > 0 else np.array([], dtype=np.float32)
    dt_rand_mean = float(np.mean(dt_rand)) if dt_rand.size > 0 else None
    dt_rand_p90 = float(np.percentile(dt_rand, 90)) if dt_rand.size > 0 else None

    corners_world = get_bwf_corners()
    Xw_ransac, _, _ = sample_model_points(points_per_meter=30.0, min_weight=0.6)
    Xw_full, w_full, names_full = sample_model_points(points_per_meter=30.0, min_weight=0.3)
    w_full, role_w_cfg = _apply_model_role_weights(names_full, w_full)
    foot_points = _foot_points_from_boxes(person_boxes, frame_bgr.shape[0])

    base_metrics = {
        "model": MODEL_ID,
        "person_box_count": int(len(foot_points)),
        "foot_points": [[float(px), float(py)] for px, py in foot_points],
        "canonical_points_norm": get_canonical_points_norm(),
        "component_id": int(component_id) if component_id is not None else None,
        "component_bbox": [int(v) for v in component_bbox] if component_bbox is not None else None,
        "component_area": int(component_area) if component_area is not None else None,
        "selected_component_id": int(component_id) if component_id is not None else None,
        "component_stats_top3": [],
        "manual_override": False,
        "manual_present": False,
        "manual_pairs_added": False,
        "anchor_prior_enabled": False,
        "used_anchor_prior": False,
        "used_manual_anchors": False,
        "dt_from": dt_from_label,
        "white_mask_sum": int(clean_sum),
        "white_mask_ratio": float(clean_ratio),
        "white_mask_raw_sum": int(raw_full_sum),
        "white_mask_raw_ratio": float(raw_full_ratio),
        "white_mask_raw_full_sum": int(raw_full_sum),
        "white_mask_raw_full_ratio": float(raw_full_ratio),
        "white_mask_raw_floor_sum": int(raw_floor_pre_sum),
        "white_mask_raw_floor_ratio": float(raw_floor_pre_ratio),
        "white_mask_raw_floor_preblob_ratio": float(raw_floor_pre_ratio),
        "white_mask_raw_floor_postblob_ratio": float(raw_floor_post_ratio),
        "white_mask_raw_floor_postblob_sum": int(raw_floor_post_sum),
        "white_mask_raw_floor_noblob_sum": int(raw_floor_noblob_sum),
        "white_mask_raw_floor_noblob_ratio": float(raw_floor_noblob_ratio),
        "white_mask_clean_sum": int(clean_sum),
        "white_mask_clean_ratio": float(clean_ratio),
        "mask_raw_ratio": float(raw_floor_pre_ratio),
        "mask_clean_ratio": float(clean_ratio),
        "max_cc_area": int(max_cc_area),
        "floor_bbox": [int(v) for v in floor_bbox],
        "floor_roi_ratio": float(floor_ratio),
        "floor_roi_source": floor_debug.get("floor_roi_source"),
        "floor_roi_green_ratio": floor_debug.get("floor_roi_green_ratio"),
        "white_mask_blob_removed_count": int(blob_stats.get("removed_count", 0))
        if isinstance(blob_stats, dict)
        else None,
        "white_mask_blob_removed_topk": blob_stats.get("removed_topk", [])
        if isinstance(blob_stats, dict)
        else [],
        "white_raw_full_bbox": [int(v) for v in raw_full_bbox],
        "fallback_floor_roi": bool(floor_debug.get("fallback_floor_roi", False)),
        "fallback_mode": floor_debug.get("fallback_mode"),
        "floor_y_cut": floor_debug.get("floor_y_cut"),
        "floor_x_tightened": floor_debug.get("floor_x_tightened"),
        "floor_x_range": floor_debug.get("floor_x_range"),
        "floor_x_support_ratio": floor_debug.get("floor_x_support_ratio"),
        "floor_x_reject_reason": floor_debug.get("floor_x_reject_reason"),
        "floor_cc_scores": floor_debug.get("floor_cc_scores"),
        "floor_strip_thr": floor_debug.get("floor_strip_thr"),
        "floor_strip_y0": floor_debug.get("floor_strip_y0"),
        "lab_seed_ok": floor_debug.get("lab_seed_ok"),
        "lab_seed_bbox": floor_debug.get("lab_seed_bbox"),
        "lab_seed_y0": floor_debug.get("lab_seed_y0"),
        "dt_min": dt_min,
        "dt_mean": dt_mean,
        "dt_p90": dt_p90,
        "dt_rand_mean": dt_rand_mean,
        "dt_rand_p90": dt_rand_p90,
        "debug_image_path": None,
        "white_mask_path": None,
        "white_mask_raw_path": None,
        "white_mask_clean_path": None,
        "lsd_dirA_path": None,
        "lsd_dirB_path": None,
        "ori_maskA_path": None,
        "ori_maskB_path": None,
        "hough_lines_path": None,
        "debug_init_path": None,
        "dt_debug_path": None,
        "linepix_green_only_used": False,
        "max_inside_count": 0,
        "selection_reason": "not_selected",
        "model_role_weights": role_w_cfg,
    }
    # ===== Farin2005 meters-based matcher hook (after noblob, needs base_metrics) =====
    matcher = os.environ.get("BADC_COURT_MATCHER", "legacy").strip().lower()
    if matcher in ("farin2005", "farin2005_m", "farin_m"):
        try:
            from src.vision.court_match_farin2005 import fit_model_farin2005_m
        except Exception:
            from .court_match_farin2005 import fit_model_farin2005_m  # type: ignore

        res = fit_model_farin2005_m(
            mask_noblob=white_mask_raw_floor_noblob,
            img_shape=white_mask_raw_floor_noblob.shape[:2],
        )

        if res.H is None or res.corners is None:
            return CourtFitResult(
                H=None,
                corners=None,
                confidence=0.0,
                metrics={
                    **base_metrics,
                    "farin2005_m": res.metrics,
                    "E": float(res.E),
                },
                reason="R_fit_poor",
                method_used="farin2005_m",
                white_mask_raw_floor_preblob=white_mask_raw_floor_preblob,
                white_mask_raw_floor_postblob=white_mask_raw_floor_postblob,
                white_mask_raw_floor_noblob=white_mask_raw_floor_noblob,
                floor_roi_mask=floor_mask,
                floor_roi_overlay=floor_roi_overlay,
            )

        conf = float(np.exp(-0.02 * float(res.E)))

        return CourtFitResult(
            H=res.H,
            corners=res.corners,
            confidence=conf,
            metrics={
                **base_metrics,
                "farin2005_m": res.metrics,
                "E": float(res.E),
            },
            reason="R_ok",
            method_used="farin2005_m",
            white_mask_raw_floor_preblob=white_mask_raw_floor_preblob,
            white_mask_raw_floor_postblob=white_mask_raw_floor_postblob,
            white_mask_raw_floor_noblob=white_mask_raw_floor_noblob,
            floor_roi_mask=floor_mask,
            floor_roi_overlay=floor_roi_overlay,
        )
    # ===== end hook =====
    if skip_component_split and component_id is not None:
        base_metrics["selected_component_id"] = int(component_id)
    base_metrics["score_base"] = None
    base_metrics["score_total"] = None
    base_metrics["inside_count"] = int(0)
    base_metrics["inside_bonus"] = 0.0
    dt_debug = cv2.normalize(dt, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8) if dt.size > 0 else None

    def _annotate_failure(reason: str) -> np.ndarray:
        img = frame_bgr.copy()
        label = f"Hough failed: {reason}"
        cv2.putText(img, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4)
        cv2.putText(img, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        return img

    def _fail_result(
        reason: str,
        *,
        maskA: Optional[np.ndarray] = None,
        maskB: Optional[np.ndarray] = None,
        hough_lines_img: Optional[np.ndarray] = None,
        linepix_mask: Optional[np.ndarray] = None,
        linepix_mask_pre: Optional[np.ndarray] = None,
        floor_gate_mask: Optional[np.ndarray] = None,
        linepix_overlay: Optional[np.ndarray] = None,
        ransac_lines_img: Optional[np.ndarray] = None,
        extra_metrics: Optional[Dict[str, Any]] = None,
        metrics_override: Optional[Dict[str, Any]] = None,
    ) -> CourtFitResult:
        metrics_base = dict(metrics_override) if metrics_override is not None else dict(base_metrics)
        if maskA is None:
            maskA = np.zeros_like(white_mask_clean)
        if maskB is None:
            maskB = np.zeros_like(white_mask_clean)
        if hough_lines_img is None:
            hough_lines_img = _annotate_failure(reason)
        metrics = {
            **metrics_base,
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
        if metrics.get("selected_component_id") is None:
            metrics["selected_component_id"] = int(metrics_base.get("component_id") or 0)
        if isinstance(extra_metrics, dict):
            metrics.update(extra_metrics)
        # Debug-visibility normalization: ensure masks are uint8 {0,255}.
        def _vis_mask(m: Optional[np.ndarray]) -> Optional[np.ndarray]:
            if not isinstance(m, np.ndarray):
                return None
            mm = m
            if mm.ndim == 3:
                mm = cv2.cvtColor(mm, cv2.COLOR_BGR2GRAY)
            if mm.dtype != np.uint8:
                mm = mm.astype(np.uint8)
            return ((mm > 0).astype(np.uint8) * 255)

        linepix_mask_vis = _vis_mask(linepix_mask)
        linepix_mask_pre_vis = _vis_mask(linepix_mask_pre)
        floor_gate_mask_vis = _vis_mask(floor_gate_mask)
        debug_fail = _annotate_failure(reason)
        return CourtFitResult(
            H=None,
            corners=None,
            confidence=0.0,
            metrics=metrics,
            reason=reason,
            debug_image=debug_fail,
            debug_image_init=debug_fail,
            white_mask=white_mask_clean,
            white_mask_raw=white_mask_raw_full,
            white_mask_clean=white_mask_clean,
            white_mask_raw_full=white_mask_raw_full,
            white_mask_raw_floor=white_mask_raw_floor,
            white_mask_raw_floor_noblob=white_mask_raw_floor_noblob,
            white_mask_raw_floor_preblob=white_mask_raw_floor_preblob,
            white_mask_raw_floor_postblob=white_mask_raw_floor_postblob,
            floor_roi_mask=floor_mask,
            floor_roi_overlay=floor_roi_overlay,
            seed_bottom_mask=floor_debug.get("seed_bottom_mask"),
            green_mask=floor_debug.get("green_mask"),
            largest_cc_mask=None,
            exg_row_plot=floor_debug.get("exg_row_plot"),
            linepix_mask=linepix_mask_vis,
            linepix_mask_pre=linepix_mask_pre_vis,
            floor_gate_mask=floor_gate_mask_vis,
            linepix_overlay=linepix_overlay,
            ransac_lines_img=ransac_lines_img,
            dt_debug=dt_debug,
            ori_mask_a=maskA,
            ori_mask_b=maskB,
            hough_lines_img=hough_lines_img,
        )

    if clean_ratio < 0.001 or clean_ratio > 0.25:
        return _fail_result("R_mask_invalid")

    if dt_rand_p90 is not None and dt_rand_mean is not None:
        if dt_rand_p90 < 0.5 or dt_rand_mean < 0.2:
            return _fail_result("R_dt_invalid")

    tau_line = 6
    disable_floor_gate = os.getenv(
        "BADC_LINEPIX_DISABLE_FLOOR_GATE",
        os.getenv("BADC_DISABLE_FLOOR_GATE", "1"),  # default: disable extra floor_gate AND
    ).strip() == "1"

    # linepix base & floor ROI application
    # IMPORTANT: "BADC_LINEPIX_SOURCE" is a legacy knob that previously selected the base.
    # If the new knob "BADC_LINEPIX_MASK_BASE" is explicitly provided, it MUST take precedence.
    force_postblob = os.getenv("BADC_FORCE_LINEPIX_POSTBLOB", "").strip() == "1"
    apply_floor_roi = os.getenv("BADC_LINEPIX_APPLY_FLOOR_ROI_TO_MASK", "1").strip() != "0"
    # Keep this opt-in; default off so linepix/overlay behavior is not silently overridden.
    linepix_direct_noblob = _env_flag("BADC_LINEPIX_DIRECT_NO_BLOB", False)
    mask_base = os.getenv("BADC_LINEPIX_MASK_BASE", "noblob").strip().lower()
    mask_base_explicit = ("BADC_LINEPIX_MASK_BASE" in os.environ)
    legacy_source = None
    if (not mask_base_explicit) and ("BADC_LINEPIX_SOURCE" in os.environ):
        legacy_source = os.getenv("BADC_LINEPIX_SOURCE", "").strip().lower()
        if legacy_source in ("mask", "postblob"):
            mask_base = "postblob"
        elif legacy_source in ("noblob", "raw_noblob"):
            mask_base = "noblob"
        elif legacy_source in ("preblob", "raw_preblob"):
            mask_base = "preblob"
        elif legacy_source in ("clean",):
            mask_base = "clean"
        elif legacy_source in ("edges",):
            mask_base = "edges"
    if force_postblob:
        mask_base = "postblob"

    use_bbox_for_linepix_gate = _env_flag("BADC_LINEPIX_FLOOR_GATE_USE_BBOX", True)

    def _pick_linepix_base(base: str) -> Optional[np.ndarray]:
        b = (base or "").lower()
        if b in ("noblob", "raw_noblob"):
            return None if white_mask_raw_floor_noblob is None else white_mask_raw_floor_noblob.copy()
        if b in ("preblob", "raw_preblob"):
            return None if white_mask_raw_floor_preblob is None else white_mask_raw_floor_preblob.copy()
        if b in ("postblob", "mask"):
            return None if white_mask_raw_floor_postblob is None else white_mask_raw_floor_postblob.copy()
        if b in ("clean", "white_mask_clean"):
            return None if white_mask_clean is None else white_mask_clean.copy()
        if b in ("edges",):
            return None
        return None if white_mask_raw_floor_noblob is None else white_mask_raw_floor_noblob.copy()

    linepix_mask = None
    linepix_pre_gate: Optional[np.ndarray] = None
    linepix_source = f"base:{mask_base}"
    if linepix_mask_override is not None:
        linepix_source = "override"
        linepix_mask = linepix_mask_override.copy()
        if linepix_mask.shape[:2] != floor_mask.shape[:2]:
            linepix_mask = cv2.resize(
                linepix_mask, (floor_mask.shape[1], floor_mask.shape[0]), interpolation=cv2.INTER_NEAREST
            )
        if linepix_mask.ndim == 3:
            linepix_mask = cv2.cvtColor(linepix_mask, cv2.COLOR_BGR2GRAY)
        linepix_mask = (linepix_mask > 0).astype(np.uint8) * 255
    else:
        linepix_mask = _pick_linepix_base(mask_base)
        min_lp = int(0.001 * float(h * w))
        if linepix_mask is not None and int(np.count_nonzero(linepix_mask)) < min_lp:
            linepix_mask = None
        if linepix_mask is None:
            linepix_source = "canny"
            if frame_bgr is not None:
                edges = cv2.Canny(gray, 30, 110)
                linepix_mask = edges
            else:
                linepix_mask = white_mask_clean.copy()

    if linepix_direct_noblob and linepix_mask_override is None and isinstance(white_mask_raw_floor_noblob, np.ndarray):
        linepix_mask = white_mask_raw_floor_noblob.copy()
        linepix_source = "direct_noblob"
        # Keep direct noblob untouched unless user explicitly disables this bypass.
        apply_floor_roi = False

    # Pixel-count diagnostics (for report.json)
    def _nz_u8(m: Optional[np.ndarray]) -> int:
        if not isinstance(m, np.ndarray) or m.size == 0:
            return 0
        mm = m
        if mm.ndim == 3:
            mm = cv2.cvtColor(mm, cv2.COLOR_BGR2GRAY)
        if mm.dtype != np.uint8:
            mm = mm.astype(np.uint8)
        return int(np.count_nonzero(mm))

    try:
        base_metrics["pix_white_mask_raw_floor_noblob"] = _nz_u8(white_mask_raw_floor_noblob)
    except Exception:
        pass
    try:
        base_metrics["pix_linepix_base_before_roi"] = _nz_u8(linepix_mask)
    except Exception:
        pass

    # Keep a copy before further gates for optional RANSAC source selection.
    linepix_pre_gate = linepix_mask.copy() if isinstance(linepix_mask, np.ndarray) else None

    # Apply floor ROI to mask base (default ON)
    if apply_floor_roi and linepix_mask is not None and floor_mask is not None:
        # NOTE: strict AND with floor_mask can erase boundary lines that sit on (or slightly outside)
        # the green/lab floor ROI. To avoid losing the outer sidelines near the image border, we
        # optionally gate by an *expanded bbox* of the floor ROI instead of the ROI mask itself.
        pad_px = int(os.environ.get("BADC_LINEPIX_FLOOR_BBOX_PAD_PX", "40") or "40")
        if pad_px > 0:
            ys, xs = np.where(floor_mask > 0)
            if xs.size > 0 and ys.size > 0:
                x0, x1 = int(xs.min()), int(xs.max())
                y0, y1 = int(ys.min()), int(ys.max())
                x0 = max(0, x0 - pad_px); y0 = max(0, y0 - pad_px)
                x1 = min(linepix_mask.shape[1] - 1, x1 + pad_px)
                y1 = min(linepix_mask.shape[0] - 1, y1 + pad_px)
                bbox_mask = np.zeros_like(floor_mask)
                bbox_mask[y0:y1+1, x0:x1+1] = 255
                linepix_mask = cv2.bitwise_and(linepix_mask, bbox_mask)
            else:
                # Fallback to original behavior if floor_mask is empty.
                linepix_mask = cv2.bitwise_and(linepix_mask, floor_mask)
        else:
            linepix_mask = cv2.bitwise_and(linepix_mask, floor_mask)

    try:
        base_metrics["pix_linepix_after_roi"] = _nz_u8(linepix_mask)
    except Exception:
        pass

    # Record selection for later debug; metrics may not exist yet at this scope.
    if "metrics" in locals() and isinstance(metrics, dict):
        metrics["linepix_mask_base"] = mask_base
        metrics["linepix_apply_floor_roi"] = int(apply_floor_roi)
        metrics["linepix_direct_noblob"] = int(linepix_direct_noblob)
        if legacy_source is not None:
            metrics["linepix_source_legacy"] = legacy_source
        metrics["linepix_source"] = linepix_source
    else:
        floor_debug["linepix_mask_base"] = mask_base
        floor_debug["linepix_apply_floor_roi"] = int(apply_floor_roi)
        floor_debug["linepix_direct_noblob"] = int(linepix_direct_noblob)
        if legacy_source is not None:
            floor_debug["linepix_source_legacy"] = legacy_source
        floor_debug["linepix_source"] = linepix_source
    # Normalize to binary uint8 {0,255} for downstream ops and debug visibility.
    if isinstance(linepix_mask, np.ndarray):
        if linepix_mask.ndim == 3:
            linepix_mask = cv2.cvtColor(linepix_mask, cv2.COLOR_BGR2GRAY)
        if linepix_mask.dtype != np.uint8:
            linepix_mask = linepix_mask.astype(np.uint8)
        linepix_mask = ((linepix_mask > 0).astype(np.uint8) * 255)
    num_linepix_points_pre = int(np.count_nonzero(linepix_mask))
    linepix_mask_pre = linepix_mask.copy()
    # floor_gate is now for metrics only; do NOT crop linepix to avoid losing far-end lines.
    floor_gate_keep_ratio = 1.0
    floor_gate_applied = False
    num_linepix_points_post_floor = num_linepix_points_pre
    floor_gate_mask = floor_mask.copy()
    # Treat floor_bbox rectangle as valid floor area for diagnostics.
    bx0, by0, bx1, by1 = floor_bbox
    bx0 = max(0, min(bx0, w - 1)); bx1 = max(0, min(bx1, w))
    by0 = max(0, min(by0, h - 1)); by1 = max(0, min(by1, h))
    if bx1 > bx0 and by1 > by0:
        floor_gate_mask[by0:by1, bx0:bx1] = 255
    green_mask = floor_debug.get("green_mask")
    if not isinstance(green_mask, np.ndarray) or green_mask.size == 0:
        b, g, r = cv2.split(frame_bgr)
        green_score = g.astype(np.int16) - np.maximum(r, b).astype(np.int16)
        green_mask = ((green_score > 24) & (g > 90)).astype(np.uint8) * 255
        if floor_mask is not None:
            green_mask = cv2.bitwise_and(green_mask, floor_mask)
    if green_mask.shape[:2] != linepix_mask.shape[:2]:
        green_mask = cv2.resize(
            green_mask, (linepix_mask.shape[1], linepix_mask.shape[0]), interpolation=cv2.INTER_NEAREST
        )
    if green_mask.ndim == 3:
        green_mask = cv2.cvtColor(green_mask, cv2.COLOR_BGR2GRAY)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    green_dil = cv2.dilate(green_mask, np.ones((5, 5), np.uint8), iterations=1)

    # Green-mask gating with fallback
    # If apply_floor_roi is ON and the user did NOT explicitly set BADC_LINEPIX_GREEN_ONLY,
    # default green_only to OFF to avoid wiping the ROI expansion.
    green_only_env_raw = os.environ.get("BADC_LINEPIX_GREEN_ONLY", None)
    green_only_defaulted_off_due_to_roi = False
    green_only_defaulted_off_due_to_direct_noblob = False
    if linepix_direct_noblob and (green_only_env_raw is None):
        green_only = False
        green_only_env = "(default-off-due-to-direct_noblob)"
        green_only_defaulted_off_due_to_direct_noblob = True
    elif apply_floor_roi and (green_only_env_raw is None):
        green_only = False
        green_only_env = "(default-off-due-to-apply_floor_roi)"
        green_only_defaulted_off_due_to_roi = True
    else:
        green_only_env = (green_only_env_raw if green_only_env_raw is not None else "1").strip()
        green_only = green_only_env not in ("0", "false", "False", "no", "NO", "off", "OFF")
    green_keep_min = float(os.environ.get("BADC_GREEN_KEEP_MIN", "0.60"))
    green_gate_mode = os.environ.get("BADC_GREEN_GATE_MODE", "erode").strip().lower()
    erode_k = int(os.environ.get("BADC_GREEN_ERODE_K", "7"))
    dilate_k = int(os.environ.get("BADC_GREEN_DILATE_K", "13"))
    erode_k = max(0, erode_k | 1)
    dilate_k = max(0, dilate_k | 1)

    if green_gate_mode in ("none", "raw", "off"):
        green_mask_for_gate = green_mask
    elif green_gate_mode in ("dilate", "dilation"):
        if dilate_k > 0:
            green_mask_for_gate = cv2.dilate(
                green_mask,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_k, dilate_k)),
                iterations=1,
            )
        else:
            green_mask_for_gate = green_mask
    else:  # default erode
        if erode_k > 0:
            green_mask_for_gate = cv2.erode(
                green_mask,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_k, erode_k)),
                iterations=1,
            )
        else:
            green_mask_for_gate = green_mask

    green_applied = False
    _before = int(cv2.countNonZero(linepix_mask))
    _after = _before
    _keep = 1.0
    bypass_cleanup = _env_flag("BADC_LINEPIX_BYPASS_CLEANUP", False) or linepix_direct_noblob
    if (not bypass_cleanup) and green_only:
        _gated = cv2.bitwise_and(linepix_mask, green_mask_for_gate)
        _after = int(cv2.countNonZero(_gated))
        _keep = float(_after) / float(max(1, _before))
        if _keep >= float(green_keep_min):
            linepix_mask = _gated
            green_applied = True
    base_metrics["linepix_green_only_used"] = bool(green_applied)
    base_metrics["linepix_green_only_env"] = green_only_env_raw if green_only_env_raw is not None else "<unset>"
    base_metrics["linepix_green_only_effective"] = bool(green_only)
    base_metrics["linepix_green_only_defaulted_off_due_to_roi"] = bool(green_only_defaulted_off_due_to_roi)
    base_metrics["linepix_green_only_defaulted_off_due_to_direct_noblob"] = bool(
        green_only_defaulted_off_due_to_direct_noblob
    )
    base_metrics["green_keep_ratio"] = float(_keep)
    base_metrics["green_linepix_before"] = int(_before)
    base_metrics["green_linepix_after"] = int(_after)
    base_metrics["pix_linepix_before_green"] = int(_before)
    base_metrics["pix_linepix_after_green"] = int(_after)
    line_like_mask = _line_like_mask(linepix_mask)
    if not bypass_cleanup:
        # Linepix cleanup can accidentally delete weak far-court horizontals.
        # Make it tunable so we can diagnose "mask has it" vs "cleanup killed it".
        lp_speckle_min_area = int(os.getenv("BADC_LINEPIX_SPECKLE_MIN_AREA", "8"))
        lp_speckle_min_aspect = float(os.getenv("BADC_LINEPIX_SPECKLE_MIN_ASPECT", "3.0"))
        lp_speckle_min_long = int(os.getenv("BADC_LINEPIX_SPECKLE_MIN_LONG", "10"))
        linepix_mask = _remove_small_speckles(
            linepix_mask,
            min_area=lp_speckle_min_area,
            min_aspect=lp_speckle_min_aspect,
            min_long=lp_speckle_min_long,
        )
    num_linepix_points = int(np.count_nonzero(linepix_mask))
    try:
        base_metrics["pix_linepix_final"] = int(num_linepix_points)
    except Exception:
        pass
    h_lp, w_lp = linepix_mask.shape[:2]
    total_lp = max(1, num_linepix_points)

    # Compute ratios inside floor bbox/ROI to avoid bias when ROI sits mostly at bottom.
    x0, y0, x1, y1 = 0, 0, w_lp, h_lp
    try:
        if isinstance(floor_bbox, (list, tuple)) and len(floor_bbox) == 4:
            fx0, fy0, fx1, fy1 = [int(round(v)) for v in floor_bbox]
            fx0 = max(0, min(w_lp, fx0))
            fx1 = max(0, min(w_lp, fx1))
            fy0 = max(0, min(h_lp, fy0))
            fy1 = max(0, min(h_lp, fy1))
            if fx1 > fx0 and fy1 > fy0:
                x0, y0, x1, y1 = fx0, fy0, fx1, fy1
    except Exception:
        pass

    roi_lp = linepix_mask[y0:y1, x0:x1]
    h_r, w_r = roi_lp.shape[:2] if roi_lp is not None else (0, 0)
    if h_r <= 0 or w_r <= 0:
        roi_lp = linepix_mask
        h_r, w_r = roi_lp.shape[:2]

    left_lp = int(np.count_nonzero(roi_lp[:, : w_r // 2]))
    right_lp = int(np.count_nonzero(roi_lp[:, w_r // 2 :]))
    top_lp = int(np.count_nonzero(roi_lp[: h_r // 2, :]))
    bottom_lp = int(np.count_nonzero(roi_lp[h_r // 2 :, :]))
    if frame_bgr is not None:
        # Visualization only: optionally draw linepix on top of floor_roi_overlay (raw frame + green ROI overlay).
        overlay_base = os.getenv("BADC_LINEPIX_OVERLAY_BASE", "raw").strip().lower()
        floor_overlay = locals().get("floor_roi_overlay", None)
        if overlay_base in ("floor", "floor_roi", "roi", "floor_roi_overlay") and floor_overlay is not None:
            linepix_overlay = floor_overlay.copy()
        else:
            linepix_overlay = frame_bgr.copy()
        # Visual-only: choose which mask to draw as red.
        #   linepix (default): post-gate linepix_mask
        #   noblob: white_mask_raw_floor_noblob
        #   pre: linepix_mask_pre if exists (pre floor-gate)
        overlay_mask_mode = os.getenv("BADC_LINEPIX_OVERLAY_MASK", "linepix").strip().lower()
        if overlay_mask_mode in ("noblob", "white_noblob"):
            overlay_mask = white_mask_raw_floor_noblob
        elif overlay_mask_mode in ("pre", "pre_gate", "pregate"):
            overlay_mask = locals().get("linepix_mask_pre", linepix_mask)
        else:
            overlay_mask = linepix_mask
        dil = int(os.getenv("BADC_LINEPIX_OVERLAY_DILATE", "0"))
        if dil > 0:
            k = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * dil + 1, 2 * dil + 1))
            overlay_mask = cv2.dilate(overlay_mask.astype(np.uint8), k, iterations=1)
        linepix_overlay[overlay_mask > 0] = (0, 0, 255)
    else:
        linepix_overlay = cv2.cvtColor(linepix_mask, cv2.COLOR_GRAY2BGR)

    # Debug dump for gate/pre/post masks to visualize line breaks near people/ROI.
    if debug_dir is not None:
        def _dump_mask(name: str, m: Optional[np.ndarray]) -> None:
            if m is None or not isinstance(m, np.ndarray):
                return
            mm = m
            if mm.ndim == 3:
                mm = cv2.cvtColor(mm, cv2.COLOR_BGR2GRAY)
            if mm.dtype != np.uint8:
                mm = mm.astype(np.uint8)
            mm = ((mm > 0).astype(np.uint8) * 255)
            cv2.imwrite(os.path.join(debug_dir, f"{debug_prefix}_{name}.jpeg"), mm)

        _dump_mask("floor_gate_mask", floor_gate_mask)
        _dump_mask("linepix_pre_gate", linepix_mask_pre)
        _dump_mask("linepix_post_gate", linepix_mask)

    # Component split is helpful when the mask is very noisy, but it can also
    # collapse coverage to only one side of the court if bridged CCs split at the net/occluders.
    # Default: disable when apply_floor_roi_to_mask is true (we want full-ROI coverage).
    comp_split_default = "0" if (apply_floor_roi or linepix_direct_noblob) else "1"
    comp_split_enabled = bool(int(os.getenv("BADC_LINEPIX_COMPONENT_SPLIT", comp_split_default)))
    if (not skip_component_split) and comp_split_enabled:
        # Bridge fragmented far-court lines into larger components for Hough.
        cc_kernel_sz = int(os.getenv("BADC_LINEPIX_CC_KERNEL", "7"))
        cc_dilate_iters = int(os.getenv("BADC_LINEPIX_CC_DILATE_ITERS", "2"))
        cc_kernel = np.ones((cc_kernel_sz, cc_kernel_sz), np.uint8)
        cc_mask = cv2.dilate(linepix_mask, cc_kernel, iterations=cc_dilate_iters)
        num_cc, labels_cc, stats_cc, _ = cv2.connectedComponentsWithStats(cc_mask, connectivity=8)
        components: list[Tuple[int, int, Tuple[int, int, int, int]]] = []
        for idx in range(1, int(num_cc)):
            area = int(stats_cc[idx, cv2.CC_STAT_AREA])
            x = int(stats_cc[idx, cv2.CC_STAT_LEFT])
            y = int(stats_cc[idx, cv2.CC_STAT_TOP])
            w_cc = int(stats_cc[idx, cv2.CC_STAT_WIDTH])
            h_cc = int(stats_cc[idx, cv2.CC_STAT_HEIGHT])
            if area <= 0 or w_cc <= 0 or h_cc <= 0:
                continue
            bbox_xyxy = (x, y, x + w_cc - 1, y + h_cc - 1)
            components.append((idx, area, bbox_xyxy))
        components.sort(key=lambda item: item[1], reverse=True)
        top_k = int(os.getenv("BADC_LINEPIX_CC_TOPK", "6"))
        component_stats_top3: list[Dict[str, Any]] = []
        component_results: list[Dict[str, Any]] = []
        img_area = float(max(1, linepix_mask.shape[0] * linepix_mask.shape[1]))
        for comp_rank, (idx, area, bbox_xyxy) in enumerate(components[:top_k]):
            comp_mask_label = np.zeros_like(linepix_mask)
            comp_mask_label[labels_cc == idx] = 255
            comp_mask = cv2.bitwise_and(linepix_mask, comp_mask_label)
            x0, y0, x1, y1 = bbox_xyxy
            bbox_area = float(max(0, x1 - x0 + 1) * max(0, y1 - y0 + 1))
            bbox_area_ratio = float(bbox_area / img_area)
            if bbox_area_ratio < 0.06:
                continue
            comp_linepix = int(np.count_nonzero(comp_mask))
            green_overlap = 0.0
            if comp_linepix > 0:
                green_overlap = float(np.count_nonzero(cv2.bitwise_and(comp_mask, green_dil))) / float(
                    max(comp_linepix, 1)
                )
            if green_overlap < 0.18:
                continue
            res_comp = _fit_court_homography_hough(
                frame_bgr,
                person_boxes=person_boxes,
                linepix_mask_override=comp_mask,
                skip_component_split=True,
                component_id=int(comp_rank),
                component_bbox=bbox_xyxy,
                component_area=int(area),
            )
            metrics_c = res_comp.metrics if isinstance(res_comp.metrics, dict) else {}
            score_base_val = metrics_c.get("score_base", metrics_c.get("score_final"))
            try:
                score_base = float(score_base_val) if score_base_val is not None else float("-inf")
            except Exception:
                score_base = float("-inf")
            score_total = metrics_c.get("score_total")
            try:
                score_total_val = float(score_total)
            except Exception:
                score_total_val = float(score_base)
            if not math.isfinite(score_total_val):
                continue
            inside_ct = metrics_c.get("inside_count")
            num_hough = metrics_c.get("num_hough_lines") or metrics_c.get("num_ransac_lines")
            selection_score = float(score_total_val) + float(25.0 * bbox_area_ratio)
            comp_stat = {
                "component_id": int(comp_rank),
                "area": int(area),
                "area_px": int(area),
                "bbox": [int(v) for v in bbox_xyxy],
                "num_linepix_points": comp_linepix,
                "num_hough_lines": int(num_hough) if num_hough is not None else None,
                "bbox_area_ratio": float(bbox_area_ratio),
                "base_score": None if not math.isfinite(score_base) else float(score_base),
                "inside_count": int(inside_ct) if inside_ct is not None else 0,
                "score_total": float(score_total_val),
                "score_breakdown": {
                    "base": None if not math.isfinite(score_base) else float(score_base),
                    "inside_bonus": float(10.0 * (inside_ct or 0)),
                    "score_total": float(score_total_val),
                },
                "green_overlap": float(green_overlap),
                "score_total_sel": float(selection_score),
            }
            component_stats_top3.append(comp_stat)
            if res_comp.corners is None:
                continue
            component_results.append(
                {
                    "score_total": float(selection_score),
                    "score_total_raw": float(score_total_val),
                    "inside_count": int(inside_ct) if inside_ct is not None else 0,
                    "bbox_area_ratio": float(bbox_area_ratio),
                    "res": res_comp,
                    "stat": comp_stat,
                }
            )
        max_inside_count = 0
        selection_reason = "none"
        best_stat = {}
        if component_results:
            max_inside_count = max(item["inside_count"] for item in component_results)
            if max_inside_count > 0:
                candidates = [item for item in component_results if item["inside_count"] == max_inside_count]
                selection_reason = "max_inside"
            else:
                candidates = component_results
                selection_reason = "bbox_area_then_score"
            candidates = sorted(
                candidates,
                key=lambda item: (item.get("bbox_area_ratio", 0.0), item.get("score_total", -1e9)),
                reverse=True,
            )
            best_item = candidates[0]
            best_res = best_item["res"]
            best_stat = best_item["stat"]
            best_score = best_item["score_total"]
            best_metrics = best_res.metrics if isinstance(best_res.metrics, dict) else {}
            if isinstance(best_metrics, dict):
                best_metrics["selected_component_id"] = int(best_stat.get("component_id"))
                best_metrics["component_stats_top3"] = component_stats_top3
                best_metrics.setdefault("score_total", float(best_item.get("score_total_raw", best_score)))
                best_metrics["score_total_sel"] = float(best_score)
                best_metrics["max_inside_count"] = int(max_inside_count)
                best_metrics["selection_reason"] = selection_reason
            # If no inside counts, keep full-mask path instead of early-return.
        if max_inside_count <= 0:
            base_metrics["component_split_fallback_full"] = True
            base_metrics["component_split_selected_component_id"] = int(best_stat.get("component_id", -1))
            base_metrics["component_split_selection_reason"] = selection_reason
        else:
            return best_res
    # Fall back to single-component processing if none found.

    # ------------------ Optional short-circuit: use raw-floor Hough fit only ------------------
    if _env_flag("BADC_FORCE_RAW_FLOOR_ONLY", False):
        H_raw, raw_metrics, raw_debug = _fit_court_homography_from_raw_floor_debug(
            white_mask_raw_floor_postblob,
            frame_bgr.shape,
            floor_roi_mask=floor_mask,
            white_mask_raw_floor_noblob=white_mask_raw_floor_noblob,
        )
        metrics = {**base_metrics, **(raw_metrics or {})}
        if H_raw is not None:
            corners_raw = project_points(H_raw, get_bwf_corners()).astype(np.float32)
            raw_score = float(raw_metrics.get("raw_floor_best_score", 0.0)) if isinstance(raw_metrics, dict) else 0.0
            score_center = float(_env_float("BADC_RAW_FLOOR_SCORE_CENTER", 40.0))
            score_scale = float(max(1e-3, _env_float("BADC_RAW_FLOOR_SCORE_SCALE", 12.0)))
            z = float((raw_score - score_center) / score_scale)
            z = max(-60.0, min(60.0, z))
            conf = float(1.0 / (1.0 + math.exp(-z)))
            return CourtFitResult(
                H=H_raw,
                corners=corners_raw,
                confidence=conf,
                metrics=metrics,
                reason="R_ok_raw_floor_forced",
                method_used="raw_floor_forced",
                white_mask_raw_floor_preblob=white_mask_raw_floor_preblob,
                white_mask_raw_floor_postblob=white_mask_raw_floor_postblob,
                white_mask_raw_floor_noblob=white_mask_raw_floor_noblob,
                floor_roi_mask=floor_mask,
                floor_roi_overlay=floor_roi_overlay,
            )
        else:
            return CourtFitResult(
                H=None,
                corners=None,
                confidence=0.0,
                metrics=metrics,
                reason="R_raw_floor_failed_forced",
                method_used="raw_floor_forced",
                white_mask_raw_floor_preblob=white_mask_raw_floor_preblob,
                white_mask_raw_floor_postblob=white_mask_raw_floor_postblob,
                white_mask_raw_floor_noblob=white_mask_raw_floor_noblob,
                floor_roi_mask=floor_mask,
                floor_roi_overlay=floor_roi_overlay,
            )

# Select mask for RANSAC (diagnostic knob to see if cleanup killed weak lines)
    ransac_mask_source = _env_str("BADC_RANSAC_MASK_SOURCE", "linepix").strip().lower()
    ransac_mask = linepix_mask
    if ransac_mask_source in ("noblob", "raw_floor_noblob", "rawfloor_noblob"):
        if white_mask_raw_floor_noblob is not None:
            ransac_mask = white_mask_raw_floor_noblob
        elif linepix_pre_gate is not None:
            ransac_mask = linepix_pre_gate
    elif ransac_mask_source in ("pre_gate", "linepix_pre_gate", "pregate"):
        if linepix_pre_gate is not None:
            ransac_mask = linepix_pre_gate

    try:
        base_metrics["ransac_mask_source"] = ransac_mask_source
        base_metrics["ransac_mask_nz"] = int(np.count_nonzero(ransac_mask)) if isinstance(ransac_mask, np.ndarray) else 0
        base_metrics["linepix_mask_nz"] = int(np.count_nonzero(linepix_mask)) if isinstance(linepix_mask, np.ndarray) else 0
        base_metrics["linepix_pre_gate_nz"] = int(np.count_nonzero(linepix_pre_gate)) if isinstance(linepix_pre_gate, np.ndarray) else None
        base_metrics["white_mask_raw_floor_noblob_nz"] = int(np.count_nonzero(white_mask_raw_floor_noblob)) if isinstance(white_mask_raw_floor_noblob, np.ndarray) else None
    except Exception:
        pass

    segments, tls_stats = ransac_line_segments_from_mask(
        ransac_mask,
        floor_mask,
        max_lines=30,
        iters=800,
        inlier_thr=2.0,
        min_inliers=250,
        min_length_ratio=0.08,
        seed=0,
    )
    ransac_lines_img = (
        frame_bgr.copy() if frame_bgr is not None else cv2.cvtColor(linepix_mask, cv2.COLOR_GRAY2BGR)
    )
    rng_lines = np.random.default_rng(0)
    for seg in segments:
        color = tuple(int(v) for v in rng_lines.integers(0, 255, size=3))
        cv2.line(
            ransac_lines_img,
            (int(round(seg.x1)), int(round(seg.y1))),
            (int(round(seg.x2)), int(round(seg.y2))),
            color,
            2,
        )
    lines_all = [(seg.x1, seg.y1, seg.x2, seg.y2) for seg in segments]
    line_ids_all = list(range(len(lines_all)))
    x0, y0, x1, y1 = floor_bbox
    # Use frame size if available; otherwise fall back to mask size.
    if frame_bgr is not None:
        H_img, W_img = frame_bgr.shape[:2]
    else:
        H_img, W_img = linepix_mask.shape[:2]

    # Optional right-strip rescue:
    # - debug overlay: draw extra right-side lines on ransac_lines_img
    # - apply-to-fit: inject rescued lines into lines_all/line_ids_all for downstream A/B split and fitting
    debug_right_rescue_enabled = _env_flag("BADC_RANSAC_DEBUG_RIGHT_RESCUE", False)
    apply_right_rescue_to_fit = _env_flag(
        "BADC_RANSAC_RIGHT_RESCUE_APPLY",
        _env_flag("BADC_RANSAC_RIGHT_RESCUE", False),
    )
    run_right_rescue = bool(debug_right_rescue_enabled or apply_right_rescue_to_fit)
    debug_right_rescue_used = False
    debug_right_rescue_lines: list[Tuple[float, float, float, float]] = []
    right_rescue_fit_lines: list[Tuple[float, float, float, float]] = []
    right_rescue_applied_to_fit = False
    right_rescue_fit_added = 0
    debug_right_rescue_mode = _env_str("BADC_RANSAC_DEBUG_RIGHT_RESCUE_MODE", "baseline").strip().lower()
    if debug_right_rescue_mode not in ("vertical", "baseline", "auto"):
        debug_right_rescue_mode = "baseline"
    debug_right_rescue_mask_source = _env_str(
        "BADC_RANSAC_DEBUG_RIGHT_RESCUE_MASK_SOURCE",
        _env_str("BADC_RANSAC_RIGHT_RESCUE_MASK_SOURCE", "preblob"),
    ).strip().lower()
    debug_right_rescue_strip: Optional[list[int]] = None
    debug_right_rescue_target_theta_deg: Optional[float] = None
    if run_right_rescue:
        rescue_mask = ransac_mask
        if debug_right_rescue_mask_source in ("linepix", "linepix_mask"):
            rescue_mask = linepix_mask
        elif debug_right_rescue_mask_source in ("pre_gate", "linepix_pre_gate", "pregate"):
            if linepix_pre_gate is not None:
                rescue_mask = linepix_pre_gate
        elif debug_right_rescue_mask_source in ("noblob", "raw_floor_noblob", "rawfloor_noblob"):
            if white_mask_raw_floor_noblob is not None:
                rescue_mask = white_mask_raw_floor_noblob
        elif debug_right_rescue_mask_source in ("preblob", "raw_floor_preblob", "rawfloor_preblob"):
            if white_mask_raw_floor_preblob is not None:
                rescue_mask = white_mask_raw_floor_preblob

        if isinstance(rescue_mask, np.ndarray):
            rescue_u8 = rescue_mask
            if rescue_u8.ndim == 3:
                rescue_u8 = cv2.cvtColor(rescue_u8, cv2.COLOR_BGR2GRAY)
            rescue_u8 = ((rescue_u8 > 0).astype(np.uint8) * 255)

            fb_w = max(1, int(x1 - x0))
            fb_h = max(1, int(y1 - y0))
            strip_frac = float(
                _env_float(
                    "BADC_RANSAC_DEBUG_RIGHT_RESCUE_STRIP_FRAC",
                    _env_float("BADC_RANSAC_RIGHT_RESCUE_STRIP_FRAC", 0.20),
                )
            )
            top_expand_frac = float(
                _env_float(
                    "BADC_RANSAC_DEBUG_RIGHT_RESCUE_TOP_EXPAND_FRAC",
                    _env_float("BADC_RANSAC_RIGHT_RESCUE_TOP_EXPAND_FRAC", 0.40),
                )
            )
            strip_w = max(8, int(round(strip_frac * float(fb_w))))
            xs0 = max(0, int(round(float(x1) - float(strip_w))))
            xs1 = min(int(W_img - 1), int(x1))
            ys0 = max(0, int(round(float(y0) - float(top_expand_frac) * float(fb_h))))
            ys1 = min(int(H_img - 1), int(y1))
            if xs1 > xs0 and ys1 > ys0:
                debug_right_rescue_strip = [int(xs0), int(ys0), int(xs1), int(ys1)]
                roi = np.zeros_like(rescue_u8)
                roi[int(ys0) : int(ys1) + 1, int(xs0) : int(xs1) + 1] = 255

                canny_low = int(max(10, _env_int("BADC_RANSAC_DEBUG_RIGHT_RESCUE_CANNY_LOW", 40)))
                canny_high = int(max(canny_low + 1, _env_int("BADC_RANSAC_DEBUG_RIGHT_RESCUE_CANNY_HIGH", 120)))
                edges_rescue = cv2.Canny(rescue_u8, canny_low, canny_high)
                edges_rescue = cv2.bitwise_and(edges_rescue, roi)

                rescue_thr = int(max(6, _env_int("BADC_RANSAC_DEBUG_RIGHT_RESCUE_THRESHOLD", 10)))
                rescue_minlen_ratio = float(
                    _env_float(
                        "BADC_RANSAC_DEBUG_RIGHT_RESCUE_MIN_LENGTH_RATIO",
                        _env_float("BADC_RANSAC_RIGHT_RESCUE_MIN_LENGTH_RATIO", 0.02),
                    )
                )
                rescue_minlen_px = int(max(12, round(float(min(H_img, W_img)) * float(rescue_minlen_ratio))))
                rescue_max_gap_px = int(max(2, _env_int("BADC_RANSAC_DEBUG_RIGHT_RESCUE_MAX_GAP_PX", 6)))
                rescue_lines = cv2.HoughLinesP(
                    edges_rescue,
                    1,
                    np.pi / 180.0,
                    rescue_thr,
                    minLineLength=rescue_minlen_px,
                    maxLineGap=rescue_max_gap_px,
                )

                angle_tol_deg = float(
                    _env_float(
                        "BADC_RANSAC_DEBUG_RIGHT_RESCUE_ANGLE_TOL_DEG",
                        _env_float("BADC_RANSAC_RIGHT_RESCUE_ANGLE_TOL_DEG", 28.0),
                    )
                )
                baseline_angle_tol_deg = float(
                    _env_float("BADC_RANSAC_DEBUG_RIGHT_RESCUE_BASELINE_ANGLE_TOL_DEG", max(8.0, angle_tol_deg))
                )
                baseline_min_cos = float(
                    _env_float("BADC_RANSAC_DEBUG_RIGHT_RESCUE_BASELINE_MIN_COS", 0.60)
                )
                baseline_min_cos = float(max(0.05, min(0.98, baseline_min_cos)))
                baseline_extend_left_strips = float(
                    _env_float("BADC_RANSAC_DEBUG_RIGHT_RESCUE_BASELINE_EXTEND_LEFT_STRIPS", 1.6)
                )
                baseline_extend_right_strips = float(
                    _env_float("BADC_RANSAC_DEBUG_RIGHT_RESCUE_BASELINE_EXTEND_RIGHT_STRIPS", 0.1)
                )
                baseline_right_trim_strips = float(
                    _env_float("BADC_RANSAC_DEBUG_RIGHT_RESCUE_BASELINE_RIGHT_TRIM_STRIPS", 0.50)
                )
                baseline_min_span_px = float(
                    _env_float("BADC_RANSAC_DEBUG_RIGHT_RESCUE_BASELINE_MIN_SPAN_PX", 220.0)
                )
                baseline_auto_len = _env_flag("BADC_RANSAC_DEBUG_RIGHT_RESCUE_BASELINE_AUTO_LEN", True)
                baseline_scan_left_strips = float(
                    _env_float("BADC_RANSAC_DEBUG_RIGHT_RESCUE_BASELINE_SCAN_LEFT_STRIPS", 4.0)
                )
                baseline_scan_right_strips = float(
                    _env_float("BADC_RANSAC_DEBUG_RIGHT_RESCUE_BASELINE_SCAN_RIGHT_STRIPS", 0.6)
                )
                baseline_scan_samples = int(max(80, _env_int("BADC_RANSAC_DEBUG_RIGHT_RESCUE_BASELINE_SCAN_SAMPLES", 220)))
                baseline_support_radius = int(max(1, _env_int("BADC_RANSAC_DEBUG_RIGHT_RESCUE_BASELINE_SUPPORT_RADIUS", 3)))
                baseline_support_thr = float(
                    _env_float("BADC_RANSAC_DEBUG_RIGHT_RESCUE_BASELINE_SUPPORT_THR", 0.024)
                )
                baseline_smooth_k = int(max(3, _env_int("BADC_RANSAC_DEBUG_RIGHT_RESCUE_BASELINE_SMOOTH_K", 11)))
                if (baseline_smooth_k % 2) == 0:
                    baseline_smooth_k += 1
                baseline_min_run = int(max(4, _env_int("BADC_RANSAC_DEBUG_RIGHT_RESCUE_BASELINE_MIN_RUN", 10)))
                baseline_support_pad_px = float(
                    _env_float("BADC_RANSAC_DEBUG_RIGHT_RESCUE_BASELINE_SUPPORT_PAD_PX", 12.0)
                )
                baseline_auto_min_left_strips = float(
                    _env_float("BADC_RANSAC_DEBUG_RIGHT_RESCUE_BASELINE_AUTO_MIN_LEFT_STRIPS", 2.5)
                )
                baseline_perp_tol_deg = float(
                    _env_float("BADC_RANSAC_DEBUG_RIGHT_RESCUE_BASELINE_PERP_TOL_DEG", 24.0)
                )
                baseline_inter_y_margin_px = float(
                    _env_float("BADC_RANSAC_DEBUG_RIGHT_RESCUE_BASELINE_INTER_Y_MARGIN_PX", 60.0)
                )
                baseline_left_cross_backoff_px = float(
                    _env_float("BADC_RANSAC_DEBUG_RIGHT_RESCUE_BASELINE_LEFT_CROSS_BACKOFF_PX", 300.0)
                )
                dedup_tol_px = float(
                    _env_float(
                        "BADC_RANSAC_DEBUG_RIGHT_RESCUE_DEDUP_TOL_PX",
                        _env_float("BADC_RANSAC_RIGHT_RESCUE_DEDUP_TOL_PX", 8.0),
                    )
                )
                rescue_max_keep = int(max(1, _env_int("BADC_RANSAC_DEBUG_RIGHT_RESCUE_MAX_LINES", 1)))

                # Near-vertical in image => |dx| / len should be small.
                cos_abs_max = float(max(0.0, min(1.0, math.sin(math.radians(angle_tol_deg)))))
                use_vertical_mode = (debug_right_rescue_mode == "vertical")
                if debug_right_rescue_mode in ("baseline", "auto"):
                    use_vertical_mode = False
                target_theta = None
                if not use_vertical_mode:
                    base_angles = []
                    base_weights = []
                    for x1s, y1s, x2s, y2s in lines_all:
                        dxs = float(x2s - x1s)
                        dys = float(y2s - y1s)
                        ll = float(math.hypot(dxs, dys))
                        if ll <= 1e-3:
                            continue
                        if (abs(dxs) / ll) < baseline_min_cos:
                            continue
                        base_angles.append(float(_seg_theta_undirected(x1s, y1s, x2s, y2s)))
                        base_weights.append(ll)
                    if base_angles:
                        ang_arr = np.array(base_angles, dtype=np.float32)
                        w_arr = np.array(base_weights, dtype=np.float32)
                        c2 = float(np.sum(w_arr * np.cos(2.0 * ang_arr)))
                        s2 = float(np.sum(w_arr * np.sin(2.0 * ang_arr)))
                        if abs(c2) + abs(s2) > 1e-6:
                            target_theta = float(0.5 * math.atan2(s2, c2))
                            if target_theta < 0.0:
                                target_theta += math.pi
                    if target_theta is None:
                        best_line = None
                        best_key = -1.0
                        for x1s, y1s, x2s, y2s in lines_all:
                            dxs = float(x2s - x1s)
                            dys = float(y2s - y1s)
                            ll = float(math.hypot(dxs, dys))
                            if ll <= 1e-3:
                                continue
                            key = (abs(dxs) / ll) * ll
                            if key > best_key:
                                best_key = key
                                best_line = (x1s, y1s, x2s, y2s)
                        if best_line is not None:
                            target_theta = float(_seg_theta_undirected(*best_line))
                    if target_theta is None:
                        use_vertical_mode = True
                    else:
                        debug_right_rescue_target_theta_deg = float(math.degrees(target_theta))

                y_ref = float(ys1)
                x_ref = float(xs1)
                existing_refs: list[float] = []
                for x1s, y1s, x2s, y2s in lines_all:
                    dxs = float(x2s - x1s)
                    dys = float(y2s - y1s)
                    ll = float(math.hypot(dxs, dys))
                    if ll <= 1e-3:
                        continue
                    line_s = _line_from_points((x1s, y1s), (x2s, y2s))
                    if use_vertical_mode:
                        if abs(dxs) / ll > cos_abs_max:
                            continue
                        xb = _x_at_y(line_s, y_ref)
                        if xb is not None and math.isfinite(float(xb)):
                            existing_refs.append(float(xb))
                    else:
                        angs = float(_seg_theta_undirected(x1s, y1s, x2s, y2s))
                        if _angle_distance(angs, float(target_theta)) > math.radians(baseline_angle_tol_deg):
                            continue
                        yb = _y_at_x(line_s, x_ref)
                        if yb is not None and math.isfinite(float(yb)):
                            existing_refs.append(float(yb))

                cand: list[Tuple[float, float, float, float, float, float, float]] = []
                if rescue_lines is not None:
                    for xr1, yr1, xr2, yr2 in rescue_lines[:, 0]:
                        x1r = float(xr1)
                        y1r = float(yr1)
                        x2r = float(xr2)
                        y2r = float(yr2)
                        dxr = x2r - x1r
                        dyr = y2r - y1r
                        ll = float(math.hypot(dxr, dyr))
                        if ll < float(rescue_minlen_px):
                            continue
                        line_r = _line_from_points((x1r, y1r), (x2r, y2r))
                        ang = float(_seg_theta_undirected(x1r, y1r, x2r, y2r))
                        mxr = 0.5 * (x1r + x2r)
                        maxx = max(x1r, x2r)
                        if mxr < float(xs0) and maxx < float(xs0):
                            continue
                        if use_vertical_mode:
                            if abs(dxr) / max(ll, 1e-6) > cos_abs_max:
                                continue
                            ref_val = _x_at_y(line_r, y_ref)
                            if ref_val is None or (not math.isfinite(float(ref_val))):
                                ref_val = mxr
                        else:
                            if _angle_distance(ang, float(target_theta)) > math.radians(baseline_angle_tol_deg):
                                continue
                            if (abs(dxr) / max(ll, 1e-6)) < (baseline_min_cos * 0.7):
                                continue
                            ref_val = _y_at_x(line_r, x_ref)
                            if ref_val is None or (not math.isfinite(float(ref_val))):
                                ref_val = 0.5 * (y1r + y2r)
                        ref_val = float(ref_val)
                        if any(abs(ref_val - ex) <= dedup_tol_px for ex in existing_refs):
                            continue
                        if any(abs(ref_val - c[4]) <= dedup_tol_px for c in cand):
                            continue
                        cand.append((x1r, y1r, x2r, y2r, ref_val, ll, ang))

                if use_vertical_mode:
                    # Prefer the rightmost candidates, then longer segments.
                    cand.sort(key=lambda t: (t[4], t[5]), reverse=True)
                else:
                    # Prefer long lines around the current baseline family.
                    y_med = float(np.median(existing_refs)) if existing_refs else float(0.5 * (ys0 + ys1))
                    cand.sort(key=lambda t: (-t[5], abs(t[4] - y_med), -max(t[0], t[2])))

                for x1r, y1r, x2r, y2r, ref_val, _ll, _ang in cand[:rescue_max_keep]:
                    existing_refs.append(float(ref_val))
                    debug_right_rescue_lines.append((x1r, y1r, x2r, y2r))
                    line_dbg = _line_from_points((x1r, y1r), (x2r, y2r))
                    if use_vertical_mode:
                        x_top = _x_at_y(line_dbg, float(ys0))
                        x_bot = _x_at_y(line_dbg, float(ys1))
                        if (
                            x_top is not None
                            and x_bot is not None
                            and math.isfinite(float(x_top))
                            and math.isfinite(float(x_bot))
                        ):
                            pt1 = (int(round(float(x_top))), int(round(float(ys0))))
                            pt2 = (int(round(float(x_bot))), int(round(float(ys1))))
                        else:
                            pt1 = (int(round(x1r)), int(round(y1r)))
                            pt2 = (int(round(x2r)), int(round(y2r)))
                    else:
                        scan_xl = float(max(0, x0, float(xs0) - baseline_scan_left_strips * float(strip_w)))
                        scan_xr = float(min(W_img - 1, x1, float(xs1) + baseline_scan_right_strips * float(strip_w)))
                        if scan_xr <= scan_xl + 8.0:
                            scan_xl = float(max(0, x0, float(xs0) - baseline_extend_left_strips * float(strip_w)))
                            scan_xr = float(min(W_img - 1, x1, float(xs1) + baseline_extend_right_strips * float(strip_w)))

                        xl = float(max(0, x0, float(xs0) - baseline_extend_left_strips * float(strip_w)))
                        xr = float(
                            min(
                                W_img - 1,
                                x1,
                                float(xs1) + baseline_extend_right_strips * float(strip_w)
                                - baseline_right_trim_strips * float(strip_w),
                            )
                        )
                        if xr <= xl + 8.0:
                            xr = float(min(W_img - 1, x1, float(xs1)))

                        if baseline_auto_len and scan_xr > scan_xl + 8.0:
                            xs_scan = np.linspace(scan_xl, scan_xr, baseline_scan_samples, dtype=np.float32)
                            sup = np.zeros((baseline_scan_samples,), dtype=np.float32)
                            for i_s, x_s in enumerate(xs_scan):
                                y_s = _y_at_x(line_dbg, float(x_s))
                                if y_s is None or (not math.isfinite(float(y_s))):
                                    continue
                                xi = int(round(float(x_s)))
                                yi = int(round(float(y_s)))
                                if xi < 0 or xi >= int(W_img) or yi < 0 or yi >= int(H_img):
                                    continue
                                xlo = max(0, xi - baseline_support_radius)
                                xhi = min(int(W_img - 1), xi + baseline_support_radius)
                                ylo = max(0, yi - baseline_support_radius)
                                yhi = min(int(H_img - 1), yi + baseline_support_radius)
                                patch = rescue_u8[ylo : yhi + 1, xlo : xhi + 1]
                                if patch.size <= 0:
                                    continue
                                sup[i_s] = float(np.count_nonzero(patch)) / float(patch.size)

                            if baseline_smooth_k > 1:
                                kernel = np.ones((baseline_smooth_k,), dtype=np.float32) / float(baseline_smooth_k)
                                sup_s = np.convolve(sup, kernel, mode="same")
                            else:
                                sup_s = sup

                            active = sup_s >= float(baseline_support_thr)
                            runs: list[Tuple[int, int]] = []
                            st = None
                            for ii, ok in enumerate(active.tolist()):
                                if ok and st is None:
                                    st = ii
                                if (not ok) and st is not None:
                                    if (ii - st) >= baseline_min_run:
                                        runs.append((int(st), int(ii - 1)))
                                    st = None
                            if st is not None and (len(active) - st) >= baseline_min_run:
                                runs.append((int(st), int(len(active) - 1)))

                            if runs:
                                x_mid = float(0.5 * (x1r + x2r))
                                best_run = None
                                best_score = -1e9
                                for rs, re in runs:
                                    x_rs = float(xs_scan[rs])
                                    x_re = float(xs_scan[re])
                                    run_len = max(1.0, x_re - x_rs)
                                    mean_sup = float(np.mean(sup_s[rs : re + 1]))
                                    contains_mid = (x_rs <= x_mid <= x_re)
                                    score = float(run_len + 260.0 * mean_sup + 0.06 * x_re + (120.0 if contains_mid else 0.0))
                                    if score > best_score:
                                        best_score = score
                                        best_run = (x_rs, x_re)
                                if best_run is not None:
                                    xl = float(best_run[0] - baseline_support_pad_px)
                                    xr = float(best_run[1] + baseline_support_pad_px)

                        # Further clip by intersections with roughly perpendicular existing lines.
                        x_mid = float(0.5 * (x1r + x2r))
                        perp_xs: list[float] = []
                        for x1s, y1s, x2s, y2s in lines_all:
                            ang_s = float(_seg_theta_undirected(x1s, y1s, x2s, y2s))
                            dperp = abs(_angle_distance(ang_s, float(target_theta)) - 0.5 * math.pi)
                            if dperp > math.radians(baseline_perp_tol_deg):
                                continue
                            ln_s = _line_from_points((x1s, y1s), (x2s, y2s))
                            p_int = _intersect_lines(line_dbg, ln_s)
                            if p_int is None:
                                continue
                            px = float(p_int[0])
                            py = float(p_int[1])
                            if not (math.isfinite(px) and math.isfinite(py)):
                                continue
                            if px < float(x0 - 40) or px > float(x1 + 40):
                                continue
                            if py < float(y0 - baseline_inter_y_margin_px) or py > float(y1 + baseline_inter_y_margin_px):
                                continue
                            perp_xs.append(px)
                        if perp_xs:
                            left_cross = max([v for v in perp_xs if v <= x_mid], default=None)
                            right_cross = min([v for v in perp_xs if v >= x_mid], default=None)
                            if left_cross is not None:
                                xl = max(xl, float(left_cross) - baseline_left_cross_backoff_px)
                            if right_cross is not None:
                                xr = min(xr, float(right_cross))

                        if baseline_auto_len:
                            min_left_cover_x = float(xs0) - baseline_auto_min_left_strips * float(strip_w)
                            xl = min(xl, float(min_left_cover_x))

                        xl = float(max(0, x0, min(xl, float(W_img - 1))))
                        xr = float(max(0, x0, min(xr, float(W_img - 1))))
                        if xr <= xl + 8.0:
                            xr = float(min(W_img - 1, x1))
                            xl = float(max(0, x0, xr - baseline_min_span_px))
                        yl = _y_at_x(line_dbg, xl)
                        yr = _y_at_x(line_dbg, xr)
                        if (
                            yl is not None
                            and yr is not None
                            and math.isfinite(float(yl))
                            and math.isfinite(float(yr))
                        ):
                            pt1 = (int(round(xl)), int(round(float(yl))))
                            pt2 = (int(round(xr)), int(round(float(yr))))
                        else:
                            pt1 = (int(round(x1r)), int(round(y1r)))
                            pt2 = (int(round(x2r)), int(round(y2r)))
                    seg_fit = (float(pt1[0]), float(pt1[1]), float(pt2[0]), float(pt2[1]))
                    if math.hypot(seg_fit[2] - seg_fit[0], seg_fit[3] - seg_fit[1]) >= 8.0:
                        right_rescue_fit_lines.append(seg_fit)

                    if debug_right_rescue_enabled or apply_right_rescue_to_fit:
                        cv2.line(
                            ransac_lines_img,
                            pt1,
                            pt2,
                            (40, 255, 40),
                            4,
                        )
                        cv2.circle(ransac_lines_img, pt1, 5, (40, 255, 40), -1)
                        cv2.circle(ransac_lines_img, pt2, 5, (40, 255, 40), -1)
                debug_right_rescue_used = len(right_rescue_fit_lines) > 0

    # Optionally inject rescued lines into the actual fitting pool.
    if apply_right_rescue_to_fit and right_rescue_fit_lines:
        apply_angle_tol_deg = float(_env_float("BADC_RANSAC_RIGHT_RESCUE_APPLY_ANGLE_TOL_DEG", 3.0))
        apply_dist_tol_px = float(_env_float("BADC_RANSAC_RIGHT_RESCUE_APPLY_DIST_TOL_PX", 10.0))
        apply_angle_tol = math.radians(apply_angle_tol_deg)

        existing_lines = list(lines_all)
        next_line_id = (max(line_ids_all) + 1) if line_ids_all else 0

        for x1r, y1r, x2r, y2r in right_rescue_fit_lines:
            dxr = float(x2r - x1r)
            dyr = float(y2r - y1r)
            llr = float(math.hypot(dxr, dyr))
            if llr < 8.0:
                continue

            ang_r = float(np.mod(math.atan2(dyr, dxr), math.pi))
            mxr = 0.5 * (float(x1r) + float(x2r))
            myr = 0.5 * (float(y1r) + float(y2r))
            dup = False

            for x1e, y1e, x2e, y2e in existing_lines:
                dxe = float(x2e - x1e)
                dye = float(y2e - y1e)
                lle = float(math.hypot(dxe, dye))
                if lle <= 1e-3:
                    continue
                ang_e = float(np.mod(math.atan2(dye, dxe), math.pi))
                if _angle_distance(ang_r, ang_e) > apply_angle_tol:
                    continue
                line_e = _line_from_points((x1e, y1e), (x2e, y2e))
                d_mid = abs(_line_signed_distance(line_e, (mxr, myr)))
                if d_mid <= apply_dist_tol_px:
                    dup = True
                    break

            if dup:
                continue

            lines_all.append((float(x1r), float(y1r), float(x2r), float(y2r)))
            line_ids_all.append(int(next_line_id))
            existing_lines.append((float(x1r), float(y1r), float(x2r), float(y2r)))
            next_line_id += 1
            right_rescue_fit_added += 1

        right_rescue_applied_to_fit = right_rescue_fit_added > 0

    # Expand floor bbox upward/lateral to keep far-court lines that sit just above green ROI.
    expand_top_frac = float(os.environ.get("BADC_FLOOR_BBOX_EXPAND_TOP_FRAC", "0.35"))
    expand_lr_frac = float(os.environ.get("BADC_FLOOR_BBOX_EXPAND_LR_FRAC", "0.05"))
    bb_w = max(1, int(x1 - x0))
    bb_h = max(1, int(y1 - y0))
    x0e = max(0, int(round(x0 - expand_lr_frac * bb_w)))
    x1e = min(W_img - 1, int(round(x1 + expand_lr_frac * bb_w)))
    y0e = max(0, int(round(y0 - expand_top_frac * bb_h)))
    y1e = min(H_img - 1, int(round(y1)))

    # --- PATCH: do NOT hard-kill lines near the top of floor bbox ---
    # Old behavior: reject any segment whose midpoint is within top 5% of floor bbox.
    # That kills many far-court lines => A/B too few.
    #
    # New behavior:
    # - Top-guard ratio is configurable via env var (default 0.0 = disabled)
    # - Even if within guard band, only reject if the segment is very short (likely noise)
    top_guard_ratio = float(os.environ.get("BADC_FLOOR_TOP_GUARD_RATIO", "0.0"))
    y_margin = int(top_guard_ratio * max(1, (y1 - y0)))

    # "short" threshold: reject tiny segments near top, keep long court lines
    diag = float(math.hypot(float(x1 - x0), float(y1 - y0)))
    short_len_thr = float(os.environ.get("BADC_FLOOR_TOP_SHORT_THR", "0.06")) * diag

    reject_mid_outside = 0
    reject_mid_topshort = 0
    filtered = []
    filtered_ids: list[int] = []
    # Optionally bypass pruning: keep all RANSAC lines to avoid empty A/B groups when lines are sparse.
    force_ransac_ab = _env_flag("BADC_RAW_FLOOR_FORCE_RANSAC_AB", True)

    for idx, (x1s, y1s, x2s, y2s) in enumerate(lines_all):
        mx = 0.5 * (float(x1s) + float(x2s))
        my = 0.5 * (float(y1s) + float(y2s))

        if force_ransac_ab:
            filtered.append((float(x1s), float(y1s), float(x2s), float(y2s)))
            if idx < len(line_ids_all):
                filtered_ids.append(line_ids_all[idx])
            continue

        # must be inside expanded floor bbox
        if mx < float(x0e) or mx > float(x1e) or my < float(y0e) or my > float(y1e):
            reject_mid_outside += 1
            continue

        # within top guard band: only reject if the segment is very short
        if y_margin > 0 and my < float(y0 + y_margin):
            seg_len = float(math.hypot(float(x2s - x1s), float(y2s - y1s)))
            if seg_len < short_len_thr:
                reject_mid_topshort += 1
                continue

        filtered.append((x1s, y1s, x2s, y2s))
        if idx < len(line_ids_all):
            filtered_ids.append(line_ids_all[idx])

    lines_all = filtered
    line_ids_all = filtered_ids
    angles = []
    weights = []
    line_ids_angles: list[int] = []
    lines_all_filtered: list[Tuple[float, float, float, float]] = []
    line_ids_filtered: list[int] = []
    for (x1s, y1s, x2s, y2s), lid in zip(lines_all, line_ids_all):
        dx = float(x2s) - float(x1s)
        dy = float(y2s) - float(y1s)
        length = float(math.hypot(dx, dy))
        if length <= 1e-3:
            continue
        ang = math.atan2(dy, dx)
        ang = float(np.mod(ang, math.pi))
        angles.append(ang)
        weights.append(length)
        line_ids_angles.append(int(lid))
        lines_all_filtered.append((x1s, y1s, x2s, y2s))
        line_ids_filtered.append(int(lid))
    lines_all = lines_all_filtered
    line_ids_all = line_ids_filtered
    if not angles:
        maskA = np.zeros_like(white_mask_clean)
        maskB = np.zeros_like(white_mask_clean)
        return _fail_result(
            "R_hough_lines",
            maskA=maskA,
            maskB=maskB,
            linepix_mask=linepix_mask,
            linepix_overlay=linepix_overlay,
            ransac_lines_img=ransac_lines_img,
            extra_metrics={
                "num_linepix_points": int(num_linepix_points),
                "num_ransac_lines": int(len(segments)),
                "top10_line_lengths": [],
            },
        )
    angles = np.array(angles, dtype=np.float32)
    weights = np.array(weights, dtype=np.float32)
    base_metrics.update(
        {
            "num_linepix_points": int(num_linepix_points),
            "num_linepix_points_pre_person": int(num_linepix_points_pre),
            "num_linepix_points_post_floor": int(num_linepix_points_post_floor),
            "num_linepix_points_post_person": int(num_linepix_points),
            "linepix_source": str(linepix_source),
            "linepix_green_only_used": bool(green_only),
            "linepix_green_gate_mode": str(green_gate_mode),
            "linepix_green_keep_min": float(green_keep_min),
            "linepix_green_erode_k": int(erode_k),
            "linepix_green_dilate_k": int(dilate_k),
            "linepix_green_gate_applied": bool(green_applied),
            "linepix_green_keep_ratio": float(_keep),
            "linepix_floor_gate_applied": bool(floor_gate_applied),
            "linepix_floor_gate_disabled": bool(disable_floor_gate),
            "linepix_floor_gate_keep_ratio": float(floor_gate_keep_ratio)
            if floor_gate_keep_ratio is not None
            else None,
            "linepix_ratio": float(num_linepix_points) / float(h_lp * w_lp),
            "linepix_left_ratio": float(left_lp) / float(total_lp),
            "linepix_right_ratio": float(right_lp) / float(total_lp),
            "linepix_top_ratio": float(top_lp) / float(total_lp),
            "linepix_bottom_ratio": float(bottom_lp) / float(total_lp),
            "num_ransac_lines": int(len(segments)),
            "top10_line_lengths": [
                float(v)
                for v in sorted([seg.length for seg in segments], reverse=True)[:10]
            ],
            "debug_right_rescue_enabled": bool(debug_right_rescue_enabled),
            "debug_right_rescue_used": bool(debug_right_rescue_used),
            "debug_right_rescue_mode": str(debug_right_rescue_mode),
            "debug_right_rescue_num_lines": int(len(debug_right_rescue_lines)),
            "debug_right_rescue_mask_source": str(debug_right_rescue_mask_source),
            "debug_right_rescue_strip": debug_right_rescue_strip,
            "debug_right_rescue_target_theta_deg": float(debug_right_rescue_target_theta_deg)
            if debug_right_rescue_target_theta_deg is not None
            else None,
            "right_rescue_apply_to_fit_enabled": bool(apply_right_rescue_to_fit),
            "right_rescue_applied_to_fit": bool(right_rescue_applied_to_fit),
            "right_rescue_fit_added": int(right_rescue_fit_added),
            "reject_mid_outside": int(reject_mid_outside),
            "num_hough_lines": int(len(lines_all)),
            "tls_num_input_pts": int(tls_stats.get("tls_num_input_pts", 0)),
            "tls_num_finite_pts": int(tls_stats.get("tls_num_finite_pts", 0)),
            "tls_skipped_degenerate": int(tls_stats.get("tls_skipped_degenerate", 0)),
        }
    )
    if manual:
        base_metrics["manual_present"] = True
    if manual_metrics:
        base_metrics.update(manual_metrics)
    if not disable_anchor_prior:
        base_metrics["used_manual_anchors"] = bool(manual_lt and manual_rt and manual_rb)
    hist, edges = np.histogram(angles, bins=180, range=(0.0, math.pi), weights=weights)
    idx1 = int(np.argmax(hist))
    center1 = float((edges[idx1] + edges[idx1 + 1]) * 0.5)
    centers = (edges[:-1] + edges[1:]) * 0.5
    best_center2 = None
    best_center2_score = -1.0
    for c, hval in zip(centers, hist):
        diff = _angle_distance(float(c), center1)
        if diff < math.radians(20.0):
            continue
        ortho_score = max(0.0, 1.0 - abs(diff - (0.5 * math.pi)) / (0.5 * math.pi))
        score = float(hval) * (0.6 + 0.4 * ortho_score)
        if score > best_center2_score:
            best_center2_score = score
            best_center2 = float(c)
    if best_center2 is None:
        best_center2 = float((center1 + math.pi * 0.5) % math.pi)
    center2 = best_center2
    ori_centers_force_ortho = _env_flag("BADC_ORI_CENTERS_FORCE_ORTHO", False)
    if legacy_ori_split and ("BADC_ORI_CENTERS_FORCE_ORTHO" not in os.environ):
        ori_centers_force_ortho = True
    ori_centers_ortho_max_dev_deg = float(_env_float("BADC_ORI_CENTERS_ORTHO_MAX_DEV_DEG", 20.0))
    ori_center_diff = float(_angle_distance(center1, center2))
    ori_center_diff_deg = float(math.degrees(ori_center_diff))
    ori_center_forced_orth = False
    # Robustness guard: when the two dominant peaks collapse to near-parallel groups,
    # force the second center to an orthogonal direction.
    if ori_centers_force_ortho:
        if abs(ori_center_diff - 0.5 * math.pi) > math.radians(max(5.0, ori_centers_ortho_max_dev_deg)):
            center2 = float((center1 + math.pi * 0.5) % math.pi)
            ori_center_forced_orth = True

    # Orientation band for splitting RANSAC lines into two families (A/B).
    # In strong perspective, the same world-parallel family can span a wider angle range
    # (near side more "vertical", far side more "slanted"). A too-tight band can drop
    # critical boundary lines (e.g., near-left sideline).
    band_deg = float(os.getenv("BADC_ORI_BAND_DEG", "25.0"))
    band = math.radians(band_deg)
    band_soft_deg = float(os.getenv("BADC_ORI_BAND_SOFT_DEG", "45.0"))
    band_soft = math.radians(band_soft_deg)
    soft_minlen_frac = float(os.getenv("BADC_ORI_SOFT_MINLEN_FRAC", "0.45"))
    soft_minlen_max_px = float(os.getenv("BADC_ORI_SOFT_MINLEN_MAX_PX", "300.0"))
    ori_assign_nearest = _env_flag("BADC_ORI_ASSIGN_NEAREST", False)
    ori_enable_soft_assign = _env_flag("BADC_ORI_ENABLE_SOFT_ASSIGN", True)
    ori_soft_keep_boundary = _env_flag("BADC_ORI_SOFT_KEEP_BOUNDARY", True)
    ori_soft_boundary_margin_frac = float(os.getenv("BADC_ORI_SOFT_BOUNDARY_MARGIN_FRAC", "0.08"))
    ori_boundary_fallback = _env_flag("BADC_ORI_BOUNDARY_FALLBACK", True)
    ori_boundary_fallback_max_angle_deg = float(os.getenv("BADC_ORI_BOUNDARY_FALLBACK_MAX_ANGLE_DEG", str(band_soft_deg)))
    ori_force_nearest_remaining = _env_flag("BADC_ORI_FORCE_NEAREST_REMAINING", True)
    ori_force_nearest_max_add = int(max(0, _env_int("BADC_ORI_FORCE_NEAREST_MAX_ADD", 4)))
    ori_force_nearest_minlen_frac = float(os.getenv("BADC_ORI_FORCE_NEAREST_MINLEN_FRAC", "0.40"))
    ori_force_nearest_max_angle_deg = float(os.getenv("BADC_ORI_FORCE_NEAREST_MAX_ANGLE_DEG", "58.0"))
    if legacy_ori_split:
        # Legacy mode: prefer clean horizontal/vertical separation.
        if "BADC_ORI_ASSIGN_NEAREST" not in os.environ:
            ori_assign_nearest = False
        if "BADC_ORI_ENABLE_SOFT_ASSIGN" not in os.environ:
            ori_enable_soft_assign = False
        if "BADC_ORI_FORCE_NEAREST_REMAINING" not in os.environ:
            ori_force_nearest_remaining = False
    bbox_x0, bbox_y0, bbox_x1, bbox_y1 = [float(v) for v in floor_bbox]
    max_w = float(np.max(weights)) if isinstance(weights, np.ndarray) and weights.size > 0 else 1.0
    linesA = []
    linesB = []
    line_ids_A: list[int] = []
    line_ids_B: list[int] = []
    maskA = np.zeros_like(white_mask_clean)
    maskB = np.zeros_like(white_mask_clean)
    ori_nearest_added = 0
    unassigned = []  # (x1,y1,x2,y2, ang, lid, w)
    for (x1s, y1s, x2s, y2s), ang, lid, w in zip(lines_all, angles, line_ids_angles, weights):
        d1 = _angle_distance(float(ang), center1)
        d2 = _angle_distance(float(ang), center2)
        if d1 <= band and d1 <= d2:
            linesA.append((x1s, y1s, x2s, y2s))
            line_ids_A.append(int(lid))
            cv2.line(maskA, (int(x1s), int(y1s)), (int(x2s), int(y2s)), 255, 1)
        elif d2 <= band:
            linesB.append((x1s, y1s, x2s, y2s))
            line_ids_B.append(int(lid))
            cv2.line(maskB, (int(x1s), int(y1s)), (int(x2s), int(y2s)), 255, 1)
        else:
            if ori_assign_nearest:
                if d1 <= d2:
                    linesA.append((x1s, y1s, x2s, y2s))
                    line_ids_A.append(int(lid))
                    cv2.line(maskA, (int(x1s), int(y1s)), (int(x2s), int(y2s)), 255, 1)
                else:
                    linesB.append((x1s, y1s, x2s, y2s))
                    line_ids_B.append(int(lid))
                    cv2.line(maskB, (int(x1s), int(y1s)), (int(x2s), int(y2s)), 255, 1)
                ori_nearest_added += 1
            else:
                unassigned.append((x1s, y1s, x2s, y2s, float(ang), int(lid), float(w)))

    ori_unassigned_initial = int(len(unassigned))

    # Soft-assign long unassigned lines back to the nearest family with a wider band.
    # This prevents missing critical boundaries under perspective.
    ori_soft_added = 0
    if ori_enable_soft_assign and unassigned and band_soft > band:
        min_w = min(soft_minlen_frac * max_w, soft_minlen_max_px)
        boundary_margin_px = max(4.0, ori_soft_boundary_margin_frac * float(max(1.0, bbox_x1 - bbox_x0)))
        for x1s, y1s, x2s, y2s, ang, lid, w in unassigned:
            d1 = _angle_distance(float(ang), center1)
            d2 = _angle_distance(float(ang), center2)
            if min(d1, d2) > band_soft:
                continue
            if w < min_w:
                if not ori_soft_keep_boundary:
                    continue
                near_left = min(float(x1s), float(x2s)) <= bbox_x0 + boundary_margin_px
                near_right = max(float(x1s), float(x2s)) >= bbox_x1 - boundary_margin_px
                if not (near_left or near_right):
                    continue
            if d1 <= d2:
                linesA.append((x1s, y1s, x2s, y2s))
                line_ids_A.append(int(lid))
                cv2.line(maskA, (int(x1s), int(y1s)), (int(x2s), int(y2s)), 255, 1)
            else:
                linesB.append((x1s, y1s, x2s, y2s))
                line_ids_B.append(int(lid))
                cv2.line(maskB, (int(x1s), int(y1s)), (int(x2s), int(y2s)), 255, 1)
            ori_soft_added += 1

    # Boundary fallback: keep near-left/right boundary lines even when soft assign is disabled.
    ori_boundary_fallback_added = 0
    if ori_boundary_fallback and unassigned:
        remaining = []
        boundary_margin_px = max(4.0, ori_soft_boundary_margin_frac * float(max(1.0, bbox_x1 - bbox_x0)))
        fallback_max_angle = math.radians(float(max(5.0, ori_boundary_fallback_max_angle_deg)))
        for x1s, y1s, x2s, y2s, ang, lid, w in unassigned:
            near_left = min(float(x1s), float(x2s)) <= bbox_x0 + boundary_margin_px
            near_right = max(float(x1s), float(x2s)) >= bbox_x1 - boundary_margin_px
            if not (near_left or near_right):
                remaining.append((x1s, y1s, x2s, y2s, ang, lid, w))
                continue
            d1 = _angle_distance(float(ang), center1)
            d2 = _angle_distance(float(ang), center2)
            if min(d1, d2) > fallback_max_angle:
                remaining.append((x1s, y1s, x2s, y2s, ang, lid, w))
                continue
            if d1 <= d2:
                linesA.append((x1s, y1s, x2s, y2s))
                line_ids_A.append(int(lid))
                cv2.line(maskA, (int(x1s), int(y1s)), (int(x2s), int(y2s)), 255, 1)
            else:
                linesB.append((x1s, y1s, x2s, y2s))
                line_ids_B.append(int(lid))
                cv2.line(maskB, (int(x1s), int(y1s)), (int(x2s), int(y2s)), 255, 1)
            ori_boundary_fallback_added += 1
        unassigned = remaining

    # Final rescue: if some lines are still unassigned, keep a few long ones by nearest center.
    ori_force_nearest_added = 0
    if ori_force_nearest_remaining and unassigned and ori_force_nearest_max_add > 0:
        remaining = []
        min_w_force = max(20.0, ori_force_nearest_minlen_frac * max_w)
        max_angle_force = math.radians(float(max(10.0, ori_force_nearest_max_angle_deg)))
        cand = sorted(unassigned, key=lambda t: float(t[6]), reverse=True)
        for x1s, y1s, x2s, y2s, ang, lid, w in cand:
            if ori_force_nearest_added >= ori_force_nearest_max_add:
                remaining.append((x1s, y1s, x2s, y2s, ang, lid, w))
                continue
            d1 = _angle_distance(float(ang), center1)
            d2 = _angle_distance(float(ang), center2)
            if float(w) < min_w_force or min(d1, d2) > max_angle_force:
                remaining.append((x1s, y1s, x2s, y2s, ang, lid, w))
                continue
            if d1 <= d2:
                linesA.append((x1s, y1s, x2s, y2s))
                line_ids_A.append(int(lid))
                cv2.line(maskA, (int(x1s), int(y1s)), (int(x2s), int(y2s)), 255, 1)
            else:
                linesB.append((x1s, y1s, x2s, y2s))
                line_ids_B.append(int(lid))
                cv2.line(maskB, (int(x1s), int(y1s)), (int(x2s), int(y2s)), 255, 1)
            ori_force_nearest_added += 1
        unassigned = remaining

    if len(linesA) < 3 or len(linesB) < 3:
        band = math.radians(30.0)
        linesA = []
        linesB = []
        line_ids_A = []
        line_ids_B = []
        maskA = np.zeros_like(white_mask_clean)
        maskB = np.zeros_like(white_mask_clean)
        for (x1, y1, x2, y2), ang, lid in zip(lines_all, angles, line_ids_angles):
            d1 = _angle_distance(float(ang), center1)
            d2 = _angle_distance(float(ang), center2)
            if d1 <= band and d1 <= d2:
                linesA.append((x1, y1, x2, y2))
                line_ids_A.append(int(lid))
                cv2.line(maskA, (int(x1), int(y1)), (int(x2), int(y2)), 255, 1)
            elif d2 <= band:
                linesB.append((x1, y1, x2, y2))
                line_ids_B.append(int(lid))
                cv2.line(maskB, (int(x1), int(y1)), (int(x2), int(y2)), 255, 1)
        if len(linesA) < 3 or len(linesB) < 3:
            # Last-resort: avoid kmeans here.
            # kmeans on (cos2θ,sin2θ) can split a sparse orientation into two non-orthogonal clusters.
            # Force an orthogonal pair and soft-assign by nearest center.
            base_metrics["angle_cluster_kmeans_used"] = False
            center2 = float((center1 + math.pi * 0.5) % math.pi)

            linesA = []
            linesB = []
            line_ids_A = []
            line_ids_B = []
            maskA = np.zeros_like(white_mask_clean)
            maskB = np.zeros_like(white_mask_clean)

            for (x1, y1, x2, y2), ang, lid in zip(lines_all, angles, line_ids_angles):
                d1 = _angle_distance(float(ang), center1)
                d2 = _angle_distance(float(ang), center2)
                if d1 <= d2:
                    linesA.append((x1, y1, x2, y2))
                    line_ids_A.append(int(lid))
                    cv2.line(maskA, (int(x1), int(y1)), (int(x2), int(y2)), 255, 1)
                else:
                    linesB.append((x1, y1, x2, y2))
                    line_ids_B.append(int(lid))
                    cv2.line(maskB, (int(x1), int(y1)), (int(x2), int(y2)), 255, 1)
    hough_lines_img = frame_bgr.copy()
    for x1, y1, x2, y2 in linesA:
        cv2.line(hough_lines_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
    for x1, y1, x2, y2 in linesB:
        cv2.line(hough_lines_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
    base_metrics["num_hough_lines_a"] = int(len(linesA))
    base_metrics["num_hough_lines_b"] = int(len(linesB))
    base_metrics["ori_center1_deg"] = float(np.degrees(center1))
    base_metrics["ori_center2_deg"] = float(np.degrees(center2))
    base_metrics["ori_center_diff_deg"] = float(ori_center_diff_deg)
    base_metrics["ori_centers_force_ortho"] = bool(ori_centers_force_ortho)
    base_metrics["ori_centers_ortho_max_dev_deg"] = float(ori_centers_ortho_max_dev_deg)
    base_metrics["ori_center_forced_orth"] = bool(ori_center_forced_orth)
    base_metrics["ori_legacy_hv_split"] = bool(legacy_ori_split)
    base_metrics["ori_band_deg"] = float(band_deg)
    base_metrics["ori_band_soft_deg"] = float(band_soft_deg)
    base_metrics["ori_soft_minlen_frac"] = float(soft_minlen_frac)
    base_metrics["ori_soft_minlen_max_px"] = float(soft_minlen_max_px)
    base_metrics["ori_assign_nearest"] = bool(ori_assign_nearest)
    base_metrics["ori_enable_soft_assign"] = bool(ori_enable_soft_assign)
    base_metrics["ori_soft_keep_boundary"] = bool(ori_soft_keep_boundary)
    base_metrics["ori_boundary_fallback"] = bool(ori_boundary_fallback)
    base_metrics["ori_boundary_fallback_max_angle_deg"] = float(ori_boundary_fallback_max_angle_deg)
    base_metrics["ori_force_nearest_remaining"] = bool(ori_force_nearest_remaining)
    base_metrics["ori_force_nearest_max_add"] = int(ori_force_nearest_max_add)
    base_metrics["ori_force_nearest_minlen_frac"] = float(ori_force_nearest_minlen_frac)
    base_metrics["ori_force_nearest_max_angle_deg"] = float(ori_force_nearest_max_angle_deg)
    base_metrics["ori_nearest_added"] = int(ori_nearest_added)
    base_metrics["ori_soft_added"] = int(ori_soft_added)
    base_metrics["ori_boundary_fallback_added"] = int(ori_boundary_fallback_added)
    base_metrics["ori_force_nearest_added"] = int(ori_force_nearest_added)
    base_metrics["ori_unassigned_initial"] = int(ori_unassigned_initial)
    base_metrics["ori_unassigned_final"] = int(len(unassigned))
    if len(linesA) < 3 or len(linesB) < 3:
        return _fail_result(
            "R_hough_lines",
            maskA=maskA,
            maskB=maskB,
            hough_lines_img=hough_lines_img,
            linepix_mask=linepix_mask,
            linepix_overlay=linepix_overlay,
            ransac_lines_img=ransac_lines_img,
            extra_metrics={
                "num_hough_lines_a": int(len(linesA)),
                "num_hough_lines_b": int(len(linesB)),
            },
        )

    manual_left_lines = None
    manual_bottom_lines = None
    manual_metrics.update(
        {
            "num_selected_left": 0,
            "num_selected_bottom": 0,
            "vp_left_error": None,
            "vp_bottom_error": None,
        }
    )
    if manual_vp_left and manual_vp_bottom and manual_lt and manual_rb:
        vp_left = (float(manual_vp_left.get("x")), float(manual_vp_left.get("y")))
        vp_bottom = (float(manual_vp_bottom.get("x")), float(manual_vp_bottom.get("y")))
        anchor_lt = (float(manual_lt.get("x")), float(manual_lt.get("y")))
        anchor_rt = (float(manual_rt.get("x")), float(manual_rt.get("y"))) if manual_rt else None
        anchor_rb = (float(manual_rb.get("x")), float(manual_rb.get("y")))
        vp_thresh = 140.0
        anchor_thresh = 140.0
        diag = float(math.hypot(frame_bgr.shape[1], frame_bgr.shape[0]))

        def _score_lines(lines):
            out = []
            for seg in lines:
                x1s, y1s, x2s, y2s = seg
                line = _line_from_points((x1s, y1s), (x2s, y2s))
                d_vp_left = _line_point_dist(line, vp_left)
                d_vp_bottom = _line_point_dist(line, vp_bottom)
                d_lt = _line_point_dist(line, anchor_lt)
                d_rt = _line_point_dist(line, anchor_rt) if anchor_rt else float("inf")
                d_rb = _line_point_dist(line, anchor_rb)
                out.append((seg, line, d_vp_left, d_vp_bottom, d_lt, d_rt, d_rb))
            return out

        scoredA = _score_lines(linesA)
        scoredB = _score_lines(linesB)
        median_left_A = np.median([v[2] for v in scoredA]) if scoredA else float("inf")
        median_left_B = np.median([v[2] for v in scoredB]) if scoredB else float("inf")
        use_A_as_left = median_left_A <= median_left_B
        left_scored = scoredA if use_A_as_left else scoredB
        bottom_scored = scoredB if use_A_as_left else scoredA
        left_lines = [
            seg
            for (seg, _, d_vpl, _, dlt, drt, _) in left_scored
            if d_vpl <= vp_thresh and min(dlt, drt) <= anchor_thresh
        ]
        bottom_lines = [
            seg for (seg, _, _, dvp, *_rest, drb) in bottom_scored if dvp <= vp_thresh and drb <= anchor_thresh
        ]
        if not left_lines and left_scored:
            left_scored_sorted = sorted(left_scored, key=lambda v: float(v[2] + min(v[4], v[5])))
            left_lines = [seg for (seg, *_rest) in left_scored_sorted[:8]]
            manual_metrics["reject_reason"] = "R_manual_line_filter_loosen_left"
        if not bottom_lines and bottom_scored:
            bottom_scored_sorted = sorted(bottom_scored, key=lambda v: float(v[3] + v[5]))
            bottom_lines = [seg for (seg, *_rest) in bottom_scored_sorted[:10]]
            manual_metrics["reject_reason"] = "R_manual_line_filter_loosen_bottom"
        if left_lines and bottom_lines:
            manual_left_lines = left_lines
            manual_bottom_lines = bottom_lines
            manual_metrics["num_selected_left"] = int(len(left_lines))
            manual_metrics["num_selected_bottom"] = int(len(bottom_lines))
            vp_left_raw = float(np.median([v[2] for v in left_scored])) if left_scored else None
            vp_bottom_raw = float(np.median([v[3] for v in bottom_scored])) if bottom_scored else None
            manual_metrics["vp_left_error_norm"] = (
                float(vp_left_raw / diag) if vp_left_raw is not None else None
            )
            manual_metrics["vp_bottom_error_norm"] = (
                float(vp_bottom_raw / diag) if vp_bottom_raw is not None else None
            )
            manual_metrics["manual_left_group"] = "A" if use_A_as_left else "B"
        else:
            manual_metrics["reject_reason"] = "R_manual_line_filter_empty"
    if manual_metrics:
        base_metrics.update(manual_metrics)
    if len(linesA) > 20:
        pairedA = list(zip(linesA, line_ids_A))
        pairedA.sort(key=lambda ln: math.hypot(ln[0][2] - ln[0][0], ln[0][3] - ln[0][1]), reverse=True)
        pairedA = pairedA[:20]
        linesA = [p[0] for p in pairedA]
        line_ids_A = [p[1] for p in pairedA]
    if len(linesB) > 20:
        pairedB = list(zip(linesB, line_ids_B))
        pairedB.sort(key=lambda ln: math.hypot(ln[0][2] - ln[0][0], ln[0][3] - ln[0][1]), reverse=True)
        pairedB = pairedB[:20]
        linesB = [p[0] for p in pairedB]
        line_ids_B = [p[1] for p in pairedB]

    lines_left = manual_left_lines if manual_left_lines else linesA
    lines_bottom = manual_bottom_lines if manual_bottom_lines else linesB
    line_ids_left = [None] * len(manual_left_lines) if manual_left_lines else line_ids_A
    line_ids_bottom = [None] * len(manual_bottom_lines) if manual_bottom_lines else line_ids_B

    def _min_point_line_dist(point_xy: Optional[Tuple[float, float]],
                             segments: list[Tuple[float, float, float, float]]) -> Optional[float]:
        if point_xy is None or not segments:
            return None
        px, py = float(point_xy[0]), float(point_xy[1])
        best = None
        for x1, y1, x2, y2 in segments:
            line = _line_from_points((x1, y1), (x2, y2))
            dist = abs(_line_signed_distance(line, (px, py)))
            if best is None or dist < best:
                best = dist
        return float(best) if best is not None else None

    if manual_lt and manual_rt and manual_rb:
        base_metrics["min_dist_line_to_LT"] = _min_point_line_dist(
            (manual_lt.get("x"), manual_lt.get("y")), lines_left
        )
        base_metrics["min_dist_line_to_RT"] = _min_point_line_dist(
            (manual_rt.get("x"), manual_rt.get("y")), lines_left
        )
        base_metrics["min_dist_line_to_RB_bottom"] = _min_point_line_dist(
            (manual_rb.get("x"), manual_rb.get("y")), lines_bottom
        )
        base_metrics["min_dist_line_to_RB_right"] = _min_point_line_dist(
            (manual_rb.get("x"), manual_rb.get("y")), lines_left
        )
    manual_left_group = manual_metrics.get("manual_left_group")
    if manual_left_group == "B":
        mask_left = maskB
        mask_bottom = maskA
        center_left = center2
    else:
        mask_left = maskA
        mask_bottom = maskB
        center_left = center1

    # --- FIX: left/right boundaries are NOT parallel under perspective ---
    # Prefer VP-based pairing to avoid selecting two nearly-parallel left lines.
    pairsA = []
    if not manual_left_lines:
        pairsA = candidate_outer_pairs_vp(
            lines_left,
            mask_left,
            floor_bbox,
            (h, w),
            top_k=16,
            max_pairs=6,
        )

    # Fallback to angle-based pairing (kept for safety / manual mode)
    if not pairsA:
        pairsA = candidate_outer_pairs(
            lines_left,
            mask_left,
            angle_ref=center_left,
            top_k=12 if manual_left_lines else 12,
            max_pairs=6,
            max_angle_deg=90.0 if manual_left_lines else 25.0,
        )
    if manual_bottom_lines:
        center_bottom = center2 if center_left == center1 else center1
        pairsB = candidate_outer_pairs(
            lines_bottom,
            mask_bottom,
            angle_ref=center_bottom,
            top_k=12,
            max_pairs=6,
            max_angle_deg=40.0,
        )
    else:
        pairsB = candidate_outer_pairs_vp(lines_bottom, mask_bottom, floor_bbox, (h, w), top_k=16, max_pairs=6)
    if not pairsB:
        center_bottom = center2 if center_left == center1 else center1
        pairsB = candidate_outer_pairs(
            lines_bottom,
            mask_bottom,
            angle_ref=center_bottom,
            top_k=10 if manual_bottom_lines else 6,
            max_pairs=3,
            max_angle_deg=90.0 if manual_bottom_lines else 15.0,
        )
    # Force extreme boundary combinations to be considered
    if len(lines_left) >= 2:
        # Pick the two most separated parallels by signed distance to the floor bbox center.
        # (Using min(x)/min(y) is unstable under perspective and for diagonal stray segments.)
        cx = 0.5 * (float(floor_bbox[0]) + float(floor_bbox[2]))
        cy = 0.5 * (float(floor_bbox[1]) + float(floor_bbox[3]))
        scored = []
        for x1s, y1s, x2s, y2s in lines_left:
            line = _line_from_points((x1s, y1s), (x2s, y2s))
            if "_canonicalize_line_abc" in globals():
                try:
                    line = _canonicalize_line_abc(*line)
                except Exception:
                    pass
            d = _line_signed_distance(line, (cx, cy))
            scored.append((d, line))
        scored.sort(key=lambda t: t[0])
        line_min = scored[0][1]
        line_max = scored[-1][1]
        if line_min != line_max:
            pairsA.append(((line_min, line_max), 1e6))
    if len(lines_bottom) >= 2:
        # Same idea for the orthogonal direction.
        cx = 0.5 * (float(floor_bbox[0]) + float(floor_bbox[2]))
        cy = 0.5 * (float(floor_bbox[1]) + float(floor_bbox[3]))
        scored = []
        for x1s, y1s, x2s, y2s in lines_bottom:
            line = _line_from_points((x1s, y1s), (x2s, y2s))
            if "_canonicalize_line_abc" in globals():
                try:
                    line = _canonicalize_line_abc(*line)
                except Exception:
                    pass
            d = _line_signed_distance(line, (cx, cy))
            scored.append((d, line))
        scored.sort(key=lambda t: t[0])
        line_min = scored[0][1]
        line_max = scored[-1][1]
        if line_min != line_max:
            pairsB.append(((line_min, line_max), 1e6))
    if manual_lt and manual_rt and manual_rb and manual_left_lines and manual_bottom_lines:
        def _seg_line_and_dist(segments, pt):
            px, py = float(pt[0]), float(pt[1])
            scored = []
            for x1s, y1s, x2s, y2s in segments:
                line = _line_from_points((x1s, y1s), (x2s, y2s))
                dist = abs(_line_signed_distance(line, (px, py)))
                scored.append((dist, line))
            return scored

        scored_lt = sorted(_seg_line_and_dist(lines_left, (manual_lt["x"], manual_lt["y"])), key=lambda v: v[0])
        scored_rt = sorted(_seg_line_and_dist(lines_left, (manual_rt["x"], manual_rt["y"])), key=lambda v: v[0])
        if scored_lt and scored_rt:
            line_lt = scored_lt[0][1]
            line_rt = next((ln for _, ln in scored_rt if ln != line_lt), None)
            if line_rt is not None:
                pairsA.append(((line_lt, line_rt), 1e6))

        scored_rb = _seg_line_and_dist(lines_bottom, (manual_rb["x"], manual_rb["y"]))
        if scored_rb:
            scored_rb.sort(key=lambda v: v[0])
            line_rb = scored_rb[0][1]
            line_far = max(scored_rb, key=lambda v: v[0])[1]
            if line_far != line_rb:
                pairsB.append(((line_rb, line_far), 1e6))
    if manual_lb and manual_rb and manual_rt and manual_lt:
        left_line = _line_from_points((manual_lt["x"], manual_lt["y"]), (manual_lb["x"], manual_lb["y"]))
        right_line = _line_from_points((manual_rt["x"], manual_rt["y"]), (manual_rb["x"], manual_rb["y"]))
        bottom_line = _line_from_points((manual_lb["x"], manual_lb["y"]), (manual_rb["x"], manual_rb["y"]))
        top_line = _line_from_points((manual_lt["x"], manual_lt["y"]), (manual_rt["x"], manual_rt["y"]))
        pairsA.append(((left_line, right_line), 1e6))
        pairsB.append(((bottom_line, top_line), 1e6))
        base_metrics["manual_pairs_added"] = True
    base_metrics["num_hough_pairs_a"] = int(len(pairsA))
    base_metrics["num_hough_pairs_b"] = int(len(pairsB))
    if not pairsA or not pairsB:
        hough_lines_img = frame_bgr.copy()
        for x1, y1, x2, y2 in linesA:
            cv2.line(hough_lines_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        for x1, y1, x2, y2 in linesB:
            cv2.line(hough_lines_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
        return _fail_result(
            "R_hough_pair",
            maskA=maskA,
            maskB=maskB,
            hough_lines_img=hough_lines_img,
            linepix_mask=linepix_mask,
            linepix_overlay=linepix_overlay,
            ransac_lines_img=ransac_lines_img,
            extra_metrics={
                "num_hough_lines_a": int(len(linesA)),
                "num_hough_lines_b": int(len(linesB)),
                "num_hough_pairs_a": int(len(pairsA)),
                "num_hough_pairs_b": int(len(pairsB)),
            },
        )
    best_pairA = None
    best_pairB = None
    best_ordered = None
    best_area_ratio = None
    best_inside_ratio = None
    best_score = float("-inf")
    best_score_base = None
    best_inside_count: Optional[int] = None
    best_inside_bonus: Optional[float] = None
    reject_aspect_ratio = 0
    reject_roi_corner = 0
    h, w = white_mask_clean.shape[:2]
    diag_img = float(max(1.0, math.hypot(float(w), float(h))))
    margin = 20
    img_bbox = (margin, margin, int(w - 1 - margin), int(h - 1 - margin))
    bbox_w = max(1, int(x1 - x0))
    bbox_h = max(1, int(y1 - y0))
    corner_margin_x = max(24, int(round(0.05 * float(bbox_w))))
    corner_margin_y = max(24, int(round(0.05 * float(bbox_h))))
    reject_manual_anchor = 0
    best_anchor_score = None
    best_mixer_parts: Optional[Dict[str, float]] = None
    best_area_ratio_img: Optional[float] = None
    best_ymax_ratio: Optional[float] = None
    best_top_edge_floor_ratio: Optional[float] = None
    best_bottom_support: Optional[float] = None
    reject_area_ratio_low = 0
    reject_area_ratio_high = 0
    reject_ymax_ratio_low = 0
    reject_degenerate_bbox = 0
    reject_degenerate_min_edge = 0
    reject_corner_oob = 0
    reject_bottom_span_floor_ratio = 0
    reject_top_edge_ratio_high = 0
    top3_main: list[Tuple[float, float, float, float]] = []
    relaxed_pass = 0
    vp_a = _estimate_vanishing_point(
        [_line_from_points((x1, y1), (x2, y2)) for x1, y1, x2, y2 in linesA], rng
    )
    vp_b = _estimate_vanishing_point(
        [_line_from_points((x1, y1), (x2, y2)) for x1, y1, x2, y2 in linesB], rng
    )
    role_prior_enabled = bool(_env_flag("BADC_ROLE_PRIOR_ENABLE", False))
    role_prior_mask_mode = _env_str("BADC_ROLE_PRIOR_MASK", "linepix").strip().lower()
    role_prior_mask = linepix_mask if role_prior_mask_mode == "linepix" else white_mask_clean
    if role_prior_mask is None or role_prior_mask.size == 0 or int(np.count_nonzero(role_prior_mask)) == 0:
        role_prior_mask = white_mask_clean
        role_prior_mask_mode = "white_mask_clean"
    base_metrics["role_prior_enabled"] = bool(role_prior_enabled)
    base_metrics["role_prior_mask"] = str(role_prior_mask_mode)
    qrt_min_bbox_w_frac = float(_env_float("BADC_QRT_MIN_BBOX_W_FRAC", 0.08))
    qrt_min_bbox_h_frac = float(_env_float("BADC_QRT_MIN_BBOX_H_FRAC", 0.06))
    qrt_min_edge_frac = float(_env_float("BADC_QRT_MIN_EDGE_FRAC", 0.05))
    ratio_verify_enabled = bool(_env_flag("BADC_COURT_RATIO_VERIFY", False))
    ratio_verify_hard_gate = bool(_env_flag("BADC_COURT_RATIO_VERIFY_HARD_GATE", False))
    ratio_verify_weight = float(_env_float("BADC_COURT_RATIO_VERIFY_W", 10.0))
    ratio_verify_min_score = float(_env_float("BADC_COURT_RATIO_VERIFY_MIN_SCORE", 0.56))
    ratio_verify_min_h = int(_env_int("BADC_COURT_RATIO_MIN_H_LINES", 4))
    ratio_verify_min_v = int(_env_int("BADC_COURT_RATIO_MIN_V_LINES", 3))
    ratio_verify_angle_tol = float(_env_float("BADC_COURT_RATIO_ANGLE_TOL_DEG", 12.0))
    ratio_verify_tol_h = float(_env_float("BADC_COURT_RATIO_TOL_H_M", 0.24))
    ratio_verify_tol_v = float(_env_float("BADC_COURT_RATIO_TOL_V_M", 0.22))
    ratio_verify_min_seg = float(_env_float("BADC_COURT_RATIO_MIN_SEG_M", 0.55))
    ratio_verify_expand = bool(_env_flag("BADC_COURT_RATIO_VERIFY_EXPAND", True))
    ratio_verify_expand_scales = _parse_float_csv(
        _env_str("BADC_COURT_RATIO_VERIFY_SCALES", "1.00,1.03,0.97,1.06,0.94"),
        [1.0, 1.03, 0.97, 1.06, 0.94],
    )
    ratio_verify_expand_min_gain = float(_env_float("BADC_COURT_RATIO_VERIFY_EXPAND_MIN_GAIN", 0.03))
    base_metrics["court_ratio_verify_enabled"] = bool(ratio_verify_enabled)
    base_metrics["court_ratio_verify_hard_gate"] = bool(ratio_verify_hard_gate)
    base_metrics["court_ratio_verify_weight"] = float(ratio_verify_weight)
    base_metrics["court_ratio_verify_min_score"] = float(ratio_verify_min_score)
    base_metrics["court_ratio_verify_min_h_lines"] = int(ratio_verify_min_h)
    base_metrics["court_ratio_verify_min_v_lines"] = int(ratio_verify_min_v)
    base_metrics["court_ratio_verify_expand"] = bool(ratio_verify_expand)
    base_metrics["court_ratio_verify_expand_scales"] = [float(v) for v in ratio_verify_expand_scales]
    best_ratio_meta: Optional[Dict[str, Any]] = None
    best_ratio_bonus: Optional[float] = None
    ratio_refine_used = False
    semantic_rerank_used = False
    semantic_rerank_switched = False
    semantic_rerank_meta: Optional[Dict[str, Any]] = None
    min_bottom_span_floor_ratio_hard = float(
        _env_float("BADC_QRT_MIN_BOTTOM_SPAN_FLOOR_RATIO", _env_float("BADC_FIT_MIN_BOTTOM_SPAN_FLOOR_RATIO", 0.45))
    )
    for relax_scale in (1.0, 0.7):
        if relax_scale < 1.0:
            relaxed_pass = 1
        min_bottom_span_floor_ratio_loop = float(max(0.28, min_bottom_span_floor_ratio_hard * float(relax_scale)))
        best_pairA = None
        best_pairB = None
        best_ordered = None
        best_area_ratio = None
        best_inside_ratio = None
        best_score = float("-inf")
        best_mixer_parts = None
        best_area_ratio_img = None
        best_ymax_ratio = None
        best_top_edge_floor_ratio = None
        best_bottom_support = None
        best_role_prior_bonus = None
        best_role_prior_meta = None
        best_ratio_meta = None
        best_ratio_bonus = None
        candidate_pool: list[Dict[str, Any]] = []
        top3_main = []
        deg_bbox_w = float(max(1.0, qrt_min_bbox_w_frac * float(w) * float(relax_scale)))
        deg_bbox_h = float(max(1.0, qrt_min_bbox_h_frac * float(h) * float(relax_scale)))
        deg_edge = float(max(1.0, qrt_min_edge_frac * float(min(h, w)) * float(relax_scale)))
        min_area_ratio_img = float(os.environ.get("BADC_QRT_MIN_AREA_RATIO_IMG", "0.04"))
        aspect_min = float(os.environ.get("BADC_QRT_ASPECT_MIN", "1.4"))
        aspect_max = float(os.environ.get("BADC_QRT_ASPECT_MAX", "3.0"))
        max_corner_oob_px = float(
            _env_float("BADC_QRT_MAX_CORNER_OOB_PX", max(160.0, 0.26 * float(max(h, w))))
        )
        corner_oob_soft_k = float(_env_float("BADC_QRT_CORNER_OOB_SOFT_K", 9.0))
        floor_oob_soft_k = float(_env_float("BADC_QRT_FLOOR_OOB_SOFT_K", 16.0))
        roi_corner_soft_bias = float(_env_float("BADC_QRT_ROI_CORNER_SOFT_BIAS", 2.0))
        max_area_ratio_floor = float(_env_float("BADC_QRT_MAX_AREA_RATIO_FLOOR", 0.45))
        max_top_edge_floor_ratio = float(_env_float("BADC_QRT_MAX_TOP_EDGE_FLOOR_RATIO", 0.58))
        top_edge_soft_ref = float(_env_float("BADC_QRT_TOP_EDGE_SOFT_REF", 0.22))
        top_edge_soft_w = float(_env_float("BADC_QRT_TOP_EDGE_SOFT_W", 18.0))
        for pairA, _scoreA in pairsA:
            for pairB, _scoreB in pairsB:
                quad = _intersections_from_pairs(pairA, pairB)
                if quad is None:
                    continue
                ordered = _order_corners_lb_rb_rt_lt(quad)
                if not _is_convex_quad(ordered):
                    continue
                if _quad_area(ordered) < 1.0:
                    continue
                bottom_span_floor_ratio = _bottom_span_floor_ratio(ordered, floor_bbox)
                if bottom_span_floor_ratio < min_bottom_span_floor_ratio_loop:
                    reject_bottom_span_floor_ratio += 1
                    continue
                ok, _gate_reason, _gate_metrics = _passes_geom_gates(ordered, floor_bbox)
                if not ok:
                    continue
                x = ordered[:, 0]
                y = ordered[:, 1]
                oob_l = np.maximum(0.0, -x)
                oob_r = np.maximum(0.0, x - float(max(0, w - 1)))
                oob_t = np.maximum(0.0, -y)
                oob_b = np.maximum(0.0, y - float(max(0, h - 1)))
                corner_oob_each = oob_l + oob_r + oob_t + oob_b
                corner_oob_sum = float(np.sum(corner_oob_each))
                if np.any(
                    corner_oob_each > float(max_corner_oob_px)
                ):
                    reject_corner_oob += 1
                    continue
                roi_corner_ok = _corners_inside_floor_roi(
                    ordered,
                    floor_bbox,
                    margin_x=float(corner_margin_x),
                    margin_y=float(corner_margin_y),
                )
                if not roi_corner_ok:
                    reject_roi_corner += 1
                x0f, y0f, x1f, y1f = [float(v) for v in floor_bbox]
                floor_oob_l = np.maximum(0.0, (x0f - float(corner_margin_x)) - x)
                floor_oob_r = np.maximum(0.0, x - (x1f + float(corner_margin_x)))
                floor_oob_t = np.maximum(0.0, (y0f - float(corner_margin_y)) - y)
                floor_oob_b = np.maximum(0.0, y - (y1f + float(corner_margin_y)))
                floor_oob_sum = float(np.sum(floor_oob_l + floor_oob_r + floor_oob_t + floor_oob_b))
                bbox_w = float(np.max(ordered[:, 0]) - np.min(ordered[:, 0]))
                bbox_h = float(np.max(ordered[:, 1]) - np.min(ordered[:, 1]))
                min_edge = float(np.min(_quad_edges(ordered)))
                if bbox_w < deg_bbox_w or bbox_h < deg_bbox_h:
                    reject_degenerate_bbox += 1
                    continue
                if min_edge < deg_edge:
                    reject_degenerate_min_edge += 1
                    continue
                quad_clip = ordered.copy()
                quad_clip[:, 0] = np.clip(quad_clip[:, 0], 0.0, float(w - 1))
                quad_clip[:, 1] = np.clip(quad_clip[:, 1], 0.0, float(h - 1))
                quad_area = float(_quad_area(quad_clip))
                area_ratio_img = quad_area / float(max(h * w, 1.0))
                if area_ratio_img < float(min_area_ratio_img):
                    reject_area_ratio_low += 1
                    continue
                w1 = float(np.linalg.norm(ordered[1] - ordered[0]))
                w2 = float(np.linalg.norm(ordered[2] - ordered[3]))
                h1 = float(np.linalg.norm(ordered[3] - ordered[0]))
                h2 = float(np.linalg.norm(ordered[2] - ordered[1]))
                mean_w = 0.5 * (w1 + w2)
                mean_h = 0.5 * (h1 + h2)
                aspect_ratio = max(mean_w, mean_h) / max(min(mean_w, mean_h), 1e-6)
                if aspect_ratio < float(aspect_min) or aspect_ratio > float(aspect_max):
                    reject_aspect_ratio += 1
                    continue
                y_vals = ordered[:, 1]
                ymax_ratio = max(0.0, min(1.0, float(np.max(y_vals)) / float(max(h - 1, 1))))
                ymean_ratio = max(0.0, min(1.0, float(np.mean(y_vals)) / float(max(h - 1, 1))))
                top_edge_floor_ratio = _top_edge_floor_ratio(ordered, floor_bbox)
                if ymax_ratio < 0.70:
                    reject_ymax_ratio_low += 1
                if top_edge_floor_ratio > max_top_edge_floor_ratio:
                    reject_top_edge_ratio_high += 1
                    continue
                H_cand = cv2.getPerspectiveTransform(get_bwf_corners().astype(np.float32), ordered.astype(np.float32))
                if not np.all(np.isfinite(H_cand)):
                    continue
                loss_info = _compute_loss_terms(
                    H_cand,
                    dt=dt,
                    Xw=Xw_full,
                    weights=w_full,
                    cover_points=None,
                    floor_bbox=floor_bbox,
                    tau_px=6.0,
                    dt_oob=dt_oob,
                )
                area_ratio = float(loss_info.get("area_ratio", 0.0))
                if area_ratio > max_area_ratio_floor:
                    reject_area_ratio_high += 1
                    continue
                score, mixer_parts = _mixer_score_candidate(
                    ordered,
                    loss_info,
                    floor_bbox,
                    vp_long=vp_a,
                    vp_short=vp_b,
                )
                bottom_support = _bottom_support_from_loss(loss_info, ymax_ratio, h, tau_px=6.0)
                top_edge_soft_penalty = float(
                    max(0.0, float(top_edge_floor_ratio) - float(top_edge_soft_ref)) * float(top_edge_soft_w)
                )
                score_final = float(
                    score
                    + 6.0 * math.log(max(area_ratio_img, 1e-6))
                    + 3.0 * ymean_ratio
                    + 2.0 * ymax_ratio
                    + 4.0 * bottom_support
                    - top_edge_soft_penalty
                )
                inside_count_val, _inside_flags = _count_points_inside_quad(ordered, foot_points)
                inside_bonus = 10.0 * float(inside_count_val)
                if manual_lt and manual_rt and manual_rb:
                    err_map = _corner_errors_to_manual(ordered, manual.get("court_corners") if manual else None)
                    err_lt = err_map.get("err_LT")
                    err_rt = err_map.get("err_RT")
                    err_rb = err_map.get("err_RB")
                    if (err_rt is not None and err_rt > 300.0) or (err_rb is not None and err_rb > 300.0):
                        reject_manual_anchor += 1
                        continue
                    err_vals = [v for v in [err_lt, err_rt, err_rb] if v is not None]
                    err_mean = float(np.mean(err_vals)) if err_vals else 999.0
                    anchor_score = float(math.exp(-err_mean / 200.0))
                    score_final *= (0.8 + 0.2 * anchor_score)
                    best_anchor_score = anchor_score
                area_w = _env_float("BADC_QUAD_AREA_PRIOR_W", 2.5)
                area_min = _env_float("BADC_QUAD_AREA_PRIOR_MIN", 0.10)
                area_max = _env_float("BADC_QUAD_AREA_PRIOR_MAX", 0.42)
                area_high_penalty_mult = _env_float("BADC_QUAD_AREA_PRIOR_HIGH_PENALTY_MULT", 8.0)
                area_bonus = area_w * min(area_ratio, area_max)
                if area_ratio < area_min:
                    area_bonus -= area_w * 3.0 * (area_min - area_ratio)
                if area_ratio > area_max:
                    area_bonus -= area_w * area_high_penalty_mult * (area_ratio - area_max)
                role_bonus = 0.0
                role_meta: Dict[str, float] = {}
                if role_prior_enabled:
                    role_bonus, role_meta = _court_role_prior_bonus(H_cand, role_prior_mask)
                ratio_meta: Optional[Dict[str, Any]] = None
                ratio_bonus = 0.0
                ratio_score_val = 0.0
                if ratio_verify_enabled:
                    ratio_meta = _evaluate_court_ratio_structure(
                        ordered,
                        lines_all,
                        img_w=w,
                        img_h=h,
                        angle_tol_deg=ratio_verify_angle_tol,
                        tol_h_m=ratio_verify_tol_h,
                        tol_v_m=ratio_verify_tol_v,
                        min_seg_len_m=ratio_verify_min_seg,
                    )
                    ratio_score_val = float(ratio_meta.get("score", 0.0))
                    ratio_h_ok = int(ratio_meta.get("h_present", 0)) >= int(max(1, ratio_verify_min_h))
                    ratio_v_ok = int(ratio_meta.get("v_present", 0)) >= int(max(1, ratio_verify_min_v))
                    ratio_pass = bool(ratio_score_val >= ratio_verify_min_score and ratio_h_ok and ratio_v_ok)
                    ratio_bonus = float(ratio_verify_weight * ratio_score_val)
                    ratio_meta = dict(ratio_meta)
                    ratio_meta["pass"] = bool(ratio_pass)
                    ratio_meta["bonus"] = float(ratio_bonus)
                    if ratio_verify_hard_gate and not ratio_pass:
                        continue
                geom_soft_penalty = float(
                    corner_oob_soft_k * (corner_oob_sum / diag_img)
                    + floor_oob_soft_k * (floor_oob_sum / diag_img)
                    + (roi_corner_soft_bias if not roi_corner_ok else 0.0)
                )
                score_total = float(
                    score_final + inside_bonus + area_bonus + float(role_bonus) + float(ratio_bonus) - geom_soft_penalty
                )
                candidate_pool.append(
                    {
                        "pairA": pairA,
                        "pairB": pairB,
                        "ordered": np.asarray(ordered, dtype=np.float32).copy(),
                        "score_total": float(score_total),
                        "score_base": float(score_final),
                        "area_ratio_floor": float(loss_info.get("area_ratio", 0.0)),
                        "inside_ratio": float(_points_inside_ratio(ordered, img_bbox)),
                        "area_ratio_img": float(area_ratio_img),
                        "ymax_ratio": float(ymax_ratio),
                        "top_edge_floor_ratio": float(top_edge_floor_ratio),
                        "bottom_support": float(bottom_support),
                        "inside_count": int(inside_count_val),
                        "inside_bonus": float(inside_bonus),
                        "role_bonus": float(role_bonus),
                        "role_meta": dict(role_meta) if role_meta else None,
                        "ratio_bonus": float(ratio_bonus),
                        "ratio_meta": dict(ratio_meta) if isinstance(ratio_meta, dict) else None,
                        "mixer_parts": dict(mixer_parts) if isinstance(mixer_parts, dict) else None,
                        "top_edge_soft_penalty": float(top_edge_soft_penalty),
                        "geom_soft_penalty": float(geom_soft_penalty),
                        "corner_oob_sum": float(corner_oob_sum),
                        "floor_oob_sum": float(floor_oob_sum),
                        "roi_corner_ok": bool(roi_corner_ok),
                    }
                )
                if len(candidate_pool) > 32:
                    candidate_pool.sort(key=lambda it: float(it.get("score_total", -1e9)), reverse=True)
                    candidate_pool = candidate_pool[:32]
                if score_total > best_score:
                    best_score = score_total
                    best_score_base = float(score_final)
                    best_pairA = pairA
                    best_pairB = pairB
                    best_ordered = ordered
                    best_area_ratio = float(loss_info.get("area_ratio", 0.0))
                    best_inside_ratio = _points_inside_ratio(ordered, img_bbox)
                    best_mixer_parts = mixer_parts
                    best_area_ratio_img = float(area_ratio_img)
                    best_ymax_ratio = float(ymax_ratio)
                    best_top_edge_floor_ratio = float(top_edge_floor_ratio)
                    best_bottom_support = float(bottom_support)
                    best_inside_count = int(inside_count_val)
                    best_inside_bonus = float(inside_bonus)
                    best_role_prior_bonus = float(role_bonus)
                    best_role_prior_meta = dict(role_meta) if role_meta else None
                    best_ratio_bonus = float(ratio_bonus)
                    best_ratio_meta = dict(ratio_meta) if isinstance(ratio_meta, dict) else None
                    base_metrics["best_geom_soft_penalty"] = float(geom_soft_penalty)
                    base_metrics["best_corner_oob_sum"] = float(corner_oob_sum)
                    base_metrics["best_floor_oob_sum"] = float(floor_oob_sum)
                    base_metrics["best_roi_corner_ok"] = bool(roi_corner_ok)
                top3_main.append(
                    (score_total, area_ratio_img, ymax_ratio, bottom_support, int(inside_count_val), float(ratio_score_val))
                )
        if best_pairA is not None and best_pairB is not None and best_ordered is not None:
            break

    if (
        ratio_verify_enabled
        and ratio_verify_expand
        and best_ordered is not None
        and isinstance(best_ratio_meta, dict)
        and not bool(best_ratio_meta.get("pass", False))
    ):
        min_area_ratio_img_ref = float(os.environ.get("BADC_QRT_MIN_AREA_RATIO_IMG", "0.04"))
        aspect_min_ref = float(os.environ.get("BADC_QRT_ASPECT_MIN", "1.4"))
        aspect_max_ref = float(os.environ.get("BADC_QRT_ASPECT_MAX", "3.0"))
        min_bbox_w_ref = float(max(1.0, qrt_min_bbox_w_frac * float(w)))
        min_bbox_h_ref = float(max(1.0, qrt_min_bbox_h_frac * float(h)))
        min_edge_ref = float(max(1.0, qrt_min_edge_frac * float(min(h, w))))
        refined_quad, refined_meta, refined = _refine_quad_by_structure_scale(
            best_ordered,
            lines_all,
            floor_bbox=floor_bbox,
            img_w=w,
            img_h=h,
            scales=ratio_verify_expand_scales,
            min_gain=ratio_verify_expand_min_gain,
            angle_tol_deg=ratio_verify_angle_tol,
            tol_h_m=ratio_verify_tol_h,
            tol_v_m=ratio_verify_tol_v,
            min_seg_len_m=ratio_verify_min_seg,
            min_area_ratio_img=min_area_ratio_img_ref,
            aspect_min=aspect_min_ref,
            aspect_max=aspect_max_ref,
            min_bbox_w=min_bbox_w_ref,
            min_bbox_h=min_bbox_h_ref,
            min_edge=min_edge_ref,
        )
        if refined and float(refined_meta.get("score", 0.0)) > float(best_ratio_meta.get("score", 0.0)):
            best_ordered = refined_quad
            best_ratio_meta = dict(refined_meta)
            best_ratio_bonus = float(ratio_verify_weight * float(best_ratio_meta.get("score", 0.0)))
            ratio_refine_used = True

    semantic_rerank_enable = bool(_env_flag("BADC_SEMANTIC_RERANK_ENABLE", True))
    semantic_rerank_hard_only = bool(_env_flag("BADC_SEMANTIC_RERANK_HARD_ONLY", True))
    semantic_rerank_topk = int(max(1, _env_int("BADC_SEMANTIC_RERANK_TOPK", 14)))
    semantic_rerank_weight = float(_env_float("BADC_SEMANTIC_RERANK_W", 2.2))
    if semantic_rerank_enable and best_ordered is not None and candidate_pool:
        semantic_mask_rank = (
            linepix_mask
            if isinstance(linepix_mask, np.ndarray) and int(np.count_nonzero(linepix_mask)) > 0
            else white_mask_clean
        )
        ranked_candidates = sorted(candidate_pool, key=lambda it: float(it.get("score_total", -1e9)), reverse=True)[
            :semantic_rerank_topk
        ]
        rank_rows: list[Dict[str, Any]] = []
        selected_cand: Optional[Dict[str, Any]] = None
        selected_sem: Optional[Dict[str, Any]] = None
        selected_total = float("-inf")
        for ridx, cand in enumerate(ranked_candidates, start=1):
            q = np.asarray(cand.get("ordered"), dtype=np.float32).reshape(4, 2)
            H_cand = cv2.getPerspectiveTransform(get_bwf_corners().astype(np.float32), q.astype(np.float32))
            _, sem_meta = _semantic_refine_homography(H_cand, semantic_mask_rank)
            sem_score = float(sem_meta.get("final_score", sem_meta.get("score", -1e9)))
            hard_ok = bool(sem_meta.get("hard_pass", True))
            total_rank = float(cand.get("score_total", -1e9)) + float(semantic_rerank_weight) * sem_score
            rank_rows.append(
                {
                    "rank": int(ridx),
                    "base_score": float(cand.get("score_total", -1e9)),
                    "semantic_score": float(sem_score),
                    "rank_score": float(total_rank),
                    "hard_pass": bool(hard_ok),
                    "hard_reason": sem_meta.get("hard_reason"),
                }
            )
            if semantic_rerank_hard_only and not hard_ok:
                continue
            if total_rank > selected_total:
                selected_total = total_rank
                selected_cand = cand
                selected_sem = sem_meta
        if selected_cand is None and ranked_candidates:
            for cand, row in zip(ranked_candidates, rank_rows):
                total_rank = float(row.get("rank_score", -1e9))
                if total_rank > selected_total:
                    selected_total = total_rank
                    selected_cand = cand
                    selected_sem = None

        if selected_cand is not None:
            semantic_rerank_used = True
            prev_ordered = np.asarray(best_ordered, dtype=np.float32).reshape(4, 2)
            new_ordered = np.asarray(selected_cand.get("ordered"), dtype=np.float32).reshape(4, 2)
            switched = not np.allclose(prev_ordered, new_ordered, atol=1.0)
            semantic_rerank_switched = bool(switched)
            if switched:
                best_pairA = selected_cand.get("pairA")
                best_pairB = selected_cand.get("pairB")
                best_ordered = new_ordered
                best_score = float(selected_cand.get("score_total", best_score))
                best_score_base = float(selected_cand.get("score_base", best_score_base or 0.0))
                best_area_ratio = float(selected_cand.get("area_ratio_floor", best_area_ratio or 0.0))
                best_inside_ratio = float(selected_cand.get("inside_ratio", best_inside_ratio or 0.0))
                best_mixer_parts = (
                    dict(selected_cand.get("mixer_parts"))
                    if isinstance(selected_cand.get("mixer_parts"), dict)
                    else best_mixer_parts
                )
                best_area_ratio_img = float(selected_cand.get("area_ratio_img", best_area_ratio_img or 0.0))
                best_ymax_ratio = float(selected_cand.get("ymax_ratio", best_ymax_ratio or 0.0))
                best_top_edge_floor_ratio = float(
                    selected_cand.get("top_edge_floor_ratio", best_top_edge_floor_ratio or 0.0)
                )
                best_bottom_support = float(selected_cand.get("bottom_support", best_bottom_support or 0.0))
                best_inside_count = int(selected_cand.get("inside_count", best_inside_count or 0))
                best_inside_bonus = float(selected_cand.get("inside_bonus", best_inside_bonus or 0.0))
                best_role_prior_bonus = float(selected_cand.get("role_bonus", best_role_prior_bonus or 0.0))
                best_role_prior_meta = (
                    dict(selected_cand.get("role_meta"))
                    if isinstance(selected_cand.get("role_meta"), dict)
                    else best_role_prior_meta
                )
                best_ratio_bonus = float(selected_cand.get("ratio_bonus", best_ratio_bonus or 0.0))
                best_ratio_meta = (
                    dict(selected_cand.get("ratio_meta"))
                    if isinstance(selected_cand.get("ratio_meta"), dict)
                    else best_ratio_meta
                )
                base_metrics["best_geom_soft_penalty"] = float(selected_cand.get("geom_soft_penalty", 0.0))
                base_metrics["best_corner_oob_sum"] = float(selected_cand.get("corner_oob_sum", 0.0))
                base_metrics["best_floor_oob_sum"] = float(selected_cand.get("floor_oob_sum", 0.0))
                base_metrics["best_roi_corner_ok"] = bool(selected_cand.get("roi_corner_ok", False))

            semantic_rerank_meta = {
                "enabled": True,
                "used": bool(semantic_rerank_used),
                "switched": bool(semantic_rerank_switched),
                "hard_only": bool(semantic_rerank_hard_only),
                "topk": int(semantic_rerank_topk),
                "weight": float(semantic_rerank_weight),
                "rows": rank_rows,
                "selected_rank_score": float(selected_total),
                "selected_hard_pass": bool((selected_sem or {}).get("hard_pass", True)),
                "selected_hard_reason": (selected_sem or {}).get("hard_reason"),
            }

    base_metrics["reject_roi_corner"] = int(reject_roi_corner)
    base_metrics["reject_manual_anchor"] = int(reject_manual_anchor)
    base_metrics["reject_area_ratio_low"] = int(reject_area_ratio_low)
    base_metrics["reject_area_ratio_high"] = int(reject_area_ratio_high)
    base_metrics["reject_ymax_ratio_low"] = int(reject_ymax_ratio_low)
    base_metrics["reject_degenerate_bbox"] = int(reject_degenerate_bbox)
    base_metrics["reject_degenerate_min_edge"] = int(reject_degenerate_min_edge)
    base_metrics["reject_corner_oob"] = int(reject_corner_oob)
    base_metrics["reject_bottom_span_floor_ratio_low"] = int(reject_bottom_span_floor_ratio)
    base_metrics["reject_top_edge_floor_ratio_high"] = int(reject_top_edge_ratio_high)
    base_metrics["gate_min_bottom_span_floor_ratio"] = float(min_bottom_span_floor_ratio_hard)
    base_metrics["gate_max_top_edge_floor_ratio"] = float(_env_float("BADC_QRT_MAX_TOP_EDGE_FLOOR_RATIO", 0.58))
    base_metrics["reject_aspect_ratio"] = int(reject_aspect_ratio)
    base_metrics["relaxed_pass"] = int(relaxed_pass)
    top3_main.sort(key=lambda item: item[0], reverse=True)
    base_metrics["top3_mainfield"] = [
        {
            "score_final": float(s),
            "score_total": float(s),
            "area_ratio": float(a),
            "ymax_ratio": float(y),
            "bottom_support": float(b),
            "inside_count": int(ic),
            "ratio_struct_score": float(rs),
        }
        for s, a, y, b, ic, rs in top3_main[:3]
    ]
    base_metrics["selected_area_ratio"] = float(best_area_ratio_img) if best_area_ratio_img is not None else None
    base_metrics["selected_area_ratio_floor"] = float(best_area_ratio) if best_area_ratio is not None else None
    base_metrics["selected_ymax_ratio"] = float(best_ymax_ratio) if best_ymax_ratio is not None else None
    base_metrics["selected_top_edge_floor_ratio"] = (
        float(best_top_edge_floor_ratio) if best_top_edge_floor_ratio is not None else None
    )
    base_metrics["selected_bottom_support"] = (
        float(best_bottom_support) if best_bottom_support is not None else None
    )
    if best_anchor_score is not None:
        base_metrics["best_anchor_score"] = float(best_anchor_score)
    if isinstance(best_mixer_parts, dict):
        base_metrics.update(best_mixer_parts)
    base_metrics["score_base"] = float(best_score_base) if best_score_base is not None else None
    base_metrics["score_total"] = float(best_score) if best_score != float("-inf") else None
    base_metrics["role_prior_bonus"] = float(best_role_prior_bonus) if best_role_prior_bonus is not None else None
    base_metrics["role_prior_meta"] = best_role_prior_meta if isinstance(best_role_prior_meta, dict) else None
    base_metrics["court_ratio_refine_used"] = bool(ratio_refine_used)
    base_metrics["court_ratio_bonus"] = float(best_ratio_bonus) if best_ratio_bonus is not None else None
    base_metrics["court_ratio_meta"] = best_ratio_meta if isinstance(best_ratio_meta, dict) else None
    base_metrics["semantic_rerank_used"] = bool(semantic_rerank_used)
    base_metrics["semantic_rerank_switched"] = bool(semantic_rerank_switched)
    base_metrics["semantic_rerank_meta"] = semantic_rerank_meta if isinstance(semantic_rerank_meta, dict) else None
    base_metrics["inside_count"] = int(best_inside_count) if best_inside_count is not None else 0
    base_metrics["inside_bonus"] = float(best_inside_bonus) if best_inside_bonus is not None else 0.0
    base_metrics["score_final"] = base_metrics["score_base"]
    if base_metrics.get("max_inside_count") is None:
        base_metrics["max_inside_count"] = int(base_metrics["inside_count"])
    if base_metrics.get("selection_reason") is None:
        base_metrics["selection_reason"] = "single_component"
    if best_score_base is not None or best_inside_bonus is not None:
        base_metrics["score_breakdown"] = {
            "base": float(best_score_base) if best_score_base is not None else None,
            "inside_bonus": float(best_inside_bonus) if best_inside_bonus is not None else None,
            "score_total": float(best_score) if best_score != float("-inf") else None,
        }

    if best_pairA is None or best_pairB is None or best_ordered is None:
        H_raw, raw_metrics, raw_debug = _fit_court_homography_from_raw_floor_debug(
            white_mask_raw_floor_postblob,
            frame_bgr.shape,
            floor_roi_mask=floor_mask,
            white_mask_raw_floor_noblob=white_mask_raw_floor_noblob,
        )
        if H_raw is not None:
            semantic_mask_raw = (
                white_mask_raw_floor_postblob
                if isinstance(white_mask_raw_floor_postblob, np.ndarray)
                and int(np.count_nonzero(white_mask_raw_floor_postblob)) > 0
                else white_mask_raw_floor_noblob
            )
            H_raw_refined, semantic_meta_raw = _semantic_refine_homography(H_raw, semantic_mask_raw)
            if bool(semantic_meta_raw.get("improved", False)):
                H_raw = H_raw_refined
            base_metrics["semantic_refine_enabled"] = bool(semantic_meta_raw.get("enabled", False))
            base_metrics["semantic_refine_applied"] = bool(semantic_meta_raw.get("applied", False))
            base_metrics["semantic_refine_meta"] = semantic_meta_raw if isinstance(semantic_meta_raw, dict) else None

            dt_raw = build_distance_transform(white_mask_raw_floor_postblob)
            dt_raw_min = float(np.min(dt_raw)) if dt_raw.size > 0 else None
            dt_raw_mean = float(np.mean(dt_raw)) if dt_raw.size > 0 else None
            dt_raw_p90 = float(np.percentile(dt_raw, 90)) if dt_raw.size > 0 else None
            cover_points_raw = _sample_white_points(white_mask_raw_floor_postblob, max_points=1500, rng=rng)
            loss_info = _compute_loss_terms(
                H_raw,
                dt=dt_raw,
                Xw=Xw_full,
                weights=w_full,
                cover_points=cover_points_raw,
                floor_bbox=floor_bbox,
                tau_px=3.0,
                dt_oob=dt_oob,
            )
            p90_raw = float(
                loss_info.get("sample_dist_p90_raw_weighted")
                or loss_info.get("sample_dist_p90_raw")
                or loss_info.get("sample_dist_p90")
                or 999.0
            )
            p90_conf = float(
                loss_info.get("sample_dist_p90_weighted")
                or loss_info.get("sample_dist_p90")
                or min(float(p90_raw), 18.0)
            )
            cover_ratio = float(loss_info.get("cover_ratio", 0.0))
            inlier_ratio = float(loss_info.get("inlier_ratio", 0.0))
            cost = 0.6 * p90_raw + 0.4 * (1.0 - cover_ratio) * 50.0 + (1.0 - inlier_ratio) * 30.0
            conf_terms = _confidence_terms(
                inlier_ratio,
                cover_ratio,
                p90_conf,
                float(loss_info.get("area_ratio", 0.0)),
            )
            conf = float(conf_terms["confidence"])
            score1 = raw_metrics.get("raw_floor_best_score")
            top5 = raw_metrics.get("raw_floor_top5") if isinstance(raw_metrics, dict) else None
            score2 = None
            if isinstance(top5, list) and len(top5) > 1:
                try:
                    score2 = float(top5[1].get("score"))
                except Exception:
                    score2 = None
            score_norm_conf = None
            rank_gap_conf = None
            rank_gap_input = None
            if score1 is not None and isinstance(score1, (int, float)):
                score_center = float(_env_float("BADC_RAW_FLOOR_SCORE_CENTER", 40.0))
                score_scale = float(max(1e-3, _env_float("BADC_RAW_FLOOR_SCORE_SCALE", 12.0)))
                z_score = float((float(score1) - score_center) / score_scale)
                z_score = max(-60.0, min(60.0, z_score))
                score_norm_conf = float(1.0 / (1.0 + math.exp(-z_score)))
                gap = float(score1 - score2) if score2 is not None else 0.0
                gap_mid = float(_env_float("BADC_RAW_FLOOR_SCORE_GAP_MID", 0.12))
                gap_scale = float(max(1e-3, _env_float("BADC_RAW_FLOOR_SCORE_GAP_SCALE", 0.08)))
                z = float((gap - gap_mid) / gap_scale)
                z = max(-60.0, min(60.0, z))
                rank_gap_input = float(z)
                rank_gap_conf = float(1.0 / (1.0 + math.exp(-z)))
                rank_mix = float(
                    0.5 * float(score_norm_conf)
                    + 0.5 * float(rank_gap_conf)
                )
                conf_blend = float(max(0.0, min(0.8, _env_float("BADC_RAW_FLOOR_CONF_BLEND", 0.25))))
                conf = float((1.0 - conf_blend) * conf + conf_blend * rank_mix)
            conf_auto = float(conf)
            raw_conf_ok_thr = float(
                _env_float(
                    "BADC_RAW_FLOOR_CONF_OK_THR",
                    _env_float("BADC_FIT_CONF_OK_THR", 0.35),
                )
            )
            reason = "OK" if conf >= raw_conf_ok_thr else "R_fit_poor"
            corners_uv = project_points(H_raw, corners_world)
            ordered_corners = _order_corners_lb_rb_rt_lt(corners_uv)
            span_ok, span_metrics = _quad_span_ok(ordered_corners, frame_bgr.shape[1], frame_bgr.shape[0])
            base_metrics.update(span_metrics)
            ordered_clip = ordered_corners.copy()
            ordered_clip[:, 0] = np.clip(ordered_clip[:, 0], 0.0, float(frame_bgr.shape[1] - 1))
            ordered_clip[:, 1] = np.clip(ordered_clip[:, 1], 0.0, float(frame_bgr.shape[0] - 1))
            selected_area_ratio_guard = float(
                _quad_area(ordered_clip) / float(max(frame_bgr.shape[0] * frame_bgr.shape[1], 1))
            )
            min_selected_area_ratio_guard = float(_env_float("BADC_FIT_MIN_SELECTED_AREA_RATIO", 0.12))
            base_metrics["selected_area_ratio_guard"] = float(selected_area_ratio_guard)
            base_metrics["selected_area_ratio_guard_min"] = float(min_selected_area_ratio_guard)
            min_raw_floor_area_ratio = float(_env_float("BADC_RAW_FLOOR_MIN_AREA_RATIO", 0.20))
            raw_floor_area_ratio = float(loss_info.get("area_ratio", 0.0))
            base_metrics["raw_floor_min_area_ratio"] = float(min_raw_floor_area_ratio)
            base_metrics["raw_floor_area_ratio"] = float(raw_floor_area_ratio)
            min_bottom_span_floor_ratio_guard_base = float(
                _env_float("BADC_FIT_MIN_BOTTOM_SPAN_FLOOR_RATIO", 0.45)
            )
            min_bottom_span_floor_ratio_guard = float(
                max(
                    0.28,
                    min_bottom_span_floor_ratio_guard_base * (0.7 if int(relaxed_pass) > 0 else 1.0),
                )
            )
            bottom_span_floor_ratio_guard = _bottom_span_floor_ratio(ordered_corners, floor_bbox)
            max_top_edge_floor_ratio_guard = float(_env_float("BADC_FIT_MAX_TOP_EDGE_FLOOR_RATIO", 0.30))
            top_edge_floor_ratio_guard = _top_edge_floor_ratio(ordered_corners, floor_bbox)
            base_metrics["bottom_span_floor_ratio"] = float(bottom_span_floor_ratio_guard)
            base_metrics["bottom_span_floor_ratio_min"] = float(min_bottom_span_floor_ratio_guard)
            base_metrics["bottom_span_floor_ratio_min_base"] = float(min_bottom_span_floor_ratio_guard_base)
            base_metrics["top_edge_floor_ratio"] = float(top_edge_floor_ratio_guard)
            base_metrics["top_edge_floor_ratio_max"] = float(max_top_edge_floor_ratio_guard)
            corners_out = ordered_corners.astype(np.float32)
            deg_bbox_w = float(np.max(ordered_corners[:, 0]) - np.min(ordered_corners[:, 0]))
            deg_bbox_h = float(np.max(ordered_corners[:, 1]) - np.min(ordered_corners[:, 1]))
            deg_min_edge = float(np.min(_quad_edges(ordered_corners)))
            base_metrics["degenerate_predicted_corners_bbox"] = [float(deg_bbox_w), float(deg_bbox_h)]
            base_metrics["degenerate_predicted_corners_min_edge"] = float(deg_min_edge)
            if deg_bbox_w < 0.08 * float(frame_bgr.shape[1]) or deg_bbox_h < 0.06 * float(frame_bgr.shape[0]) or (
                deg_min_edge < 0.05 * float(min(frame_bgr.shape[0], frame_bgr.shape[1]))
            ):
                reason = "R_degenerate_quad"
                conf = 0.0
                conf_terms = {"confidence": 0.0, "s_inlier": 0.0, "s_cover": 0.0, "s_p90": 0.0, "s_area": 0.0}
                corners_out = None
            if corners_out is not None and bottom_span_floor_ratio_guard < min_bottom_span_floor_ratio_guard:
                reason = "R_bottom_span_small"
                conf = 0.0
                conf_terms = {"confidence": 0.0, "s_inlier": 0.0, "s_cover": 0.0, "s_p90": 0.0, "s_area": 0.0}
                corners_out = None
            if corners_out is not None and top_edge_floor_ratio_guard > max_top_edge_floor_ratio_guard:
                reason = "R_top_edge_low"
                conf = 0.0
                conf_terms = {"confidence": 0.0, "s_inlier": 0.0, "s_cover": 0.0, "s_p90": 0.0, "s_area": 0.0}
                corners_out = None
            if corners_out is not None and selected_area_ratio_guard < min_selected_area_ratio_guard:
                reason = "R_area_small"
                conf = 0.0
                conf_terms = {"confidence": 0.0, "s_inlier": 0.0, "s_cover": 0.0, "s_p90": 0.0, "s_area": 0.0}
                corners_out = None
            if corners_out is not None and raw_floor_area_ratio < min_raw_floor_area_ratio:
                reason = "R_small_area"
                conf = 0.0
                conf_terms = {"confidence": 0.0, "s_inlier": 0.0, "s_cover": 0.0, "s_p90": 0.0, "s_area": 0.0}
                corners_out = None
            if corners_out is not None and isinstance(semantic_meta_raw, dict) and not bool(
                semantic_meta_raw.get("hard_pass", True)
            ):
                bypass_conf = float(_env_float("BADC_SEMANTIC_HARD_BYPASS_CONF", 0.65))
                bypass_inlier = float(_env_float("BADC_SEMANTIC_HARD_BYPASS_INLIER", 0.70))
                bypass_cover = float(_env_float("BADC_SEMANTIC_HARD_BYPASS_COVER", 0.70))
                hard_bypass = bool(
                    float(conf) >= bypass_conf
                    and float(inlier_ratio) >= bypass_inlier
                    and float(cover_ratio) >= bypass_cover
                )
                base_metrics["semantic_hard_bypass"] = bool(hard_bypass)
                base_metrics["semantic_hard_bypass_conf"] = float(bypass_conf)
                base_metrics["semantic_hard_bypass_inlier"] = float(bypass_inlier)
                base_metrics["semantic_hard_bypass_cover"] = float(bypass_cover)
                if not hard_bypass:
                    reason = str(semantic_meta_raw.get("hard_reason") or "R_semantic")
                    conf = 0.0
                    conf_terms = {"confidence": 0.0, "s_inlier": 0.0, "s_cover": 0.0, "s_p90": 0.0, "s_area": 0.0}
                    corners_out = None
            if corners_out is not None and reason == "R_fit_poor":
                fit_poor_rescue_enable = bool(_env_flag("BADC_RAW_FLOOR_FIT_POOR_RESCUE_ENABLE", True))
                fit_poor_rescue_min_area = float(_env_float("BADC_RAW_FLOOR_FIT_POOR_RESCUE_MIN_AREA", 0.36))
                fit_poor_rescue_min_bottom = float(_env_float("BADC_RAW_FLOOR_FIT_POOR_RESCUE_MIN_BOTTOM_SPAN", 0.62))
                fit_poor_rescue_max_top = float(_env_float("BADC_RAW_FLOOR_FIT_POOR_RESCUE_MAX_TOP_EDGE", 0.26))
                fit_poor_rescue_min_conf = float(_env_float("BADC_RAW_FLOOR_FIT_POOR_RESCUE_MIN_CONF", 0.20))
                sem_hard_pass = bool((semantic_meta_raw or {}).get("hard_pass", True))
                fit_poor_rescue_ok = bool(
                    fit_poor_rescue_enable
                    and sem_hard_pass
                    and float(raw_floor_area_ratio) >= fit_poor_rescue_min_area
                    and float(bottom_span_floor_ratio_guard) >= fit_poor_rescue_min_bottom
                    and float(top_edge_floor_ratio_guard) <= fit_poor_rescue_max_top
                )
                base_metrics["raw_floor_fit_poor_rescue_enable"] = bool(fit_poor_rescue_enable)
                base_metrics["raw_floor_fit_poor_rescue_ok"] = bool(fit_poor_rescue_ok)
                base_metrics["raw_floor_fit_poor_rescue_sem_hard_pass"] = bool(sem_hard_pass)
                base_metrics["raw_floor_fit_poor_rescue_min_area"] = float(fit_poor_rescue_min_area)
                base_metrics["raw_floor_fit_poor_rescue_min_bottom_span"] = float(fit_poor_rescue_min_bottom)
                base_metrics["raw_floor_fit_poor_rescue_max_top_edge"] = float(fit_poor_rescue_max_top)
                if fit_poor_rescue_ok:
                    reason = "OK"
                    conf = max(float(conf), fit_poor_rescue_min_conf)
                    conf_terms = dict(conf_terms)
                    conf_terms["confidence"] = float(conf)
            if corners_out is not None and reason != "OK":
                corners_out = None
            if corners_out is not None and not manual:
                fb_w = max(1, int(floor_bbox[2] - floor_bbox[0]))
                fb_h = max(1, int(floor_bbox[3] - floor_bbox[1]))
                mx = max(24, int(round(0.05 * float(fb_w))))
                my = max(24, int(round(0.05 * float(fb_h))))
                if not _corners_inside_floor_roi(corners_out, floor_bbox, margin_x=mx, margin_y=my):
                    base_metrics["corners_outside_floor_roi"] = True
            pred_corners = _format_predicted_corners(ordered_corners, frame_bgr.shape[1], frame_bgr.shape[0])
            err_map = _corner_errors_to_manual(ordered_corners, manual.get("court_corners") if manual else None)
            err_lt = err_map.get("err_LT")
            err_rt = err_map.get("err_RT")
            err_rb = err_map.get("err_RB")
            err_vals = [v for v in [err_lt, err_rt, err_rb] if v is not None]
            err_mean = float(np.mean(err_vals)) if err_vals else 999.0
            conf_geom = float(math.exp(-err_mean / 200.0)) if manual_vis else 1.0
            conf_support = float(min(1.0, max(0.0, 0.5 * (inlier_ratio + cover_ratio))))
            conf_visible = float(sum(1 for v in pred_corners if v.get("visible")) / 4.0)
            anchor_tol = 80.0
            if reason == "OK" and manual_vis:
                if (manual_vis.get("LT") and err_lt is not None and err_lt > anchor_tol) or (
                    manual_vis.get("RT") and err_rt is not None and err_rt > anchor_tol
                ) or (manual_vis.get("RB") and err_rb is not None and err_rb > anchor_tol):
                    base_metrics["manual_anchor_violation"] = True
                    conf = float(conf) * 0.7
            if manual_vis and not base_metrics.get("manual_override", False):
                if base_metrics.get("manual_anchor_violation") or (
                    err_rt is not None and err_rt > 120.0
                ) or (err_rb is not None and err_rb > 150.0):
                    reason = "R_manual_anchor_violation"
                    corners_out = None
                    conf = 0.0
                    conf_terms = {"confidence": 0.0, "s_inlier": 0.0, "s_cover": 0.0, "s_p90": 0.0, "s_area": 0.0}
            if reason != "OK" and manual_quad is not None:
                H_manual = cv2.getPerspectiveTransform(get_bwf_corners().astype(np.float32), manual_quad.astype(np.float32))
                manual_loss = _compute_loss_terms(
                    H_manual,
                    dt=dt_raw,
                    Xw=Xw_full,
                    weights=w_full,
                    cover_points=cover_points_raw,
                    floor_bbox=floor_bbox,
                    tau_px=3.0,
                    dt_oob=dt_oob,
                )
                p90_manual = float(
                    manual_loss.get("sample_dist_p90_raw_weighted")
                    or manual_loss.get("sample_dist_p90_raw")
                    or manual_loss.get("sample_dist_p90")
                    or 999.0
                )
                cover_manual = float(manual_loss.get("cover_ratio", 0.0))
                inlier_manual = float(manual_loss.get("inlier_ratio", 0.0))
                conf_manual_terms = _confidence_terms(
                    inlier_manual, cover_manual, p90_manual, float(manual_loss.get("area_ratio", 0.0))
                )
                base_metrics["manual_candidate_confidence"] = float(conf_manual_terms["confidence"])
                base_metrics["manual_candidate_cost"] = float(
                    0.6 * p90_manual + 0.4 * (1.0 - cover_manual) * 50.0 + (1.0 - inlier_manual) * 30.0
                )
            frame_overlay = _draw_model_lines(frame_bgr, H_raw, (0, 255, 0))
            raw_overlay = raw_debug.get("overlay_raw")
            raw_best_mixer = raw_metrics.get("raw_floor_best_mixer") if isinstance(raw_metrics, dict) else None
            if isinstance(raw_best_mixer, dict):
                base_metrics.update(raw_best_mixer)
            metrics = {
                **base_metrics,
                **raw_metrics,
                "raw_floor_fit_used": True,
                "H": [float(v) for v in H_raw.reshape(-1)],
                "dt_min": dt_raw_min,
                "dt_mean": dt_raw_mean,
                "dt_p90": dt_raw_p90,
                "cost_total": float(cost),
                "confidence": float(conf),
                "s_inlier": float(conf_terms["s_inlier"]),
                "s_cover": float(conf_terms["s_cover"]),
                "s_p90": float(conf_terms["s_p90"]),
                "s_area": float(conf_terms["s_area"]),
                "inlier_ratio": float(inlier_ratio),
                "mean_dist_px": float(loss_info.get("sample_dist_mean") or 0.0),
                "p90_dist_px": float(p90_raw),
                "num_inliers": int(loss_info.get("num_inliers", 0)),
                "num_valid_samples": int(loss_info.get("num_valid_samples", 0)),
                "num_samples": int(loss_info.get("num_samples", Xw_full.shape[0])),
                "dt_oob_count": int(loss_info.get("dt_oob_count", 0)),
                "dt_oob_ratio": float(loss_info.get("dt_oob_count", 0)) / float(max(loss_info.get("num_samples", 1), 1)),
                "tau_px": 3.0,
                "l_dist": float(loss_info.get("l_dist", 0.0)),
                "l_cover": float(loss_info.get("l_cover", 0.0)),
                "l_reg": float(loss_info.get("l_reg", 0.0)),
                "area_ratio": float(loss_info.get("area_ratio", 0.0)),
                "cover_ratio": float(cover_ratio),
                "p90_dist_px_conf": float(p90_conf),
                "raw_floor_score_norm_conf": float(score_norm_conf) if score_norm_conf is not None else None,
                "raw_floor_rank_gap_conf": float(rank_gap_conf) if rank_gap_conf is not None else None,
                "raw_floor_rank_gap_input": float(rank_gap_input) if rank_gap_input is not None else None,
                "raw_floor_conf_final": float(conf),
                "raw_floor_conf_ok_thr": float(raw_conf_ok_thr),
                "confidence_auto": float(conf_auto),
                "confidence_final": float(conf),
                "conf_geom": float(conf_geom),
                "conf_support": float(conf_support),
                "conf_visible": float(conf_visible),
                "predicted_corners": pred_corners,
                "err_LT": err_lt,
                "err_RT": err_rt,
                "err_RB": err_rb,
            }
            if metrics.get("selected_component_id") is None:
                metrics["selected_component_id"] = int(base_metrics.get("component_id") or 0)
            raw_debug_save = raw_debug
            return CourtFitResult(
                H=H_raw,
                corners=corners_out,
                confidence=conf,
                metrics=metrics,
                reason=reason,
                debug_image=frame_overlay,
                white_mask=white_mask_clean,
                debug_image_init=frame_overlay,
                white_mask_raw=white_mask_raw_full,
                white_mask_clean=white_mask_clean,
                white_mask_raw_full=white_mask_raw_full,
                white_mask_raw_floor=white_mask_raw_floor,
                white_mask_raw_floor_noblob=white_mask_raw_floor_noblob,
                white_mask_raw_floor_preblob=white_mask_raw_floor_preblob,
                white_mask_raw_floor_postblob=white_mask_raw_floor_postblob,
                floor_roi_mask=floor_mask,
                floor_roi_overlay=floor_roi_overlay,
                seed_bottom_mask=floor_debug.get("seed_bottom_mask"),
                green_mask=floor_debug.get("green_mask"),
                largest_cc_mask=floor_debug.get("largest_cc_mask"),
                exg_row_plot=floor_debug.get("exg_row_plot"),
                linepix_mask=linepix_mask,
                linepix_overlay=linepix_overlay,
                ransac_lines_img=ransac_lines_img,
                raw_floor_hough_lines_img=raw_debug.get("hough_lines"),
                raw_floor_dt_debug=raw_debug.get("dt"),
                raw_floor_model_overlay=raw_overlay,
                raw_floor_top5_overlay=raw_debug.get("overlay_top5"),
                raw_floor_preprocessed=raw_debug.get("preprocessed"),
                raw_floor_edges=raw_debug.get("edges"),
                raw_floor_hough_lines_a=raw_debug.get("hough_lines_a"),
                raw_floor_hough_lines_b=raw_debug.get("hough_lines_b"),
                frame_model_overlay=frame_overlay,
                dt_debug=raw_debug.get("dt"),
                ori_mask_a=maskA,
                ori_mask_b=maskB,
                hough_lines_img=hough_lines_img,
            )
        fail = _fail_result(
            "R_hough_geometry",
            maskA=maskA,
            maskB=maskB,
            linepix_mask=linepix_mask,
            linepix_overlay=linepix_overlay,
            ransac_lines_img=ransac_lines_img,
            extra_metrics={
                "num_hough_lines_a": int(len(linesA)),
                "num_hough_lines_b": int(len(linesB)),
                "num_hough_pairs_a": int(len(pairsA)),
                "num_hough_pairs_b": int(len(pairsB)),
                **raw_metrics,
            },
        )
        fail.raw_floor_hough_lines_img = raw_debug.get("hough_lines")
        fail.raw_floor_dt_debug = raw_debug.get("dt")
        fail.raw_floor_model_overlay = raw_debug.get("overlay_raw")
        fail.raw_floor_top5_overlay = raw_debug.get("overlay_top5")
        fail.raw_floor_preprocessed = raw_debug.get("preprocessed")
        fail.raw_floor_edges = raw_debug.get("edges")
        fail.raw_floor_hough_lines_a = raw_debug.get("hough_lines_a")
        fail.raw_floor_hough_lines_b = raw_debug.get("hough_lines_b")
        return fail

    pairA = best_pairA
    pairB = best_pairB
    ordered = best_ordered
    area_ratio = float(best_area_ratio) if best_area_ratio is not None else 0.0
    inside_ratio = float(best_inside_ratio) if best_inside_ratio is not None else 0.0

    completion_used = False
    completion_meta: Dict[str, Any] = {}
    completion_reject_reason = None

    def _detect_baseline_ambiguity_for_completion(
        ordered_quad: np.ndarray,
        candidate_lines: Sequence[Tuple[float, float, float, float]],
        floor_bbox_xyxy: Tuple[int, int, int, int],
        line_mask_local: Optional[np.ndarray],
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Detect whether current bottom edge is likely an inner service line instead of the true baseline.
        Trigger condition (image-space): a sufficiently strong near-parallel line exists below current bottom.
        """
        meta_local: Dict[str, Any] = {"enabled": True, "ambiguous": False}
        if ordered_quad is None or len(candidate_lines) == 0:
            meta_local["reason"] = "no_input"
            return False, meta_local
        try:
            q = np.asarray(ordered_quad, dtype=np.float32).reshape(4, 2)
        except Exception:
            meta_local["reason"] = "bad_quad"
            return False, meta_local
        if q.shape != (4, 2) or not np.all(np.isfinite(q)):
            meta_local["reason"] = "bad_quad"
            return False, meta_local

        x0f, y0f, x1f, y1f = [float(v) for v in floor_bbox_xyxy]
        floor_w = float(max(1.0, x1f - x0f))
        x_ref = float(0.5 * (x0f + x1f))
        curr_bottom = _line_from_points((float(q[0, 0]), float(q[0, 1])), (float(q[1, 0]), float(q[1, 1])))
        curr_ang = float(np.mod(math.atan2(float(q[1, 1] - q[0, 1]), float(q[1, 0] - q[0, 0])), math.pi))
        y_curr = _y_at_x(curr_bottom, x_ref)
        if y_curr is None or not math.isfinite(float(y_curr)):
            y_curr = float(0.5 * (float(q[0, 1]) + float(q[1, 1])))

        angle_tol_deg = float(_env_float("BADC_COMPLETION_BASELINE_AMBIG_ANGLE_TOL_DEG", 14.0))
        min_below_px = float(_env_float("BADC_COMPLETION_BASELINE_AMBIG_MIN_BELOW_PX", 10.0))
        max_below_px = float(_env_float("BADC_COMPLETION_BASELINE_AMBIG_MAX_BELOW_PX", 180.0))
        min_support = float(max(0.0, min(1.0, _env_float("BADC_COMPLETION_BASELINE_AMBIG_MIN_SUPPORT", 0.10))))
        min_len_frac = float(max(0.10, min(1.0, _env_float("BADC_COMPLETION_BASELINE_AMBIG_MIN_LEN_FRAC", 0.28))))
        expected_gap_m = float(_env_float("BADC_POSTFIT_BASELINE_TARGET_GAP_M", 0.76))
        expected_gap_tol_ratio = float(max(0.20, _env_float("BADC_COMPLETION_BASELINE_AMBIG_GAP_TOL_RATIO", 0.55)))

        # Estimate expected ~0.76m vertical gap in image for extra confidence.
        expected_gap_px = None
        try:
            H_m2i = cv2.getPerspectiveTransform(get_bwf_corners().astype(np.float32), q.astype(np.float32))
            model_w = float(np.max(get_bwf_corners()[:, 0]) - np.min(get_bwf_corners()[:, 0]))
            p_gap = project_points(
                H_m2i,
                np.array([[0.5 * model_w, 0.0], [0.5 * model_w, float(expected_gap_m)]], dtype=np.float32),
            )
            if p_gap.shape == (2, 2):
                expected_gap_px = float(abs(float(p_gap[1, 1]) - float(p_gap[0, 1])))
        except Exception:
            expected_gap_px = None
        if expected_gap_px is not None and (not math.isfinite(expected_gap_px) or expected_gap_px < 4.0):
            expected_gap_px = None

        best: Optional[Dict[str, Any]] = None
        ang_tol = math.radians(max(3.0, angle_tol_deg))
        min_len_px = float(min_len_frac * floor_w)
        for seg in candidate_lines:
            x1s, y1s, x2s, y2s = [float(v) for v in seg]
            seg_len = float(math.hypot(float(x2s - x1s), float(y2s - y1s)))
            if seg_len < min_len_px:
                continue
            seg_ang = float(np.mod(math.atan2(float(y2s - y1s), float(x2s - x1s)), math.pi))
            if _angle_distance(seg_ang, curr_ang) > ang_tol:
                continue
            line_i = _line_from_points((x1s, y1s), (x2s, y2s))
            yi = _y_at_x(line_i, x_ref)
            if yi is None or not math.isfinite(float(yi)):
                yi = float(0.5 * (y1s + y2s))
            dy = float(yi - float(y_curr))
            if dy < min_below_px or dy > max_below_px:
                continue
            sup = _line_support((x1s, y1s, x2s, y2s), line_mask_local) if line_mask_local is not None else None
            sup_val = float(sup) if sup is not None else 0.0
            if sup_val < min_support:
                continue
            gap_bonus = 0.0
            if expected_gap_px is not None:
                tol_px = max(6.0, expected_gap_tol_ratio * expected_gap_px)
                gap_bonus = float(math.exp(-abs(dy - expected_gap_px) / max(1e-6, tol_px)))
            score = float(0.70 * sup_val + 0.20 * min(1.0, seg_len / max(1.0, floor_w)) + 0.10 * gap_bonus)
            cand = {
                "dy_px": float(dy),
                "support": float(sup_val),
                "seg_len_px": float(seg_len),
                "score": float(score),
                "gap_bonus": float(gap_bonus),
                "x1": float(x1s),
                "y1": float(y1s),
                "x2": float(x2s),
                "y2": float(y2s),
            }
            if best is None or float(cand["score"]) > float(best["score"]):
                best = cand

        meta_local["current_bottom_y_ref"] = float(y_curr)
        meta_local["x_ref"] = float(x_ref)
        meta_local["expected_gap_px"] = float(expected_gap_px) if expected_gap_px is not None else None
        meta_local["num_candidates"] = int(len(candidate_lines))
        if best is None:
            meta_local["reason"] = "no_lower_parallel_line"
            return False, meta_local
        meta_local["best_candidate"] = best
        meta_local["ambiguous"] = True
        meta_local["reason"] = "lower_parallel_line_detected"
        return True, meta_local

    if ordered is not None and lines_left and lines_bottom:
        floor_w = float(max(1.0, float(floor_bbox[2] - floor_bbox[0])))
        floor_h = float(max(1.0, float(floor_bbox[3] - floor_bbox[1])))
        completion_left_min_span_ratio = float(
            max(0.0, min(1.0, _env_float("BADC_COMPLETION_LEFT_FILTER_MIN_SPAN_RATIO", 0.55)))
        )
        completion_left_min_span_px = float(max(80.0, _env_float("BADC_COMPLETION_LEFT_FILTER_MIN_SPAN_PX", 260.0)))
        completion_left_extreme_keep = int(max(1, _env_int("BADC_COMPLETION_LEFT_FILTER_EXTREME_KEEP", 2)))
        comp_left_margin_x = float(
            max(
                8.0,
                _env_float(
                    "BADC_COMPLETION_LEFT_MARGIN_X_PX",
                    _env_float("BADC_COMPLETION_LEFT_MARGIN_X_FRAC", 0.60) * floor_w,
                ),
            )
        )
        comp_left_margin_top = float(
            max(8.0, _env_float("BADC_COMPLETION_LEFT_MARGIN_TOP_PX", _env_float("BADC_COMPLETION_LEFT_MARGIN_TOP_FRAC", 0.15) * floor_h))
        )
        comp_left_margin_bottom = float(
            max(8.0, _env_float("BADC_COMPLETION_LEFT_MARGIN_BOTTOM_PX", _env_float("BADC_COMPLETION_LEFT_MARGIN_BOTTOM_FRAC", 0.35) * floor_h))
        )
        lines_left_comp, line_ids_left_comp = _filter_lines_for_completion(
            lines_left,
            line_ids_left,
            ordered,
            margin_px=8.0,
            margin_x_px=comp_left_margin_x,
            margin_top_px=comp_left_margin_top,
            margin_bottom_px=comp_left_margin_bottom,
            expand_span_axis="x",
            min_span_ratio=completion_left_min_span_ratio,
            min_span_px=completion_left_min_span_px,
            extreme_keep=completion_left_extreme_keep,
        )
        comp_bottom_extra_px = _env_float(
            "BADC_COMPLETION_BOTTOM_EXTRA_PX",
            _env_float("BADC_COMPLETION_BOTTOM_EXTRA_FRAC", 0.22) * floor_h,
        )
        comp_bottom_margin_x = float(
            max(
                8.0,
                _env_float(
                    "BADC_COMPLETION_BOTTOM_MARGIN_X_PX",
                    _env_float("BADC_COMPLETION_BOTTOM_MARGIN_X_FRAC", 0.65) * floor_w,
                ),
            )
        )
        comp_bottom_margin_top = float(
            max(8.0, _env_float("BADC_COMPLETION_BOTTOM_MARGIN_TOP_PX", _env_float("BADC_COMPLETION_BOTTOM_MARGIN_TOP_FRAC", 0.12) * floor_h))
        )
        lines_bottom_comp, line_ids_bottom_comp = _filter_lines_for_completion(
            lines_bottom,
            line_ids_bottom,
            ordered,
            margin_px=8.0,
            margin_x_px=comp_bottom_margin_x,
            margin_top_px=comp_bottom_margin_top,
            margin_bottom_px=max(8.0, float(comp_bottom_extra_px)),
        )
        base_metrics["completion_bottom_extra_px"] = float(max(8.0, float(comp_bottom_extra_px)))
        base_metrics["completion_bottom_margin_x_px"] = float(comp_bottom_margin_x)
        base_metrics["completion_bottom_margin_top_px"] = float(comp_bottom_margin_top)
        base_metrics["completion_bottom_lines_before"] = int(len(lines_bottom))
        base_metrics["completion_bottom_lines_after"] = int(len(lines_bottom_comp))
        base_metrics["completion_left_lines_before"] = int(len(lines_left))
        base_metrics["completion_left_lines_after"] = int(len(lines_left_comp))
        base_metrics["completion_left_filter_min_span_ratio"] = float(completion_left_min_span_ratio)
        base_metrics["completion_left_filter_min_span_px"] = float(completion_left_min_span_px)
        base_metrics["completion_left_filter_extreme_keep"] = int(completion_left_extreme_keep)
        base_metrics["completion_left_margin_x_px"] = float(comp_left_margin_x)
        base_metrics["completion_left_margin_top_px"] = float(comp_left_margin_top)
        base_metrics["completion_left_margin_bottom_px"] = float(comp_left_margin_bottom)
        completion_lines_preview_limit = int(max(4, _env_int("BADC_COMPLETION_LINES_PREVIEW_LIMIT", 24)))

        def _completion_line_preview(
            segs: Sequence[Tuple[float, float, float, float]],
            ids: Sequence[Optional[int]],
        ) -> list[Dict[str, Any]]:
            out: list[Dict[str, Any]] = []
            for idx, seg in enumerate(segs[:completion_lines_preview_limit]):
                x1s, y1s, x2s, y2s = [float(v) for v in seg]
                theta = float(np.mod(math.degrees(math.atan2(float(y2s - y1s), float(x2s - x1s))), 180.0))
                length = float(math.hypot(float(x2s - x1s), float(y2s - y1s)))
                sup = _line_support((x1s, y1s, x2s, y2s), linepix_mask) if linepix_mask is not None else None
                line_abc = _line_from_points((x1s, y1s), (x2s, y2s))
                x_ref_at_bottom = _x_at_y(line_abc, float(floor_bbox[3] - 1.0))
                out.append(
                    {
                        "line_id": ids[idx] if idx < len(ids) else None,
                        "theta_deg": float(theta),
                        "length": float(length),
                        "support": float(sup) if sup is not None else None,
                        "x1": float(x1s),
                        "y1": float(y1s),
                        "x2": float(x2s),
                        "y2": float(y2s),
                        "x_at_floor_bottom": float(x_ref_at_bottom)
                        if x_ref_at_bottom is not None and math.isfinite(float(x_ref_at_bottom))
                        else None,
                    }
                )
            return out

        base_metrics["completion_left_lines_preview"] = _completion_line_preview(lines_left_comp, line_ids_left_comp)
        base_metrics["completion_bottom_lines_preview"] = _completion_line_preview(lines_bottom_comp, line_ids_bottom_comp)
        y_sorted = np.sort(ordered[:, 1])
        bottom_min = float(y_sorted[-2]) if y_sorted.size >= 2 else float(np.max(ordered[:, 1]))
        out_of_frame = np.any(
            (ordered[:, 0] < -0.05 * w)
            | (ordered[:, 0] > 1.05 * w)
            | (ordered[:, 1] < -0.05 * h)
            | (ordered[:, 1] > 1.05 * h)
        )
        baseline_ambiguous, baseline_ambig_meta = _detect_baseline_ambiguity_for_completion(
            ordered,
            lines_bottom,
            floor_bbox,
            linepix_mask,
        )
        base_metrics["completion_baseline_ambiguous"] = bool(baseline_ambiguous)
        base_metrics["completion_baseline_ambiguous_meta"] = baseline_ambig_meta
        if (not lines_left_comp or not lines_bottom_comp) and completion_reject_reason is None:
            completion_reject_reason = "completion_no_inlier_lines"
        completion_force = bool(_env_flag("BADC_COMPLETION_FORCE", False))
        completion_trigger = bool(
            completion_force
            or out_of_frame
            or baseline_ambiguous
            or (bottom_min < (float(floor_bbox[1]) + 0.65 * floor_h))
        )
        base_metrics["completion_force"] = bool(completion_force)
        base_metrics["completion_triggered"] = bool(completion_trigger)
        base_metrics["completion_trigger_reasons"] = {
            "force": bool(completion_force),
            "out_of_frame": bool(out_of_frame),
            "baseline_ambiguous": bool(baseline_ambiguous),
            "bottom_too_high": bool(bottom_min < (float(floor_bbox[1]) + 0.65 * floor_h)),
        }
        if completion_trigger and lines_left_comp and lines_bottom_comp:
            left_union_enabled = _env_flag("BADC_COMPLETION_LEFT_USE_UNION", True)
            all_union_enabled = _env_flag("BADC_COMPLETION_USE_ALL_LINES", baseline_ambiguous)
            lines_left_completion: list[Tuple[float, float, float, float]] = list(lines_left_comp)
            line_ids_left_completion: list[Optional[int]] = list(line_ids_left_comp)
            lines_all_comp: list[Tuple[float, float, float, float]] = []
            line_ids_all_comp: list[Optional[int]] = []
            if all_union_enabled:
                comp_all_margin_x = float(
                    max(
                        8.0,
                        _env_float(
                            "BADC_COMPLETION_ALL_MARGIN_X_PX",
                            _env_float("BADC_COMPLETION_ALL_MARGIN_X_FRAC", 0.70) * floor_w,
                        ),
                    )
                )
                comp_all_margin_top = float(
                    max(8.0, _env_float("BADC_COMPLETION_ALL_MARGIN_TOP_PX", _env_float("BADC_COMPLETION_ALL_MARGIN_TOP_FRAC", 0.18) * floor_h))
                )
                comp_all_margin_bottom = float(
                    max(
                        8.0,
                        _env_float(
                            "BADC_COMPLETION_ALL_MARGIN_BOTTOM_PX",
                            max(32.0, _env_float("BADC_COMPLETION_ALL_MARGIN_BOTTOM_FRAC", 0.40) * floor_h),
                        ),
                    )
                )
                comp_all_min_span_ratio = float(
                    max(
                        completion_left_min_span_ratio,
                        min(1.0, _env_float("BADC_COMPLETION_ALL_MIN_SPAN_RATIO", 0.70)),
                    )
                )
                comp_all_min_span_px = float(
                    max(
                        completion_left_min_span_px,
                        max(120.0, _env_float("BADC_COMPLETION_ALL_MIN_SPAN_PX", 320.0)),
                    )
                )
                comp_all_extreme_keep = int(
                    max(
                        completion_left_extreme_keep,
                        max(1, _env_int("BADC_COMPLETION_ALL_EXTREME_KEEP", 3)),
                    )
                )
                lines_all_comp, line_ids_all_comp = _filter_lines_for_completion(
                    lines_all,
                    line_ids_all,
                    ordered,
                    margin_px=8.0,
                    margin_x_px=comp_all_margin_x,
                    margin_top_px=comp_all_margin_top,
                    margin_bottom_px=comp_all_margin_bottom,
                    expand_span_axis="x",
                    min_span_ratio=comp_all_min_span_ratio,
                    min_span_px=comp_all_min_span_px,
                    extreme_keep=comp_all_extreme_keep,
                )
            if left_union_enabled:
                seen_keys: set[Tuple[int, int, int, int]] = set()
                merged_lines: list[Tuple[float, float, float, float]] = []
                merged_ids: list[Optional[int]] = []
                merged_input_lines = list(lines_left_comp) + list(lines_bottom_comp) + list(lines_all_comp)
                merged_input_ids = list(line_ids_left_comp) + list(line_ids_bottom_comp) + list(line_ids_all_comp)
                for seg, lid in zip(merged_input_lines, merged_input_ids):
                    x1s, y1s, x2s, y2s = [float(v) for v in seg]
                    p1 = (x1s, y1s)
                    p2 = (x2s, y2s)
                    if (p2[0], p2[1]) < (p1[0], p1[1]):
                        p1, p2 = p2, p1
                    # Coarse quantization for dedup across nearly identical segments.
                    key = (
                        int(round(p1[0] / 2.0)),
                        int(round(p1[1] / 2.0)),
                        int(round(p2[0] / 2.0)),
                        int(round(p2[1] / 2.0)),
                    )
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)
                    merged_lines.append((x1s, y1s, x2s, y2s))
                    merged_ids.append(lid)
                lines_left_completion = merged_lines
                line_ids_left_completion = merged_ids
            base_metrics["completion_left_union_enabled"] = bool(left_union_enabled)
            base_metrics["completion_all_union_enabled"] = bool(all_union_enabled)
            base_metrics["completion_left_lines_base"] = int(len(lines_left_comp))
            base_metrics["completion_left_lines_merged"] = int(len(lines_left_completion))
            base_metrics["completion_all_lines_base"] = int(len(lines_all))
            base_metrics["completion_all_lines_filtered"] = int(len(lines_all_comp))
            completed, ok, meta = _complete_corners_from_lines(
                ordered,
                lines_left=lines_left_completion,
                lines_bottom=lines_bottom_comp,
                floor_bbox=floor_bbox,
                img_w=w,
                img_h=h,
                line_mask=linepix_mask,
                line_ids_left=line_ids_left_completion,
                line_ids_bottom=line_ids_bottom_comp,
                component_id=component_id,
            )
            if ok:
                comp_bbox_w = float(np.max(completed[:, 0]) - np.min(completed[:, 0]))
                comp_bbox_h = float(np.max(completed[:, 1]) - np.min(completed[:, 1]))
                comp_min_edge = float(np.min(_quad_edges(completed)))
                comp_area = float(_quad_area(completed))
                comp_area_ratio = comp_area / float(max(w * h, 1.0))
                base_metrics["completion_candidate_bbox_w"] = float(comp_bbox_w)
                base_metrics["completion_candidate_bbox_h"] = float(comp_bbox_h)
                base_metrics["completion_candidate_min_edge"] = float(comp_min_edge)
                base_metrics["completion_candidate_area_ratio"] = float(comp_area_ratio)
                comp_min_edge_frac = float(max(0.0, _env_float("BADC_COMPLETION_MIN_EDGE_FRAC", 0.05)))
                comp_min_edge_relaxed_frac = float(
                    max(0.0, _env_float("BADC_COMPLETION_MIN_EDGE_RELAXED_FRAC", 0.045))
                )
                comp_min_edge_relax_area = float(
                    max(0.0, _env_float("BADC_COMPLETION_MIN_EDGE_RELAX_AREA_RATIO", 0.08))
                )
                comp_min_edge_req = float(comp_min_edge_frac * float(min(w, h)))
                if comp_area_ratio >= comp_min_edge_relax_area:
                    comp_min_edge_req = min(comp_min_edge_req, float(comp_min_edge_relaxed_frac * float(min(w, h))))
                base_metrics["completion_min_edge_req"] = float(comp_min_edge_req)
                if isinstance(meta, dict):
                    base_metrics["completion_candidate_meta"] = meta
                if comp_min_edge < comp_min_edge_req:
                    completion_reject_reason = "completion_min_edge"
                elif comp_area_ratio < 0.02:
                    completion_reject_reason = "completion_area_small"
                elif comp_bbox_h < 0.06 * float(h):
                    completion_reject_reason = "completion_bbox_h_small"
                else:
                    comp_floor_margin_px = float(max(8.0, _env_float("BADC_COMPLETION_BOTTOM_FLOOR_MARGIN_PX", 24.0)))
                    comp_bottom = np.asarray(completed[:2], dtype=np.float32)
                    x0f, y0f, x1f, y1f = [float(v) for v in floor_bbox]
                    bottom_x = comp_bottom[:, 0].astype(np.float32)
                    bottom_y = comp_bottom[:, 1].astype(np.float32)
                    oob_l = np.maximum(0.0, (x0f - comp_floor_margin_px) - bottom_x)
                    oob_r = np.maximum(0.0, bottom_x - (x1f + comp_floor_margin_px))
                    oob_t = np.maximum(0.0, (y0f - comp_floor_margin_px) - bottom_y)
                    oob_b = np.maximum(0.0, bottom_y - (y1f + comp_floor_margin_px))
                    bottom_oob = oob_l + oob_r + oob_t + oob_b
                    bottom_oob_max = float(np.max(bottom_oob)) if bottom_oob.size else 0.0
                    bottom_oob_sum = float(np.sum(bottom_oob)) if bottom_oob.size else 0.0
                    bottom_inside_floor = bool(
                        np.all(
                            (comp_bottom[:, 0] >= x0f - comp_floor_margin_px)
                            & (comp_bottom[:, 0] <= x1f + comp_floor_margin_px)
                            & (comp_bottom[:, 1] >= y0f - comp_floor_margin_px)
                            & (comp_bottom[:, 1] <= y1f + comp_floor_margin_px)
                        )
                    )
                    base_metrics["completion_bottom_inside_floor_bbox"] = bool(bottom_inside_floor)
                    base_metrics["completion_bottom_floor_margin_px"] = float(comp_floor_margin_px)
                    base_metrics["completion_bottom_floor_oob_max_px"] = float(bottom_oob_max)
                    base_metrics["completion_bottom_floor_oob_sum_px"] = float(bottom_oob_sum)
                    allow_bottom_oob = bool(_env_flag("BADC_COMPLETION_BOTTOM_ALLOW_OOB", True))
                    oob_max_px = float(
                        max(
                            80.0,
                            _env_float(
                                "BADC_COMPLETION_BOTTOM_OOB_MAX_PX",
                                max(220.0, 1.70 * float(floor_h)),
                            ),
                        )
                    )
                    oob_sum_px = float(
                        max(
                            oob_max_px,
                            _env_float(
                                "BADC_COMPLETION_BOTTOM_OOB_SUM_MAX_PX",
                                max(420.0, 3.30 * float(floor_h)),
                            ),
                        )
                    )
                    oob_min_area_ratio = float(max(0.0, _env_float("BADC_COMPLETION_BOTTOM_OOB_MIN_AREA_RATIO", 0.10)))
                    oob_min_edge_frac = float(max(0.0, _env_float("BADC_COMPLETION_BOTTOM_OOB_MIN_EDGE_FRAC", 0.08)))
                    oob_min_edge_px = float(oob_min_edge_frac * float(min(w, h)))
                    baseline_refined = bool(
                        isinstance(meta, dict)
                        and isinstance(meta.get("baseline_refine_meta"), dict)
                        and bool(meta.get("baseline_refine_meta", {}).get("used"))
                    )
                    bottom_oob_accepted = bool(
                        (not bottom_inside_floor)
                        and allow_bottom_oob
                        and baseline_refined
                        and (comp_area_ratio >= oob_min_area_ratio)
                        and (comp_min_edge >= oob_min_edge_px)
                        and (bottom_oob_max <= oob_max_px)
                        and (bottom_oob_sum <= oob_sum_px)
                    )
                    base_metrics["completion_bottom_oob_accepted"] = bool(bottom_oob_accepted)
                    base_metrics["completion_bottom_oob_allow_enabled"] = bool(allow_bottom_oob)
                    base_metrics["completion_bottom_oob_max_px"] = float(oob_max_px)
                    base_metrics["completion_bottom_oob_sum_px"] = float(oob_sum_px)
                    base_metrics["completion_bottom_oob_min_area_ratio"] = float(oob_min_area_ratio)
                    base_metrics["completion_bottom_oob_min_edge_px"] = float(oob_min_edge_px)
                    if (not bottom_inside_floor) and (not bottom_oob_accepted):
                        completion_reject_reason = "completion_bottom_outside_floor_bbox"
                if completion_reject_reason is None:
                    ordered = completed
                    completion_used = True
                    completion_meta = meta
    base_metrics["corner_completion_used"] = bool(completion_used)
    if completion_reject_reason:
        base_metrics["corner_completion_rejected_reason"] = completion_reject_reason
    if completion_used and isinstance(completion_meta, dict):
        def _line_to_list(line_val: Optional[Tuple[float, float, float]]) -> Optional[list]:
            if line_val is None:
                return None
            return [float(line_val[0]), float(line_val[1]), float(line_val[2])]

        base_metrics["completed_LB"] = [float(ordered[0, 0]), float(ordered[0, 1])]
        base_metrics["completed_RB"] = [float(ordered[1, 0]), float(ordered[1, 1])]
        base_metrics["completed_RT"] = [float(ordered[2, 0]), float(ordered[2, 1])]
        base_metrics["completed_LT"] = [float(ordered[3, 0]), float(ordered[3, 1])]
        base_metrics["baseline_line"] = _line_to_list(completion_meta.get("baseline_line"))
        base_metrics["left_sideline_line"] = _line_to_list(completion_meta.get("left_sideline"))
        base_metrics["right_sideline_line"] = _line_to_list(completion_meta.get("right_sideline"))
        base_metrics["top_line"] = _line_to_list(completion_meta.get("top_line"))
        base_metrics["baseline_line_meta"] = completion_meta.get("baseline_meta")
        base_metrics["left_sideline_meta"] = completion_meta.get("left_sideline_meta")
        base_metrics["right_sideline_meta"] = completion_meta.get("right_sideline_meta")
        base_metrics["top_line_meta"] = completion_meta.get("top_line_meta")

    postfit_baseline_used = False
    postfit_baseline_meta: Dict[str, Any] = {}
    if ordered is not None:
        ordered_before_postfit = np.asarray(ordered, dtype=np.float32).copy()
        refined_ordered, postfit_baseline_used, postfit_baseline_meta = _refine_bottom_baseline_postfit(
            ordered,
            lines_bottom=lines_bottom,
            lines_all=lines_all,
            floor_bbox=floor_bbox,
            line_mask=linepix_mask,
        )
        if postfit_baseline_used:
            def _quad_image_oob_stats(quad_xy: np.ndarray, img_w: int, img_h: int) -> Tuple[float, float]:
                q = np.asarray(quad_xy, dtype=np.float32).reshape(4, 2)
                x = q[:, 0]
                y = q[:, 1]
                oob_l = np.maximum(0.0, -x)
                oob_r = np.maximum(0.0, x - float(max(0, img_w - 1)))
                oob_t = np.maximum(0.0, -y)
                oob_b = np.maximum(0.0, y - float(max(0, img_h - 1)))
                per_corner = oob_l + oob_r + oob_t + oob_b
                return float(np.max(per_corner)), float(np.sum(per_corner))

            def _quad_floor_oob_sum(quad_xy: np.ndarray, bbox: Tuple[int, int, int, int], margin: float) -> float:
                q = np.asarray(quad_xy, dtype=np.float32).reshape(4, 2)
                x0f, y0f, x1f, y1f = [float(v) for v in bbox]
                x = q[:, 0]
                y = q[:, 1]
                oob_l = np.maximum(0.0, (x0f - float(margin)) - x)
                oob_r = np.maximum(0.0, x - (x1f + float(margin)))
                oob_t = np.maximum(0.0, (y0f - float(margin)) - y)
                oob_b = np.maximum(0.0, y - (y1f + float(margin)))
                return float(np.sum(oob_l + oob_r + oob_t + oob_b))

            h_img, w_img = frame_bgr.shape[:2]
            pre_oob_max, pre_oob_sum = _quad_image_oob_stats(ordered_before_postfit, w_img, h_img)
            post_oob_max, post_oob_sum = _quad_image_oob_stats(refined_ordered, w_img, h_img)
            floor_h_px = float(max(1.0, float(floor_bbox[3] - floor_bbox[1])))
            floor_margin = float(_env_float("BADC_POSTFIT_BASELINE_OOB_FLOOR_MARGIN_FRAC", 0.12)) * floor_h_px
            pre_floor_oob = _quad_floor_oob_sum(ordered_before_postfit, floor_bbox, floor_margin)
            post_floor_oob = _quad_floor_oob_sum(refined_ordered, floor_bbox, floor_margin)
            max_inc = float(_env_float("BADC_POSTFIT_BASELINE_OOB_MAX_INC_PX", 60.0))
            sum_inc = float(_env_float("BADC_POSTFIT_BASELINE_OOB_SUM_INC_PX", 160.0))
            abs_sum_max = float(_env_float("BADC_POSTFIT_BASELINE_OOB_ABS_SUM_MAX_PX", 260.0))
            floor_sum_inc = float(_env_float("BADC_POSTFIT_BASELINE_FLOOR_OOB_SUM_INC_PX", 140.0))
            floor_abs_max = float(_env_float("BADC_POSTFIT_BASELINE_FLOOR_OOB_ABS_MAX_PX", 220.0))

            reject_postfit = False
            if (post_oob_max > pre_oob_max + max_inc) or (post_oob_sum > pre_oob_sum + sum_inc):
                reject_postfit = True
            if post_oob_sum > abs_sum_max:
                reject_postfit = True
            if (post_floor_oob > pre_floor_oob + floor_sum_inc) or (post_floor_oob > floor_abs_max):
                reject_postfit = True

            postfit_baseline_meta = dict(postfit_baseline_meta or {})
            postfit_baseline_meta["pre_oob_max"] = float(pre_oob_max)
            postfit_baseline_meta["pre_oob_sum"] = float(pre_oob_sum)
            postfit_baseline_meta["post_oob_max"] = float(post_oob_max)
            postfit_baseline_meta["post_oob_sum"] = float(post_oob_sum)
            postfit_baseline_meta["pre_floor_oob_sum"] = float(pre_floor_oob)
            postfit_baseline_meta["post_floor_oob_sum"] = float(post_floor_oob)
            postfit_baseline_meta["oob_guard_rejected"] = bool(reject_postfit)
            postfit_baseline_meta["oob_guard_thresholds"] = {
                "max_inc": float(max_inc),
                "sum_inc": float(sum_inc),
                "abs_sum_max": float(abs_sum_max),
                "floor_sum_inc": float(floor_sum_inc),
                "floor_abs_max": float(floor_abs_max),
                "floor_margin_px": float(floor_margin),
            }

            if reject_postfit:
                postfit_baseline_used = False
            else:
                ordered = refined_ordered
    base_metrics["postfit_baseline_verify_used"] = bool(postfit_baseline_used)
    base_metrics["postfit_baseline_verify_meta"] = postfit_baseline_meta if isinstance(postfit_baseline_meta, dict) else None

    H_best = cv2.getPerspectiveTransform(get_bwf_corners().astype(np.float32), ordered.astype(np.float32))
    semantic_mask_main = (
        linepix_mask
        if isinstance(linepix_mask, np.ndarray) and int(np.count_nonzero(linepix_mask)) > 0
        else white_mask_clean
    )
    H_best_refined, semantic_meta_main = _semantic_refine_homography(H_best, semantic_mask_main)
    if bool(semantic_meta_main.get("improved", False)):
        H_best = H_best_refined
    base_metrics["semantic_refine_enabled"] = bool(semantic_meta_main.get("enabled", False))
    base_metrics["semantic_refine_applied"] = bool(semantic_meta_main.get("applied", False))
    base_metrics["semantic_refine_meta"] = semantic_meta_main if isinstance(semantic_meta_main, dict) else None

    cover_points = _sample_white_points(white_mask_clean, max_points=1500, rng=rng)
    tau_px = 6.0
    loss_info = _compute_loss_terms(
        H_best,
        dt=dt,
        Xw=Xw_full,
        weights=w_full,
        cover_points=cover_points,
        floor_bbox=floor_bbox,
        tau_px=tau_px,
        dt_oob=dt_oob,
    )
    p90_raw = float(
        loss_info.get("sample_dist_p90_raw_weighted")
        or loss_info.get("sample_dist_p90_raw")
        or loss_info.get("sample_dist_p90")
        or 999.0
    )
    cover_ratio = float(loss_info.get("cover_ratio", 0.0))
    inlier_ratio = float(loss_info.get("inlier_ratio", 0.0))
    cost = 0.6 * p90_raw + 0.4 * (1.0 - cover_ratio) * 50.0 + (1.0 - inlier_ratio) * 30.0

    conf_terms = _confidence_terms(inlier_ratio, cover_ratio, p90_raw, float(loss_info.get("area_ratio", 0.0)))
    conf = float(conf_terms["confidence"])
    conf_auto = float(conf)
    fit_conf_ok_thr = float(_env_float("BADC_FIT_CONF_OK_THR", 0.35))
    reason = "OK" if conf >= fit_conf_ok_thr else "R_fit_poor"
    dt_oob_count = int(loss_info.get("dt_oob_count", 0))
    num_samples = int(loss_info.get("num_samples", 0))
    dt_oob_ratio = float(dt_oob_count) / float(max(num_samples, 1))
    if dt_oob_ratio > 0.2:
        reason = "R_dt_oob"
        conf = 0.0
        conf_terms = {"confidence": 0.0, "s_inlier": 0.0, "s_cover": 0.0, "s_p90": 0.0, "s_area": 0.0}

    corners_uv = project_points(H_best, corners_world)
    ordered_corners = _order_corners_lb_rb_rt_lt(corners_uv)
    span_ok, span_metrics = _quad_span_ok(ordered_corners, w, h)
    base_metrics.update(span_metrics)
    ordered_clip = ordered_corners.copy()
    ordered_clip[:, 0] = np.clip(ordered_clip[:, 0], 0.0, float(w - 1))
    ordered_clip[:, 1] = np.clip(ordered_clip[:, 1], 0.0, float(h - 1))
    selected_area_ratio_guard = float(_quad_area(ordered_clip) / float(max(h * w, 1)))
    min_selected_area_ratio_guard = float(_env_float("BADC_FIT_MIN_SELECTED_AREA_RATIO", 0.12))
    min_bottom_span_floor_ratio_guard_base = float(
        _env_float("BADC_FIT_MIN_BOTTOM_SPAN_FLOOR_RATIO", 0.45)
    )
    min_bottom_span_floor_ratio_guard = float(
        max(
            0.28,
            min_bottom_span_floor_ratio_guard_base * (0.7 if int(relaxed_pass) > 0 else 1.0),
        )
    )
    max_top_edge_floor_ratio_guard = float(
        _env_float("BADC_FIT_MAX_TOP_EDGE_FLOOR_RATIO", 0.60)
    )
    bottom_span_floor_ratio_guard = _bottom_span_floor_ratio(ordered_corners, floor_bbox)
    top_edge_floor_ratio_guard = _top_edge_floor_ratio(ordered_corners, floor_bbox)
    base_metrics["bottom_span_floor_ratio"] = float(bottom_span_floor_ratio_guard)
    base_metrics["bottom_span_floor_ratio_min"] = float(min_bottom_span_floor_ratio_guard)
    base_metrics["bottom_span_floor_ratio_min_base"] = float(min_bottom_span_floor_ratio_guard_base)
    base_metrics["selected_area_ratio_guard"] = float(selected_area_ratio_guard)
    base_metrics["selected_area_ratio_guard_min"] = float(min_selected_area_ratio_guard)
    base_metrics["top_edge_floor_ratio"] = float(top_edge_floor_ratio_guard)
    base_metrics["top_edge_floor_ratio_max"] = float(max_top_edge_floor_ratio_guard)
    corners_out = ordered_corners.astype(np.float32)
    deg_bbox_w = float(np.max(ordered_corners[:, 0]) - np.min(ordered_corners[:, 0]))
    deg_bbox_h = float(np.max(ordered_corners[:, 1]) - np.min(ordered_corners[:, 1]))
    deg_min_edge = float(np.min(_quad_edges(ordered_corners)))
    base_metrics["degenerate_predicted_corners_bbox"] = [float(deg_bbox_w), float(deg_bbox_h)]
    base_metrics["degenerate_predicted_corners_min_edge"] = float(deg_min_edge)
    if deg_bbox_w < 0.08 * float(w) or deg_bbox_h < 0.06 * float(h) or (
        deg_min_edge < 0.05 * float(min(h, w))
    ):
        reason = "R_degenerate_quad"
        conf = 0.0
        conf_terms = {"confidence": 0.0, "s_inlier": 0.0, "s_cover": 0.0, "s_p90": 0.0, "s_area": 0.0}
        corners_out = None
    if corners_out is not None and bottom_span_floor_ratio_guard < min_bottom_span_floor_ratio_guard:
        reason = "R_bottom_span_small"
        conf = 0.0
        conf_terms = {"confidence": 0.0, "s_inlier": 0.0, "s_cover": 0.0, "s_p90": 0.0, "s_area": 0.0}
        corners_out = None
    if corners_out is not None and selected_area_ratio_guard < min_selected_area_ratio_guard:
        reason = "R_area_small"
        conf = 0.0
        conf_terms = {"confidence": 0.0, "s_inlier": 0.0, "s_cover": 0.0, "s_p90": 0.0, "s_area": 0.0}
        corners_out = None
    if corners_out is not None and top_edge_floor_ratio_guard > max_top_edge_floor_ratio_guard:
        reason = "R_top_edge_too_low"
        conf = 0.0
        conf_terms = {"confidence": 0.0, "s_inlier": 0.0, "s_cover": 0.0, "s_p90": 0.0, "s_area": 0.0}
        corners_out = None
    if corners_out is not None and isinstance(semantic_meta_main, dict) and not bool(semantic_meta_main.get("hard_pass", True)):
        bypass_conf = float(_env_float("BADC_SEMANTIC_HARD_BYPASS_CONF", 0.65))
        bypass_inlier = float(_env_float("BADC_SEMANTIC_HARD_BYPASS_INLIER", 0.70))
        bypass_cover = float(_env_float("BADC_SEMANTIC_HARD_BYPASS_COVER", 0.70))
        hard_bypass = bool(
            float(conf) >= bypass_conf
            and float(inlier_ratio) >= bypass_inlier
            and float(cover_ratio) >= bypass_cover
        )
        base_metrics["semantic_hard_bypass"] = bool(hard_bypass)
        base_metrics["semantic_hard_bypass_conf"] = float(bypass_conf)
        base_metrics["semantic_hard_bypass_inlier"] = float(bypass_inlier)
        base_metrics["semantic_hard_bypass_cover"] = float(bypass_cover)
        if not hard_bypass:
            reason = str(semantic_meta_main.get("hard_reason") or "R_semantic")
            conf = 0.0
            conf_terms = {"confidence": 0.0, "s_inlier": 0.0, "s_cover": 0.0, "s_p90": 0.0, "s_area": 0.0}
            corners_out = None
    if corners_out is not None and not manual:
        fb_w = max(1, int(floor_bbox[2] - floor_bbox[0]))
        fb_h = max(1, int(floor_bbox[3] - floor_bbox[1]))
        mx = max(24, int(round(0.05 * float(fb_w))))
        my = max(24, int(round(0.05 * float(fb_h))))
        if not _corners_inside_floor_roi(corners_out, floor_bbox, margin_x=mx, margin_y=my):
            base_metrics["corners_outside_floor_roi"] = True
    pred_corners = _format_predicted_corners(ordered_corners, frame_bgr.shape[1], frame_bgr.shape[0])
    err_map = _corner_errors_to_manual(ordered_corners, manual.get("court_corners") if manual else None)
    err_lt = err_map.get("err_LT")
    err_rt = err_map.get("err_RT")
    err_rb = err_map.get("err_RB")
    err_vals = [v for v in [err_lt, err_rt, err_rb] if v is not None]
    err_mean = float(np.mean(err_vals)) if err_vals else 999.0
    conf_geom = float(math.exp(-err_mean / 200.0)) if manual_vis else 1.0
    if best_anchor_score is not None:
        conf_geom = max(conf_geom, float(best_anchor_score))
    conf_support = float(min(1.0, max(0.0, 0.5 * (inlier_ratio + cover_ratio))))
    conf_visible = float(sum(1 for v in pred_corners if v.get("visible")) / 4.0)
    anchor_prior = 0.0
    anchor_prior_enabled = False
    used_anchor_prior = False
    if best_anchor_score is not None and inlier_ratio >= 0.20 and p90_raw <= 80.0:
        anchor_prior = float(0.35 * float(best_anchor_score))
        anchor_prior_enabled = True
        used_anchor_prior = True
    conf_final = max(conf_auto, conf_support * conf_geom * conf_visible, anchor_prior)
    conf = conf_final
    if reason in ("OK", "R_fit_poor"):
        reason = "OK" if conf_final >= fit_conf_ok_thr else "R_fit_poor"
    if reason != "OK":
        corners_out = None
    base_metrics["fit_conf_ok_thr"] = float(fit_conf_ok_thr)
    if reason == "OK" and corners_out is None:
        corners_out = ordered_corners.astype(np.float32)
    if manual_vis and not base_metrics.get("manual_override", False):
        if base_metrics.get("manual_anchor_violation") or (
            err_rt is not None and err_rt > 120.0
        ) or (err_rb is not None and err_rb > 150.0):
            reason = "R_manual_anchor_violation"
            conf = 0.0
            conf_terms = {"confidence": 0.0, "s_inlier": 0.0, "s_cover": 0.0, "s_p90": 0.0, "s_area": 0.0}
    if reason == "OK":
        reject_reason = base_metrics.get("reject_reason")
        if isinstance(reject_reason, str) and any(tag in reject_reason for tag in ["loosen", "fallback", "override"]):
            base_metrics["reject_reason"] = None
    if reason != "OK" and manual_quad is not None:
        H_manual = cv2.getPerspectiveTransform(get_bwf_corners().astype(np.float32), manual_quad.astype(np.float32))
        manual_loss = _compute_loss_terms(
            H_manual,
            dt=dt,
            Xw=Xw_full,
            weights=w_full,
            cover_points=cover_points,
            floor_bbox=floor_bbox,
            tau_px=tau_px,
            dt_oob=dt_oob,
        )
        p90_manual = float(
            manual_loss.get("sample_dist_p90_raw_weighted")
            or manual_loss.get("sample_dist_p90_raw")
            or manual_loss.get("sample_dist_p90")
            or 999.0
        )
        cover_manual = float(manual_loss.get("cover_ratio", 0.0))
        inlier_manual = float(manual_loss.get("inlier_ratio", 0.0))
        conf_manual_terms = _confidence_terms(
            inlier_manual, cover_manual, p90_manual, float(manual_loss.get("area_ratio", 0.0))
        )
        base_metrics["manual_candidate_confidence"] = float(conf_manual_terms["confidence"])
        base_metrics["manual_candidate_cost"] = float(
            0.6 * p90_manual + 0.4 * (1.0 - cover_manual) * 50.0 + (1.0 - inlier_manual) * 30.0
        )

    debug_image = draw_debug_overlay(
        frame_bgr,
        white_mask_clean,
        H_best,
        conf=float(conf),
        cost_total=float(cost),
        s_inlier=float(conf_terms["s_inlier"]),
        s_cover=float(conf_terms["s_cover"]),
        s_p90=float(conf_terms["s_p90"]),
        s_area=float(conf_terms["s_area"]),
        inlier_ratio=float(inlier_ratio),
        mean_dist_px=float(loss_info.get("sample_dist_mean") or 0.0),
        p90_dist_px=float(p90_raw),
        white_mask_ratio=float(clean_ratio),
        cover_ratio=float(cover_ratio),
        l_dist=float(loss_info.get("l_dist", 0.0)),
        l_cover=float(loss_info.get("l_cover", 0.0)),
        l_reg=float(loss_info.get("l_reg", 0.0)),
        area_ratio=float(loss_info.get("area_ratio", 0.0)),
        dt_min=dt_min,
        dt_mean=dt_mean,
        dt_p90=dt_p90,
        sample_dist_mean=loss_info.get("sample_dist_mean"),
        sample_dist_p50=loss_info.get("sample_dist_p50"),
        sample_dist_p90=loss_info.get("sample_dist_p90_raw_weighted")
        or loss_info.get("sample_dist_p90_raw")
        or loss_info.get("sample_dist_p90"),
        sample_dist_max=loss_info.get("sample_dist_max"),
        sample_uv=loss_info.get("sample_uv", np.zeros((0, 2), dtype=np.float32)),
        sample_dist=loss_info.get("sample_dists", np.zeros((0,), dtype=np.float32)),
        tau_px=3.0,
    )

    hough_lines_img = frame_bgr.copy()
    for x1, y1, x2, y2 in linesA:
        cv2.line(hough_lines_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
    for x1, y1, x2, y2 in linesB:
        cv2.line(hough_lines_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)

    metrics = {
        **base_metrics,
        "H": [float(v) for v in H_best.reshape(-1)],
        "cost_total": float(cost),
        "confidence": float(conf),
        "s_inlier": float(conf_terms["s_inlier"]),
        "s_cover": float(conf_terms["s_cover"]),
        "s_p90": float(conf_terms["s_p90"]),
        "s_area": float(conf_terms["s_area"]),
        "inlier_ratio": float(inlier_ratio),
        "mean_dist_px": float(loss_info.get("sample_dist_mean") or 0.0),
        "p90_dist_px": float(p90_raw),
        "num_inliers": int(loss_info.get("num_inliers", 0)),
        "num_valid_samples": int(loss_info.get("num_valid_samples", 0)),
        "num_samples": int(loss_info.get("num_samples", Xw_full.shape[0])),
        "dt_oob_count": int(loss_info.get("dt_oob_count", 0)),
        "dt_oob_ratio": float(dt_oob_ratio),
        "tau_px": float(tau_px),
        "l_dist": float(loss_info.get("l_dist", 0.0)),
        "l_cover": float(loss_info.get("l_cover", 0.0)),
        "l_reg": float(loss_info.get("l_reg", 0.0)),
        "area_ratio": float(loss_info.get("area_ratio", 0.0)),
        "cover_ratio": float(cover_ratio),
        "sample_dist_p90": float(p90_raw),
        "num_hough_lines_a": int(len(linesA)),
        "num_hough_lines_b": int(len(linesB)),
        "predicted_corners": pred_corners,
        "err_LT": err_lt,
        "err_RT": err_rt,
        "err_RB": err_rb,
        "confidence_auto": float(conf_auto),
        "confidence_final": float(conf),
        "conf_geom": float(conf_geom),
        "conf_support": float(conf_support),
        "conf_visible": float(conf_visible),
        "best_anchor_score": float(best_anchor_score) if best_anchor_score is not None else None,
        "anchor_prior_enabled": bool(anchor_prior_enabled),
        "used_anchor_prior": bool(used_anchor_prior),
    }
    if metrics.get("selected_component_id") is None:
        metrics["selected_component_id"] = int(base_metrics.get("component_id") or 0)

    # Generate raw-floor debug overlays even when main Hough path succeeds, so downstream
    # report assets (raw_floor_* overlays) are populated.
    if raw_debug_save is None and save_raw_debug:
        try:
            _Hdbg, _mdbg, raw_debug_save = _fit_court_homography_from_raw_floor_debug(
                white_mask_raw_floor_postblob,
                frame_bgr.shape,
                floor_roi_mask=floor_mask,
                white_mask_raw_floor_noblob=white_mask_raw_floor_noblob,
            )
        except Exception:
            raw_debug_save = None

    return CourtFitResult(
        H=H_best,
        corners=corners_out,
        confidence=conf,
        metrics=metrics,
        reason=reason,
        debug_image=debug_image,
        white_mask=white_mask_clean,
        debug_image_init=debug_image,
        white_mask_raw=white_mask_raw_full,
        white_mask_clean=white_mask_clean,
        white_mask_raw_full=white_mask_raw_full,
        white_mask_raw_floor=white_mask_raw_floor,
        white_mask_raw_floor_noblob=white_mask_raw_floor_noblob,
        white_mask_raw_floor_preblob=white_mask_raw_floor_preblob,
        white_mask_raw_floor_postblob=white_mask_raw_floor_postblob,
        floor_roi_mask=floor_mask,
        floor_roi_overlay=floor_roi_overlay,
        seed_bottom_mask=floor_debug.get("seed_bottom_mask"),
        green_mask=floor_debug.get("green_mask"),
        largest_cc_mask=floor_debug.get("largest_cc_mask"),
        exg_row_plot=floor_debug.get("exg_row_plot"),
        linepix_mask_pre=linepix_mask_pre,
        floor_gate_mask=floor_gate_mask,
        linepix_mask=linepix_mask,
        linepix_overlay=linepix_overlay,
        ransac_lines_img=ransac_lines_img,
        dt_debug=dt_debug,
        ori_mask_a=maskA,
        ori_mask_b=maskB,
        hough_lines_img=hough_lines_img,
        raw_floor_hough_lines_img=(raw_debug_save or {}).get("hough_lines"),
        raw_floor_dt_debug=(raw_debug_save or {}).get("dt"),
        raw_floor_model_overlay=(raw_debug_save or {}).get("overlay_raw"),
        raw_floor_top5_overlay=(raw_debug_save or {}).get("overlay_top5"),
        raw_floor_preprocessed=(raw_debug_save or {}).get("preprocessed"),
        raw_floor_edges=(raw_debug_save or {}).get("edges"),
        raw_floor_hough_lines_a=(raw_debug_save or {}).get("hough_lines_a"),
        raw_floor_hough_lines_b=(raw_debug_save or {}).get("hough_lines_b"),
    )


def fit_court_homography(
    frame_bgr: np.ndarray,
    method: str = "lsd",
    allow_fallback_lsd: bool = True,
    person_boxes: Optional[Sequence[Sequence[float]]] = None,
) -> CourtFitResult:
    method_key = str(method).lower()
    if method_key in ("hough_orient", "hough"):
        res = _fit_court_homography_hough(frame_bgr, person_boxes=person_boxes)
        res.method_used = "hough_orient"
        if isinstance(res.metrics, dict):
            res.metrics["method_used"] = "hough_orient"
        if res.reason == "OK" or not allow_fallback_lsd:
            return res
        res2 = _fit_court_homography_lsd(frame_bgr)
        res2.method_used = "lsd_fallback"
        if isinstance(res2.metrics, dict):
            res2.metrics["method_used"] = "lsd_fallback"
        return res2

    res = _fit_court_homography_lsd(frame_bgr)
    res.method_used = "lsd"
    if isinstance(res.metrics, dict):
        res.metrics["method_used"] = "lsd"
    return res


def _fit_court_homography_lsd(frame_bgr: np.ndarray) -> CourtFitResult:
    rng = np.random.default_rng(0)
    (
        white_mask_raw,
        white_mask_clean,
        floor_bbox,
        max_cc_area_raw,
        max_cc_area_clean,
    ) = build_white_mask(frame_bgr)
    raw_sum, raw_ratio, _ = _white_mask_stats(white_mask_raw)
    clean_sum, clean_ratio, _ = _white_mask_stats(white_mask_clean)
    dt = build_distance_transform(white_mask_clean)
    dt_min = float(np.min(dt)) if dt.size > 0 else None
    dt_mean = float(np.mean(dt)) if dt.size > 0 else None
    dt_p90 = float(np.percentile(dt, 90)) if dt.size > 0 else None
    dt_oob = float(max(dt_p90 or 0.0, 15.0))
    x1, y1, x2, y2 = floor_bbox
    x1 = int(np.clip(x1, 0, dt.shape[1] - 1))
    x2 = int(np.clip(x2, x1 + 1, dt.shape[1]))
    y1 = int(np.clip(y1, 0, dt.shape[0] - 1))
    y2 = int(np.clip(y2, y1 + 1, dt.shape[0]))
    roi_h = max(1, y2 - y1)
    roi_w = max(1, x2 - x1)
    rand_n = 1000
    xs = rng.integers(x1, x1 + roi_w, size=rand_n)
    ys = rng.integers(y1, y1 + roi_h, size=rand_n)
    dt_rand = dt[ys, xs].astype(np.float32) if dt.size > 0 else np.array([], dtype=np.float32)
    dt_rand_mean = float(np.mean(dt_rand)) if dt_rand.size > 0 else None
    dt_rand_p90 = float(np.percentile(dt_rand, 90)) if dt_rand.size > 0 else None

    corners_world = get_bwf_corners()
    Xw_ransac, w_ransac, names_ransac = sample_model_points(points_per_meter=30.0, min_weight=0.6)
    Xw_full, w_full, names_full = sample_model_points(points_per_meter=30.0, min_weight=0.3)
    w_ransac, role_w_cfg_ransac = _apply_model_role_weights(names_ransac, w_ransac)
    w_full, role_w_cfg_full = _apply_model_role_weights(names_full, w_full)

    base_metrics = {
        "model": MODEL_ID,
        "dt_from": "white_mask_frame",
        "white_mask_sum": int(clean_sum),
        "white_mask_ratio": float(clean_ratio),
        "white_mask_raw_sum": int(raw_sum),
        "white_mask_raw_ratio": float(raw_ratio),
        "white_mask_clean_sum": int(clean_sum),
        "white_mask_clean_ratio": float(clean_ratio),
        "mask_raw_ratio": float(raw_ratio),
        "mask_clean_ratio": float(clean_ratio),
        "max_cc_area_raw": int(max_cc_area_raw),
        "max_cc_area_clean": int(max_cc_area_clean),
        "max_cc_area": int(max_cc_area_clean),
        "floor_bbox": [int(v) for v in floor_bbox],
        "dt_min": dt_min,
        "dt_mean": dt_mean,
        "dt_p90": dt_p90,
        "dt_rand_mean": dt_rand_mean,
        "dt_rand_p90": dt_rand_p90,
        "debug_image_path": None,
        "white_mask_path": None,
        "white_mask_raw_path": None,
        "white_mask_clean_path": None,
        "lsd_dirA_path": None,
        "lsd_dirB_path": None,
        "debug_init_path": None,
        "dt_debug_path": None,
        "model_role_weights": role_w_cfg_full,
        "model_role_weights_ransac": role_w_cfg_ransac,
    }

    if clean_ratio < 0.001 or clean_ratio > 0.25:
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
            white_mask=white_mask_clean,
            white_mask_raw=white_mask_raw,
            white_mask_clean=white_mask_clean,
            dt_debug=None,
        )

    if dt_rand_p90 is not None and dt_rand_mean is not None:
        if dt_rand_p90 < 0.5 or dt_rand_mean < 0.2:
            dt_vis = cv2.normalize(dt, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
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
                reason="R_dt_invalid",
                debug_image=None,
                white_mask=white_mask_clean,
                white_mask_raw=white_mask_raw,
                white_mask_clean=white_mask_clean,
                debug_image_init=None,
                lsd_lines_a=None,
                lsd_lines_b=None,
                dt_debug=dt_vis,
            )

    cover_points_ransac = _sample_white_points(white_mask_clean, max_points=600, rng=rng)
    cover_points = _sample_white_points(white_mask_clean, max_points=1500, rng=rng)

    H_init, init_info = ransac_init_homography(
        white_mask=white_mask_clean,
        white_mask_raw=white_mask_raw,
        dt=dt,
        world_corners=corners_world,
        Xw=Xw_ransac,
        weights=w_ransac,
        cover_points=cover_points_ransac,
        floor_bbox=floor_bbox,
        iters=2000,
        tau_px=3.0,
        min_area_ratio=0.01,
        dt_oob=dt_oob,
    )
    lsd_img_a, lsd_img_b = None, None
    if isinstance(init_info, dict):
        lsd_img_a, lsd_img_b = _draw_lsd_segment_lists(
            frame_bgr, init_info.get("lsd_lines_a"), init_info.get("lsd_lines_b")
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
        if isinstance(init_info, dict):
            metrics.update(
                {
                    "vp_long": init_info.get("vp_long"),
                    "vp_short": init_info.get("vp_short"),
                    "num_lsd_raw": init_info.get("num_lsd_raw"),
                    "num_lsd_kept": init_info.get("num_lsd_kept"),
                    "dirA_count": init_info.get("dirA_count"),
                    "dirB_count": init_info.get("dirB_count"),
                    "sum_len_a": init_info.get("sum_len_a"),
                    "sum_len_b": init_info.get("sum_len_b"),
                    "cluster_ratio": init_info.get("cluster_ratio"),
                    "mean_angle_a": init_info.get("mean_angle_a"),
                    "mean_angle_b": init_info.get("mean_angle_b"),
                }
            )
        return CourtFitResult(
            H=None,
            corners=None,
            confidence=0.0,
            metrics=metrics,
            reason="R_fit_poor",
            debug_image=None,
            white_mask=white_mask_clean,
            white_mask_raw=white_mask_raw,
            white_mask_clean=white_mask_clean,
            lsd_lines_a=lsd_img_a,
            lsd_lines_b=lsd_img_b,
            dt_debug=None,
        )

    init_p90 = float(
        init_info.get("sample_dist_p90_raw_weighted")
        or init_info.get("sample_dist_p90_raw")
        or init_info.get("sample_dist_p90")
        or 999.0
    )
    init_conf_terms = _confidence_terms(
        float(init_info.get("inlier_ratio", 0.0)),
        float(init_info.get("cover_ratio", 0.0)),
        init_p90,
        float(init_info.get("area_ratio", 0.0)),
    )
    debug_image_init = draw_debug_overlay(
        frame_bgr,
        white_mask_clean,
        H_init,
        conf=float(init_conf_terms["confidence"]),
        cost_total=float(init_info.get("loss", 0.0)),
        s_inlier=float(init_conf_terms["s_inlier"]),
        s_cover=float(init_conf_terms["s_cover"]),
        s_p90=float(init_conf_terms["s_p90"]),
        s_area=float(init_conf_terms["s_area"]),
        inlier_ratio=float(init_info.get("inlier_ratio", 0.0)),
        mean_dist_px=float(init_info.get("sample_dist_mean") or 0.0),
        p90_dist_px=float(
            init_info.get("sample_dist_p90_raw_weighted")
            or init_info.get("sample_dist_p90_raw")
            or init_info.get("sample_dist_p90")
            or 0.0
        ),
        white_mask_ratio=float(clean_ratio),
        cover_ratio=float(init_info.get("cover_ratio", 0.0)),
        l_dist=float(init_info.get("l_dist", 0.0)),
        l_cover=float(init_info.get("l_cover", 0.0)),
        l_reg=float(init_info.get("l_reg", 0.0)),
        area_ratio=float(init_info.get("area_ratio", 0.0)),
        dt_min=dt_min,
        dt_mean=dt_mean,
        dt_p90=dt_p90,
        sample_dist_mean=init_info.get("sample_dist_mean"),
        sample_dist_p50=init_info.get("sample_dist_p50"),
        sample_dist_p90=init_info.get("sample_dist_p90_raw_weighted")
        or init_info.get("sample_dist_p90_raw")
        or init_info.get("sample_dist_p90"),
        sample_dist_max=init_info.get("sample_dist_max"),
        sample_uv=init_info.get("sample_uv", np.zeros((0, 2), dtype=np.float32)),
        sample_dist=init_info.get("sample_dists", np.zeros((0,), dtype=np.float32)),
        tau_px=3.0,
    )

    H_ref, lm_cost = refine_homography_lm(
        H_init,
        dt,
        Xw_full,
        w_full,
        cover_points=cover_points,
        floor_bbox=floor_bbox,
        max_nfev=50,
        dt_oob=dt_oob,
    )
    loss_info = _compute_loss_terms(
        H_ref,
        dt=dt,
        Xw=Xw_full,
        weights=w_full,
        cover_points=cover_points,
        floor_bbox=floor_bbox,
        tau_px=3.0,
        dt_oob=dt_oob,
    )
    mean_dist = loss_info.get("sample_dist_mean")
    p90_clamped = loss_info.get("sample_dist_p90")
    p90_dist = (
        loss_info.get("sample_dist_p90_raw_weighted")
        or loss_info.get("sample_dist_p90_raw")
        or p90_clamped
    )
    inlier_ratio = float(loss_info.get("inlier_ratio", 0.0))
    cover_ratio = float(loss_info.get("cover_ratio", 0.0))
    area_ratio = float(loss_info.get("area_ratio", 0.0))
    p90_val = float(p90_dist) if p90_dist is not None else 999.0
    conf_terms = _confidence_terms(inlier_ratio, cover_ratio, p90_val, area_ratio)
    conf = float(conf_terms["confidence"])
    reason = "OK" if conf >= 0.35 else "R_fit_poor"
    dt_oob_count = int(loss_info.get("dt_oob_count", 0))
    num_samples = int(loss_info.get("num_samples", 0))
    dt_oob_ratio = float(dt_oob_count) / float(max(num_samples, 1))
    dt_debug = None
    if dt_oob_ratio > 0.2:
        reason = "R_dt_oob"
        conf = 0.0
        conf_terms = {"confidence": 0.0, "s_inlier": 0.0, "s_cover": 0.0, "s_p90": 0.0, "s_area": 0.0}
        dt_debug = cv2.normalize(dt, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    corners_uv = project_points(H_ref, corners_world)
    ordered_corners = _order_corners_lb_rb_rt_lt(corners_uv)
    span_ok, span_metrics = _quad_span_ok(ordered_corners, frame_bgr.shape[1], frame_bgr.shape[0])
    base_metrics.update(span_metrics)
    corners_out = ordered_corners.astype(np.float32)

    debug_image = draw_debug_overlay(
        frame_bgr,
        white_mask_clean,
        H_ref,
        conf=float(conf),
        cost_total=float(loss_info.get("loss", 0.0)),
        s_inlier=float(conf_terms["s_inlier"]),
        s_cover=float(conf_terms["s_cover"]),
        s_p90=float(conf_terms["s_p90"]),
        s_area=float(conf_terms["s_area"]),
        inlier_ratio=float(inlier_ratio),
        mean_dist_px=float(mean_dist) if mean_dist is not None else 0.0,
        p90_dist_px=float(p90_dist) if p90_dist is not None else 0.0,
        white_mask_ratio=float(clean_ratio),
        cover_ratio=float(loss_info.get("cover_ratio", 0.0)),
        l_dist=float(loss_info.get("l_dist", 0.0)),
        l_cover=float(loss_info.get("l_cover", 0.0)),
        l_reg=float(loss_info.get("l_reg", 0.0)),
        area_ratio=float(loss_info.get("area_ratio", 0.0)),
        dt_min=dt_min,
        dt_mean=dt_mean,
        dt_p90=dt_p90,
        sample_dist_mean=loss_info.get("sample_dist_mean"),
        sample_dist_p50=loss_info.get("sample_dist_p50"),
        sample_dist_p90=loss_info.get("sample_dist_p90_raw_weighted")
        or loss_info.get("sample_dist_p90_raw")
        or loss_info.get("sample_dist_p90"),
        sample_dist_max=loss_info.get("sample_dist_max"),
        sample_uv=loss_info.get("sample_uv", np.zeros((0, 2), dtype=np.float32)),
        sample_dist=loss_info.get("sample_dists", np.zeros((0,), dtype=np.float32)),
        tau_px=3.0,
    )

    metrics = {
        **base_metrics,
        "H": [float(v) for v in H_ref.reshape(-1)],
        "cost_total": float(loss_info.get("loss", 0.0)),
        "confidence": float(conf),
        "s_inlier": float(conf_terms["s_inlier"]),
        "s_cover": float(conf_terms["s_cover"]),
        "s_p90": float(conf_terms["s_p90"]),
        "s_area": float(conf_terms["s_area"]),
        "inlier_ratio": float(inlier_ratio),
        "mean_dist_px": float(mean_dist) if mean_dist is not None else None,
        "p90_dist_px": float(p90_dist) if p90_dist is not None else None,
        "p90_dist_px_clamped": float(p90_clamped) if p90_clamped is not None else None,
        "num_inliers": int(loss_info.get("num_inliers", 0)),
        "num_valid_samples": int(loss_info.get("num_valid_samples", 0)),
        "num_samples": int(loss_info.get("num_samples", Xw_full.shape[0])),
        "dt_oob_count": int(loss_info.get("dt_oob_count", 0)),
        "dt_oob_ratio": float(dt_oob_ratio),
        "tau_px": 3.0,
        "lm_cost": float(lm_cost),
        "l_dist": float(loss_info.get("l_dist", 0.0)),
        "l_cover": float(loss_info.get("l_cover", 0.0)),
        "l_reg": float(loss_info.get("l_reg", 0.0)),
        "area_ratio": float(loss_info.get("area_ratio", 0.0)),
        "cover_ratio": float(loss_info.get("cover_ratio", 0.0)),
        "cover_penalty": float(loss_info.get("cover_penalty", 0.0)),
        "inlier_dist_p90": loss_info.get("inlier_dist_p90"),
        "sample_dist_mean": loss_info.get("sample_dist_mean"),
        "sample_dist_p50": loss_info.get("sample_dist_p50"),
        "sample_dist_p90": loss_info.get("sample_dist_p90_raw_weighted")
        or loss_info.get("sample_dist_p90_raw")
        or loss_info.get("sample_dist_p90"),
        "sample_dist_max": loss_info.get("sample_dist_max"),
        "vp_long": init_info.get("vp_long") if isinstance(init_info, dict) else None,
        "vp_short": init_info.get("vp_short") if isinstance(init_info, dict) else None,
        "num_lsd_raw": init_info.get("num_lsd_raw") if isinstance(init_info, dict) else None,
        "num_lsd_kept": init_info.get("num_lsd_kept") if isinstance(init_info, dict) else None,
        "dirA_count": init_info.get("dirA_count") if isinstance(init_info, dict) else None,
        "dirB_count": init_info.get("dirB_count") if isinstance(init_info, dict) else None,
        "sum_len_a": init_info.get("sum_len_a") if isinstance(init_info, dict) else None,
        "sum_len_b": init_info.get("sum_len_b") if isinstance(init_info, dict) else None,
        "cluster_ratio": init_info.get("cluster_ratio") if isinstance(init_info, dict) else None,
        "mean_angle_a": init_info.get("mean_angle_a") if isinstance(init_info, dict) else None,
        "mean_angle_b": init_info.get("mean_angle_b") if isinstance(init_info, dict) else None,
        "init_ratio": init_info.get("init_ratio") if isinstance(init_info, dict) else None,
        "init_inside_ratio": init_info.get("init_inside_ratio") if isinstance(init_info, dict) else None,
        "init_cover": init_info.get("cover_ratio") if isinstance(init_info, dict) else None,
        "init_p90": (
            (
                init_info.get("sample_dist_p90_raw_weighted")
                or init_info.get("sample_dist_p90_raw")
            )
            if isinstance(init_info, dict)
            else None
        ),
        "init_area_ratio": init_info.get("area_ratio") if isinstance(init_info, dict) else None,
    }
    return CourtFitResult(
        H=H_ref,
        corners=corners_out,
        confidence=conf,
        metrics=metrics,
        reason=reason,
        debug_image=debug_image,
        white_mask=white_mask_clean,
        debug_image_init=debug_image_init,
        white_mask_raw=white_mask_raw,
        white_mask_clean=white_mask_clean,
        lsd_lines_a=lsd_img_a,
        lsd_lines_b=lsd_img_b,
        dt_debug=dt_debug,
    )
