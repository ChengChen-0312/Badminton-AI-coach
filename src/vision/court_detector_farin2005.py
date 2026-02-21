from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple

import cv2
import numpy as np

from src.geometry.homography import COURT_LENGTH_M, COURT_WIDTH_M
from src.vision.court_detector import CourtDetector, CourtLines


@dataclass(frozen=True)
class _LineModel:
    # Normalized line in ax + by + c = 0 form, with sqrt(a^2+b^2)=1
    a: float
    b: float
    c: float
    # A representative segment (for debug/visualization)
    p1: tuple[float, float]
    p2: tuple[float, float]
    # Orientation in degrees in [0, 180)
    theta_deg: float
    length: float
    inliers: int
    support: float  # 0..1 (inliers / sampled points)


@dataclass(frozen=True)
class _CourtCandidate:
    corners: np.ndarray  # [4,2] in LB,RB,RT,LT
    edge_support: list[float]
    tpl_f1: float
    model_error: float
    quad_area_norm: float
    near_baseline_y_norm: float
    span_x: float
    span_y: float
    aspect_ratio: float
    s_area: float
    s_near: float
    s_support: float
    s_tpl: float
    near_score: float
    method: str
    candidate_lines: list[dict[str, Any]]


class CourtDetectorFarin2005:
    """
    Farin et al. (2005) inspired court detector:
      1) court-line pixel detection (white-ish pixels on floor, exclude large blobs)
      2) RANSAC-based dominant line estimation (extract multiple line hypotheses)
      3) combinatorial model fitting (2 horizontal + 2 vertical line combination)
      4) quick rejection (area/aspect sanity)
      5) rank candidates and pick the nearest court (largest/lowest quad + best fit)

    Output corners MUST be in LB, RB, RT, LT order (image coords).
    """

    def __init__(
        self,
        # Line pixel detection
        white_s_max: int = 110,
        white_v_min: int = 160,
        top_crop_ratio: float = 0.20,
        morph_kernel: int = 5,
        morph_close_iter: int = 1,
        morph_open_iter: int = 1,
        blob_open_kernel: int = 41,
        floor_close_kernel: int = 35,
        floor_close_iter: int = 1,
        # RANSAC line extraction
        ransac_max_lines: int = 22,
        ransac_hypotheses: int = 60,
        ransac_dist_thresh: float = 3.0,
        ransac_min_inliers: int = 180,
        max_points: int = 6000,
        # Orientation + combinations
        angle_tol_deg: float = 18.0,
        topk_lines_per_dir: int = 8,
        min_sep_y_ratio: float = 0.25,
        min_sep_x_ratio: float = 0.25,
        max_candidates: int = 160,
        # Quick rejection tests (paper-style)
        min_quad_area_norm: float = 0.10,
        aspect_ratio_model_min_frac: float = 0.10,
        court_min_span_x_norm: float = 0.55,
        court_min_span_y_norm: float = 0.55,
        # Scoring thresholds
        min_edges_supported: int = 3,
        min_edge_support: float = 0.06,
        min_tpl_f1: float = 0.06,
        min_confidence: float = 0.60,
        court_tpl_weight: float = 0.30,
        # Template matching
        template_scale_px_per_m: int = 80,
        template_dilate: int = 3,
        # Debug
        support_dilate: int = 9,
        edge_sample_points: int = 200,
        seed: int = 0,
    ) -> None:
        self.white_s_max = int(white_s_max)
        self.white_v_min = int(white_v_min)
        self.top_crop_ratio = float(top_crop_ratio)
        self.morph_kernel = int(morph_kernel)
        self.morph_close_iter = int(morph_close_iter)
        self.morph_open_iter = int(morph_open_iter)
        self.blob_open_kernel = int(blob_open_kernel)
        self.floor_close_kernel = int(floor_close_kernel)
        self.floor_close_iter = int(floor_close_iter)

        self.ransac_max_lines = int(ransac_max_lines)
        self.ransac_hypotheses = int(ransac_hypotheses)
        self.ransac_dist_thresh = float(ransac_dist_thresh)
        self.ransac_min_inliers = int(ransac_min_inliers)
        self.max_points = int(max_points)

        self.angle_tol_deg = float(angle_tol_deg)
        self.topk_lines_per_dir = int(topk_lines_per_dir)
        self.min_sep_y_ratio = float(min_sep_y_ratio)
        self.min_sep_x_ratio = float(min_sep_x_ratio)
        self.max_candidates = int(max_candidates)

        self.min_quad_area_norm = float(min_quad_area_norm)
        self.aspect_ratio_model_min_frac = float(aspect_ratio_model_min_frac)
        self.court_min_span_x_norm = float(court_min_span_x_norm)
        self.court_min_span_y_norm = float(court_min_span_y_norm)
        self.min_edges_supported = int(min_edges_supported)
        self.min_edge_support = float(min_edge_support)
        self.min_tpl_f1 = float(min_tpl_f1)
        self.min_confidence = float(min_confidence)
        self.court_tpl_weight = float(court_tpl_weight)

        self.template_scale_px_per_m = int(template_scale_px_per_m)
        self.template_dilate = int(template_dilate)

        self.support_dilate = int(support_dilate)
        self.edge_sample_points = int(edge_sample_points)

        self._rng = np.random.default_rng(int(seed))
        self._template_cache: tuple[tuple[int, int], np.ndarray] | None = None
        # Reuse the existing, battle-tested floor estimation + adaptive line-mask
        # extraction from the heuristic detector to keep line density in a sane range.
        self._mask_helper = CourtDetector(
            white_s_max=self.white_s_max,
            white_v_min=self.white_v_min,
            top_crop_ratio=self.top_crop_ratio,
            morph_kernel=self.morph_kernel,
            morph_close_iter=self.morph_close_iter,
            morph_open_iter=self.morph_open_iter,
            blob_open_kernel=self.blob_open_kernel,
            floor_close_kernel=self.floor_close_kernel,
            floor_close_iter=self.floor_close_iter,
        )

        # Diagnostics from the last detection attempt.
        self.last_confidence: float | None = None
        self.last_reason: str | None = None
        self.last_edge_support: list[float] | None = None
        self.last_method: str | None = None
        self.last_metrics: dict[str, Any] | None = None

    @staticmethod
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

    @staticmethod
    def _quad_area(pts_xy: np.ndarray) -> float:
        pts = np.array(pts_xy, dtype=np.float32).reshape(-1, 1, 2)
        return float(abs(cv2.contourArea(pts)))

    @staticmethod
    def _is_convex_quad(pts_xy: np.ndarray) -> bool:
        pts = np.array(pts_xy, dtype=np.float32).reshape(-1, 1, 2)
        try:
            return bool(cv2.isContourConvex(pts))
        except Exception:
            return False

    @staticmethod
    def _angle_distance_deg(a: float, b: float) -> float:
        d = abs(float(a) - float(b)) % 180.0
        return float(min(d, 180.0 - d))

    @staticmethod
    def _line_intersection(l1: _LineModel, l2: _LineModel) -> tuple[float, float] | None:
        A = np.array([[l1.a, l1.b], [l2.a, l2.b]], dtype=np.float64)
        det = float(np.linalg.det(A))
        if abs(det) < 1e-9:
            return None
        b = np.array([-l1.c, -l2.c], dtype=np.float64)
        x, y = np.linalg.solve(A, b).tolist()
        return float(x), float(y)

    @staticmethod
    def _line_y_at_x(line: _LineModel, x: float) -> float | None:
        if abs(line.b) < 1e-6:
            return None
        return float((-line.a * float(x) - line.c) / line.b)

    @staticmethod
    def _line_x_at_y(line: _LineModel, y: float) -> float | None:
        if abs(line.a) < 1e-6:
            return None
        return float((-line.b * float(y) - line.c) / line.a)

    def _ensure_template(self) -> tuple[tuple[int, int], np.ndarray]:
        scale = max(20, int(self.template_scale_px_per_m))
        w_px = int(round(float(COURT_WIDTH_M) * float(scale))) + 1
        h_px = int(round(float(COURT_LENGTH_M) * float(scale))) + 1
        key = (w_px, h_px)
        if self._template_cache is not None and self._template_cache[0] == key:
            return self._template_cache

        tmpl = np.zeros((h_px, w_px), dtype=np.uint8)

        def _x(m: float) -> int:
            return int(round(float(m) * float(scale)))

        def _y(m: float) -> int:
            # meters y=0 at near baseline (bottom), y increases towards far baseline (top)
            return int(h_px - 1 - round(float(m) * float(scale)))

        thick = 2
        cv2.rectangle(tmpl, (0, 0), (w_px - 1, h_px - 1), 255, thickness=thick)
        net_y = _y(float(COURT_LENGTH_M) / 2.0)
        cv2.line(tmpl, (0, net_y), (w_px - 1, net_y), 255, thickness=thick)
        short = 1.98
        cv2.line(
            tmpl,
            (0, _y(float(COURT_LENGTH_M) / 2.0 - short)),
            (w_px - 1, _y(float(COURT_LENGTH_M) / 2.0 - short)),
            255,
            thickness=thick,
        )
        cv2.line(
            tmpl,
            (0, _y(float(COURT_LENGTH_M) / 2.0 + short)),
            (w_px - 1, _y(float(COURT_LENGTH_M) / 2.0 + short)),
            255,
            thickness=thick,
        )
        long_d = 0.76
        cv2.line(tmpl, (0, _y(long_d)), (w_px - 1, _y(long_d)), 255, thickness=thick)
        cv2.line(
            tmpl,
            (0, _y(float(COURT_LENGTH_M) - long_d)),
            (w_px - 1, _y(float(COURT_LENGTH_M) - long_d)),
            255,
            thickness=thick,
        )
        singles_inset = (float(COURT_WIDTH_M) - 5.18) / 2.0
        x_l = _x(singles_inset)
        x_r = _x(float(COURT_WIDTH_M) - singles_inset)
        cv2.line(tmpl, (x_l, 0), (x_l, h_px - 1), 255, thickness=thick)
        cv2.line(tmpl, (x_r, 0), (x_r, h_px - 1), 255, thickness=thick)
        x_c = _x(float(COURT_WIDTH_M) / 2.0)
        cv2.line(tmpl, (x_c, _y(long_d)), (x_c, _y(float(COURT_LENGTH_M) / 2.0 - short)), 255, thickness=thick)
        cv2.line(tmpl, (x_c, _y(float(COURT_LENGTH_M) / 2.0 + short)), (x_c, _y(float(COURT_LENGTH_M) - long_d)), 255, thickness=thick)

        self._template_cache = (key, tmpl)
        return self._template_cache

    def _template_f1_score(self, line_mask: np.ndarray, corners_xy: np.ndarray) -> float:
        if line_mask is None or line_mask.size == 0:
            return 0.0
        h, w = line_mask.shape[:2]
        corners = np.array(corners_xy, dtype=np.float32).reshape(4, 2)

        poly = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(poly, np.rint(corners).astype(np.int32), 255)
        masked = cv2.bitwise_and(line_mask, poly)

        (tw, th), tmpl = self._ensure_template()
        dst = np.array(
            [[0.0, float(th - 1)], [float(tw - 1), float(th - 1)], [float(tw - 1), 0.0], [0.0, 0.0]],
            dtype=np.float32,
        )
        try:
            H = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
        except Exception:
            return 0.0

        warped = cv2.warpPerspective(masked, H, dsize=(tw, th), flags=cv2.INTER_NEAREST, borderValue=0)
        warped_b = (warped > 0).astype(np.uint8)
        tmpl_b = (tmpl > 0).astype(np.uint8)

        k = max(1, int(self.template_dilate))
        if k > 1:
            kernel = np.ones((k, k), np.uint8)
            warped_b = cv2.dilate(warped_b, kernel, iterations=1)
            tmpl_b = cv2.dilate(tmpl_b, kernel, iterations=1)

        tp = int(np.count_nonzero((warped_b > 0) & (tmpl_b > 0)))
        fp = int(np.count_nonzero((warped_b > 0) & (tmpl_b == 0)))
        fn = int(np.count_nonzero((warped_b == 0) & (tmpl_b > 0)))

        precision = float(tp) / float(max(tp + fp, 1))
        recall = float(tp) / float(max(tp + fn, 1))
        if precision + recall < 1e-9:
            return 0.0
        return float(2.0 * precision * recall / (precision + recall))

    def _edge_support_ratios(self, line_mask: np.ndarray, corners_xy: np.ndarray) -> list[float]:
        h, w = line_mask.shape[:2]
        if self.support_dilate > 1:
            k = int(self.support_dilate)
            support = cv2.dilate(line_mask, np.ones((k, k), np.uint8), iterations=1)
        else:
            support = line_mask

        corners = np.array(corners_xy, dtype=np.float32).reshape(4, 2)
        edges = [
            (corners[0], corners[1]),
            (corners[1], corners[2]),
            (corners[2], corners[3]),
            (corners[3], corners[0]),
        ]
        ratios: list[float] = []
        n = max(20, int(self.edge_sample_points))
        for p1, p2 in edges:
            xs = np.linspace(float(p1[0]), float(p2[0]), n)
            ys = np.linspace(float(p1[1]), float(p2[1]), n)
            ix = np.clip(np.rint(xs).astype(np.int32), 0, w - 1)
            iy = np.clip(np.rint(ys).astype(np.int32), 0, h - 1)
            hits = int(np.count_nonzero(support[iy, ix]))
            ratios.append(float(hits) / float(n))
        return ratios

    def _line_pixel_mask(self, frame_rgb: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
        """
        Court-line pixel detection: white-ish pixels with local contrast, within the
        floor region, plus morphology to keep thin structures and drop big blobs.
        """
        h, w = frame_rgb.shape[:2]
        hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)

        floor_region = self._mask_helper._estimate_floor_region(hsv)
        top = int(round(float(h) * float(self.top_crop_ratio)))
        line_mask, stats = self._mask_helper._compute_line_mask(hsv, floor_region=floor_region, top_crop_px=top)

        floor_bbox = CourtDetector._bbox_from_mask(floor_region) if floor_region is not None else None
        if floor_bbox is not None:
            x1, y1, x2, y2 = floor_bbox
            stats = {
                **stats,
                "floor_bbox_is_floor": 1.0,
                "floor_bbox_x1": float(x1) / float(max(w, 1)),
                "floor_bbox_y1": float(y1) / float(max(h, 1)),
                "floor_bbox_x2": float(x2) / float(max(w, 1)),
                "floor_bbox_y2": float(y2) / float(max(h, 1)),
            }
        else:
            stats = {**stats, "floor_bbox_is_floor": 0.0}

        return line_mask, stats

    def _ransac_extract_lines(self, pts_xy: np.ndarray) -> list[_LineModel]:
        """
        RANSAC dominant line extraction from point samples.
        Returns multiple lines with refined (TLS) parameters.
        """
        pts = np.array(pts_xy, dtype=np.float32).reshape(-1, 2)
        if pts.shape[0] < max(50, self.ransac_min_inliers):
            return []

        # Downsample for speed
        if pts.shape[0] > int(self.max_points):
            idx = self._rng.choice(pts.shape[0], size=int(self.max_points), replace=False)
            pts = pts[idx]

        total = int(pts.shape[0])
        lines: list[_LineModel] = []
        remaining = pts

        for _ in range(int(self.ransac_max_lines)):
            n = int(remaining.shape[0])
            if n < int(self.ransac_min_inliers):
                break

            best_inliers = None
            best_score = -1
            best_abcn = None

            for _h in range(int(self.ransac_hypotheses)):
                i1, i2 = self._rng.integers(0, n, size=2).tolist()
                if i1 == i2:
                    continue
                p1 = remaining[i1]
                p2 = remaining[i2]
                dx = float(p2[0] - p1[0])
                dy = float(p2[1] - p1[1])
                if abs(dx) + abs(dy) < 2.0:
                    continue

                # ax + by + c = 0 using normal (dy, -dx)
                a = float(dy)
                b = float(-dx)
                norm = float(np.hypot(a, b))
                if norm < 1e-6:
                    continue
                a /= norm
                b /= norm
                c = -a * float(p1[0]) - b * float(p1[1])

                d = np.abs(a * remaining[:, 0] + b * remaining[:, 1] + c)
                inliers = d <= float(self.ransac_dist_thresh)
                score = int(np.count_nonzero(inliers))
                if score > best_score:
                    best_score = score
                    best_inliers = inliers
                    best_abcn = (a, b, c)

            if best_inliers is None or best_abcn is None or best_score < int(self.ransac_min_inliers):
                break

            inlier_pts = remaining[best_inliers]
            # TLS refine: direction is principal component of inliers
            mean = np.mean(inlier_pts, axis=0)
            centered = (inlier_pts - mean).astype(np.float64)
            if centered.shape[0] < 2:
                break
            _, _, vt = np.linalg.svd(centered, full_matrices=False)
            direction = vt[0]
            direction = direction / max(float(np.linalg.norm(direction)), 1e-9)
            dx, dy = float(direction[0]), float(direction[1])
            # normal is perpendicular
            a = -dy
            b = dx
            norm = float(np.hypot(a, b))
            a /= max(norm, 1e-9)
            b /= max(norm, 1e-9)
            c = -a * float(mean[0]) - b * float(mean[1])

            # Avoid BLAS-backed matmul here; some macOS/Accelerate builds can emit
            # spurious floating-point warnings for small vector GEMV operations.
            proj = centered[:, 0] * float(dx) + centered[:, 1] * float(dy)
            t_min = float(np.min(proj))
            t_max = float(np.max(proj))
            p_start = (float(mean[0] + t_min * dx), float(mean[1] + t_min * dy))
            p_end = (float(mean[0] + t_max * dx), float(mean[1] + t_max * dy))
            length = float(np.hypot(p_end[0] - p_start[0], p_end[1] - p_start[1]))

            theta = float(np.degrees(np.arctan2(dy, dx))) % 180.0
            line = _LineModel(
                a=a,
                b=b,
                c=c,
                p1=p_start,
                p2=p_end,
                theta_deg=theta,
                length=length,
                inliers=int(best_score),
                support=float(best_score) / float(max(total, 1)),
            )
            lines.append(line)

            # Remove inliers (and a small margin) from remaining set.
            d_all = np.abs(a * remaining[:, 0] + b * remaining[:, 1] + c)
            keep = d_all > float(self.ransac_dist_thresh) * 1.15
            remaining = remaining[keep]

        return lines

    def _dominant_directions(self, lines: list[_LineModel]) -> tuple[float, float]:
        """
        Estimate two dominant line directions (in degrees), without assuming
        orthogonality (projective views can skew angles).

        Returns (h_dir, v_dir) where h_dir is the direction closer to image-horizontal.
        """
        if not lines:
            return 0.0, 90.0

        thetas = np.array([float(ln.theta_deg) % 180.0 for ln in lines], dtype=np.float64)
        weights = np.array([max(1.0, float(ln.inliers)) for ln in lines], dtype=np.float64)

        # Cluster directions on the unit circle using doubled-angle embedding so that
        # theta and theta+180 map to the same point.
        rad = np.deg2rad(thetas)
        X = np.stack([np.cos(2.0 * rad), np.sin(2.0 * rad)], axis=1)  # [N,2]

        # Init centers: pick highest-weight sample + farthest by cosine distance.
        i0 = int(np.argmax(weights))
        c0 = X[i0]
        sims0 = X @ c0.reshape(2, 1)
        i1 = int(np.argmin(sims0))  # lowest similarity = farthest on circle
        c1 = X[i1]
        centers = np.stack([c0, c1], axis=0)

        labels = np.zeros((X.shape[0],), dtype=np.int32)
        for _ in range(12):
            sims = X @ centers.T  # [N,2]
            labels_new = np.argmax(sims, axis=1).astype(np.int32)
            if np.array_equal(labels_new, labels):
                break
            labels = labels_new
            new_centers = []
            for k in (0, 1):
                m = labels == k
                if not bool(np.any(m)):
                    new_centers.append(centers[k])
                    continue
                vec = (weights[m, None] * X[m]).sum(axis=0)
                n = float(np.linalg.norm(vec))
                if n < 1e-9:
                    new_centers.append(centers[k])
                else:
                    new_centers.append(vec / n)
            centers = np.stack(new_centers, axis=0)

        # Convert back to degrees in [0,180)
        angles = 0.5 * np.arctan2(centers[:, 1], centers[:, 0])
        dirs = (np.degrees(angles) % 180.0).tolist()
        dir0, dir1 = float(dirs[0]), float(dirs[1])

        # Assign "horizontal" as the direction closer to 0/180.
        def _horiz_dist(d: float) -> float:
            return float(min(abs(d - 0.0), abs(d - 180.0)))

        if _horiz_dist(dir0) <= _horiz_dist(dir1):
            return dir0, dir1
        return dir1, dir0

    def _quick_reject(self, corners: np.ndarray, frame_shape: tuple[int, int]) -> tuple[bool, str, float]:
        h, w = int(frame_shape[0]), int(frame_shape[1])
        corners = np.array(corners, dtype=np.float32).reshape(4, 2)

        if not self._is_convex_quad(corners):
            return False, "R0_not_convex", 0.0

        area = self._quad_area(corners)
        area_norm = float(area) / float(max(h * w, 1))
        if area_norm < float(self.min_quad_area_norm):
            return False, "R_area_too_small", area_norm

        xs = corners[:, 0]
        ys = corners[:, 1]
        width = float(max(xs.max() - xs.min(), 1.0))
        height = float(max(ys.max() - ys.min(), 1.0))
        aspect = height / width

        model_aspect = float(COURT_LENGTH_M) / float(COURT_WIDTH_M)
        lo = float(model_aspect) * float(self.aspect_ratio_model_min_frac)
        hi = float(model_aspect) * 1.00
        if aspect < lo or aspect > hi:
            return False, "R_aspect_out_of_range", area_norm
        return True, "OK", area_norm

    def _rank_candidate(
        self,
        corners: np.ndarray,
        line_mask: np.ndarray,
        frame_shape: tuple[int, int],
        method: str,
        candidate_lines: list[dict[str, Any]],
    ) -> _CourtCandidate | None:
        h, w = int(frame_shape[0]), int(frame_shape[1])
        corners = np.array(corners, dtype=np.float32).reshape(4, 2)
        corners = self._order_corners_lb_rb_rt_lt(corners)

        ok, _reason, area_norm = self._quick_reject(corners, frame_shape=(h, w))
        if not ok:
            return None

        edge_support = self._edge_support_ratios(line_mask, corners)
        supported = sum(1 for r in edge_support if r >= float(self.min_edge_support))
        if supported < int(self.min_edges_supported):
            return None

        tpl_f1 = float(self._template_f1_score(line_mask, corners))
        if tpl_f1 < float(self.min_tpl_f1):
            return None
        model_error = float(1.0 - tpl_f1)

        near_baseline_y = float((corners[0, 1] + corners[1, 1]) * 0.5) / float(max(h, 1))
        xs = corners[:, 0]
        ys = corners[:, 1]
        width_span = float(xs.max() - xs.min()) / float(max(w, 1))
        height_span = float(ys.max() - ys.min()) / float(max(h, 1))
        span_x = float(width_span)
        span_y = float(height_span)
        min_span_x = max(0.0, float(self.court_min_span_x_norm))
        min_span_y = max(0.0, float(self.court_min_span_y_norm))
        if span_x < min_span_x or span_y < min_span_y:
            return None
        aspect = float(height_span / max(width_span, 1e-6))
        model_aspect = float(COURT_LENGTH_M) / float(COURT_WIDTH_M)
        aspect_error = abs(aspect - model_aspect) / float(max(model_aspect, 1e-6))
        aspect_penalty = float(np.clip(aspect_error, 0.0, 1.0))


        # Near-court heuristic: prefer larger quad, lower near baseline, higher support and fit.
        s_area = float(np.clip((area_norm - float(self.min_quad_area_norm)) / 0.40, 0.0, 1.0))
        s_near = float(np.clip((near_baseline_y - 0.35) / 0.55, 0.0, 1.0))
        s_support = float(np.clip(float(np.mean(edge_support)) / 0.20, 0.0, 1.0))
        s_tpl = float(np.clip(tpl_f1 / 0.25, 0.0, 1.0))
        w_area = 0.30
        w_near = 0.30
        w_support = 0.20
        w_tpl = max(0.0, float(self.court_tpl_weight))
        w_sum = float(w_area + w_near + w_support + w_tpl)
        base_score = (w_area * s_area + w_near * s_near + w_support * s_support + w_tpl * s_tpl) / float(max(w_sum, 1e-6))
        near_score = float(np.clip(base_score * (1.0 - 0.15 * aspect_penalty), 0.0, 1.0))

        return _CourtCandidate(
            corners=corners.astype(np.float32),
            edge_support=edge_support,
            tpl_f1=tpl_f1,
            model_error=model_error,
            quad_area_norm=float(area_norm),
            near_baseline_y_norm=near_baseline_y,
            span_x=span_x,
            span_y=span_y,
            aspect_ratio=aspect,
            s_area=s_area,
            s_near=s_near,
            s_support=s_support,
            s_tpl=s_tpl,
            near_score=near_score,
            method=str(method),
            candidate_lines=candidate_lines,
        )

    def _build_candidates_from_lines(
        self,
        lines: list[_LineModel],
        line_mask: np.ndarray,
        frame_shape: tuple[int, int],
    ) -> list[_CourtCandidate]:
        h, w = int(frame_shape[0]), int(frame_shape[1])
        if len(lines) < 4:
            return []

        h_dir, v_dir = self._dominant_directions(lines)

        horiz: list[_LineModel] = []
        vert: list[_LineModel] = []
        for ln in lines:
            if self._angle_distance_deg(ln.theta_deg, h_dir) <= float(self.angle_tol_deg):
                horiz.append(ln)
            elif self._angle_distance_deg(ln.theta_deg, v_dir) <= float(self.angle_tol_deg):
                vert.append(ln)

        if len(horiz) < 2 or len(vert) < 2:
            return []

        diag = float(np.hypot(w, h))

        def _score_line(ln: _LineModel) -> float:
            return float(0.75 * ln.support + 0.25 * (ln.length / max(diag, 1e-6)))

        horiz.sort(key=_score_line, reverse=True)
        vert.sort(key=_score_line, reverse=True)
        horiz = horiz[: max(2, int(self.topk_lines_per_dir))]
        vert = vert[: max(2, int(self.topk_lines_per_dir))]

        def _mid_xy(ln: _LineModel) -> tuple[float, float]:
            return (float(ln.p1[0] + ln.p2[0]) * 0.5, float(ln.p1[1] + ln.p2[1]) * 0.5)

        # Order baselines by their vertical position and sidelines by horizontal position.
        horiz.sort(key=lambda ln: _mid_xy(ln)[1])  # top -> bottom
        vert.sort(key=lambda ln: _mid_xy(ln)[0])  # left -> right

        # Candidate pools: a few most-extreme lines on each side.
        k_ext = min(3, len(horiz))
        k_ext_v = min(3, len(vert))
        if k_ext < 2 or k_ext_v < 2:
            return []
        top_lines = horiz[:k_ext]
        bottom_lines = horiz[-k_ext:]
        left_lines = vert[:k_ext_v]
        right_lines = vert[-k_ext_v:]

        min_sep_y = float(h) * float(self.min_sep_y_ratio)
        min_sep_x = float(w) * float(self.min_sep_x_ratio)

        candidates: list[_CourtCandidate] = []
        combos = 0
        for top_ln in top_lines:
            for bot_ln in bottom_lines:
                if float(_mid_xy(bot_ln)[1] - _mid_xy(top_ln)[1]) < min_sep_y:
                    continue
                for left_ln in left_lines:
                    for right_ln in right_lines:
                        if float(_mid_xy(right_ln)[0] - _mid_xy(left_ln)[0]) < min_sep_x:
                            continue

                        combos += 1
                        if combos > int(self.max_candidates) * 5:
                            break

                        lb = self._line_intersection(bot_ln, left_ln)
                        rb = self._line_intersection(bot_ln, right_ln)
                        rt = self._line_intersection(top_ln, right_ln)
                        lt = self._line_intersection(top_ln, left_ln)
                        if lb is None or rb is None or rt is None or lt is None:
                            continue
                        cand = np.array([lb, rb, rt, lt], dtype=np.float32).reshape(4, 2)

                        # Coarse bounds: allow some extrapolation but reject extreme intersections.
                        if np.any(cand[:, 0] < -0.5 * w) or np.any(cand[:, 0] > 1.5 * w):
                            continue
                        if np.any(cand[:, 1] < -0.5 * h) or np.any(cand[:, 1] > 1.6 * h):
                            continue

                        candidate_lines = [
                            {"role": "bottom", "p1": list(bot_ln.p1), "p2": list(bot_ln.p2), "support": bot_ln.support},
                            {"role": "top", "p1": list(top_ln.p1), "p2": list(top_ln.p2), "support": top_ln.support},
                            {"role": "left", "p1": list(left_ln.p1), "p2": list(left_ln.p2), "support": left_ln.support},
                            {"role": "right", "p1": list(right_ln.p1), "p2": list(right_ln.p2), "support": right_ln.support},
                        ]
                        c = self._rank_candidate(
                            cand,
                            line_mask=line_mask,
                            frame_shape=(h, w),
                            method="farin2005_2h2v",
                            candidate_lines=candidate_lines,
                        )
                        if c is not None:
                            candidates.append(c)
                    if combos > int(self.max_candidates) * 5:
                        break
                if combos > int(self.max_candidates) * 5:
                    break

        # Multi-court selection rule: prefer nearer baseline, then larger area, then better template fit.
        candidates.sort(key=lambda c: (c.near_baseline_y_norm, c.quad_area_norm, c.tpl_f1), reverse=True)
        return candidates[: int(self.max_candidates)]

    @staticmethod
    def _quad_metrics(corners_xy: np.ndarray, frame_shape: Tuple[int, int]) -> dict[str, float]:
        h, w = int(frame_shape[0]), int(frame_shape[1])
        corners = np.array(corners_xy, dtype=np.float32).reshape(4, 2)
        xs = corners[:, 0]
        ys = corners[:, 1]
        area = float(abs(cv2.contourArea(corners.reshape(-1, 1, 2))))
        # Edge length ratios (perspective sanity diagnostics)
        v_bottom = corners[1] - corners[0]
        v_top = corners[3] - corners[2]
        v_left = corners[3] - corners[0]
        v_right = corners[2] - corners[1]
        bottom_len = float(np.linalg.norm(v_bottom))
        top_len = float(np.linalg.norm(v_top))
        left_len = float(np.linalg.norm(v_left))
        right_len = float(np.linalg.norm(v_right))
        tb_ratio = float(top_len / max(bottom_len, 1e-6))
        lr_ratio = float(left_len / max(right_len, 1e-6))
        top_y = float((corners[2, 1] + corners[3, 1]) * 0.5)
        bottom_y = float((corners[0, 1] + corners[1, 1]) * 0.5)
        return {
            "area_norm": float(area) / float(max(h * w, 1)),
            "width_span": float(xs.max() - xs.min()) / float(max(w, 1)),
            "height_span": float(ys.max() - ys.min()) / float(max(h, 1)),
            "min_y": float(ys.min()) / float(max(h, 1)),
            "max_y": float(ys.max()) / float(max(h, 1)),
            "top_y": float(top_y) / float(max(h, 1)),
            "bottom_y": float(bottom_y) / float(max(h, 1)),
            "tb_ratio": float(tb_ratio),
            "lr_ratio": float(lr_ratio),
        }

    def detect_court(self, frame: np.ndarray) -> Optional[CourtLines]:
        self.last_confidence = None
        self.last_reason = None
        self.last_edge_support = None
        self.last_method = None
        self.last_metrics = None

        if frame is None or frame.ndim != 3:
            self.last_confidence = 0.0
            self.last_reason = "invalid_frame"
            return None
        h, w = frame.shape[:2]
        if h < 32 or w < 32:
            self.last_confidence = 0.0
            self.last_reason = "frame_too_small"
            return None

        line_mask, mask_stats = self._line_pixel_mask(frame)
        pts = cv2.findNonZero(line_mask)
        if pts is None or int(pts.shape[0]) < 400:
            self.last_confidence = 0.0
            self.last_reason = "not_enough_line_pixels"
            self.last_metrics = dict(mask_stats)
            return None

        pts_xy = pts.reshape(-1, 2).astype(np.float32)
        lines = self._ransac_extract_lines(pts_xy)
        if len(lines) < 4:
            self.last_confidence = 0.0
            self.last_reason = "not_enough_lines"
            self.last_metrics = {**mask_stats, "num_lines": float(len(lines))}
            return None

        candidates = self._build_candidates_from_lines(lines, line_mask=line_mask, frame_shape=(h, w))
        if not candidates:
            self.last_confidence = 0.0
            self.last_reason = "no_valid_candidate"
            self.last_metrics = {**mask_stats, "num_lines": float(len(lines))}
            return None

        chosen = candidates[0]
        conf = float(np.clip(chosen.near_score, 0.0, 1.0))
        topk_n = min(len(candidates), 5)
        topk: list[dict[str, Any]] = []
        for i, cand in enumerate(candidates[:topk_n]):
            topk.append(
                {
                    "idx": int(i),
                    "near_score": float(cand.near_score),
                    "model_error": float(cand.model_error),
                    "tpl_f1": float(cand.tpl_f1),
                    "area_norm": float(cand.quad_area_norm),
                    "near_baseline_y": float(cand.near_baseline_y_norm),
                    "span_x": float(cand.span_x),
                    "span_y": float(cand.span_y),
                    "corners": cand.corners.astype(np.float32).tolist(),
                }
            )

        if conf < float(self.min_confidence):
            self.last_confidence = conf
            self.last_reason = "low_confidence"
            self.last_metrics = {**mask_stats, "num_candidates": float(len(candidates)), "candidates_topk": topk}
            return None

        self.last_confidence = conf
        self.last_reason = "OK"
        self.last_edge_support = chosen.edge_support
        self.last_method = chosen.method

        metrics = self._quad_metrics(chosen.corners, frame_shape=(h, w))
        metrics.update(mask_stats)
        metrics["tpl_f1"] = float(chosen.tpl_f1)
        metrics["model_error"] = float(chosen.model_error)
        metrics["near_score"] = float(chosen.near_score)
        metrics["near_baseline_y"] = float(chosen.near_baseline_y_norm)
        metrics["span_x"] = float(chosen.span_x)
        metrics["span_y"] = float(chosen.span_y)
        metrics["s_area"] = float(chosen.s_area)
        metrics["s_near"] = float(chosen.s_near)
        metrics["s_support"] = float(chosen.s_support)
        metrics["s_tpl"] = float(chosen.s_tpl)
        metrics["num_candidates"] = float(len(candidates))
        metrics["chosen_idx"] = 0.0
        metrics["candidate_lines"] = chosen.candidate_lines
        # Compact top-k summary (for optional debug tooling)
        metrics["candidates_topk"] = topk
        self.last_metrics = metrics

        return CourtLines(corners=chosen.corners.astype(np.float32))
