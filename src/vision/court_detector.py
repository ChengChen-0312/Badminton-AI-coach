from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import cv2
import numpy as np


@dataclass
class CourtLines:
    """Key points / lines of the court in image coordinates."""

    # [4, 2] in image (x, y) order: LB, RB, RT, LT (matches CourtHomography.from_corners)
    corners: np.ndarray


class CourtDetector:
    """
    Court detector using white line segmentation + contour geometry.

    Returns a best-effort estimate of the outer doubles boundary corners in
    image coordinates.
    """

    def __init__(
        self,
        white_s_max: int = 110,
        white_v_min: int = 160,
        top_crop_ratio: float = 0.20,
        morph_kernel: int = 9,
        morph_close_iter: int = 3,
        morph_open_iter: int = 1,
        min_contour_area_ratio: float = 0.01,
        min_bbox_width_ratio: float = 0.25,
        min_bbox_height_ratio: float = 0.25,
        # Validation / scoring (reject common failure modes like roof/wall lines)
        floor_y_min_ratio: float = 0.10,
        min_quad_area_ratio: float = 0.01,
        max_parallel_deg: float = 25.0,
        min_edge_support_strong: float = 0.12,
        min_edge_support_weak: float = 0.05,
        edge_top_min: float = 0.20,
        edge_bottom_min: float = 0.18,
        edge_min_floor: float = 0.10,
        net_suppress_y_min: float = 0.35,
        net_suppress_y_max: float = 0.55,
        tpl_net_reject_max: float = 0.14,
        top_edge_net_reject_max: float = 0.18,
        min_strong_edges: int = 2,
        min_confidence: float = 0.55,
        support_dilate: int = 5,
        edge_sample_points: int = 200,
        # Floor region estimation (used to mask out roof/wall structures)
        floor_sat_min: int = 35,
        floor_val_min: int = 35,
        floor_seed_y_ratio: float = 0.35,
        floor_close_kernel: int = 35,
    ) -> None:
        self.white_s_max = int(white_s_max)
        self.white_v_min = int(white_v_min)
        self.top_crop_ratio = float(top_crop_ratio)
        self.morph_kernel = int(morph_kernel)
        self.morph_close_iter = int(morph_close_iter)
        self.morph_open_iter = int(morph_open_iter)
        self.min_contour_area_ratio = float(min_contour_area_ratio)
        self.min_bbox_width_ratio = float(min_bbox_width_ratio)
        self.min_bbox_height_ratio = float(min_bbox_height_ratio)
        self.floor_y_min_ratio = float(floor_y_min_ratio)
        self.min_quad_area_ratio = float(min_quad_area_ratio)
        self.max_parallel_deg = float(max_parallel_deg)
        self.min_edge_support_strong = float(min_edge_support_strong)
        self.min_edge_support_weak = float(min_edge_support_weak)
        self.edge_top_min = float(edge_top_min)
        self.edge_bottom_min = float(edge_bottom_min)
        self.edge_min_floor = float(edge_min_floor)
        self.net_suppress_y_min = float(net_suppress_y_min)
        self.net_suppress_y_max = float(net_suppress_y_max)
        self.tpl_net_reject_max = float(tpl_net_reject_max)
        self.top_edge_net_reject_max = float(top_edge_net_reject_max)
        self.min_strong_edges = int(min_strong_edges)
        self.min_confidence = float(min_confidence)
        self.support_dilate = int(support_dilate)
        self.edge_sample_points = int(edge_sample_points)
        self.floor_sat_min = int(floor_sat_min)
        self.floor_val_min = int(floor_val_min)
        self.floor_seed_y_ratio = float(floor_seed_y_ratio)
        self.floor_close_kernel = int(floor_close_kernel)

        # Diagnostics from the last detection attempt.
        self.last_confidence: float | None = None
        self.last_reason: str | None = None
        self.last_edge_support: list[float] | None = None
        self.last_metrics: dict | None = None

    @staticmethod
    def _order_corners_lb_rb_rt_lt(pts_xy: np.ndarray) -> np.ndarray:
        """Order 4 points into LB, RB, RT, LT (image x right, y down)."""
        pts = np.array(pts_xy, dtype=np.float32).reshape(4, 2)
        # Bottom two = largest y
        idx = np.argsort(pts[:, 1])
        top = pts[idx[:2]]
        bottom = pts[idx[2:]]
        bottom = bottom[np.argsort(bottom[:, 0])]  # left, right
        top = top[np.argsort(top[:, 0])]  # left, right
        lb, rb = bottom[0], bottom[1]
        lt, rt = top[0], top[1]
        return np.stack([lb, rb, rt, lt], axis=0)

    @staticmethod
    def _angle_deg(v1: np.ndarray, v2: np.ndarray) -> float:
        n1 = float(np.linalg.norm(v1))
        n2 = float(np.linalg.norm(v2))
        if n1 < 1e-6 or n2 < 1e-6:
            return 180.0
        u1 = v1 / n1
        u2 = v2 / n2
        cos = float(abs(np.clip(np.dot(u1, u2), -1.0, 1.0)))
        return float(np.degrees(np.arccos(cos)))

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

    def _edge_support_ratios(
        self,
        white_mask: np.ndarray,
        corners_xy: np.ndarray,
    ) -> list[float]:
        h, w = white_mask.shape[:2]
        if self.support_dilate > 1:
            k = int(self.support_dilate)
            kernel = np.ones((k, k), np.uint8)
            support = cv2.dilate(white_mask, kernel, iterations=1)
        else:
            support = white_mask

        corners = np.array(corners_xy, dtype=np.float32).reshape(4, 2)
        # Order: bottom, right, top, left (LB->RB, RB->RT, RT->LT, LT->LB).
        edges = [
            (corners[0], corners[1]),  # bottom
            (corners[1], corners[2]),  # right
            (corners[2], corners[3]),  # top
            (corners[3], corners[0]),  # left
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

    def _estimate_floor_region(self, hsv: np.ndarray) -> Optional[np.ndarray]:
        """
        Estimate the floor (court plane) region as a binary mask (uint8 0/255).

        This is a heuristic to suppress false positives from roof lights / walls.
        """
        h, w = hsv.shape[:2]
        y0 = int(round(float(h) * float(self.floor_seed_y_ratio)))
        y0 = max(0, min(h - 1, y0))

        seed = np.zeros((h, w), dtype=np.uint8)
        seed[y0:, :] = 255

        base = cv2.inRange(hsv, (0, self.floor_sat_min, self.floor_val_min), (180, 255, 255))
        base = cv2.bitwise_and(base, seed)
        if int(np.count_nonzero(base)) < int(0.01 * h * w):
            return None

        k = max(3, int(self.floor_close_kernel))
        if k % 2 == 0:
            k += 1
        kernel = np.ones((k, k), np.uint8)
        closed = cv2.morphologyEx(base, cv2.MORPH_CLOSE, kernel, iterations=1)
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8), iterations=1)

        num, labels, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
        if num <= 1:
            return None

        best_lbl = None
        best_area = -1
        for lbl in range(1, num):
            x, y, ww, hh, area = stats[lbl].tolist()
            if area <= 0:
                continue
            touches_bottom = (y + hh) >= (h - 2)
            if not touches_bottom:
                continue
            if area > best_area:
                best_area = area
                best_lbl = lbl

        if best_lbl is None:
            best_lbl = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))

        return (labels == int(best_lbl)).astype(np.uint8) * 255

    @staticmethod
    def _bbox_from_mask(mask: np.ndarray) -> Optional[tuple[int, int, int, int]]:
        ys, xs = np.where(mask.astype(bool))
        if ys.size == 0:
            return None
        return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1

    @staticmethod
    def _edge_f1_score(line_mask: np.ndarray, corners_xy: np.ndarray) -> float:
        if line_mask is None or line_mask.size == 0:
            return 0.0
        h, w = line_mask.shape[:2]
        tmpl = np.zeros((h, w), dtype=np.uint8)
        pts = np.rint(np.array(corners_xy, dtype=np.float32).reshape(4, 2)).astype(np.int32)
        cv2.polylines(tmpl, [pts], True, 255, thickness=2)
        line_b = line_mask > 0
        tmpl_b = tmpl > 0
        tp = int(np.count_nonzero(line_b & tmpl_b))
        fp = int(np.count_nonzero(line_b & (~tmpl_b)))
        fn = int(np.count_nonzero((~line_b) & tmpl_b))
        denom = int(2 * tp + fp + fn)
        return float(2 * tp) / float(max(denom, 1))

    def _validate_and_score(
        self,
        corners_xy: np.ndarray,
        white_mask: np.ndarray,
        frame_shape: Tuple[int, int],
        floor_bbox_y1_norm: Optional[float] = None,
        proposal_kind: Optional[str] = None,
    ) -> tuple[float, str, list[float], dict[str, Any] | None]:
        h, w = int(frame_shape[0]), int(frame_shape[1])
        corners = np.array(corners_xy, dtype=np.float32).reshape(4, 2)

        # R0: convex + area sanity
        area = self._quad_area(corners)
        if (not self._is_convex_quad(corners)) or area < (float(h * w) * float(self.min_quad_area_ratio)):
            return 0.0, "R0_convex_or_area", [0.0, 0.0, 0.0, 0.0], None

        # R1: reject corners too high (not on floor plane)
        if float(np.min(corners[:, 1])) < float(h) * float(self.floor_y_min_ratio):
            return 0.0, "R1_not_on_floor", [0.0, 0.0, 0.0, 0.0], None

        ys = corners[:, 1]
        span_y_norm = float(ys.max() - ys.min()) / float(max(h, 1))
        top_edge_y = float((corners[2, 1] + corners[3, 1]) * 0.5)
        bottom_edge_y = float((corners[0, 1] + corners[1, 1]) * 0.5)
        top_y_norm = float(top_edge_y) / float(max(h, 1))
        bottom_y_norm = float(bottom_edge_y) / float(max(h, 1))

        # R2: require white-line support along edges
        edge_support = self._edge_support_ratios(white_mask, corners)
        top_edge_support = float(edge_support[2]) if len(edge_support) == 4 else 0.0
        bottom_edge_support = float(edge_support[0]) if len(edge_support) == 4 else 0.0
        tpl_f1 = float(self._edge_f1_score(white_mask, corners))
        net_band_hit = float(self.net_suppress_y_min) <= top_y_norm <= float(self.net_suppress_y_max)
        tpl_low_for_net = float(tpl_f1) <= float(self.tpl_net_reject_max)
        top_edge_weak_for_net = float(top_edge_support) <= float(self.top_edge_net_reject_max)
        net_like_suspect = bool(net_band_hit and tpl_low_for_net)
        span_y_low = span_y_norm < 0.45
        span_y_high = span_y_norm > 0.90
        span_y_violated = span_y_low or span_y_high
        span_y_penalty_applied = False
        if span_y_high:
            return 0.0, "R_span_y_out_of_range", edge_support, None
        if span_y_low:
            if proposal_kind == "far_suppressed":
                span_y_penalty_applied = True
            else:
                return 0.0, "R_span_y_out_of_range", edge_support, None
        if bottom_y_norm < 0.78:
            return 0.0, "R_bottom_too_high", edge_support, None
        if floor_bbox_y1_norm is not None:
            floor_y1 = max(float(floor_bbox_y1_norm), 0.25)
            if top_y_norm < float(floor_y1 - 0.02):
                return 0.0, "R_above_floor_bbox", edge_support, None
        if len(edge_support) == 4:
            if top_edge_support < float(self.edge_top_min):
                return 0.0, "R_top_edge_weak", edge_support, None
            if bottom_edge_support < float(self.edge_bottom_min):
                return 0.0, "R_bottom_edge_weak", edge_support, None
            if min(edge_support) < float(self.edge_min_floor):
                return 0.0, "R_edge_too_weak", edge_support, None
        strong = sum(1 for r in edge_support if r >= float(self.min_edge_support_strong))
        if strong < int(self.min_strong_edges) or min(edge_support) < float(self.min_edge_support_weak):
            return 0.0, "R2_weak_line_support", edge_support, None

        # R3: parallelism constraints
        v_bottom = corners[1] - corners[0]
        v_top = corners[3] - corners[2]
        v_left = corners[3] - corners[0]
        v_right = corners[2] - corners[1]
        ang_tb = self._angle_deg(v_bottom, v_top)
        ang_lr = self._angle_deg(v_left, v_right)
        if ang_tb > float(self.max_parallel_deg) or ang_lr > float(self.max_parallel_deg):
            return 0.0, "R3_not_parallel", edge_support, None

        # R4: aspect sanity (avoid extreme trapezoids)
        bottom_len = float(np.linalg.norm(v_bottom))
        top_len = float(np.linalg.norm(v_top))
        left_len = float(np.linalg.norm(v_left))
        right_len = float(np.linalg.norm(v_right))
        if min(bottom_len, top_len, left_len, right_len) < 1.0:
            return 0.0, "R4_degenerate_edges", edge_support, None
        tb_ratio = top_len / bottom_len
        lr_ratio = left_len / right_len
        if not (0.20 <= tb_ratio <= 2.50 and 0.20 <= lr_ratio <= 2.50):
            return 0.0, "R4_aspect_out_of_range", edge_support, None

        # Soft confidence scoring
        target = 0.15
        s_line = float(np.clip(np.mean([r / target for r in edge_support]), 0.0, 1.0))
        s_geom = float(np.clip(1.0 - 0.5 * (ang_tb + ang_lr) / float(self.max_parallel_deg), 0.0, 1.0))
        min_y = float(np.min(corners[:, 1])) / float(max(h, 1))
        s_floor = float(np.clip((min_y - float(self.floor_y_min_ratio)) / 0.40, 0.0, 1.0))
        tpl_low = tpl_f1 < 0.05
        conf_raw = 0.55 * s_line + 0.30 * s_geom + 0.15 * s_floor
        conf_bonus_far_top = 0.0
        if not net_band_hit:
            bonus_scale = float(np.clip((0.50 - top_y_norm) / 0.20, 0.0, 1.0))
            conf_bonus_far_top = 0.15 * bonus_scale
            conf_raw = float(np.clip(conf_raw + conf_bonus_far_top, 0.0, 1.0))
        conf_after_net = float(conf_raw)
        if net_like_suspect:
            conf_after_net *= 0.35
        conf_final = float(conf_after_net)
        if span_y_penalty_applied:
            conf_final *= 0.5
        if tpl_low:
            conf_final *= 0.85
        conf_final = float(np.clip(conf_final, 0.0, 1.0))
        score_debug = {
            "conf_raw": float(conf_raw),
            "conf_after_net": float(conf_after_net),
            "conf_final": float(conf_final),
            "conf_bonus_far_top": float(conf_bonus_far_top),
            "net_like_suspect": bool(net_like_suspect),
            "net_band_hit": bool(net_band_hit),
            "tpl_low_for_net": bool(tpl_low_for_net),
            "top_edge_weak_for_net": bool(top_edge_weak_for_net),
            "span_y_violated": bool(span_y_violated),
            "span_y_penalty_applied": bool(span_y_penalty_applied),
        }
        return conf_final, "OK", edge_support, score_debug

    def detect_court(self, frame: np.ndarray) -> Optional[CourtLines]:
        """
        Args:
            frame: RGB frame [H, W, 3]

        Returns:
            CourtLines or None if failed.
        """
        self.last_confidence = None
        self.last_reason = None
        self.last_edge_support = None
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

        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        floor_region = self._estimate_floor_region(hsv)
        # White-ish lines: low saturation, high value.
        white = cv2.inRange(hsv, (0, 0, self.white_v_min), (180, self.white_s_max, 255))
        if floor_region is not None:
            white = cv2.bitwise_and(white, floor_region)

        # Ignore the roof / lights region which often creates false positives.
        top = int(round(h * self.top_crop_ratio))
        if top > 0:
            white[:top, :] = 0

        kernel = np.ones((self.morph_kernel, self.morph_kernel), np.uint8)
        if self.morph_close_iter > 0:
            white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, kernel, iterations=self.morph_close_iter)
        if self.morph_open_iter > 0:
            white = cv2.morphologyEx(white, cv2.MORPH_OPEN, kernel, iterations=self.morph_open_iter)

        mask_density = float(np.count_nonzero(white)) / float(max(h * w, 1))
        floor_bbox = self._bbox_from_mask(floor_region) if floor_region is not None else None
        if floor_bbox is None:
            floor_bbox = (0, top, w, h)
        fx1, fy1, fx2, fy2 = floor_bbox
        mask_stats: dict[str, Any] = {
            "mask_density": float(mask_density),
            "top_crop_px": float(top),
            "floor_bbox": [
                float(fx1) / float(max(w, 1)),
                float(fy1) / float(max(h, 1)),
                float(fx2) / float(max(w, 1)),
                float(fy2) / float(max(h, 1)),
            ],
            "floor_bbox_y1": float(fy1) / float(max(h, 1)),
            "candidates_topk": [],
        }

        def _push_candidate(
            ordered: np.ndarray,
            conf: float,
            reason: str,
            edge_support: list[float],
            score_debug: Optional[dict[str, Any]] = None,
            proposal_kind: Optional[str] = None,
        ) -> dict[str, Any]:
            xs = ordered[:, 0]
            ys = ordered[:, 1]
            top_y_norm = float((ordered[2, 1] + ordered[3, 1]) * 0.5) / float(max(h, 1))
            bottom_y_norm = float((ordered[0, 1] + ordered[1, 1]) * 0.5) / float(max(h, 1))
            tpl_f1 = float(self._edge_f1_score(white, ordered))
            span_y_norm = float(ys.max() - ys.min()) / float(max(h, 1))
            top_edge_support = float(edge_support[2]) if len(edge_support) == 4 else 0.0
            edge_support_by_side = None
            if len(edge_support) == 4:
                edge_support_by_side = {
                    "bottom": float(edge_support[0]),
                    "right": float(edge_support[1]),
                    "top": float(edge_support[2]),
                    "left": float(edge_support[3]),
                }
            net_band_hit = float(self.net_suppress_y_min) <= float(top_y_norm) <= float(self.net_suppress_y_max)
            tpl_low_for_net = float(tpl_f1) <= float(self.tpl_net_reject_max)
            top_edge_weak_for_net = float(top_edge_support) <= float(self.top_edge_net_reject_max)
            net_like_suspect = bool(net_band_hit and tpl_low_for_net)
            span_y_violated = bool(span_y_norm < 0.45 or span_y_norm > 0.90)
            span_y_penalty_applied = False
            conf_raw = float(conf)
            conf_after_net = float(conf)
            conf_final = float(conf)
            conf_bonus_far_top = 0.0
            if isinstance(score_debug, dict):
                conf_raw = float(score_debug.get("conf_raw", conf_raw))
                conf_after_net = float(score_debug.get("conf_after_net", conf_after_net))
                conf_final = float(score_debug.get("conf_final", conf_final))
                conf_bonus_far_top = float(score_debug.get("conf_bonus_far_top", conf_bonus_far_top))
                net_like_suspect = bool(score_debug.get("net_like_suspect", net_like_suspect))
                net_band_hit = bool(score_debug.get("net_band_hit", net_band_hit))
                tpl_low_for_net = bool(score_debug.get("tpl_low_for_net", tpl_low_for_net))
                top_edge_weak_for_net = bool(score_debug.get("top_edge_weak_for_net", top_edge_weak_for_net))
                span_y_violated = bool(score_debug.get("span_y_violated", span_y_violated))
                span_y_penalty_applied = bool(score_debug.get("span_y_penalty_applied", span_y_penalty_applied))
            info = {
                "tpl_f1": float(tpl_f1),
                "top_y_norm": float(top_y_norm),
                "bottom_y_norm": float(bottom_y_norm),
                "span_y_norm": float(span_y_norm),
                "top_edge_support": float(top_edge_support),
                "net_like_reject_triggered": False,
                "net_like_suspect": bool(net_like_suspect),
                "net_band_hit": bool(net_band_hit),
                "tpl_low_for_net": bool(tpl_low_for_net),
                "top_edge_weak_for_net": bool(top_edge_weak_for_net),
                "tpl_low": bool(tpl_f1 < 0.05),
                "span_y_violated": bool(span_y_violated),
                "span_y_penalty_applied": bool(span_y_penalty_applied),
                "conf_raw": float(conf_raw),
                "conf_after_net": float(conf_after_net),
                "conf_final": float(conf_final),
                "conf_bonus_far_top": float(conf_bonus_far_top),
                "proposal_kind": proposal_kind,
            }
            record_reason = str(reason)
            record = {
                "corners": ordered.astype(np.float32).tolist(),
                "conf": float(conf),
                "conf_raw": float(conf_raw),
                "conf_after_net": float(conf_after_net),
                "conf_final": float(conf_final),
                "conf_bonus_far_top": float(conf_bonus_far_top),
                "reason": record_reason,
                "top_y_norm": float(top_y_norm),
                "bottom_y_norm": float(bottom_y_norm),
                "span_x": float(xs.max() - xs.min()) / float(max(w, 1)),
                "span_y": float(ys.max() - ys.min()) / float(max(h, 1)),
                "span_y_norm": float(span_y_norm),
                "tpl_f1": float(tpl_f1),
                "top_edge_support": float(top_edge_support),
                "edge_support": [float(v) for v in edge_support],
                "edge_support_by_side": edge_support_by_side,
                "net_like_suspect": bool(net_like_suspect),
                "net_band_hit": bool(net_band_hit),
                "tpl_low_for_net": bool(tpl_low_for_net),
                "top_edge_weak_for_net": bool(top_edge_weak_for_net),
                "span_y_violated": bool(span_y_violated),
                "span_y_penalty_applied": bool(span_y_penalty_applied),
                "proposal_kind": proposal_kind,
            }
            record.update(info)
            record["penalty_terms"] = {
                "span_y_penalty_applied": bool(record.get("span_y_penalty_applied", False)),
                "tpl_low": bool(record.get("tpl_low", False)),
                "edge_penalty_applied": False,
            }
            cand_list = list(mask_stats.get("candidates_topk", []))
            cand_list.append(record)
            cand_list.sort(key=lambda r: float(r.get("conf", 0.0)), reverse=True)
            cand_list = cand_list[:5]
            if proposal_kind == "far_suppressed_high" and float(top_y_norm) <= 0.50:
                has_high = any(r.get("proposal_kind") == "far_suppressed_high" for r in cand_list)
                if not has_high:
                    if len(cand_list) < 5:
                        cand_list.append(record)
                    else:
                        cand_list[-1] = record
            mask_stats["candidates_topk"] = cand_list
            return info

        def _set_metrics(edge_support: list[float], info: Optional[dict[str, Any]]) -> None:
            edge_support_by_side = None
            if len(edge_support) == 4:
                edge_support_by_side = {
                    "bottom": float(edge_support[0]),
                    "right": float(edge_support[1]),
                    "top": float(edge_support[2]),
                    "left": float(edge_support[3]),
                }
            metrics: dict[str, Any] = {
                "edge_support_by_side": edge_support_by_side,
                "cfg_edge_top_min": float(self.edge_top_min),
                "cfg_edge_bottom_min": float(self.edge_bottom_min),
                "cfg_edge_min_floor": float(self.edge_min_floor),
                "cfg_net_band": [float(self.net_suppress_y_min), float(self.net_suppress_y_max)],
                "cfg_tpl_net_reject_max": float(self.tpl_net_reject_max),
                "cfg_top_edge_net_reject_max": float(self.top_edge_net_reject_max),
                "min_confidence": float(self.min_confidence),
            }
            if isinstance(info, dict):
                if info.get("tpl_f1") is not None:
                    metrics["tpl_f1"] = float(info.get("tpl_f1"))
                if info.get("top_y_norm") is not None:
                    metrics["top_y_norm"] = float(info.get("top_y_norm"))
                if info.get("bottom_y_norm") is not None:
                    metrics["bottom_y_norm"] = float(info.get("bottom_y_norm"))
                if info.get("span_y_norm") is not None:
                    metrics["span_y_norm"] = float(info.get("span_y_norm"))
                if info.get("top_edge_support") is not None:
                    metrics["top_edge_support"] = float(info.get("top_edge_support"))
                metrics["net_like_reject_triggered"] = bool(info.get("net_like_reject_triggered", False))
                metrics["net_like_suspect"] = bool(info.get("net_like_suspect", False))
                metrics["net_band_hit"] = bool(info.get("net_band_hit", False))
                metrics["tpl_low_for_net"] = bool(info.get("tpl_low_for_net", False))
                metrics["top_edge_weak_for_net"] = bool(info.get("top_edge_weak_for_net", False))
                metrics["tpl_low"] = bool(info.get("tpl_low", False))
                metrics["span_y_violated"] = bool(info.get("span_y_violated", False))
                metrics["span_y_penalty_applied"] = bool(info.get("span_y_penalty_applied", False))
                metrics["penalty_terms"] = {
                    "span_y_penalty_applied": bool(info.get("span_y_penalty_applied", False)),
                    "tpl_low": bool(info.get("tpl_low", False)),
                    "edge_penalty_applied": False,
                }
                if info.get("conf_raw") is not None:
                    metrics["conf_raw"] = float(info.get("conf_raw"))
                if info.get("conf_after_net") is not None:
                    metrics["conf_after_net"] = float(info.get("conf_after_net"))
                if info.get("conf_final") is not None:
                    metrics["conf_final"] = float(info.get("conf_final"))
                    metrics["conf_margin"] = float(info.get("conf_final")) - float(self.min_confidence)
                if info.get("conf_bonus_far_top") is not None:
                    metrics["conf_bonus_far_top"] = float(info.get("conf_bonus_far_top"))
                if info.get("proposal_kind") is not None:
                    metrics["proposal_kind"] = info.get("proposal_kind")
                metrics["accepted_below_min_conf"] = bool(info.get("accepted_below_min_conf", False))
            metrics["fallback_triggered"] = False
            metrics.update(mask_stats)
            self.last_metrics = metrics

        # Connected-component selection is more robust than raw contour sorting for thin line masks.
        num, labels, stats, centroids = cv2.connectedComponentsWithStats(white, connectivity=8)
        if num <= 1:
            self.last_confidence = 0.0
            self.last_reason = "no_components"
            self.last_metrics = mask_stats
            return None

        min_area = float(h * w) * self.min_contour_area_ratio
        anchor = np.array([w * 0.5, h * 0.75], dtype=np.float32)

        best_label = None
        best_cost = None
        diag = float(np.hypot(w, h))
        frame_area = float(max(h * w, 1))
        for lbl in range(1, num):
            area = float(stats[lbl, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            cx, cy = centroids[lbl]
            dist = float(np.hypot(cx - anchor[0], cy - anchor[1]))
            dist_norm = dist / diag
            area_ratio = area / frame_area
            # Cost: prefer large connected components close to the court region.
            cost = dist_norm - 2.0 * area_ratio
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_label = lbl

        if best_label is None:
            # Fallback: take the largest component.
            best_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))

        comp = (labels == int(best_label)).astype(np.uint8) * 255
        contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            self.last_confidence = 0.0
            self.last_reason = "no_contours"
            self.last_metrics = mask_stats
            return None
        best_cnt = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(best_cnt)
        peri = cv2.arcLength(hull, True)
        if peri < 1e-6:
            self.last_confidence = 0.0
            self.last_reason = "degenerate_hull"
            self.last_metrics = mask_stats
            return None

        quad = None
        # Try to find a 4-vertex approximation. Increase epsilon gradually.
        for eps_frac in np.linspace(0.01, 0.08, 20):
            approx = cv2.approxPolyDP(hull, eps_frac * peri, True)
            if len(approx) == 4:
                quad = approx.reshape(4, 2)
                break

        candidates: list[tuple[np.ndarray, Optional[str]]] = []
        if quad is not None:
            candidates.append((quad, None))
        rect = cv2.minAreaRect(hull)
        candidates.append((cv2.boxPoints(rect).reshape(4, 2), None))
        rect_cnt = cv2.minAreaRect(best_cnt)
        candidates.append((cv2.boxPoints(rect_cnt).reshape(4, 2), None))
        x, y, ww, hh = cv2.boundingRect(hull)
        bbox_quad = np.array(
            [[float(x), float(y + hh - 1)], [float(x + ww - 1), float(y + hh - 1)], [float(x + ww - 1), float(y)], [float(x), float(y)]],
            dtype=np.float32,
        )
        candidates.append((bbox_quad, None))

        def _extract_far_candidates(mask: np.ndarray, prefer_top: bool = False) -> list[np.ndarray]:
            out: list[np.ndarray] = []
            try:
                num_f, labels_f, stats_f, centroids_f = cv2.connectedComponentsWithStats(mask, connectivity=8)
                if num_f <= 1:
                    return out
                best_label_f = None
                best_cost_f = None
                for lbl in range(1, num_f):
                    area_f = float(stats_f[lbl, cv2.CC_STAT_AREA])
                    if area_f < min_area:
                        continue
                    cx, cy = centroids_f[lbl]
                    if prefer_top:
                        cost = float(cy)
                    else:
                        dist = float(np.hypot(cx - anchor[0], cy - anchor[1]))
                        dist_norm = dist / diag
                        area_ratio = area_f / frame_area
                        cost = dist_norm - 2.0 * area_ratio
                    if best_cost_f is None or cost < best_cost_f:
                        best_cost_f = cost
                        best_label_f = lbl
                if best_label_f is None:
                    best_label_f = 1 + int(np.argmax(stats_f[1:, cv2.CC_STAT_AREA]))
                comp_f = (labels_f == int(best_label_f)).astype(np.uint8) * 255
                contours_f, _ = cv2.findContours(comp_f, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours_f:
                    best_cnt_f = max(contours_f, key=cv2.contourArea)
                    hull_f = cv2.convexHull(best_cnt_f)
                    peri_f = cv2.arcLength(hull_f, True)
                    if peri_f > 1e-6:
                        quad_f = None
                        for eps_frac in np.linspace(0.01, 0.08, 20):
                            approx = cv2.approxPolyDP(hull_f, eps_frac * peri_f, True)
                            if len(approx) == 4:
                                quad_f = approx.reshape(4, 2)
                                break
                        if quad_f is not None:
                            out.append(quad_f)
                        rect_f = cv2.minAreaRect(hull_f)
                        out.append(cv2.boxPoints(rect_f).reshape(4, 2))
                        rect_cnt_f = cv2.minAreaRect(best_cnt_f)
                        out.append(cv2.boxPoints(rect_cnt_f).reshape(4, 2))
                        x_f, y_f, ww_f, hh_f = cv2.boundingRect(hull_f)
                        bbox_f = np.array(
                            [
                                [float(x_f), float(y_f + hh_f - 1)],
                                [float(x_f + ww_f - 1), float(y_f + hh_f - 1)],
                                [float(x_f + ww_f - 1), float(y_f)],
                                [float(x_f), float(y_f)],
                            ],
                            dtype=np.float32,
                        )
                        out.append(bbox_f)
            except Exception:
                return []
            return out

        net_y1 = int(round(float(h) * float(self.net_suppress_y_min)))
        net_y2 = int(round(float(h) * float(self.net_suppress_y_max)))
        y1 = max(0, min(h, min(net_y1, net_y2)))
        y2 = max(0, min(h, max(net_y1, net_y2)))

        white_far = white.copy()
        if y2 > y1:
            white_far[y1:y2, :] = 0
        for cand in _extract_far_candidates(white_far):
            candidates.append((cand, "far_suppressed"))

        white_far_high = white.copy()
        floor_bbox = mask_stats.get("floor_bbox", None)
        floor_y1_norm = float(mask_stats.get("floor_bbox_y1", 0.0))
        floor_y2_norm = float(floor_bbox[3]) if isinstance(floor_bbox, list) and len(floor_bbox) == 4 else 1.0
        floor_y1_px = int(round(float(h) * floor_y1_norm))
        floor_y2_px = int(round(float(h) * floor_y2_norm))
        floor_y1_px = max(0, min(h, floor_y1_px))
        floor_y2_px = max(0, min(h, floor_y2_px))
        roi_height = max(1, floor_y2_px - floor_y1_px)
        high_end = int(round(float(floor_y1_px) + 0.35 * float(roi_height)))
        high_end = max(0, min(h, high_end))
        if floor_y1_px > 0:
            white_far_high[:floor_y1_px, :] = 0
        if high_end < h:
            white_far_high[high_end:, :] = 0
        ys_high, xs_high = np.where(white_far_high > 0)
        if ys_high.size > 0 and xs_high.size > 0:
            x1_h = float(xs_high.min())
            x2_h = float(xs_high.max())
            y1_h = float(ys_high.min())
            y2_h = float(ys_high.max())
            if x2_h > x1_h and y2_h > y1_h:
                bbox_high = np.array(
                    [
                        [x1_h, y2_h],
                        [x2_h, y2_h],
                        [x2_h, y1_h],
                        [x1_h, y1_h],
                    ],
                    dtype=np.float32,
                )
                candidates.append((bbox_high, "far_suppressed_high"))
        for cand in _extract_far_candidates(white_far_high, prefer_top=True):
            candidates.append((cand, "far_suppressed_high"))

        min_conf = float(self.min_confidence)
        near_min_conf = float(min(min_conf, 0.45))
        best: tuple[float, str, list[float], np.ndarray, dict[str, Any], bool] | None = None
        best_reject: tuple[float, str, list[float], np.ndarray, dict[str, Any]] | None = None
        best_suspect = True
        for cand, proposal_kind in candidates:
            ordered = self._order_corners_lb_rb_rt_lt(cand)
            xs = ordered[:, 0]
            ys = ordered[:, 1]
            if (float(xs.max() - xs.min()) / float(w)) < self.min_bbox_width_ratio:
                edge_support = self._edge_support_ratios(white, ordered)
                info = _push_candidate(ordered, 0.0, "bbox_too_narrow", edge_support, proposal_kind=proposal_kind)
                if best_reject is None or 0.0 > float(best_reject[0]):
                    best_reject = (0.0, "bbox_too_narrow", edge_support, ordered, info)
                continue
            if (float(ys.max() - ys.min()) / float(h)) < self.min_bbox_height_ratio:
                edge_support = self._edge_support_ratios(white, ordered)
                info = _push_candidate(ordered, 0.0, "bbox_too_short", edge_support, proposal_kind=proposal_kind)
                if best_reject is None or 0.0 > float(best_reject[0]):
                    best_reject = (0.0, "bbox_too_short", edge_support, ordered, info)
                continue

            conf, reason, edge_support, score_debug = self._validate_and_score(
                ordered,
                white_mask=white,
                frame_shape=(h, w),
                floor_bbox_y1_norm=mask_stats.get("floor_bbox_y1"),
                proposal_kind=proposal_kind,
            )
            info = _push_candidate(
                ordered,
                conf,
                reason,
                edge_support,
                score_debug=score_debug,
                proposal_kind=proposal_kind,
            )
            if best_reject is None or float(conf) > float(best_reject[0]):
                best_reject = (float(conf), str(reason), edge_support, ordered, info)
            near_threshold_ok = False
            if reason == "OK":
                if proposal_kind == "far_suppressed" and not bool(info.get("net_like_suspect", False)):
                    if conf >= (near_min_conf - 0.05):
                        near_threshold_ok = True
            if reason != "OK" or (conf < min_conf and not near_threshold_ok):
                continue
            if near_threshold_ok and conf < min_conf:
                info["accepted_below_min_conf"] = True
                info["min_confidence"] = float(min_conf)
                info["conf_margin"] = float(conf - min_conf)

            cand_suspect = bool(info.get("net_like_suspect", False))
            if best is None or (best_suspect and not cand_suspect) or (cand_suspect == best_suspect and conf > float(best[0])):
                best = (float(conf), str(reason), edge_support, ordered, info, cand_suspect)
                best_suspect = cand_suspect

        if best is None:
            if best_reject is not None:
                conf, reason, edge_support, ordered, info = best_reject
                self.last_confidence = float(conf)
                self.last_reason = str(reason)
                self.last_edge_support = edge_support
                _set_metrics(edge_support, info)
            return None

        conf, reason, edge_support, ordered, info, _suspect = best
        self.last_confidence = float(conf)
        self.last_reason = str(reason)
        self.last_edge_support = edge_support
        _set_metrics(edge_support, info)

        cand_topk = mask_stats.get("candidates_topk")
        all_net_like = False
        if isinstance(cand_topk, list) and cand_topk:
            all_net_like = all(bool(c.get("net_like_suspect", False)) for c in cand_topk)
        if all_net_like and not bool(getattr(self, "_fallback_active", False)):
            pre_reason = str(reason)
            pre_conf = float(conf)
            pre_top_y = info.get("top_y_norm") if isinstance(info, dict) else None
            pre_bottom_y = info.get("bottom_y_norm") if isinstance(info, dict) else None
            pre_metrics = dict(self.last_metrics) if isinstance(self.last_metrics, dict) else None
            prev_top = float(self.top_crop_ratio)
            prev_close = int(self.morph_close_iter)
            prev_seed = float(self.floor_seed_y_ratio)
            self._fallback_active = True
            try:
                self.top_crop_ratio = 0.0
                self.morph_close_iter = int(max(1, prev_close + 1))
                self.floor_seed_y_ratio = float(max(0.20, prev_seed - 0.10))
                result = self.detect_court(frame)
            finally:
                self.top_crop_ratio = prev_top
                self.morph_close_iter = prev_close
                self.floor_seed_y_ratio = prev_seed
                self._fallback_active = False
            if isinstance(self.last_metrics, dict):
                self.last_metrics["fallback_triggered"] = True
                self.last_metrics["fallback_pre_reason"] = pre_reason
                self.last_metrics["fallback_pre_conf"] = pre_conf
                self.last_metrics["fallback_pre_top_y_norm"] = pre_top_y
                self.last_metrics["fallback_pre_bottom_y_norm"] = pre_bottom_y
                self.last_metrics["fallback_best_reason"] = self.last_reason
                self.last_metrics["fallback_best_conf"] = self.last_confidence
                self.last_metrics["fallback_best_top_y_norm"] = self.last_metrics.get("top_y_norm")
                self.last_metrics["fallback_best_bottom_y_norm"] = self.last_metrics.get("bottom_y_norm")
            if result is not None:
                return result
            if pre_metrics is not None:
                pre_metrics["fallback_triggered"] = True
                pre_metrics["fallback_pre_reason"] = pre_reason
                pre_metrics["fallback_pre_conf"] = pre_conf
                pre_metrics["fallback_pre_top_y_norm"] = pre_top_y
                pre_metrics["fallback_pre_bottom_y_norm"] = pre_bottom_y
                pre_metrics["fallback_best_reason"] = self.last_reason
                pre_metrics["fallback_best_conf"] = self.last_confidence
                if isinstance(self.last_metrics, dict):
                    pre_metrics["fallback_best_top_y_norm"] = self.last_metrics.get("top_y_norm")
                    pre_metrics["fallback_best_bottom_y_norm"] = self.last_metrics.get("bottom_y_norm")
                self.last_metrics = pre_metrics

        return CourtLines(corners=ordered.astype(np.float32))
