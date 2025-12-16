from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

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

    def detect_court(self, frame: np.ndarray) -> Optional[CourtLines]:
        """
        Args:
            frame: RGB frame [H, W, 3]

        Returns:
            CourtLines or None if failed.
        """
        if frame is None or frame.ndim != 3:
            return None
        h, w = frame.shape[:2]
        if h < 32 or w < 32:
            return None

        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        # White-ish lines: low saturation, high value.
        white = cv2.inRange(hsv, (0, 0, self.white_v_min), (180, self.white_s_max, 255))

        # Ignore the roof / lights region which often creates false positives.
        top = int(round(h * self.top_crop_ratio))
        if top > 0:
            white[:top, :] = 0

        kernel = np.ones((self.morph_kernel, self.morph_kernel), np.uint8)
        if self.morph_close_iter > 0:
            white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, kernel, iterations=self.morph_close_iter)
        if self.morph_open_iter > 0:
            white = cv2.morphologyEx(white, cv2.MORPH_OPEN, kernel, iterations=self.morph_open_iter)

        # Connected-component selection is more robust than raw contour sorting for thin line masks.
        num, labels, stats, centroids = cv2.connectedComponentsWithStats(white, connectivity=8)
        if num <= 1:
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
            return None
        best_cnt = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(best_cnt)
        peri = cv2.arcLength(hull, True)
        if peri < 1e-6:
            return None

        quad = None
        # Try to find a 4-vertex approximation. Increase epsilon gradually.
        for eps_frac in np.linspace(0.01, 0.08, 20):
            approx = cv2.approxPolyDP(hull, eps_frac * peri, True)
            if len(approx) == 4:
                quad = approx.reshape(4, 2)
                break

        if quad is None:
            rect = cv2.minAreaRect(hull)
            quad = cv2.boxPoints(rect).reshape(4, 2)

        ordered = self._order_corners_lb_rb_rt_lt(quad)
        xs = ordered[:, 0]
        ys = ordered[:, 1]
        if (float(xs.max() - xs.min()) / float(w)) < self.min_bbox_width_ratio:
            return None
        if (float(ys.max() - ys.min()) / float(h)) < self.min_bbox_height_ratio:
            return None
        return CourtLines(corners=ordered.astype(np.float32))
