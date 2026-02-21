#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np
import yaml

from src.vision.court_detector import CourtDetector
from src.vision.court_detector_farin2005 import CourtDetectorFarin2005


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def _read_frame(video_path: Path, frame_idx: int) -> tuple[bool, Any]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    cap.release()
    return ok, frame


def _draw_quad(img_bgr: Any, corners: list[list[float]], color: tuple[int, int, int], thickness: int) -> None:
    if len(corners) != 4:
        return
    pts = np.array([(int(round(float(x))), int(round(float(y)))) for x, y in corners], dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(img_bgr, [pts], True, color, thickness)


def _draw_text(img_bgr: Any, text: str, org: tuple[int, int], color: tuple[int, int, int]) -> None:
    cv2.putText(img_bgr, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, lineType=cv2.LINE_AA)
    cv2.putText(img_bgr, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, lineType=cv2.LINE_AA)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Debug court detector top-k candidates on a single frame.")
    p.add_argument("--video", type=str, required=True, help="Path to the video file.")
    p.add_argument("--frame", type=int, default=0, help="Frame index to visualize.")
    p.add_argument("--config", type=str, default="src/config/v3_realtime.yaml", help="YAML config path.")
    p.add_argument("--topk", type=int, default=5, help="How many candidates to draw (if available).")
    p.add_argument(
        "--out-image",
        type=str,
        default="reports/court_candidates_debug.jpg",
        help="Where to save the debug image.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = _load_yaml(args.config)
    vision_cfg = cfg.get("vision", {})
    method = str(vision_cfg.get("court_detector_method", "heuristic")).strip().lower()
    top_crop_ratio = float(vision_cfg.get("top_crop_ratio", 0.20))
    floor_close_kernel = int(vision_cfg.get("floor_close_kernel", 35))
    floor_close_iter = int(vision_cfg.get("floor_close_iter", 1))
    min_confidence = float(vision_cfg.get("min_confidence", 0.60))
    court_min_span_x_norm = float(vision_cfg.get("court_min_span_x_norm", 0.55))
    court_min_span_y_norm = float(vision_cfg.get("court_min_span_y_norm", 0.55))
    court_tpl_weight = float(vision_cfg.get("court_tpl_weight", 0.30))
    net_suppress_y_min = float(vision_cfg.get("net_suppress_y_min", 0.35))
    net_suppress_y_max = float(vision_cfg.get("net_suppress_y_max", 0.55))
    net_penalty = float(vision_cfg.get("net_penalty", 0.60))
    court_bottom_min_y_norm = float(vision_cfg.get("court_bottom_min_y_norm", 0.78))
    gap_scan_enable = bool(vision_cfg.get("gap_scan_enable", True))
    gap_row_density_thresh = float(vision_cfg.get("gap_row_density_thresh", 0.003))
    gap_min_height_norm = float(vision_cfg.get("gap_min_height_norm", 0.03))
    gap_penalty_or_reject = str(vision_cfg.get("gap_penalty_or_reject", "reject"))
    gap_penalty = float(vision_cfg.get("gap_penalty", 0.60))
    video_path = Path(args.video)
    ok, frame_bgr = _read_frame(video_path, int(args.frame))
    if not ok or frame_bgr is None:
        raise RuntimeError(f"Failed to read frame {args.frame} from {video_path}")

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    detector: Any
    if method == "farin2005":
        detector = CourtDetectorFarin2005(
            top_crop_ratio=top_crop_ratio,
            min_confidence=min_confidence,
            court_min_span_x_norm=court_min_span_x_norm,
            court_min_span_y_norm=court_min_span_y_norm,
            court_tpl_weight=court_tpl_weight,
            floor_close_kernel=floor_close_kernel,
            floor_close_iter=floor_close_iter,
        )
    else:
        detector = CourtDetector(
            top_crop_ratio=top_crop_ratio,
            min_confidence=min_confidence,
            floor_close_kernel=floor_close_kernel,
            floor_close_iter=floor_close_iter,
            net_suppress_y_min=net_suppress_y_min,
            net_suppress_y_max=net_suppress_y_max,
            net_penalty=net_penalty,
            court_bottom_min_y_norm=court_bottom_min_y_norm,
            gap_scan_enable=gap_scan_enable,
            gap_row_density_thresh=gap_row_density_thresh,
            gap_min_height_norm=gap_min_height_norm,
            gap_penalty_or_reject=gap_penalty_or_reject,
            gap_penalty=gap_penalty,
        )
        method = "heuristic"

    lines = detector.detect_court(rgb)
    conf = float(getattr(detector, "last_confidence", 0.0) or 0.0)
    reason = str(getattr(detector, "last_reason", "unknown") or "unknown")
    det_method = getattr(detector, "last_method", None)
    metrics = getattr(detector, "last_metrics", None)

    out = frame_bgr.copy()
    header = f"court_detector_method={method} conf={conf:.2f} reason={reason} method={det_method}"
    _draw_text(out, header, (20, 40), (255, 255, 255))

    corners: Optional[list[list[float]]] = None
    if lines is not None and getattr(lines, "corners", None) is not None:
        corners = lines.corners.tolist()
        _draw_quad(out, corners, (0, 255, 0), 3)
        for (x, y), label in zip(corners, ["LB", "RB", "RT", "LT"]):
            cv2.circle(out, (int(round(float(x))), int(round(float(y)))), 6, (0, 0, 255), -1)
            _draw_text(out, str(label), (int(round(float(x) + 8)), int(round(float(y) - 8))), (0, 0, 255))

    # If the detector exposes candidates_topk, render them too (requires corners).
    if isinstance(metrics, dict):
        cand_list = metrics.get("candidates_topk")
        if isinstance(cand_list, list) and cand_list:
            colors = [
                (0, 255, 0),  # best
                (0, 165, 255),
                (255, 0, 0),
                (255, 0, 255),
                (0, 255, 255),
            ]
            topk = max(1, int(args.topk))
            for i, cand in enumerate(cand_list[:topk]):
                if not isinstance(cand, dict):
                    continue
                cc = cand.get("corners")
                if not (isinstance(cc, list) and len(cc) == 4):
                    continue
                color = colors[i] if i < len(colors) else (200, 200, 200)
                _draw_quad(out, cc, color, 2 if i > 0 else 3)
                xs = [float(p[0]) for p in cc]
                ys = [float(p[1]) for p in cc]
                x_txt = int(round(min(xs)))
                y_txt = int(round(min(ys) - 8))
                if y_txt < 20:
                    y_txt = int(round(min(ys) + 22))
                try:
                    near = cand.get("near_score")
                    conf = cand.get("conf")
                    span_x = cand.get("span_x")
                    span_y = cand.get("span_y")
                    tpl = cand.get("tpl_f1")
                    top_y_norm = cand.get("top_y_norm")
                    bottom_y_norm = cand.get("bottom_y_norm")
                    reason = cand.get("reason")
                    label = f"cand[{i}]"
                    if near is not None:
                        label += f" near={float(near):.2f}"
                    elif conf is not None:
                        label += f" conf={float(conf):.2f}"
                    if tpl is not None:
                        label += f" tpl={float(tpl):.2f}"
                    if span_x is not None and span_y is not None:
                        label += f" span=({float(span_x):.2f},{float(span_y):.2f})"
                    details = []
                    if top_y_norm is not None:
                        details.append(f"top={float(top_y_norm):.2f}")
                    if bottom_y_norm is not None:
                        details.append(f"bottom={float(bottom_y_norm):.2f}")
                    if reason is not None:
                        short_reason = str(reason)
                        if len(short_reason) > 40:
                            short_reason = short_reason[:37] + "..."
                        details.append(f"reason={short_reason}")
                    _draw_text(out, label, (x_txt, y_txt), color)
                    if details:
                        _draw_text(out, " ".join(details), (x_txt, y_txt + 20), color)
                except Exception:
                    _draw_text(out, f"cand[{i}]", (x_txt, y_txt), color)

    out_path = Path(args.out_image)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), out)
    print(f"[OK] Wrote: {out_path}")


if __name__ == "__main__":
    main()
