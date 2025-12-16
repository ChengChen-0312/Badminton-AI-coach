#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import cv2


def _read_frame(video_path: Path, frame_idx: int) -> Tuple[bool, any]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    cap.release()
    return ok, frame


def _draw_overlay(img_bgr, points: List[Tuple[int, int]], order_label: str) -> any:
    out = img_bgr.copy()
    for i, (x, y) in enumerate(points):
        cv2.circle(out, (int(x), int(y)), 7, (0, 0, 255), -1)
        cv2.putText(
            out,
            str(i + 1),
            (int(x) + 10, int(y) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2,
        )
    if len(points) >= 2:
        for (x1, y1), (x2, y2) in zip(points[:-1], points[1:]):
            cv2.line(out, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    if len(points) == 4:
        cv2.line(out, points[-1], points[0], (0, 255, 0), 2)

    help_lines = [
        "Click 4 court corners in this order:",
        f"  1) {order_label}",
        "Keys: [r] reset  [u] undo  [q] quit",
    ]
    y0 = 30
    for i, t in enumerate(help_lines):
        cv2.putText(
            out,
            t,
            (20, y0 + i * 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            lineType=cv2.LINE_AA,
        )
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Interactively calibrate badminton court corners.")
    p.add_argument("--video", type=str, required=True, help="Path to the video file.")
    p.add_argument("--frame", type=int, default=0, help="Frame index to use for calibration.")
    p.add_argument(
        "--out-image",
        type=str,
        default="reports/court_calibration_debug.jpg",
        help="Where to save a debug image with the selected corners.",
    )
    p.add_argument(
        "--print-yaml",
        action="store_true",
        help="Print YAML snippet for src/config/*.yaml (vision.court_corners).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    video_path = Path(args.video).expanduser().resolve()
    ok, frame_bgr = _read_frame(video_path, args.frame)
    if not ok or frame_bgr is None:
        raise RuntimeError(f"Failed to read frame {args.frame} from: {video_path}")

    # Expected order used by CourtHomography.from_corners: LB, RB, RT, LT.
    order_label = "LB -> RB -> RT -> LT (outer doubles boundary)"

    points: List[Tuple[int, int]] = []
    win = "Court Calibration (click 4 points)"

    def on_mouse(event, x, y, _flags, _param):
        nonlocal points
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(points) < 4:
                points.append((int(x), int(y)))

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)

    while True:
        overlay = _draw_overlay(frame_bgr, points, order_label=order_label)
        cv2.imshow(win, overlay)
        key = cv2.waitKey(20) & 0xFF
        if key == ord("q") or key == 27:
            break
        if key == ord("r"):
            points = []
        if key == ord("u") and points:
            points.pop()
        if len(points) == 4:
            break

    cv2.destroyAllWindows()
    if len(points) != 4:
        print("[INFO] Calibration canceled (did not select 4 points).", file=sys.stderr)
        return

    out_image = Path(args.out_image)
    out_image.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_image), _draw_overlay(frame_bgr, points, order_label=order_label))
    print(f"[OK] Saved debug image: {out_image}")

    corners = [[float(x), float(y)] for x, y in points]
    if args.print_yaml:
        print("\n# Paste into your config under: vision:")
        print("vision:")
        print("  court_corners:")
        for x, y in corners:
            print(f"    - [{x:.1f}, {y:.1f}]")
        print("  # court_corners order: LB, RB, RT, LT")

    else:
        print("[OK] court_corners:", corners)


if __name__ == "__main__":
    main()

