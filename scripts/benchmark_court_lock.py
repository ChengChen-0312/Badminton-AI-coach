#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.vision.court_fit_homography import fit_court_homography
from src.vision.court_env_profile import apply_court_env_overrides, parse_env_file


ORDER = ("LB", "RB", "RT", "LT")


def _read_frame(video_path: Path, frame_idx: int) -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(max(0, frame_idx)))
        ok, bgr = cap.read()
        return bgr if ok and bgr is not None else None
    finally:
        cap.release()


def _order_gt_corners(sample: Dict[str, Any]) -> Tuple[List[Optional[List[float]]], List[bool]]:
    corners = sample.get("corners", {}) if isinstance(sample.get("corners"), dict) else {}
    pts: List[Optional[List[float]]] = []
    vis: List[bool] = []
    for k in ORDER:
        v = corners.get(k)
        if not isinstance(v, dict):
            pts.append(None)
            vis.append(False)
            continue
        x = v.get("x")
        y = v.get("y")
        if x is None or y is None:
            pts.append(None)
            vis.append(False)
            continue
        pts.append([float(x), float(y)])
        vis.append(bool(v.get("visible", True)))
    return pts, vis


def _l2(a: List[float], b: List[float]) -> float:
    return float(math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1])))


def _eval_case(repo_root: Path, sample: Dict[str, Any], method: str, allow_fallback_lsd: bool) -> Dict[str, Any]:
    sid = str(sample.get("id", "unknown"))
    video_rel = str(sample.get("video", "") or "").strip()
    frame_idx = int(sample.get("frame", 0) or 0)
    if not video_rel:
        return {"id": sid, "status": "invalid", "error": "missing_video_path"}
    video_path = Path(video_rel)
    if not video_path.is_absolute():
        video_path = repo_root / video_path
    if not video_path.exists():
        return {"id": sid, "video_path": str(video_path), "status": "missing_video"}

    frame = _read_frame(video_path, frame_idx=frame_idx)
    if frame is None:
        return {"id": sid, "video_path": str(video_path), "status": "frame_read_failed"}

    fit = fit_court_homography(
        frame,
        method=method,
        allow_fallback_lsd=bool(allow_fallback_lsd),
    )
    gt_pts, gt_visible = _order_gt_corners(sample)
    out: Dict[str, Any] = {
        "id": sid,
        "video_path": str(video_path),
        "frame": int(frame_idx),
        "status": "ok" if fit.corners is not None else "no_corners",
        "reason": str(fit.reason),
        "confidence": float(fit.confidence),
        "method_used": str(fit.method_used),
    }
    if fit.corners is None:
        return out

    pred = np.asarray(fit.corners, dtype=np.float32).reshape(4, 2).tolist()
    errs: List[Optional[float]] = []
    for i in range(4):
        if not gt_visible[i] or gt_pts[i] is None:
            errs.append(None)
            continue
        errs.append(_l2(pred[i], gt_pts[i]))  # type: ignore[arg-type]
    visible_errs = [e for e in errs if e is not None]
    out["err_px_lb_rb_rt_lt"] = [None if e is None else round(float(e), 2) for e in errs]
    out["mean_err_visible"] = round(float(np.mean(visible_errs)), 2) if visible_errs else None
    out["max_err_visible"] = round(float(np.max(visible_errs)), 2) if visible_errs else None
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Court lock regression benchmark against GT corners.")
    ap.add_argument("--batch-gt", type=str, default="archive/gt_corners_batch.json")
    ap.add_argument("--method", type=str, default="hough_orient")
    ap.add_argument("--allow-fallback-lsd", action="store_true")
    ap.add_argument(
        "--env-file",
        type=str,
        default="",
        help="Optional .env file with BADC_* court knobs (export KEY=VALUE supported).",
    )
    ap.add_argument("--out", type=str, default="reports/court_regression_summary.json")
    ap.add_argument("--max-mean-err", type=float, default=30.0)
    ap.add_argument("--strict", action="store_true", help="Exit non-zero if benchmark fails threshold.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    env_profile = None
    env_num_vars = 0
    if str(args.env_file or "").strip():
        env_path = Path(str(args.env_file)).expanduser()
        if not env_path.is_absolute():
            env_path = repo_root / env_path
        if not env_path.exists():
            raise SystemExit(f"env file not found: {env_path}")
        env_overrides = parse_env_file(env_path)
        applied = apply_court_env_overrides(env_overrides)
        env_profile = str(env_path)
        env_num_vars = int(len(applied))
        print(f"[COURT_ENV] applied {env_num_vars} vars from {env_profile}")

    gt_path = Path(args.batch_gt)
    if not gt_path.is_absolute():
        gt_path = repo_root / gt_path
    if not gt_path.exists():
        raise SystemExit(f"GT file not found: {gt_path}")

    gt_data = json.loads(gt_path.read_text(encoding="utf-8"))
    samples = gt_data.get("samples", []) if isinstance(gt_data, dict) else []
    if not isinstance(samples, list):
        samples = []

    cases = [_eval_case(repo_root, s, method=args.method, allow_fallback_lsd=args.allow_fallback_lsd) for s in samples]
    valid = [c for c in cases if c.get("status") == "ok" and c.get("mean_err_visible") is not None]
    mean_err = float(np.mean([float(c["mean_err_visible"]) for c in valid])) if valid else None

    summary = {
        "batch_gt": str(gt_path),
        "method": args.method,
        "allow_fallback_lsd": bool(args.allow_fallback_lsd),
        "court_env_profile": env_profile,
        "court_env_num_vars": int(env_num_vars),
        "num_cases": len(cases),
        "num_valid": len(valid),
        "mean_err_visible_over_cases": round(mean_err, 2) if mean_err is not None else None,
        "max_mean_err_threshold": float(args.max_mean_err),
        "pass": bool(mean_err is not None and mean_err <= float(args.max_mean_err)),
        "cases": cases,
    }

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = repo_root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.strict and not summary["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
