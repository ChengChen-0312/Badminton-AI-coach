#!/usr/bin/env python3
import argparse
import json
import math
from typing import Dict, List, Optional


def _order_corners_lb_rb_rt_lt(pts: List[List[float]]) -> List[List[float]]:
    import numpy as np

    arr = np.array(pts, dtype=np.float32).reshape(4, 2)
    idx = np.argsort(arr[:, 1])
    top = arr[idx[:2]]
    bottom = arr[idx[2:]]
    bottom = bottom[np.argsort(bottom[:, 0])]
    top = top[np.argsort(top[:, 0])]
    lb, rb = bottom[0], bottom[1]
    lt, rt = top[0], top[1]
    return [
        [float(lb[0]), float(lb[1])],
        [float(rb[0]), float(rb[1])],
        [float(rt[0]), float(rt[1])],
        [float(lt[0]), float(lt[1])],
    ]


def _l2(a: List[float], b: List[float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _select_sample(gt: Dict[str, object], sample_id: str) -> Optional[Dict[str, object]]:
    samples = gt.get("samples", [])
    if not isinstance(samples, list):
        return None
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        if sample.get("id") == sample_id:
            return sample
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default="reports/demo_friend/demo_friend_report.json")
    ap.add_argument("--gt", required=True, help="Batch GT JSON file")
    ap.add_argument("--sample", required=True, help="Sample id in GT JSON")
    args = ap.parse_args()

    with open(args.report, "r", encoding="utf-8") as f:
        rep = json.load(f)
    with open(args.gt, "r", encoding="utf-8") as f:
        gt = json.load(f)

    sample = _select_sample(gt, args.sample)
    if not sample:
        raise SystemExit(f"sample id not found: {args.sample}")

    pred = rep.get("court_detection", {}).get("corners")
    if not pred:
        raise SystemExit("no predicted corners in report")

    pred = _order_corners_lb_rb_rt_lt(pred)
    corners = sample.get("corners", {})
    order = ["LB", "RB", "RT", "LT"]
    gt_list = []
    vis_flags = []
    for key in order:
        pt = corners.get(key)
        if isinstance(pt, dict):
            gt_list.append([float(pt.get("x")), float(pt.get("y"))])
            vis_flags.append(bool(pt.get("visible", True)))
        else:
            gt_list.append(pt)
            vis_flags.append(True)

    errs = []
    for idx, gt_pt in enumerate(gt_list):
        if not vis_flags[idx]:
            errs.append(None)
            continue
        if gt_pt is None:
            errs.append(None)
            continue
        errs.append(_l2(pred[idx], gt_pt))

    visible_errs = [e for e in errs if e is not None]
    if not visible_errs:
        raise SystemExit("no visible GT corners to evaluate")
    mean_err = sum(visible_errs) / float(len(visible_errs))
    max_err = max(visible_errs)

    metrics = rep.get("court_detection", {}).get("last_metrics", {})
    print("report:", args.report)
    print("sample:", args.sample)
    print("err_px_LB,RB,RT,LT:", [None if e is None else round(e, 2) for e in errs])
    print("mean_err_visible:", round(mean_err, 2), "max_err_visible:", round(max_err, 2))
    print("area_ratio:", metrics.get("area_ratio"))
    print("inlier_ratio:", metrics.get("inlier_ratio"))
    print("p90_dist_px:", metrics.get("p90_dist_px"))


if __name__ == "__main__":
    main()
