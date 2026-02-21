#!/usr/bin/env python3
import argparse
import json
import math
import sys


def _l2(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default="reports/demo_friend/demo_friend_report.json")
    ap.add_argument("--gt", default="assets/gt/demo_friend_gt.json")
    ap.add_argument("--sample_id", required=True)
    args = ap.parse_args()

    with open(args.gt, "r", encoding="utf-8") as f:
        gt = json.load(f)
    samples = gt.get("samples", [])
    sample = next((s for s in samples if s.get("id") == args.sample_id), None)
    if sample is None:
        print(f"sample_id not found: {args.sample_id}")
        return 2

    with open(args.report, "r", encoding="utf-8") as f:
        rep = json.load(f)
    corners = rep.get("court_detection", {}).get("corners")
    if not corners:
        print("no corners in report")
        return 3

    order = ["LB", "RB", "RT", "LT"]
    gt_corners = sample.get("court_corners", {})
    pred = corners
    errs = []
    visible_errs = []
    for idx, k in enumerate(order):
        g = gt_corners.get(k, {})
        gx = g.get("x")
        gy = g.get("y")
        if gx is None or gy is None:
            continue
        px, py = pred[idx]
        e = _l2((px, py), (gx, gy))
        errs.append((k, e))
        if g.get("visible", True):
            visible_errs.append(e)

    if not visible_errs:
        print("no visible GT corners")
        return 4

    mean_err = sum(visible_errs) / len(visible_errs)
    max_err = max(visible_errs)
    print("report:", args.report)
    print("sample_id:", args.sample_id)
    print("err_px (LB,RB,RT,LT):", [round(v, 2) for _, v in errs])
    print("mean_err_visible:", round(mean_err, 2), "max_err_visible:", round(max_err, 2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
