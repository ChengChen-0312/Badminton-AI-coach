import json
import math
import sys
from pathlib import Path

DATA = {
    "corners_order": ["LB","RB","RT","LT"],
    "corners_px": {"LB":[832,1770], "RB":[1650,790], "RT":[670,630], "LT":[226,655]},
    "segments": [
        {"name":"bottom","p0":"LB","p1":"RB"},
        {"name":"right","p0":"RB","p1":"RT"},
        {"name":"top","p0":"RT","p1":"LT"},
        {"name":"left","p0":"LT","p1":"LB"}
    ],
    "vanishing_points_px": {"VP_left":[457,1080], "VP_bottom":[1408,1080]}
}


def load_report(path: Path):
    with path.open("r") as f:
        data = json.load(f)
    cd = data.get("court_detection", {})
    lm = cd.get("last_metrics", {})
    return cd, lm


def dist(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])


def main(paths):
    reports = []
    for p in paths:
        cd, lm = load_report(Path(p))
        reports.append((p, cd, lm))

    # S1-S6 per report
    s_results = []
    for p, cd, lm in reports:
        reason = cd.get("reason")
        manual_override = lm.get("manual_override")
        inlier_ratio = lm.get("inlier_ratio")
        p90 = lm.get("p90_dist_px")
        conf_auto = lm.get("confidence_auto")
        corners = cd.get("corners")
        s_ok = [
            reason == "OK",
            manual_override is False,
            (inlier_ratio is not None and inlier_ratio >= 0.25),
            (p90 is not None and p90 <= 80),
            (conf_auto is not None and conf_auto >= 0.25),
            (isinstance(corners, list) and len(corners) == 4),
        ]
        s_results.append(s_ok)

    # T2 drift
    base_corners = reports[0][1].get("corners")
    drifts = []
    if isinstance(base_corners, list) and len(base_corners) == 4:
        for p, cd, lm in reports:
            corners = cd.get("corners")
            if not (isinstance(corners, list) and len(corners) == 4):
                drifts.append(None)
                continue
            per_point = [dist(corners[i], base_corners[i]) for i in range(4)]
            drifts.append(per_point)
    else:
        drifts = [None]*len(reports)

    # E1/E2
    e_results = []
    for p, cd, lm in reports:
        e1 = (lm.get("anchor_prior_enabled") is False and lm.get("used_anchor_prior") is False)
        e2 = (lm.get("manual_present") is False and lm.get("used_manual_anchors") is False and lm.get("manual_override") is False)
        e3 = None
        conf_auto = lm.get("confidence_auto")
        conf_final = lm.get("confidence_final")
        if conf_auto is not None and conf_final is not None:
            e3 = abs(conf_final - conf_auto) <= 1e-6
        e_results.append((e1, e2, e3))

    # DATA error eval
    data_errs = []
    for p, cd, lm in reports:
        corners = cd.get("corners")
        if not (isinstance(corners, list) and len(corners) == 4):
            data_errs.append(None)
            continue
        order = DATA["corners_order"]
        errs = {}
        for idx, name in enumerate(order):
            target = DATA["corners_px"][name]
            errs[f"err_{name}"] = dist(corners[idx], target)
        data_errs.append(errs)

    # Print summary
    for i, (p, cd, lm) in enumerate(reports):
        print(f"Report: {p}")
        print("  S1-6:", s_results[i])
        if drifts[i] is not None:
            print("  drift_from_first:", drifts[i])
        else:
            print("  drift_from_first: None")
        print("  E1/E2/E3:", e_results[i])
        print("  DATA_errs:", data_errs[i])

    # Overall T1-T3
    t1 = all(all(s) for s in s_results)
    t2 = all(d is not None and all(x <= 5.0 for x in d) for d in drifts)
    t3 = all(
        (lm.get("reject_reason") is None) or (isinstance(lm.get("reject_reason"), str) and not any(tag in lm.get("reject_reason").lower() for tag in ["loosen","fallback","override"]))
        for _,_,lm in reports
    )
    e1_all = all(e[0] for e in e_results)
    e2_all = all(e[1] for e in e_results)
    e3_all = all(e[2] for e in e_results if e[2] is not None)

    print("T1:", t1)
    print("T2:", t2)
    print("T3:", t3)
    print("E1:", e1_all)
    print("E2:", e2_all)
    print("E3:", e3_all)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: stageB_eval_only.py <report1> <report2> <report3>")
        sys.exit(1)
    main(sys.argv[1:])
