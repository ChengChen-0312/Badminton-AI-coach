#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _load_report(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return {"strokes": data}
    if not isinstance(data, dict):
        return {"strokes": []}
    return data


def _compute_metrics(report: Dict[str, Any]) -> Dict[str, Any]:
    strokes = report.get("strokes")
    if not isinstance(strokes, list):
        strokes = []

    stroke_count = int(report.get("stroke_count", len(strokes)) or len(strokes))
    unknown = 0
    with_classifier = 0
    low_conf_rejected = 0
    conf_values: List[float] = []

    for s in strokes:
        if not isinstance(s, dict):
            continue
        if str(s.get("final_type", "")).strip().lower() in ("", "unknown"):
            unknown += 1
        if s.get("classifier_label") is not None:
            with_classifier += 1
        if bool(s.get("classifier_rejected_low_conf")):
            low_conf_rejected += 1
        try:
            c = float(s.get("confidence"))
            if c >= 0.0:
                conf_values.append(c)
        except Exception:
            pass

    court_detection = report.get("court_detection") if isinstance(report.get("court_detection"), dict) else {}
    court_conf = court_detection.get("confidence")
    try:
        court_conf_f = float(court_conf) if court_conf is not None else None
    except Exception:
        court_conf_f = None

    mean_conf = (sum(conf_values) / len(conf_values)) if conf_values else None
    unknown_ratio = (unknown / stroke_count) if stroke_count > 0 else 1.0
    classifier_coverage = (with_classifier / stroke_count) if stroke_count > 0 else 0.0

    return {
        "stroke_count": stroke_count,
        "unknown_count": int(unknown),
        "unknown_ratio": float(round(unknown_ratio, 4)),
        "classifier_coverage": float(round(classifier_coverage, 4)),
        "classifier_rejected_low_conf_count": int(low_conf_rejected),
        "mean_stroke_confidence": round(float(mean_conf), 4) if mean_conf is not None else None,
        "court_confidence": court_conf_f,
        "court_reason": court_detection.get("reason"),
        "label_space_version": report.get("label_space_version"),
    }


def _evaluate(metrics: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    checks = {
        "stroke_count": bool(metrics["stroke_count"] >= int(args.min_stroke_count)),
        "unknown_ratio": bool(metrics["unknown_ratio"] <= float(args.max_unknown_ratio)),
        "court_confidence": bool(
            metrics.get("court_confidence") is None or metrics.get("court_confidence") >= float(args.min_court_confidence)
        ),
    }
    return {
        "pass": bool(all(checks.values())),
        "checks": checks,
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Monitor report quality with regression-style thresholds.")
    ap.add_argument("--report", type=str, required=True, help="Path to report JSON.")
    ap.add_argument("--out", type=str, default="", help="Optional output JSON path.")
    ap.add_argument("--min-stroke-count", type=int, default=1)
    ap.add_argument("--max-unknown-ratio", type=float, default=0.80)
    ap.add_argument("--min-court-confidence", type=float, default=0.20)
    ap.add_argument("--strict", action="store_true", help="Exit non-zero when checks fail.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    report_path = Path(args.report).expanduser()
    report = _load_report(report_path)
    metrics = _compute_metrics(report)
    verdict = _evaluate(metrics, args)

    output = {
        "report": str(report_path),
        "metrics": metrics,
        "verdict": verdict,
    }

    if args.out:
        out_path = Path(args.out).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(output, indent=2, ensure_ascii=False))

    if args.strict and not verdict["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
