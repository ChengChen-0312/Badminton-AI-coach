#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


def _run(cmd: List[str], cwd: Path) -> None:
    print(f"[RUN] {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(cwd))
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run local regression suite (compile + unit tests + report monitor).")
    ap.add_argument("--repo-root", type=str, default=".")
    ap.add_argument("--report", type=str, default="tests/assets/sample_report.json")
    ap.add_argument("--min-stroke-count", type=int, default=1)
    ap.add_argument("--max-unknown-ratio", type=float, default=0.80)
    ap.add_argument("--min-court-confidence", type=float, default=0.20)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    repo = Path(args.repo_root).expanduser().resolve()
    py = sys.executable

    _run([py, "-m", "compileall", "src", "scripts"], cwd=repo)
    _run([py, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py", "-v"], cwd=repo)
    _run(
        [
            py,
            "scripts/monitor_report_quality.py",
            "--report",
            args.report,
            "--min-stroke-count",
            str(args.min_stroke_count),
            "--max-unknown-ratio",
            str(args.max_unknown_ratio),
            "--min-court-confidence",
            str(args.min_court_confidence),
            "--strict",
        ],
        cwd=repo,
    )
    print("[OK] regression suite passed")


if __name__ == "__main__":
    main()
