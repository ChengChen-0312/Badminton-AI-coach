#!/usr/bin/env python3
"""CLI for full match analysis (v3 placeholder)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure repo root on path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import yaml

from src.pipeline.analyse_video import VideoAnalyser


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyse a badminton match/video.")
    parser.add_argument("--video", type=str, required=True, help="Path to video file.")
    parser.add_argument("--config", type=str, default="src/config/v3.yaml", help="Config path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    analyser = VideoAnalyser(config)
    result = analyser.analyse(args.video)
    print(f"Analysis finished. Summary: {result}")


if __name__ == "__main__":
    main()
