#!/usr/bin/env python3
"""Quick FPS benchmark for the realtime pipeline."""

from __future__ import annotations

import time
import sys
from pathlib import Path

import yaml

# Ensure repo root on path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.pipeline.analyse_video import analyse_video


def test(video: str = "archive/forehand_drive/048.mp4", cfg_path: str = "src/config/v3_realtime.yaml") -> None:
    cfg = yaml.safe_load(open(cfg_path, "r"))

    t0 = time.time()
    result = analyse_video(video, config=cfg)
    t1 = time.time()

    frames = len(result.frame_results)
    fps = frames / max(t1 - t0, 1e-6)

    print(f"Processed {frames} frames in {t1 - t0:.2f}s -> FPS={fps:.2f}")


if __name__ == "__main__":
    test()
