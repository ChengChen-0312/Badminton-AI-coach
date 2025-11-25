#!/usr/bin/env python3
"""CLI entrypoint for Version 2 training with advanced schedulers/AMP."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure repo root on path when running as a script
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.training.train import load_config, train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train badminton stroke classifier (V2).")
    parser.add_argument("--config", type=str, default="src/config/default.yaml", help="Path to YAML config.")
    parser.add_argument("--epochs", type=int, help="Override number of epochs.")
    parser.add_argument("--batch-size", type=int, help="Override batch size.")
    parser.add_argument("--learning-rate", type=float, help="Override learning rate.")
    parser.add_argument("--device", type=str, help="Override device, e.g., mps or cpu.")
    parser.add_argument("--scheduler", type=str, help='Scheduler override: "none", "cosine", or "onecycle".')
    parser.add_argument("--backbone", type=str, help="Backbone override, e.g., x3d_s, r3d_18, resnet50.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(Path(args.config))

    if args.epochs is not None:
        config["training"]["num_epochs"] = args.epochs
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        config["training"]["learning_rate"] = args.learning_rate
    if args.device is not None:
        config["training"]["device"] = args.device
    if args.scheduler is not None:
        config["training"]["scheduler"] = args.scheduler
    if args.backbone is not None:
        config["model"]["backbone"] = args.backbone

    train(config)


if __name__ == "__main__":
    main()
