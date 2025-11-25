#!/usr/bin/env python3
"""CLI entrypoint to train the badminton stroke classifier."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training.train import load_config, train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train badminton stroke classifier.")
    parser.add_argument("--config", type=str, default="src/config/default.yaml", help="Path to YAML config.")
    parser.add_argument("--epochs", type=int, help="Override number of epochs.")
    parser.add_argument("--batch-size", type=int, help="Override batch size.")
    parser.add_argument("--learning-rate", type=float, help="Override learning rate.")
    parser.add_argument("--device", type=str, help="Override device, e.g., cuda or cpu.")
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

    train(config)


if __name__ == "__main__":
    main()
