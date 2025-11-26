#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.ai_score.distill_trainer import DistillTrainer


def load_samples(path: str):
    samples = []
    with open(path, "r") as f:
        for line in f:
            obj = json.loads(line)
            samples.append({"description": obj["description"], "teacher_output": obj["teacher_output"]})
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Distill student model from teacher labels.")
    parser.add_argument("--data", required=True, help="JSONL with description and teacher_output.")
    parser.add_argument("--save_dir", required=True, help="Where to save distilled student.")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    args = parser.parse_args()

    samples = load_samples(args.data)
    trainer = DistillTrainer(lr=args.lr)
    trainer.train(samples, batch_size=args.batch_size, epochs=args.epochs)
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    trainer.save(args.save_dir)


if __name__ == "__main__":
    main()
