#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.ai_score.distill_trainer_hf import HFDistillTrainer


def load_samples(path: str):
    samples = []
    with open(path, "r") as f:
        for line in f:
            obj = json.loads(line)
            samples.append({"description": obj["description"], "teacher_output": obj["teacher_output"]})
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Distill HF student model from teacher labels.")
    parser.add_argument("--data", required=True, help="JSONL with description and teacher_output.")
    parser.add_argument("--save_dir", required=True, help="Where to save distilled student.")
    parser.add_argument("--student_model", default="Qwen/Qwen2.5-VL-4B-Instruct", help="HF model id or local path.")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    args = parser.parse_args()

    samples = load_samples(args.data)
    trainer = HFDistillTrainer(student_model=args.student_model, lr=args.lr)
    trainer.train(samples, batch_size=args.batch_size, epochs=args.epochs, grad_accum_steps=args.grad_accum_steps)
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    trainer.save(args.save_dir)


if __name__ == "__main__":
    main()
