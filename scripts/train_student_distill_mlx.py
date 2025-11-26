#!/usr/bin/env python3
"""
Placeholder for MLX-based finetuning of Qwen3-VL-4B.

NOTE: Full parameter-efficient finetuning for mlx_vlm is not implemented here.
This script documents the expected workflow and will raise NotImplementedError
so you can wire in your preferred MLX finetuning library when available.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="MLX finetune placeholder for Qwen3-VL-4B.")
    parser.add_argument("--data", required=True, help="JSONL with description and teacher_output.")
    parser.add_argument("--save_dir", required=True, help="Where to save distilled student.")
    parser.add_argument(
        "--base_model",
        default="/Users/chencheng/llm/qwen3-4b",
        help="Local path to MLX Qwen3-VL-4B (e.g., mlx-community/Qwen3-VL-4B-Instruct-4bit).",
    )
    args = parser.parse_args()

    # Load dataset
    samples = []
    with open(args.data, "r") as f:
        for line in f:
            obj = json.loads(line)
            samples.append(obj)

    print(f"Loaded {len(samples)} samples from {args.data}")
    print("Base model:", args.base_model)
    print("Requested save_dir:", args.save_dir)
    raise NotImplementedError(
        "MLX finetuning is not implemented in this repository. "
        "Integrate your preferred MLX finetune library here to train and save a LoRA/adapter."
    )


if __name__ == "__main__":
    main()
