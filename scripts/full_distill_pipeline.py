#!/usr/bin/env python3
"""
Full MLX distillation pipeline:
1) (optional) generate teacher labels (already available)
2) prepare data for mlx-lm (train/valid jsonl)
3) run mlx-lm LoRA training (command invocation)
4) fuse adapter (optional)
5) quick eval (optional)

This script assembles shell commands; you can run or copy them.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def run_cmd(cmd: str) -> None:
    print(f"[CMD] {cmd}")
    subprocess.run(cmd, shell=True, check=True)


def prepare_data(input_file: str = "data/distill/distill_data_chat.jsonl", out_dir: str = "data/mlx_train") -> str:
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    data = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))

    if not data:
        raise SystemExit(f"[ERROR] no data in {input_file}")

    split_idx = int(len(data) * 0.9)
    train = data[:split_idx]
    valid = data[split_idx:]

    def _write(path: Path, items):
        with path.open("w", encoding="utf-8") as f:
            for it in items:
                msgs = it.get("messages")
                if not msgs:
                    continue
                f.write(json.dumps({"messages": msgs}, ensure_ascii=False) + "\n")

    _write(out_dir_path / "train.jsonl", train)
    _write(out_dir_path / "valid.jsonl", valid)
    print(f"[DATA] train={len(train)} valid={len(valid)} written to {out_dir_path}")
    return str(out_dir_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Full MLX distillation pipeline helper.")
    parser.add_argument("--data", default="data/distill/distill_data_chat.jsonl", help="Chat-format dataset.")
    parser.add_argument("--student_model", default="/Users/chencheng/llm/qwen3-4b", help="Base MLX student model path.")
    parser.add_argument("--adapter_out", default="outputs/lora_adapters", help="Where to save LoRA adapters.")
    parser.add_argument("--fused_out", default="outputs/fused_student_model", help="Where to save fused model.")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--steps_per_eval", type=int, default=50)
    args = parser.parse_args()

    # Step 1: prepare data for mlx-lm
    data_dir = prepare_data(args.data, "data/mlx_train")

    # Step 2: train LoRA with mlx_lm.lora
    lora_cmd = (
        f"python -m mlx_lm lora "
        f"--model {args.student_model} "
        f"--train "
        f"--data {data_dir} "
        f"--batch-size {args.batch_size} "
        f"--num-layers {args.num_layers} "
        f"--iters {args.iters} "
        f"--learning-rate {args.lr} "
        f"--steps-per-eval {args.steps_per_eval} "
        f"--adapter-path {args.adapter_out}"
    )
    run_cmd(lora_cmd)

    # Step 3: fuse (optional but helpful for deployment)
    fuse_cmd = (
        f"python -m mlx_lm fuse "
        f"--model {args.student_model} "
        f"--adapter-path {args.adapter_out} "
        f"--save-path {args.fused_out}"
    )
    run_cmd(fuse_cmd)

    print("\n[DONE] Distillation pipeline finished.")
    print(f"Adapter: {args.adapter_out}")
    print(f"Fused model: {args.fused_out}")


if __name__ == "__main__":
    main()
