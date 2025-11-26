#!/usr/bin/env python3
"""
Convert distill_data_chat.jsonl into mlx-lm chat format (train/valid split).

Input: data/distill/distill_data_chat.jsonl
Output: data/mlx_train/train.jsonl and valid.jsonl
Each line in output: {"messages": [...]}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any


def load_data(path: Path) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                data.append(obj)
            except json.JSONDecodeError:
                continue
    return data


def convert_to_mlx_format(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Flatten messages into plain text for mlx_lm.
    We emit a single 'text' field combining prompt and completion.
    """
    converted: List[Dict[str, Any]] = []
    for item in items:
        msgs = item.get("messages")
        if not msgs:
            continue
        user_msg = next((m.get("content") for m in msgs if m.get("role") == "user"), None)
        assistant_msg = next((m.get("content") for m in msgs if m.get("role") == "assistant"), None)
        if not user_msg or not assistant_msg:
            continue
        text = f"USER: {user_msg}\nASSISTANT: {assistant_msg}"
        converted.append({"text": text})
    return converted


def main() -> None:
    input_path = Path("data/distill/distill_data_chat.jsonl")
    output_dir = Path("data/mlx_train")
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = load_data(input_path)
    if not raw:
        raise SystemExit(f"[ERROR] No data found at {input_path}")

    split_idx = int(len(raw) * 0.9)
    train_data = convert_to_mlx_format(raw[:split_idx])
    valid_data = convert_to_mlx_format(raw[split_idx:])

    train_out = output_dir / "train.jsonl"
    valid_out = output_dir / "valid.jsonl"

    with train_out.open("w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with valid_out.open("w", encoding="utf-8") as f:
        for item in valid_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"[DONE] train={len(train_data)} -> {train_out}")
    print(f"[DONE] valid={len(valid_data)} -> {valid_out}")


if __name__ == "__main__":
    main()
