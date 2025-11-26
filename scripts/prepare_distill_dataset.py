#!/usr/bin/env python3
"""
Convert teacher_labels.jsonl into chat-style JSONL for distillation.

Input (per line):
  {
    "video_path": "...",
    "summary": {...},
    "prompt": "...",
    "teacher_output": <str | dict | list | other>
  }

Output (per line):
  {
    "messages": [
      {"role": "system", "content": "..."},
      {"role": "user", "content": "<prompt>"},
      {"role": "assistant", "content": "<teacher_output text>"}
    ]
  }
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def teacher_output_to_text(teacher_output) -> str:
    """Normalize teacher output to a string."""
    if isinstance(teacher_output, str):
        return teacher_output.strip()
    try:
        return json.dumps(teacher_output, ensure_ascii=False)
    except Exception:
        return str(teacher_output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare chat-style distillation dataset from teacher labels.")
    parser.add_argument("--input", type=str, default="data/distill/teacher_labels.jsonl", help="Teacher labels JSONL")
    parser.add_argument(
        "--output", type=str, default="data/distill/distill_data_chat.jsonl", help="Output chat JSONL path"
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=(
            "You are a professional badminton coach. "
            "You evaluate strokes, point out mistakes, and give concise, practical suggestions."
        ),
        help="System prompt/persona for the student model.",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    num_in = 0
    num_out = 0
    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            num_in += 1
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                print(f"[WARN] Skip invalid JSON line #{num_in}")
                continue

            prompt = item.get("prompt", "").strip()
            teacher_output = item.get("teacher_output", "")
            if not prompt:
                print(f"[WARN] Skip item #{num_in} because prompt is empty.")
                continue

            assistant_text = teacher_output_to_text(teacher_output)
            if not assistant_text:
                print(f"[WARN] Skip item #{num_in} because teacher_output is empty.")
                continue

            chat_item = {
                "messages": [
                    {"role": "system", "content": args.system_prompt},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": assistant_text},
                ]
            }
            fout.write(json.dumps(chat_item, ensure_ascii=False) + "\n")
            num_out += 1

    print(f"[DONE] Read {num_in} items, wrote {num_out} chat examples to {out_path}")


if __name__ == "__main__":
    main()
