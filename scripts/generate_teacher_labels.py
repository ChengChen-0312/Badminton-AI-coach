#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.ai_score.teacher_local_32b import LocalTeacher32B


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate teacher labels for distillation.")
    parser.add_argument("--input", required=True, help="Path to motion description JSONL (field: description).")
    parser.add_argument("--output", required=True, help="Path to output JSONL with teacher_output.")
    args = parser.parse_args()

    teacher = LocalTeacher32B()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(args.input, "r") as f_in, open(out_path, "w") as f_out:
        for line in f_in:
            obj = json.loads(line)
            desc = obj.get("description", "")
            teacher_out = teacher.analyse_motion(desc)
            obj["teacher_output"] = teacher_out
            f_out.write(json.dumps(obj) + "\n")


if __name__ == "__main__":
    main()
