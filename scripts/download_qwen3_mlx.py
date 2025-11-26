#!/usr/bin/env python3
"""Download mlx-vlm Qwen3 models to local directories."""

from __future__ import annotations

from huggingface_hub import snapshot_download


def main() -> None:
    # Teacher 30B 4bit
    snapshot_download(
        "mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit",
        local_dir="/Users/chencheng/llm/qwen3-30b",
        ignore_patterns="*.safetensors.index.json",
    )
    # Student 4B 4bit
    snapshot_download(
        "mlx-community/Qwen3-VL-4B-Instruct-4bit",
        local_dir="/Users/chencheng/llm/qwen3-4b",
        ignore_patterns="*.safetensors.index.json",
    )
    print("Download complete to /Users/chencheng/llm/qwen3-30b and /Users/chencheng/llm/qwen3-4b")


if __name__ == "__main__":
    main()
