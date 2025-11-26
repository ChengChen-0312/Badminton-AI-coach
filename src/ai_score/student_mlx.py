from __future__ import annotations

from typing import Optional

from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config


class MLXStudent:
    """MLX-based student (e.g., Qwen3-VL-4B-Instruct-4bit)."""

    def __init__(self, model_path: str = "/Users/chencheng/llm/qwen3-4b") -> None:
        print(f"[MLXStudent] Loading MLX model from: {model_path}")
        self.model, self.processor = load(model_path)
        self.config = load_config(model_path)

    def generate_feedback(self, description: str, temperature: float = 0.2, max_tokens: int = 256) -> str:
        prompt = (
            "You are a badminton coach. Given the motion description, provide concise advice and a score (0-100).\n\n"
            f"Motion description:\n{description}\n\n"
            'Return JSON: {"advice": "...", "score": 0-100}'
        )
        formatted = apply_chat_template(self.processor, self.config, prompt, num_images=0)
        output = generate(
            self.model,
            self.processor,
            formatted,
            image=None,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return output
