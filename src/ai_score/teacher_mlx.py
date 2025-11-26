from __future__ import annotations

from typing import Optional

from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config


class MLXTeacher:
    """
    MLX-based teacher using Qwen3-VL-MoE (e.g., 30B 4bit) for motion analysis.
    """

    def __init__(self, model_path: str = "/Users/chencheng/llm/qwen3-30b") -> None:
        print(f"[MLXTeacher] Loading MLX model from: {model_path}")
        self.model, self.processor = load(model_path)
        self.config = load_config(model_path)

    def analyse_motion(self, description: str, temperature: float = 0.3, max_tokens: int = 512) -> str:
        prompt = f"""
You are a professional badminton coach.
Analyze the following motion description and give:
1) Mistake analysis
2) Correction advice
3) Score (0-100)

Return JSON:
{{
    "mistake": "...",
    "advice": "...",
    "score": 0-100
}}

Motion description:
{description}
"""
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
