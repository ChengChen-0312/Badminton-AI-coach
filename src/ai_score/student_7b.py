from __future__ import annotations

from typing import Any, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class Student7B:
    """
    Lightweight student model for action feedback (distilled).
    Default: Qwen2.5-VL-7B-Instruct (replace path if needed).
    """

    def __init__(self, model_path: str = "Qwen/Qwen2.5-VL-7B-Instruct", device: str = "mps"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto" if device == "auto" else {"": device},
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        self.model.eval()

    def generate_feedback(self, description: str, temperature: float = 0.3, max_new_tokens: int = 256) -> str:
        prompt = (
            "You are a badminton coach. Given the motion description, provide concise advice and a score (0-100).\n\n"
            f"Motion description:\n{description}\n\n"
            'Return JSON: {"advice": "...", "score": 0-100}'
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
            )
        return self.tokenizer.decode(out[0], skip_special_tokens=True)
