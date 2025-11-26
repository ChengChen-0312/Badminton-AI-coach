from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class LocalTeacher32B:
    """
    Local teacher model loader for Qwen3/Qwen2 VL 30B/32B variants.
    Default path points to your local download.
    """

    def __init__(self, model_path: str = "/Users/chencheng/llm/qwen3-30b"):
        print(f"[Teacher32B] Loading local model from: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()

    def analyse_motion(self, description: str, lang: str = "en") -> str:
        prompt = f"""
You are a professional badminton coach.
Analyze the following motion description and give:
1) Mistake analysis
2) Correction advice
3) Score (0-100)

Motion description:
{description}

Return JSON:
{{
    "mistake": "...",
    "advice": "...",
    "score": 0-100
}}
"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.3,
            )
        return self.tokenizer.decode(output[0], skip_special_tokens=True)
