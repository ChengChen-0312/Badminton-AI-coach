from __future__ import annotations

import os
from typing import Optional, Sequence

import mlx.core as mx
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config


class MLXStudent:
    """
    MLX-based student (e.g., Qwen3-VL-4B-Instruct-4bit) with optional LoRA adapter
    applied to the language model while keeping the full vision tower for image input.
    """

    def __init__(
        self,
        model_path: str = "/Users/chencheng/llm/qwen3-4b",
        adapter_path: Optional[str] = None,
    ) -> None:
        print(f"[MLXStudent] Loading base VLM from: {model_path}")
        self.model, self.processor = load(model_path)
        self.config = load_config(model_path)
        if adapter_path:
            self._load_adapters(adapter_path)

    def _load_adapters(self, adapter_path: str) -> None:
        """Load LoRA adapters and apply to the language model (strict=False)."""
        adapter_file = None
        for fname in ("adapters.safetensors", "adapters.npz"):
            candidate = os.path.join(adapter_path, fname)
            if os.path.exists(candidate):
                adapter_file = candidate
                break
        if adapter_file is None:
            print(f"[MLXStudent] Warning: no adapter found in {adapter_path}")
            return

        print(f"[MLXStudent] Loading adapters from: {adapter_file}")
        try:
            if adapter_file.endswith(".safetensors"):
                try:
                    from safetensors import safe_open

                    adapters = {}
                    with safe_open(adapter_file, framework="numpy") as f:
                        for key in f.keys():
                            adapters[key] = mx.array(f.get_tensor(key))
                except ImportError:
                    adapters = dict(mx.load(adapter_file))
            else:
                adapters = dict(mx.load(adapter_file))
        except Exception as exc:  # pragma: no cover
            print(f"[MLXStudent] Failed to load adapters: {exc}")
            return

        if hasattr(self.model, "language_model"):
            self.model.language_model.load_weights(list(adapters.items()), strict=False)
        else:
            self.model.load_weights(list(adapters.items()), strict=False)
        print(f"[MLXStudent] Applied {len(adapters)} adapter weights")

    def score_motion(
        self,
        description: str,
        image_path: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 300,
    ) -> str:
        """Evaluate motion with optional image input, returning JSON text."""
        prompt = (
            "You are a professional badminton coach.\n"
            "Given the motion description, provide mistakes, advice, and a score (0-100) in JSON.\n\n"
            f"Motion description:\n{description}\n\n"
            'Return JSON with keys: "mistake", "advice", "score".'
        )
        images: Sequence[str] | None = [image_path] if image_path else None
        formatted = apply_chat_template(
            self.processor,
            self.config,
            prompt,
            num_images=len(images) if images else 0,
        )
        result = generate(
            self.model,
            self.processor,
            formatted,
            image=images,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        if hasattr(result, "text"):
            return result.text
        return str(result)

    def score_motion_with_image(self, image_path: str, stroke_type: Optional[str] = None) -> str:
        """Direct image-based evaluation with optional stroke type hint."""
        prompt = "分析这张羽毛球动作图片，评估动作的标准程度。"
        if stroke_type:
            prompt = f"分析这张羽毛球{stroke_type}动作图片，评估动作的标准程度。"
        prompt += (
            "\n\n请以JSON格式输出评估结果，包含:\n"
            "- mistake: 动作问题描述\n- advice: 改进建议\n- score: 评分(0-100)\n\n只输出JSON，不要其他内容。"
        )
        formatted = apply_chat_template(self.processor, self.config, prompt, num_images=1)
        result = generate(
            self.model,
            self.processor,
            formatted,
            image=[image_path],
            max_tokens=300,
            temperature=0.3,
        )
        if hasattr(result, "text"):
            return result.text
        return str(result)

    def generate_feedback(self, description: str, **kwargs) -> str:
        """Backward-compatible wrapper."""
        return self.score_motion(description, **kwargs)
