from __future__ import annotations

from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from .distillation_dataset import DistillationDataset


class HFDistillTrainer:
    """
    Simple distillation trainer for HF causal LMs (e.g., Qwen2.5-VL-4B-Instruct).
    Trains on text-only teacher outputs; assumes instruction-style prompting.
    """

    def __init__(
        self,
        student_model: str = "Qwen/Qwen2.5-VL-4B-Instruct",
        lr: float = 5e-6,
        max_len: int = 512,
        device: str = "auto",
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(student_model, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            student_model,
            torch_dtype=torch.float16,
            device_map="auto" if device == "auto" else {"": device},
            trust_remote_code=True,
        )
        self.model.train()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.max_len = max_len

    def _collate(self, batch: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
        prompts = []
        for item in batch:
            prompts.append(
                f"Motion description:\n{item['description']}\n\n"
                f"Teacher feedback:\n{item['teacher_output']}\n\n"
                "Student response (match teacher style):"
            )
        enc = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_len,
        )
        input_ids = enc["input_ids"]
        labels = input_ids.clone()
        return {"input_ids": input_ids.to(self.model.device), "labels": labels.to(self.model.device)}

    def train(
        self,
        samples: List[Dict[str, str]],
        batch_size: int = 1,
        epochs: int = 1,
        warmup_ratio: float = 0.1,
        grad_accum_steps: int = 1,
    ) -> None:
        dataset = DistillationDataset([])
        dataset.samples = samples
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=self._collate)
        total_steps = len(dataloader) * epochs // max(1, grad_accum_steps)
        scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=int(total_steps * warmup_ratio), num_training_steps=total_steps
        )

        step = 0
        for _ in range(epochs):
            for i, batch in enumerate(dataloader):
                outputs = self.model(**batch)
                loss = outputs.loss / grad_accum_steps
                loss.backward()
                if (i + 1) % grad_accum_steps == 0:
                    self.optimizer.step()
                    scheduler.step()
                    self.optimizer.zero_grad()
                    step += 1
        self.model.eval()

    def save(self, path: str) -> None:
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
