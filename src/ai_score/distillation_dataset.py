from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from torch.utils.data import Dataset


@dataclass
class DistillSample:
    description: str
    teacher_output: str  # serialized JSON/text from teacher


class DistillationDataset(Dataset):
    """Simple in-memory distillation dataset."""

    def __init__(self, samples: List[DistillSample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        sample = self.samples[idx]
        return {"description": sample.description, "teacher_output": sample.teacher_output}
