from __future__ import annotations

import random
from typing import Sequence, Tuple


def train_val_split(
    samples: Sequence,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> Tuple[list, list]:
    """Reproducible train/validation split."""
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1")

    rng = random.Random(seed)
    indices = list(range(len(samples)))
    rng.shuffle(indices)
    split_idx = int(len(samples) * train_ratio)
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    train_samples = [samples[i] for i in train_indices]
    val_samples = [samples[i] for i in val_indices]
    return train_samples, val_samples
