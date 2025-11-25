from __future__ import annotations

import random
from typing import Dict, Sequence, Tuple


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


def compute_class_counts(samples: Sequence, class_to_idx: Dict[str, int]) -> Dict[int, int]:
    """Count samples per class index."""
    counts = {idx: 0 for idx in class_to_idx.values()}
    for _, class_name in samples:
        counts[class_to_idx[class_name]] += 1
    return counts


def compute_class_weights(train_counts: Dict[int, int]) -> list[float]:
    """Inverse-frequency weights for CrossEntropyLoss."""
    total = sum(train_counts.values())
    weights = []
    for idx in sorted(train_counts.keys()):
        count = max(train_counts[idx], 1)
        weights.append(total / count)
    # Normalize to keep loss scale stable
    weight_sum = sum(weights)
    if weight_sum > 0:
        weights = [w / weight_sum * len(weights) for w in weights]
    return weights
