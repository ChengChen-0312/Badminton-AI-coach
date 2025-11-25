from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_num_threads(num: int = 4) -> None:
    """Limit CPU threads to reduce contention and dataloader overhead."""
    import os

    torch.set_num_threads(num)
    os.environ["OMP_NUM_THREADS"] = str(num)
    os.environ["MKL_NUM_THREADS"] = str(num)


def accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    preds = outputs.argmax(dim=1)
    correct = (preds == targets).sum().item()
    total = targets.numel()
    return correct / total if total > 0 else 0.0


def save_checkpoint(state: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: Path,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


def get_device(device_name: str) -> torch.device:
    """Prefer MPS when available; otherwise fall back gracefully."""
    if device_name == "cpu":
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available() and device_name == "cuda":
        return torch.device("cuda")
    # default fallback
    return torch.device("cpu")


def move_to_device(
    videos: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    use_channels_last: bool = False,
    is_3d: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    videos = videos.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)

    if use_channels_last:
        if is_3d and hasattr(torch, "channels_last_3d"):
            # Convert to (B, C, T, H, W) and channels_last_3d for better MPS perf.
            videos = videos.permute(0, 2, 1, 3, 4).contiguous(memory_format=torch.channels_last_3d)
        elif not is_3d:
            # Keep layout (B, T, C, H, W); channels_last will be applied after flattening in forward.
            videos = videos.contiguous()
    return videos, labels
