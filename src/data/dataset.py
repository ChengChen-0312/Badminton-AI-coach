from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

# Supported video extensions – add more if new data arrives with different formats.
ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

# ImageNet statistics keep compatibility with torchvision backbones.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def discover_video_files(root_dir: Path | str) -> Tuple[List[Tuple[Path, str]], List[str]]:
    """Discover videos under root_dir/<class_name>/. Returns samples and class list.

    This automatically adapts to new classes added as new folders under the root directory,
    so expanding the dataset does not require any code change.
    """
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    class_names = sorted([p.name for p in root.iterdir() if p.is_dir()])
    if not class_names:
        raise ValueError(f"No class folders found in {root}")

    samples: List[Tuple[Path, str]] = []
    for class_name in class_names:
        class_dir = root / class_name
        for video_path in sorted(class_dir.iterdir()):
            if video_path.is_file() and video_path.suffix.lower() in ALLOWED_EXTENSIONS:
                samples.append((video_path, class_name))

    if not samples:
        raise ValueError(f"No video files found under {root}")

    return samples, class_names


def _default_transform(frame_size: int) -> T.Compose:
    """Basic transform used if none is provided."""
    return T.Compose(
        [
            T.ToPILImage(),
            T.Resize((frame_size, frame_size)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


class VideoDataset(Dataset):
    """Simple folder-based video dataset.

    Each item returns (video_tensor, label_index, label_name, video_path).
    """

    def __init__(
        self,
        samples: Sequence[Tuple[Path, str]],
        class_to_idx: Dict[str, int],
        num_frames: int = 16,
        frame_size: int = 224,
        frame_step: int = 1,
        transform: Optional[Callable] = None,
    ) -> None:
        self.samples = list(samples)
        self.class_to_idx = class_to_idx
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.frame_step = frame_step
        self.transform = transform or _default_transform(frame_size)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str, str]:
        video_path, label_name = self.samples[idx]
        frames = self._load_video_frames(video_path)
        processed = [self.transform(frame) for frame in frames]
        video_tensor = torch.stack(processed)  # (T, C, H, W)
        label_idx = self.class_to_idx[label_name]
        return video_tensor, label_idx, label_name, str(video_path)

    def _load_video_frames(self, video_path: Path) -> List[np.ndarray]:
        """Read video and sample frames uniformly."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        frames: List[np.ndarray] = []
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % self.frame_step == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
            frame_idx += 1
        cap.release()

        if not frames:
            raise RuntimeError(f"No frames read from video: {video_path}")

        sampled_indices = self._sample_indices(len(frames))
        return [frames[i] for i in sampled_indices]

    def _sample_indices(self, num_available: int) -> List[int]:
        """Uniformly sample frame indices to reach num_frames length."""
        if num_available == 1:
            return [0 for _ in range(self.num_frames)]

        positions = np.linspace(0, num_available - 1, self.num_frames)
        indices = [int(round(p)) for p in positions]
        indices = [min(i, num_available - 1) for i in indices]
        return indices
