from __future__ import annotations

from torchvision import transforms as T

from .dataset import IMAGENET_MEAN, IMAGENET_STD


def get_train_transforms(frame_size: int = 224) -> T.Compose:
    """Augmentations for training frames."""
    return T.Compose(
        [
            T.ToPILImage(),
            T.Resize(int(frame_size * 1.15)),
            T.RandomCrop(frame_size),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def get_val_transforms(frame_size: int = 224) -> T.Compose:
    """Validation transforms: deterministic resize + center crop."""
    return T.Compose(
        [
            T.ToPILImage(),
            T.Resize(int(frame_size * 1.05)),
            T.CenterCrop(frame_size),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
