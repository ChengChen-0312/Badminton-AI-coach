from __future__ import annotations

from torchvision import transforms as T

from .dataset import IMAGENET_MEAN, IMAGENET_STD


def get_train_transforms(frame_size: int = 224, augmentation: str = "strong", strong_aug: bool = True) -> T.Compose:
    """Augmentations for training frames."""
    if strong_aug:
        # Stronger jitter/crop for larger data regimes.
        return T.Compose(
            [
                T.ToPILImage(),
                T.RandomResizedCrop(frame_size, scale=(0.6, 1.0)),
                T.RandomHorizontalFlip(),
                T.RandomRotation(degrees=10),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
    # Mild augmentations to preserve subtle cues (e.g., backhand vs forehand).
    return T.Compose(
        [
            T.ToPILImage(),
            T.RandomResizedCrop(frame_size, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(p=0.4),
            T.RandomRotation(degrees=6),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
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
