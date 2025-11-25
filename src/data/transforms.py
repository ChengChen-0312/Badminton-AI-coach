from __future__ import annotations

from torchvision import transforms as T

from .dataset import IMAGENET_MEAN, IMAGENET_STD


def get_train_transforms(frame_size: int = 224, augmentation: str = "strong") -> T.Compose:
    """Augmentations for training frames."""
    if augmentation == "strong":
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
    # Fallback to light augmentations
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
