from __future__ import annotations

from typing import Tuple

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.models.video import (
    MC3_18_Weights,
    R3D_18_Weights,
    mc3_18,
    r3d_18,
)


def _build_backbone(name: str, pretrained: bool) -> Tuple[nn.Module, int, bool]:
    """Create backbone, returning (module, feature_dim, is_3d)."""
    name = name.lower()
    if name == "r3d_18":
        weights = R3D_18_Weights.DEFAULT if pretrained else None
        backbone = r3d_18(weights=weights)
        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        return backbone, feature_dim, True
    if name == "mc3_18":
        weights = MC3_18_Weights.DEFAULT if pretrained else None
        backbone = mc3_18(weights=weights)
        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        return backbone, feature_dim, True
    if name == "resnet18":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)
        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        return backbone, feature_dim, False
    raise ValueError(f"Unsupported backbone: {name}")


class VideoClassifier(nn.Module):
    """Wraps a torchvision backbone for video classification."""

    def __init__(
        self,
        num_classes: int,
        backbone_name: str = "r3d_18",
        pretrained: bool = True,
        use_channels_last: bool = False,
    ) -> None:
        super().__init__()
        backbone, feature_dim, is_3d = _build_backbone(backbone_name, pretrained)
        self.backbone = backbone
        self.feature_dim = feature_dim
        self.is_3d = is_3d
        self.use_channels_last = use_channels_last
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x shape: (B, T, C, H, W)."""
        if self.is_3d:
            # Accept either (B, T, C, H, W) or (B, C, T, H, W) for efficiency.
            if x.ndim != 5:
                raise ValueError(f"Expected 5D input, got {x.shape}")
            if x.shape[1] == 3:
                x = x  # already (B, C, T, H, W)
            else:
                x = x.permute(0, 2, 1, 3, 4)
            if self.use_channels_last and hasattr(torch, "channels_last_3d"):
                x = x.contiguous(memory_format=torch.channels_last_3d)
            feats = self.backbone(x)
            if feats.ndim == 5:
                # Fallback in case a custom backbone keeps spatiotemporal dims.
                feats = feats.mean(dim=[2, 3, 4])
        else:
            b, t, c, h, w = x.shape
            frames = x.view(b * t, c, h, w)
            if self.use_channels_last:
                frames = frames.contiguous(memory_format=torch.channels_last)
            feats = self.backbone(frames)
            feats = feats.view(b, t, -1).mean(dim=1)
        logits = self.classifier(feats)
        return logits
