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
    ) -> None:
        super().__init__()
        backbone, feature_dim, is_3d = _build_backbone(backbone_name, pretrained)
        self.backbone = backbone
        self.feature_dim = feature_dim
        self.is_3d = is_3d
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x shape: (B, T, C, H, W)."""
        if self.is_3d:
            # 3D backbones expect (B, C, T, H, W).
            x = x.permute(0, 2, 1, 3, 4)
            feats = self.backbone(x)
            if feats.ndim == 5:
                # Fallback in case a custom backbone keeps spatiotemporal dims.
                feats = feats.mean(dim=[2, 3, 4])
        else:
            b, t, c, h, w = x.shape
            feats = self.backbone(x.view(b * t, c, h, w))
            feats = feats.view(b, t, -1).mean(dim=1)
        logits = self.classifier(feats)
        return logits
