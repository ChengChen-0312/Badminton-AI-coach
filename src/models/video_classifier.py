from __future__ import annotations

from typing import Tuple

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, ResNet50_Weights, resnet18, resnet50
from torchvision.models.video import (
    MC3_18_Weights,
    R3D_18_Weights,
    mc3_18,
    r3d_18,
)
from torchvision.models.video import resnet as video_resnet


def _infer_feature_dim(backbone: nn.Module) -> int:
    """Best-effort feature dimension extraction and removal of the final head."""
    if hasattr(backbone, "fc"):
        dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        return dim
    if hasattr(backbone, "classifier") and isinstance(backbone.classifier, nn.Linear):
        dim = backbone.classifier.in_features
        backbone.classifier = nn.Identity()
        return dim
    if hasattr(backbone, "head") and hasattr(backbone.head, "proj"):
        proj = backbone.head.proj
        dim = proj.in_features if hasattr(proj, "in_features") else proj.out_features
        backbone.head.proj = nn.Identity()
        return dim
    if hasattr(backbone, "blocks") and hasattr(backbone.blocks[-1], "proj"):
        proj = backbone.blocks[-1].proj
        dim = proj.in_features if hasattr(proj, "in_features") else proj.out_features
        backbone.blocks[-1].proj = nn.Identity()
        return dim
    raise ValueError("Could not infer feature dimension for backbone.")


def _build_backbone(name: str, pretrained: bool) -> Tuple[nn.Module, int, bool, bool]:
    """Create backbone, returning (module, feature_dim, is_3d, is_slowfast)."""
    name = name.lower()
    if name == "r3d_18":
        weights = R3D_18_Weights.DEFAULT if pretrained else None
        backbone = r3d_18(weights=weights)
        feature_dim = _infer_feature_dim(backbone)
        return backbone, feature_dim, True, False
    if name == "mc3_18":
        weights = MC3_18_Weights.DEFAULT if pretrained else None
        backbone = mc3_18(weights=weights)
        feature_dim = _infer_feature_dim(backbone)
        return backbone, feature_dim, True, False
    if name == "r3d_34":
        # Custom 34-layer 3D ResNet (no official weights).
        backbone = video_resnet._video_resnet(
            block=video_resnet.BasicBlock,
            conv_makers=[video_resnet.Conv3DSimple] * 4,
            layers=[3, 4, 6, 3],
            stem=video_resnet.BasicStem,
            weights=None,
            progress=True,
        )
        feature_dim = _infer_feature_dim(backbone)
        return backbone, feature_dim, True, False
    if name == "resnet18":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)
        feature_dim = _infer_feature_dim(backbone)
        return backbone, feature_dim, False, False
    if name == "resnet50":
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        backbone = resnet50(weights=weights)
        feature_dim = _infer_feature_dim(backbone)
        return backbone, feature_dim, False, False

    if name in {"x3d_s", "x3d_m"}:
        try:
            from torchvision.models.video import X3D_M_Weights, X3D_S_Weights, x3d_m, x3d_s
        except Exception as e:  # pragma: no cover - optional dependency
            raise ValueError("x3d models require a newer torchvision build.") from e
        if name == "x3d_s":
            weights = X3D_S_Weights.DEFAULT if pretrained else None
            backbone = x3d_s(weights=weights)
        else:
            weights = X3D_M_Weights.DEFAULT if pretrained else None
            backbone = x3d_m(weights=weights)
        feature_dim = _infer_feature_dim(backbone)
        return backbone, feature_dim, True, False

    if name == "slowfast_r50":
        try:
            from torchvision.models.video import SlowFast_R50_Weights, slowfast_r50
        except Exception as e:  # pragma: no cover - optional dependency
            raise ValueError("slowfast_r50 requires a newer torchvision build.") from e
        weights = SlowFast_R50_Weights.DEFAULT if pretrained else None
        backbone = slowfast_r50(weights=weights)
        feature_dim = _infer_feature_dim(backbone)
        return backbone, feature_dim, True, True

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
        backbone, feature_dim, is_3d, is_slowfast = _build_backbone(backbone_name, pretrained)
        self.backbone = backbone
        self.feature_dim = feature_dim
        self.is_3d = is_3d
        self.is_slowfast = is_slowfast
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
            if self.is_slowfast:
                alpha = 4
                fast_pathway = x
                slow_pathway = x[:, :, ::alpha, :, :]
                if self.use_channels_last and hasattr(torch, "channels_last_3d"):
                    fast_pathway = fast_pathway.contiguous(memory_format=torch.channels_last_3d)
                    slow_pathway = slow_pathway.contiguous(memory_format=torch.channels_last_3d)
                feats = self.backbone([slow_pathway, fast_pathway])
            else:
                if self.use_channels_last and hasattr(torch, "channels_last_3d"):
                    x = x.contiguous(memory_format=torch.channels_last_3d)
                feats = self.backbone(x)
            if isinstance(feats, (list, tuple)):
                feats = feats[0]
            if feats.ndim == 5:
                # Fallback in case a custom backbone keeps spatiotemporal dims.
                feats = feats.mean(dim=[2, 3, 4])
        else:
            b, t, c, h, w = x.shape
            frames = x.view(b * t, c, h, w)
            if self.use_channels_last:
                frames = frames.contiguous(memory_format=torch.channels_last)
            feats = self.backbone(frames)
            if isinstance(feats, (list, tuple)):
                feats = feats[0]
            feats = feats.view(b, t, -1).mean(dim=1)
        logits = self.classifier(feats)
        return logits
