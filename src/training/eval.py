from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torch import nn

from src.models.video_classifier import VideoClassifier
from src.training.train import build_dataloaders, load_config
from src.training.utils import get_device, load_checkpoint, move_to_device


def evaluate_model(
    model: nn.Module,
    dataloader,
    device: torch.device,
    classes: List[str],
    use_channels_last: bool,
) -> Tuple[float, float, Dict[str, float], List[int], List[int]]:
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    per_class_correct = {cls: 0 for cls in classes}
    per_class_total = {cls: 0 for cls in classes}
    all_preds: List[int] = []
    all_labels: List[int] = []

    with torch.no_grad():
        for videos, labels, _, _ in dataloader:
            videos, labels = move_to_device(
                videos,
                labels,
                device,
                use_channels_last=use_channels_last,
                is_3d=model.is_3d,
            )
            outputs = model(videos)
            loss = criterion(outputs, labels)

            preds = outputs.argmax(1)
            total_loss += loss.item() * labels.size(0)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

            for pred, target in zip(preds.tolist(), labels.tolist()):
                class_name = classes[target]
                per_class_total[class_name] += 1
                per_class_correct[class_name] += int(pred == target)
                all_preds.append(pred)
                all_labels.append(target)

    avg_loss = total_loss / max(total_samples, 1)
    avg_acc = total_correct / max(total_samples, 1)
    per_class_acc = {
        cls: (per_class_correct[cls] / per_class_total[cls] if per_class_total[cls] > 0 else 0.0)
        for cls in classes
    }
    return avg_loss, avg_acc, per_class_acc, all_preds, all_labels


def run_eval(config_path: str | Path, checkpoint_path: str | Path) -> Dict[str, Any]:
    config = load_config(config_path)
    device = get_device(config["training"]["device"])

    checkpoint_meta = torch.load(checkpoint_path, map_location="cpu")
    classes_override = checkpoint_meta.get("classes")

    _, val_loader, classes = build_dataloaders(config, class_names=classes_override)
    model = VideoClassifier(
        num_classes=len(classes),
        backbone_name=config["model"]["backbone"],
        pretrained=False,
        use_channels_last=config["training"].get("use_channels_last", False),
    ).to(device)

    checkpoint = load_checkpoint(model, Path(checkpoint_path), device)
    print(f"Loaded checkpoint from {checkpoint_path}, epoch={checkpoint.get('epoch')}, val_acc={checkpoint.get('val_acc')}")

    val_loss, val_acc, per_class_acc, preds, labels = evaluate_model(
        model,
        val_loader,
        device,
        classes,
        use_channels_last=config["training"].get("use_channels_last", False),
    )
    print(f"Validation accuracy: {val_acc:.3f}, loss: {val_loss:.4f}")
    for cls, acc in per_class_acc.items():
        print(f"  {cls}: {acc:.3f}")

    return {
        "val_loss": val_loss,
        "val_acc": val_acc,
        "per_class_acc": per_class_acc,
        "preds": preds,
        "labels": labels,
        "classes": classes,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate badminton stroke classifier.")
    parser.add_argument("--config", type=str, default="src/config/default.yaml", help="Path to YAML config.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint.")
    args = parser.parse_args()

    run_eval(args.config, args.checkpoint)
