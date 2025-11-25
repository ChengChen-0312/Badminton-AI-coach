from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
) -> Tuple[float, float, float, Dict[str, float], List[int], List[int], List[str]]:
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    total_top2 = 0

    per_class_correct = {cls: 0 for cls in classes}
    per_class_total = {cls: 0 for cls in classes}
    all_preds: List[int] = []
    all_labels: List[int] = []
    all_paths: List[str] = []

    with torch.no_grad():
        for videos, labels, _, paths in dataloader:
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
            top2 = outputs.topk(2, dim=1).indices

            total_loss += loss.item() * labels.size(0)
            total_correct += (preds == labels).sum().item()
            total_top2 += sum(label.item() in top2_row.tolist() for label, top2_row in zip(labels, top2))
            total_samples += labels.size(0)

            for pred, target in zip(preds.tolist(), labels.tolist()):
                class_name = classes[target]
                per_class_total[class_name] += 1
                per_class_correct[class_name] += int(pred == target)
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())
            all_paths.extend(paths)

    avg_loss = total_loss / max(total_samples, 1)
    avg_acc = total_correct / max(total_samples, 1)
    top2_acc = total_top2 / max(total_samples, 1)
    per_class_acc = {
        cls: (per_class_correct[cls] / per_class_total[cls] if per_class_total[cls] > 0 else 0.0)
        for cls in classes
    }
    return avg_loss, avg_acc, top2_acc, per_class_acc, all_preds, all_labels, all_paths


def run_eval(config_path: str | Path, checkpoint_path: str | Path) -> Dict[str, Any]:
    config = load_config(config_path)
    device = get_device(config["training"]["device"])

    checkpoint_meta = torch.load(checkpoint_path, map_location="cpu")
    classes_override = checkpoint_meta.get("classes")

    _, val_loader, classes, _ = build_dataloaders(config, class_names=classes_override)
    model = VideoClassifier(
        num_classes=len(classes),
        backbone_name=config["model"]["backbone"],
        pretrained=False,
        use_channels_last=config["training"].get("use_channels_last", False),
    ).to(device)

    checkpoint = load_checkpoint(model, Path(checkpoint_path), device)
    print(f"Loaded checkpoint from {checkpoint_path}, epoch={checkpoint.get('epoch')}, val_acc={checkpoint.get('val_acc')}")

    val_loss, val_acc, top2_acc, per_class_acc, preds, labels, paths = evaluate_model(
        model,
        val_loader,
        device,
        classes,
        use_channels_last=config["training"].get("use_channels_last", False),
    )
    print(f"Validation accuracy: {val_acc:.3f}, top2_acc={top2_acc:.3f}, loss: {val_loss:.4f}")
    for cls, acc in per_class_acc.items():
        print(f"  {cls}: {acc:.3f}")

    output_dir = Path(config["logging"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Confusion matrix
    num_classes = len(classes)
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(labels, preds):
        cm[t, p] += 1
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
    cm_path = output_dir / "confusion_matrix.png"
    plt.tight_layout()
    plt.savefig(cm_path)
    plt.close(fig)
    print(f"Saved confusion matrix to {cm_path}")

    # Optional CSV with predictions
    csv_path = output_dir / "predictions.csv"
    df = pd.DataFrame(
        {
            "video_path": paths,
            "true_label": [classes[i] for i in labels],
            "pred_label": [classes[i] for i in preds],
        }
    )
    df.to_csv(csv_path, index=False)
    print(f"Saved predictions CSV to {csv_path}")

    # Report most confused pairs (top off-diagonal counts).
    flat_confusions = []
    for i in range(num_classes):
        for j in range(num_classes):
            if i == j:
                continue
            flat_confusions.append((cm[i, j], classes[i], classes[j]))
    flat_confusions.sort(reverse=True, key=lambda x: x[0])
    top_pairs = [f"{src}->{dst}: {cnt}" for cnt, src, dst in flat_confusions[:3] if cnt > 0]
    if top_pairs:
        print("Most confused pairs: " + ", ".join(top_pairs))

    return {
        "val_loss": val_loss,
        "val_acc": val_acc,
        "top2_acc": top2_acc,
        "per_class_acc": per_class_acc,
        "preds": preds,
        "labels": labels,
        "classes": classes,
        "confusion_matrix": cm,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate badminton stroke classifier.")
    parser.add_argument("--config", type=str, default="src/config/default.yaml", help="Path to YAML config.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint.")
    args = parser.parse_args()

    run_eval(args.config, args.checkpoint)
