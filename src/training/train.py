from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import VideoDataset, discover_video_files
from src.data.split import train_val_split
from src.data.transforms import get_train_transforms, get_val_transforms
from src.models.video_classifier import VideoClassifier
from src.training.utils import get_device, save_checkpoint, set_seed


def load_config(config_path: str | Path) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_dataloaders(
    config: Dict[str, Any],
    class_names: List[str] | None = None,
) -> Tuple[DataLoader, DataLoader, List[str]]:
    data_cfg = config["data"]
    root_dir = Path(data_cfg["root_dir"])

    samples, discovered_classes = discover_video_files(root_dir)
    classes = class_names or discovered_classes
    # Keep deterministic ordering; assumes provided class_names match discovered.
    if set(classes) != set(discovered_classes):
        raise ValueError("Provided class_names do not match dataset folders.")
    train_samples, val_samples = train_val_split(
        samples, train_ratio=data_cfg.get("train_val_split", 0.8), seed=config.get("seed", 42)
    )
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}

    train_dataset = VideoDataset(
        samples=train_samples,
        class_to_idx=class_to_idx,
        num_frames=data_cfg.get("num_frames", 16),
        frame_size=data_cfg.get("frame_size", 224),
        frame_step=data_cfg.get("frame_step", 1),
        transform=get_train_transforms(data_cfg.get("frame_size", 224)),
    )
    val_dataset = VideoDataset(
        samples=val_samples,
        class_to_idx=class_to_idx,
        num_frames=data_cfg.get("num_frames", 16),
        frame_size=data_cfg.get("frame_size", 224),
        frame_step=data_cfg.get("frame_step", 1),
        transform=get_val_transforms(data_cfg.get("frame_size", 224)),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"],
        pin_memory=True,
    )
    return train_loader, val_loader, classes


def _validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for videos, labels, _, _ in dataloader:
            videos = videos.to(device)
            labels = labels.to(device)
            outputs = model(videos)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            total_correct += (outputs.argmax(1) == labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = total_loss / max(total_samples, 1)
    avg_acc = total_correct / max(total_samples, 1)
    return avg_loss, avg_acc


def train(config: Dict[str, Any]) -> None:
    device = get_device(config["training"]["device"])
    set_seed(config.get("seed", 42))

    train_loader, val_loader, classes = build_dataloaders(config)
    model = VideoClassifier(
        num_classes=len(classes),
        backbone_name=config["model"]["backbone"],
        pretrained=config["model"].get("pretrained", True),
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    num_epochs = config["training"]["num_epochs"]
    log_interval = config["logging"]["log_interval"]
    output_dir = Path(config["logging"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        total = 0

        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}", leave=False)
        for step, (videos, labels, _, _) in enumerate(progress, start=1):
            videos = videos.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(videos)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            running_correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

            if step % log_interval == 0:
                train_loss = running_loss / max(total, 1)
                train_acc = running_correct / max(total, 1)
                progress.set_postfix({"train_loss": f"{train_loss:.4f}", "train_acc": f"{train_acc:.3f}"})

        train_loss = running_loss / max(total, 1)
        train_acc = running_correct / max(total, 1)

        val_loss, val_acc = _validate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch}: "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.3f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.3f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = output_dir / "best_model.pt"
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "classes": classes,
                    "config": config,
                },
                checkpoint_path,
            )
            print(f"Saved new best checkpoint to {checkpoint_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train badminton stroke classifier.")
    parser.add_argument("--config", type=str, default="src/config/default.yaml", help="Path to YAML config.")
    parser.add_argument("--epochs", type=int, help="Override number of epochs.")
    parser.add_argument("--batch-size", type=int, help="Override batch size.")
    parser.add_argument("--learning-rate", type=float, help="Override learning rate.")
    parser.add_argument("--device", type=str, help="Override device, e.g., cuda or cpu.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = load_config(args.config)

    if args.epochs is not None:
        config["training"]["num_epochs"] = args.epochs
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        config["training"]["learning_rate"] = args.learning_rate
    if args.device is not None:
        config["training"]["device"] = args.device

    train(config)


if __name__ == "__main__":
    main()
