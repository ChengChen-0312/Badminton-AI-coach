from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from tqdm import tqdm

from src.data.dataset import VideoDataset, discover_video_files
from src.data.split import compute_class_counts, compute_class_weights, train_val_split
from src.data.transforms import get_train_transforms, get_val_transforms
from src.models.video_classifier import VideoClassifier
from src.training.utils import (
    get_device,
    move_to_device,
    save_checkpoint,
    set_num_threads,
    set_seed,
)


def load_config(config_path: str | Path) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_dataloaders(
    config: Dict[str, Any],
    class_names: List[str] | None = None,
) -> Tuple[DataLoader, DataLoader, List[str], torch.Tensor]:
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
        root_dir=root_dir,
        num_frames=data_cfg.get("num_frames", 16),
        frame_size=data_cfg.get("frame_size", 224),
        frame_step=data_cfg.get("frame_step", 1),
        sampling=data_cfg.get("sampling", "uniform"),
        transform=get_train_transforms(
            data_cfg.get("frame_size", 224),
            augmentation=data_cfg.get("augmentation", "strong"),
            strong_aug=data_cfg.get("strong_aug", True),
        ),
        focus_court=data_cfg.get("focus_court", "full"),
        use_cache=data_cfg.get("use_cache", False),
        cache_dir=data_cfg.get("cache_dir", "frame_cache"),
    )
    val_dataset = VideoDataset(
        samples=val_samples,
        class_to_idx=class_to_idx,
        root_dir=root_dir,
        num_frames=data_cfg.get("num_frames", 16),
        frame_size=data_cfg.get("frame_size", 224),
        frame_step=data_cfg.get("frame_step", 1),
        sampling=data_cfg.get("sampling", "uniform"),
        transform=get_val_transforms(data_cfg.get("frame_size", 224)),
        focus_court=data_cfg.get("focus_court", "full"),
        use_cache=data_cfg.get("use_cache", False),
        cache_dir=data_cfg.get("cache_dir", "frame_cache"),
    )

    train_counts = compute_class_counts(train_samples, class_to_idx)
    class_weights = torch.tensor(compute_class_weights(train_counts), dtype=torch.float)

    num_workers = config["training"]["num_workers"]
    prefetch_factor = config["training"].get("prefetch_factor", 2)
    persistent_workers = config["training"].get("persistent_workers", num_workers > 0)

    # Optional weighted sampler to emphasize under-represented or difficult classes.
    sampler = None
    if config["training"].get("sampler", "none") == "weighted":
        sample_weights = []
        boost_class = config["training"].get("boost_class")
        boost_factor = config["training"].get("boost_factor", 1.0)
        for _, label_name in train_samples:
            weight = class_weights[class_to_idx[label_name]].item()
            if boost_class and label_name == boost_class:
                weight *= boost_factor
            sample_weights.append(weight)
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    common_loader_kwargs = dict(
        batch_size=config["training"]["batch_size"],
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )

    train_loader = DataLoader(
        train_dataset,
        shuffle=sampler is None,
        sampler=sampler,
        drop_last=True,
        **common_loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        drop_last=False,
        **common_loader_kwargs,
    )
    return train_loader, val_loader, classes, class_weights


def _validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_channels_last: bool,
    classes: List[str],
) -> Tuple[float, float, Dict[str, float]]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    per_class_correct = {cls: 0 for cls in classes}
    per_class_total = {cls: 0 for cls in classes}

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

            total_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
            for pred, target in zip(preds.tolist(), labels.tolist()):
                class_name = classes[target]
                per_class_total[class_name] += 1
                per_class_correct[class_name] += int(pred == target)

    avg_loss = total_loss / max(total_samples, 1)
    avg_acc = total_correct / max(total_samples, 1)
    per_class_acc = {
        cls: (per_class_correct[cls] / per_class_total[cls] if per_class_total[cls] > 0 else 0.0) for cls in classes
    }
    return avg_loss, avg_acc, per_class_acc


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_name: str,
    training_cfg: Dict[str, Any],
    steps_per_epoch: int,
) -> torch.optim.lr_scheduler._LRScheduler | None:
    scheduler_name = (scheduler_name or "none").lower()
    if scheduler_name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=training_cfg["num_epochs"])
    if scheduler_name == "onecycle":
        if steps_per_epoch == 0:
            return None
        return OneCycleLR(
            optimizer,
            max_lr=training_cfg["learning_rate"],
            steps_per_epoch=steps_per_epoch,
            epochs=training_cfg["num_epochs"],
        )
    return None


def train(config: Dict[str, Any]) -> None:
    device = get_device(config["training"]["device"])
    set_seed(config.get("seed", 42))
    set_num_threads(config["training"].get("num_workers", 4))

    print(f"Using court focus: {config['data'].get('focus_court', 'full')}")
    train_loader, val_loader, classes, class_weights = build_dataloaders(config)
    model = VideoClassifier(
        num_classes=len(classes),
        backbone_name=config["model"]["backbone"],
        pretrained=config["model"].get("pretrained", True),
        use_channels_last=config["training"].get("use_channels_last", False),
    ).to(device)

    class_weights = class_weights.to(device)
    weight_tensor = class_weights if config["training"].get("class_weighting", "auto") == "auto" else None
    criterion = nn.CrossEntropyLoss(
        weight=weight_tensor,
        label_smoothing=config["training"].get("label_smoothing", 0.0),
    )
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
    best_val_loss = float("inf")
    patience = config["training"].get("early_stopping_patience", 5)
    patience_counter = 0

    use_amp = config["training"].get("use_amp", False)
    scaler = torch.amp.GradScaler(device.type) if use_amp else None
    scheduler = _build_scheduler(
        optimizer,
        scheduler_name=config["training"].get("scheduler", "none"),
        training_cfg=config["training"],
        steps_per_epoch=len(train_loader),
    )
    grad_clip = config["training"].get("grad_clip", 0.0)

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        total = 0

        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}", leave=False)
        for step, (videos, labels, _, _) in enumerate(progress, start=1):
            videos, labels = move_to_device(
                videos,
                labels,
                device,
                use_channels_last=config["training"].get("use_channels_last", False),
                is_3d=model.is_3d,
            )

            optimizer.zero_grad()
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(videos)
                loss = criterion(outputs, labels)

            if scaler is not None:
                scaler.scale(loss).backward()
                if grad_clip and grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            running_loss += loss.item() * labels.size(0)
            running_correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

            if step % log_interval == 0:
                train_loss = running_loss / max(total, 1)
                train_acc = running_correct / max(total, 1)
                progress.set_postfix({"train_loss": f"{train_loss:.4f}", "train_acc": f"{train_acc:.3f}"})

            if scheduler and isinstance(scheduler, OneCycleLR):
                scheduler.step()

        train_loss = running_loss / max(total, 1)
        train_acc = running_correct / max(total, 1)

        val_loss, val_acc, val_per_class = _validate(
            model,
            val_loader,
            criterion,
            device,
            use_channels_last=config["training"].get("use_channels_last", False),
            classes=classes,
        )

        print(
            f"Epoch {epoch}: "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.3f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.3f}"
        )
        # Quick per-class summary to monitor weak classes.
        per_class_str = ", ".join(f"{cls}:{acc:.2f}" for cls, acc in val_per_class.items())
        print(f"Val per-class acc -> {per_class_str}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = output_dir / "best_model.pt"
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                    "classes": classes,
                    "config": config,
                },
                checkpoint_path,
            )
            print(f"Saved new best checkpoint to {checkpoint_path}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if scheduler and not isinstance(scheduler, OneCycleLR):
            scheduler.step()

        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch} epochs.")
            break


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
