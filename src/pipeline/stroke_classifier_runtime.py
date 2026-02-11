from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from src.data.transforms import get_val_transforms
from src.models.video_classifier import VideoClassifier
from src.pipeline.label_space import build_label_space_metadata


@dataclass
class ClassifierPrediction:
    label: str
    confidence: float
    topk: List[Dict[str, float]]


def _resolve_device(device: str) -> torch.device:
    name = str(device or "auto").strip().lower()
    if name in ("", "auto"):
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if name == "mps":
        return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    if name == "cuda":
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return torch.device("cpu")


class StrokeClassifierRuntime:
    """Runtime helper to classify stroke clips from a saved training checkpoint."""

    def __init__(
        self,
        model: VideoClassifier,
        classes: Sequence[str],
        device: torch.device,
        frame_size: int,
        num_frames: int,
        focus_court: str = "full",
        sampling: str = "uniform",
        topk: int = 3,
    ) -> None:
        self.model = model
        self.classes = list(classes)
        self.device = device
        self.frame_size = int(frame_size)
        self.num_frames = int(max(1, num_frames))
        self.focus_court = str(focus_court or "full").strip().lower()
        self.sampling = str(sampling or "uniform").strip().lower()
        self.topk = int(max(1, topk))
        self.transform = get_val_transforms(self.frame_size)
        self.label_space = build_label_space_metadata(self.classes)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: str = "auto",
        frame_size: Optional[int] = None,
        num_frames: Optional[int] = None,
        focus_court: Optional[str] = None,
        sampling: Optional[str] = None,
        topk: int = 3,
    ) -> "StrokeClassifierRuntime":
        ckpt_path = Path(checkpoint_path).expanduser()
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Stroke classifier checkpoint not found: {ckpt_path}")

        checkpoint = torch.load(ckpt_path, map_location="cpu")
        classes = checkpoint.get("classes")
        if not isinstance(classes, list) or len(classes) == 0:
            raise ValueError("Checkpoint missing valid `classes` list.")

        cfg = checkpoint.get("config") if isinstance(checkpoint.get("config"), dict) else {}
        model_cfg = cfg.get("model", {}) if isinstance(cfg.get("model"), dict) else {}
        data_cfg = cfg.get("data", {}) if isinstance(cfg.get("data"), dict) else {}
        training_cfg = cfg.get("training", {}) if isinstance(cfg.get("training"), dict) else {}

        backbone_name = str(model_cfg.get("backbone", "r3d_18"))
        use_channels_last = bool(training_cfg.get("use_channels_last", False))
        runtime_frame_size = int(frame_size if frame_size is not None else data_cfg.get("frame_size", 160))
        runtime_num_frames = int(num_frames if num_frames is not None else data_cfg.get("num_frames", 16))
        runtime_focus = str(focus_court if focus_court is not None else data_cfg.get("focus_court", "full"))
        runtime_sampling = str(sampling if sampling is not None else data_cfg.get("sampling", "uniform"))

        model = VideoClassifier(
            num_classes=len(classes),
            backbone_name=backbone_name,
            pretrained=False,
            use_channels_last=use_channels_last,
        )
        state_dict = checkpoint.get("model_state_dict")
        if not isinstance(state_dict, dict):
            raise ValueError("Checkpoint missing `model_state_dict`.")
        model.load_state_dict(state_dict, strict=True)
        model.eval()

        runtime_device = _resolve_device(device)
        model.to(runtime_device)

        return cls(
            model=model,
            classes=classes,
            device=runtime_device,
            frame_size=runtime_frame_size,
            num_frames=runtime_num_frames,
            focus_court=runtime_focus,
            sampling=runtime_sampling,
            topk=topk,
        )

    def _apply_focus_crop(self, frame_rgb: np.ndarray, ratio: float = 0.58) -> np.ndarray:
        if self.focus_court == "full":
            return frame_rgb
        h, _, _ = frame_rgb.shape
        cutoff = int(h * ratio)
        if self.focus_court == "far":
            return frame_rgb[:cutoff, :, :]
        if self.focus_court == "near":
            return frame_rgb[h - cutoff :, :, :]
        return frame_rgb

    def _sample_indices(self, num_available: int) -> List[int]:
        if num_available <= 1:
            return [0 for _ in range(self.num_frames)]

        if self.sampling == "strided":
            stride = max(1, num_available // self.num_frames)
            idxs = list(range(0, num_available, stride))[: self.num_frames]
            if len(idxs) < self.num_frames:
                idxs.extend([idxs[-1]] * (self.num_frames - len(idxs)))
            return idxs

        # Deterministic uniform sampling for inference.
        pos = np.linspace(0, num_available - 1, self.num_frames)
        idxs = [int(round(v)) for v in pos]
        return [max(0, min(num_available - 1, i)) for i in idxs]

    def _predict_from_frames(self, frames_rgb: Sequence[np.ndarray]) -> Optional[ClassifierPrediction]:
        if len(frames_rgb) == 0:
            return None

        processed: List[torch.Tensor] = []
        for fr in frames_rgb:
            crop = self._apply_focus_crop(fr)
            processed.append(self.transform(crop))
        clip = torch.stack(processed).unsqueeze(0).to(self.device)  # (1, T, C, H, W)

        with torch.no_grad():
            logits = self.model(clip)
            probs = F.softmax(logits, dim=1)[0].detach().cpu().numpy()

        topk = min(self.topk, len(self.classes))
        top_idx = np.argsort(-probs)[:topk]
        top_items = [
            {"label": str(self.classes[int(i)]), "confidence": float(probs[int(i)])}
            for i in top_idx
        ]
        best = top_items[0]
        return ClassifierPrediction(
            label=str(best["label"]),
            confidence=float(best["confidence"]),
            topk=top_items,
        )

    def predict_from_video_segment(
        self,
        video_path: str | Path,
        start_frame: int,
        end_frame: int,
    ) -> Optional[ClassifierPrediction]:
        s = int(start_frame)
        e = int(end_frame)
        if e < s:
            s, e = e, s

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
        try:
            total = max(1, e - s + 1)
            rel_idx = self._sample_indices(total)
            abs_idx = [s + i for i in rel_idx]
            frames_rgb: List[np.ndarray] = []
            for fi in abs_idx:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
                ok, bgr = cap.read()
                if not ok or bgr is None:
                    continue
                frames_rgb.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

            if len(frames_rgb) == 0:
                return None
            while len(frames_rgb) < self.num_frames:
                frames_rgb.append(frames_rgb[-1])
            return self._predict_from_frames(frames_rgb[: self.num_frames])
        finally:
            cap.release()

    def predict_from_frame_store(
        self,
        frame_store_rgb: Mapping[int, np.ndarray],
        start_frame: int,
        end_frame: int,
    ) -> Optional[ClassifierPrediction]:
        s = int(start_frame)
        e = int(end_frame)
        if e < s:
            s, e = e, s
        available = sorted(int(k) for k in frame_store_rgb.keys() if s <= int(k) <= e)
        if len(available) == 0:
            return None

        sample_local = self._sample_indices(len(available))
        frames_rgb: List[np.ndarray] = []
        for li in sample_local:
            fi = available[int(li)]
            fr = frame_store_rgb.get(fi)
            if isinstance(fr, np.ndarray) and fr.ndim == 3:
                frames_rgb.append(fr)
        if len(frames_rgb) == 0:
            return None
        while len(frames_rgb) < self.num_frames:
            frames_rgb.append(frames_rgb[-1])
        return self._predict_from_frames(frames_rgb[: self.num_frames])

    def predict_labels_for_segments_from_video(
        self,
        video_path: str | Path,
        segments: Sequence[Sequence[int]],
    ) -> List[Dict[str, Any]]:
        outputs: List[Dict[str, Any]] = []
        for seg in segments:
            if len(seg) < 2:
                outputs.append({"label": None, "confidence": None, "topk": []})
                continue
            pred = self.predict_from_video_segment(video_path, int(seg[0]), int(seg[1]))
            if pred is None:
                outputs.append({"label": None, "confidence": None, "topk": []})
            else:
                outputs.append(
                    {
                        "label": pred.label,
                        "confidence": float(pred.confidence),
                        "topk": pred.topk,
                    }
                )
        return outputs
