from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence


DEFAULT_LABEL_SPACE_VERSION = "stroke_6class_v1"
DEFAULT_6CLASS_LABELS = [
    "backhand_drive",
    "backhand_net_shot",
    "forehand_clear",
    "forehand_drive",
    "forehand_lift",
    "forehand_net_shot",
]


def _norm_labels(labels: Optional[Sequence[str]]) -> List[str]:
    if not labels:
        return []
    out: List[str] = []
    for item in labels:
        s = str(item or "").strip()
        if s:
            out.append(s)
    return out


def infer_label_space_version(labels: Optional[Sequence[str]], fallback: str = DEFAULT_LABEL_SPACE_VERSION) -> str:
    norm = _norm_labels(labels)
    if not norm:
        return str(fallback)
    if norm == DEFAULT_6CLASS_LABELS:
        return DEFAULT_LABEL_SPACE_VERSION
    return f"custom_{len(norm)}class"


def build_label_space_metadata(
    labels: Optional[Sequence[str]],
    *,
    version_hint: Optional[str] = None,
    expected_num_classes: Optional[int] = None,
    expected_class_names: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    class_names = _norm_labels(labels)
    expected_names = _norm_labels(expected_class_names)

    version = str(version_hint or infer_label_space_version(class_names)).strip() or DEFAULT_LABEL_SPACE_VERSION
    num_classes = int(len(class_names))
    expected_num = int(expected_num_classes) if expected_num_classes is not None else None

    mismatches: List[str] = []
    if class_names and expected_num is not None and num_classes != expected_num:
        mismatches.append(f"num_classes:{num_classes}!=expected:{expected_num}")
    if expected_names and class_names and expected_names != class_names:
        mismatches.append("class_names_mismatch")

    return {
        "version": version,
        "num_classes": num_classes,
        "class_names": class_names,
        "expected_num_classes": expected_num,
        "expected_class_names": expected_names if expected_names else None,
        "mismatch": bool(mismatches),
        "mismatch_reasons": mismatches,
    }
