import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Any


def create_stroke_timeline(
    strokes: List[Dict[str, Any]],
    out_path: str,
    figsize=(12, 2),
):
    """
    Build a stroke timeline plot.
    Requires: contact_frame, final_type, hitter_role (near/far).
    """
    if len(strokes) == 0:
        print("No strokes – skip timeline")
        return None

    xs = [s.get("contact_frame", 0) for s in strokes]
    ys = [1 if s.get("hitter_role") == "near" else 2 for s in strokes]
    labels = [s.get("final_type") for s in strokes]

    plt.figure(figsize=figsize)
    plt.title("Stroke Timeline")
    plt.scatter(xs, ys, s=60, c=ys, cmap="coolwarm")

    for x, y, label in zip(xs, ys, labels):
        plt.text(x, y + 0.05, label, fontsize=8, rotation=45)

    plt.yticks([1, 2], ["Near", "Far"])
    plt.xlabel("Frame Index")
    plt.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()

    return str(out_path)
