import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path
from typing import List, Dict, Any, Optional

# Court dimensions in meters
COURT_W = 6.1
COURT_H = 13.4
SINGLES_W = 5.18
SHORT_SERVICE = 1.98
LONG_SERVICE_DBL = 0.76


def _draw_court(
    ax,
    width: float = COURT_W,
    height: float = COURT_H,
    floor_color: str = "#2b9a5c",  # lighter green default
) -> None:
    """Draw a badminton court with key lines."""
    ax.set_facecolor(floor_color)
    # Outer doubles boundary
    ax.add_patch(Rectangle((0, 0), width, height, fill=False, edgecolor="white", linewidth=2))
    # Singles sidelines
    margin = (width - SINGLES_W) / 2.0
    ax.add_patch(Rectangle((margin, 0), SINGLES_W, height, fill=False, edgecolor="white", linewidth=1))
    # Net (center line)
    ax.plot([0, width], [height / 2.0, height / 2.0], color="white", linewidth=2)
    # Short service lines (both sides)
    ax.plot([0, width], [height / 2.0 - SHORT_SERVICE, height / 2.0 - SHORT_SERVICE], color="white", linewidth=1.2)
    ax.plot([0, width], [height / 2.0 + SHORT_SERVICE, height / 2.0 + SHORT_SERVICE], color="white", linewidth=1.2)
    # Long service lines for doubles (back boundary minus 0.76m)
    ax.plot([0, width], [height - LONG_SERVICE_DBL, height - LONG_SERVICE_DBL], color="white", linestyle="--", linewidth=1)
    ax.plot([0, width], [LONG_SERVICE_DBL, LONG_SERVICE_DBL], color="white", linestyle="--", linewidth=1)
    # Center service line
    ax.plot([width / 2.0, width / 2.0], [height / 2.0 - SHORT_SERVICE, height - LONG_SERVICE_DBL], color="white", linewidth=1)
    ax.plot([width / 2.0, width / 2.0], [LONG_SERVICE_DBL, height / 2.0 + SHORT_SERVICE], color="white", linewidth=1)
    # 3x3 grid guides (light)
    ax.vlines([width / 3.0, 2 * width / 3.0], 0, height, colors="gray", linestyles=":", linewidth=0.8)
    ax.hlines([height / 3.0, 2 * height / 3.0], 0, width, colors="gray", linestyles=":", linewidth=0.8)
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.invert_yaxis()  # keep broadcast-style top origin


def create_landing_heatmap(
    strokes: List[Dict[str, Any]],
    out_path: str,
    bins: int = 32,
    figsize: int = 6,
    court_image_path: Optional[str] = None,
    floor_color: str = "#2b9a5c",
):
    """
    Generate a landing heatmap from stroke summaries over a court background.
    Requires strokes to have landing_x and landing_y in meters.
    """
    xs, ys = [], []
    for s in strokes:
        if s.get("landing_x") is not None and s.get("landing_y") is not None:
            xs.append(s["landing_x"])
            ys.append(s["landing_y"])

    if len(xs) == 0:
        print("No landing points – skip heatmap")
        return None

    xs = np.array(xs)
    ys = np.array(ys)

    heatmap, xedges, yedges = np.histogram2d(
        xs, ys, bins=bins, range=[[0, COURT_W], [0, COURT_H]]
    )

    fig, ax = plt.subplots(figsize=(figsize, figsize * 2))
    ax.set_title("Landing Heatmap")

    if court_image_path:
        try:
            img = plt.imread(court_image_path)
            ax.imshow(img, extent=[0, COURT_W, 0, COURT_H], origin="upper", aspect="auto")
        except Exception:
            _draw_court(ax, floor_color=floor_color)
    else:
        _draw_court(ax, floor_color=floor_color)

    # Render heatmap as semi-transparent layer
    # Render heatmap only if we have more than one populated bin to avoid a single opaque block
    if np.count_nonzero(heatmap) > 1:
        hm = ax.imshow(
            heatmap.T,
            origin="lower",
            extent=[0, COURT_W, 0, COURT_H],
            cmap="hot",
            alpha=0.35,
            aspect="auto",
        )
        cbar = plt.colorbar(hm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Hit Density")

    # Overlay red circular points for individual landings
    ax.scatter(xs, ys, c="red", s=20, alpha=0.8, edgecolors="white", linewidths=0.3)

    ax.set_xlabel("Width (m)")
    ax.set_ylabel("Length (m)")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return str(out_path)
