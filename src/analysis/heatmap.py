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
    # Court coordinates: origin at near baseline (y=0), y increases toward far baseline.


def _create_xy_heatmap(
    strokes: List[Dict[str, Any]],
    out_path: str,
    x_key: str,
    y_key: str,
    title: str,
    bins: int = 32,
    figsize: int = 6,
    court_image_path: Optional[str] = None,
    floor_color: str = "#2b9a5c",
):
    """
    Generate a heatmap from stroke summaries over a court background.
    Requires `x_key`/`y_key` in meters.
    """
    xs, ys = [], []
    for s in strokes:
        if s.get(x_key) is not None and s.get(y_key) is not None:
            xs.append(s[x_key])
            ys.append(s[y_key])

    if len(xs) == 0:
        print(f"No points for {title} – skip heatmap")
        return None

    xs = np.array(xs)
    ys = np.array(ys)

    heatmap, xedges, yedges = np.histogram2d(
        xs, ys, bins=bins, range=[[0, COURT_W], [0, COURT_H]]
    )

    fig, ax = plt.subplots(figsize=(figsize, figsize * 2))
    ax.set_title(title)

    if court_image_path:
        try:
            img = plt.imread(court_image_path)
            ax.imshow(img, extent=[0, COURT_W, 0, COURT_H], origin="lower", aspect="auto")
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


def create_landing_heatmap(
    strokes: List[Dict[str, Any]],
    out_path: str,
    bins: int = 32,
    figsize: int = 6,
    court_image_path: Optional[str] = None,
    floor_color: str = "#2b9a5c",
    show_predicted: bool = True,
):
    """
    Landing (shuttle) heatmap using `landing_x/landing_y` in meters.

    Notes:
      - Strokes with `landing_predicted=True` are not included in the density grid (to avoid
        misleading "ground truth"), but can be rendered as separate markers.
    """
    observed_xs, observed_ys = [], []
    pred_xs, pred_ys = [], []
    for s in strokes:
        x = s.get("landing_x")
        y = s.get("landing_y")
        if x is None or y is None:
            continue
        if bool(s.get("landing_predicted", False)):
            pred_xs.append(x)
            pred_ys.append(y)
        else:
            observed_xs.append(x)
            observed_ys.append(y)

    if len(observed_xs) == 0 and (not show_predicted or len(pred_xs) == 0):
        print("No points for Landing Heatmap – skip heatmap")
        return None

    fig, ax = plt.subplots(figsize=(figsize, figsize * 2))
    ax.set_title("Landing Heatmap")

    if court_image_path:
        try:
            img = plt.imread(court_image_path)
            ax.imshow(img, extent=[0, COURT_W, 0, COURT_H], origin="lower", aspect="auto")
        except Exception:
            _draw_court(ax, floor_color=floor_color)
    else:
        _draw_court(ax, floor_color=floor_color)

    if observed_xs:
        xs = np.array(observed_xs)
        ys = np.array(observed_ys)
        heatmap, _, _ = np.histogram2d(xs, ys, bins=bins, range=[[0, COURT_W], [0, COURT_H]])
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
        ax.scatter(xs, ys, c="red", s=20, alpha=0.85, edgecolors="white", linewidths=0.3, label="observed")

    if show_predicted and pred_xs:
        px = np.array(pred_xs)
        py = np.array(pred_ys)
        ax.scatter(px, py, c="#ffcc00", s=28, alpha=0.9, marker="x", linewidths=1.2, label="predicted")

    if (observed_xs and pred_xs) or (pred_xs and not observed_xs):
        ax.legend(loc="upper right", framealpha=0.8)

    ax.set_xlabel("Width (m)")
    ax.set_ylabel("Length (m)")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return str(out_path)


def create_hitter_heatmap(
    strokes: List[Dict[str, Any]],
    out_path: str,
    bins: int = 32,
    figsize: int = 6,
    court_image_path: Optional[str] = None,
    floor_color: str = "#2b9a5c",
):
    """Hitter (contact) heatmap using `hitter_x/hitter_y` in meters."""
    return _create_xy_heatmap(
        strokes=strokes,
        out_path=out_path,
        x_key="hitter_x",
        y_key="hitter_y",
        title="Hitter (Contact) Heatmap",
        bins=bins,
        figsize=figsize,
        court_image_path=court_image_path,
        floor_color=floor_color,
    )
