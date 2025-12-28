from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np

MODEL_ID = "bwf_standard_v1"

# Court dimensions (meters).
L = 13.40
WD = 6.10
WS = 5.18
HALF_L = 6.70
HALF_WD = 3.05
HALF_WS = 2.59
SHORT_SERVICE = 1.98
DOUBLES_LONG_FROM_BASELINE = 0.76
DOUBLES_LONG_FROM_NET = HALF_L - DOUBLES_LONG_FROM_BASELINE


@dataclass(frozen=True)
class CourtLine:
    name: str
    p1: Tuple[float, float]
    p2: Tuple[float, float]
    weight: float


def get_bwf_corners() -> np.ndarray:
    """Return outer doubles corners ordered LB, RB, RT, LT."""
    return np.array(
        [
            [-HALF_WD, -HALF_L],
            [HALF_WD, -HALF_L],
            [HALF_WD, HALF_L],
            [-HALF_WD, HALF_L],
        ],
        dtype=np.float32,
    )


def get_bwf_lines() -> List[CourtLine]:
    return [
        CourtLine("baseline_bottom", (-HALF_WD, -HALF_L), (HALF_WD, -HALF_L), 1.0),
        CourtLine("baseline_top", (-HALF_WD, HALF_L), (HALF_WD, HALF_L), 1.0),
        CourtLine("sideline_left_d", (-HALF_WD, -HALF_L), (-HALF_WD, HALF_L), 1.0),
        CourtLine("sideline_right_d", (HALF_WD, -HALF_L), (HALF_WD, HALF_L), 1.0),
        CourtLine("sideline_left_s", (-HALF_WS, -HALF_L), (-HALF_WS, HALF_L), 0.7),
        CourtLine("sideline_right_s", (HALF_WS, -HALF_L), (HALF_WS, HALF_L), 0.7),
        CourtLine("short_near", (-HALF_WS, -SHORT_SERVICE), (HALF_WS, -SHORT_SERVICE), 0.6),
        CourtLine("short_far", (-HALF_WS, SHORT_SERVICE), (HALF_WS, SHORT_SERVICE), 0.6),
        CourtLine(
            "long_near_d",
            (-HALF_WD, -DOUBLES_LONG_FROM_NET),
            (HALF_WD, -DOUBLES_LONG_FROM_NET),
            0.4,
        ),
        CourtLine(
            "long_far_d",
            (-HALF_WD, DOUBLES_LONG_FROM_NET),
            (HALF_WD, DOUBLES_LONG_FROM_NET),
            0.4,
        ),
        CourtLine("center_near", (0.0, -SHORT_SERVICE), (0.0, -DOUBLES_LONG_FROM_NET), 0.4),
        CourtLine("center_far", (0.0, SHORT_SERVICE), (0.0, DOUBLES_LONG_FROM_NET), 0.4),
    ]


def _sample_segment(p1: Tuple[float, float], p2: Tuple[float, float], n: int) -> np.ndarray:
    xs = np.linspace(p1[0], p2[0], n, dtype=np.float32)
    ys = np.linspace(p1[1], p2[1], n, dtype=np.float32)
    return np.stack([xs, ys], axis=1)


def sample_model_points(
    points_per_meter: float = 30.0,
    min_weight: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    lines = [ln for ln in get_bwf_lines() if ln.weight >= float(min_weight)]
    pts: List[np.ndarray] = []
    weights: List[float] = []
    names: List[str] = []
    for ln in lines:
        length = float(np.hypot(ln.p2[0] - ln.p1[0], ln.p2[1] - ln.p1[1]))
        n = max(2, int(round(length * float(points_per_meter))))
        seg = _sample_segment(ln.p1, ln.p2, n)
        pts.append(seg)
        weights.extend([float(ln.weight)] * seg.shape[0])
        names.extend([ln.name] * seg.shape[0])
    if not pts:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32), []
    Xw = np.concatenate(pts, axis=0).astype(np.float32)
    w = np.array(weights, dtype=np.float32)
    return Xw, w, names
