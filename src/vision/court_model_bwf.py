from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np

MODEL_ID = "bwf_standard_v1"

# BWF标准双打场（单位：米）
# 全场长：13.40
# 全场宽（双打）：6.10
# 单打宽：5.18
# 网高：中心 1.524，网柱 1.55（2D暂不使用）
# 短发球线：距网 1.98
# 双打后发球线：距后场底线 0.76
# 中线：场地宽的一半（x=3.05）
#
# 2D模型坐标系：左下角 LB 为 (0,0)，x 沿宽，y 沿长。
L = 13.40
WD = 6.10
WS = 5.18
HALF_L = 6.70
HALF_WD = 3.05
HALF_WS = 2.59
SHORT_SERVICE = 1.98
DOUBLES_LONG_FROM_BASELINE = 0.76
NET_Y = HALF_L
SHORT_NEAR_Y = NET_Y - SHORT_SERVICE
SHORT_FAR_Y = NET_Y + SHORT_SERVICE
DOUBLES_LONG_NEAR_Y = DOUBLES_LONG_FROM_BASELINE
DOUBLES_LONG_FAR_Y = L - DOUBLES_LONG_FROM_BASELINE


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
            [0.0, 0.0],
            [WD, 0.0],
            [WD, L],
            [0.0, L],
        ],
        dtype=np.float32,
    )


def get_bwf_lines() -> List[CourtLine]:
    return [
        CourtLine("baseline_bottom", (0.0, 0.0), (WD, 0.0), 1.0),
        CourtLine("baseline_top", (0.0, L), (WD, L), 1.0),
        CourtLine("sideline_left_d", (0.0, 0.0), (0.0, L), 1.0),
        CourtLine("sideline_right_d", (WD, 0.0), (WD, L), 1.0),
        CourtLine("sideline_left_s", (HALF_WD - HALF_WS, 0.0), (HALF_WD - HALF_WS, L), 0.6),
        CourtLine("sideline_right_s", (HALF_WD + HALF_WS, 0.0), (HALF_WD + HALF_WS, L), 0.6),
        CourtLine("short_near", (HALF_WD - HALF_WS, SHORT_NEAR_Y), (HALF_WD + HALF_WS, SHORT_NEAR_Y), 0.6),
        CourtLine("short_far", (HALF_WD - HALF_WS, SHORT_FAR_Y), (HALF_WD + HALF_WS, SHORT_FAR_Y), 0.6),
        CourtLine("long_near_d", (0.0, DOUBLES_LONG_NEAR_Y), (WD, DOUBLES_LONG_NEAR_Y), 0.6),
        CourtLine("long_far_d", (0.0, DOUBLES_LONG_FAR_Y), (WD, DOUBLES_LONG_FAR_Y), 0.6),
        CourtLine("center_near", (HALF_WD, DOUBLES_LONG_NEAR_Y), (HALF_WD, SHORT_NEAR_Y), 0.6),
        CourtLine("center_far", (HALF_WD, SHORT_FAR_Y), (HALF_WD, DOUBLES_LONG_FAR_Y), 0.6),
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
