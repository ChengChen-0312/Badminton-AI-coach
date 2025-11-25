from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, List

# Heuristic offense/defense cues
OFFENSIVE_EVENTS = {"net_shot", "smash", "drive"}
DEFENSIVE_EVENTS = {"clear", "lift"}


def _region_row(region: str | None) -> str:
    """Map a 3x3 region name to front/mid/back rows."""
    if not region:
        return "unknown"
    r = region.lower()
    if "front" in r:
        return "front"
    if "back" in r:
        return "back"
    if "mid" in r:
        return "mid"
    return "unknown"


def classify_phase(stroke: Dict[str, Any]) -> str:
    """
    Roughly classify a stroke as offense/defense/neutral using event and landing region.
    Offense: offensive events or front-row landing
    Defense: defensive events or back-row landing
    Neutral: everything else
    """
    event_type = stroke.get("event_type")
    landing_region = stroke.get("landing_region")

    if event_type in OFFENSIVE_EVENTS:
        return "offense"
    if event_type in DEFENSIVE_EVENTS:
        return "defense"

    row = _region_row(landing_region)
    if row == "front":
        return "offense"
    if row == "back":
        return "defense"
    return "neutral"


def _safe_role(stroke: Dict[str, Any]) -> str:
    role = stroke.get("hitter_role") or "unknown"
    if role not in ("near", "far"):
        return "unknown"
    return role


def summarize_tactics(strokes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate tactical stats per player and globally.

    Returns:
      {
        "per_player": { "near": {...}, "far": {...}, "unknown": {...} },
        "global": {
          "total_strokes": ...,
          "phase_counts": {...},
          "control_index": {...}
        }
      }
    """
    if not strokes:
        return {
            "per_player": {},
            "global": {"total_strokes": 0, "phase_counts": {}, "control_index": {}},
        }

    phase_counts_global: Counter = Counter()
    control_score: Counter = Counter()  # offense +1, defense -1
    total_strokes = len(strokes)

    per_player: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "total_strokes": 0,
            "phase_counts": Counter(),
            "by_event_type": Counter(),
            "by_final_type": Counter(),
            "by_landing_region": Counter(),
        }
    )

    for s in strokes:
        role = _safe_role(s)
        phase = classify_phase(s)
        event_type = s.get("event_type") or "unknown"
        final_type = s.get("final_type") or s.get("classifier_label") or "unknown"
        landing_region = s.get("landing_region") or "unknown"

        # global tallies
        phase_counts_global[phase] += 1
        if phase == "offense":
            control_score[role] += 1
        elif phase == "defense":
            control_score[role] -= 1

        # per-player tallies
        stats = per_player[role]
        stats["total_strokes"] += 1
        stats["phase_counts"][phase] += 1
        stats["by_event_type"][event_type] += 1
        stats["by_final_type"][final_type] += 1
        stats["by_landing_region"][landing_region] += 1

    control_index: Dict[str, float] = {}
    for role, stats in per_player.items():
        total = stats["total_strokes"]
        control_index[role] = float(control_score[role]) / float(total) if total else 0.0

    per_player_out: Dict[str, Dict[str, Any]] = {}
    for role, stats in per_player.items():
        per_player_out[role] = {
            "total_strokes": stats["total_strokes"],
            "phase_counts": dict(stats["phase_counts"]),
            "by_event_type": dict(stats["by_event_type"]),
            "by_final_type": dict(stats["by_final_type"]),
            "by_landing_region": dict(stats["by_landing_region"]),
            "control_index": control_index.get(role, 0.0),
        }

    return {
        "per_player": per_player_out,
        "global": {
            "total_strokes": total_strokes,
            "phase_counts": dict(phase_counts_global),
            "control_index": control_index,
        },
    }
