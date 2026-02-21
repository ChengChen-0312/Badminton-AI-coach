from __future__ import annotations

import json
from typing import Any, Dict, Optional, Tuple

import numpy as np


DIM_KEYS = ("technique", "footwork", "timing", "decision", "outcome")
WEIGHTS = {
    "technique": 0.30,
    "footwork": 0.20,
    "timing": 0.20,
    "decision": 0.15,
    "outcome": 0.15,
}


def _clamp_score(v: Any) -> Optional[float]:
    try:
        x = float(v)
    except Exception:
        return None
    if not np.isfinite(x):
        return None
    return float(max(0.0, min(100.0, x)))


def _extract_json_dict(text: str) -> Tuple[Optional[Dict[str, Any]], str]:
    t = str(text or "").strip()
    if not t:
        return None, ""
    try:
        obj = json.loads(t)
        return obj if isinstance(obj, dict) else None, t
    except Exception:
        pass
    start = t.find("{")
    end = t.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = t[start : end + 1]
        try:
            obj = json.loads(candidate)
            return obj if isinstance(obj, dict) else None, t
        except Exception:
            return None, t
    return None, t


def _extract_declared_score(parsed: Optional[Dict[str, Any]]) -> Optional[float]:
    if not isinstance(parsed, dict):
        return None
    for key in ("score", "final_score", "overall_score"):
        if key in parsed:
            sv = _clamp_score(parsed.get(key))
            if sv is not None:
                return sv
    return None


def _extract_dim_scores(parsed: Optional[Dict[str, Any]]) -> Dict[str, float]:
    if not isinstance(parsed, dict):
        return {}
    node = parsed.get("scores")
    src: Dict[str, Any] = node if isinstance(node, dict) else parsed
    out: Dict[str, float] = {}
    for k in DIM_KEYS:
        sv = _clamp_score(src.get(k))
        if sv is not None:
            out[k] = sv
    return out


def _weighted_score(dim_scores: Dict[str, float]) -> Optional[float]:
    used = [k for k in DIM_KEYS if k in dim_scores]
    if len(used) < 3:
        return None
    w_sum = float(sum(WEIGHTS[k] for k in used))
    if w_sum <= 0:
        return None
    score = sum(float(dim_scores[k]) * float(WEIGHTS[k]) for k in used) / w_sum
    return float(max(0.0, min(100.0, round(score, 1))))


def _fallback_dim_scores(summary: Dict[str, Any]) -> Dict[str, float]:
    dims = {k: 70.0 for k in DIM_KEYS}
    final_type = str(summary.get("final_type", "") or "").strip().lower()
    event_type = str(summary.get("event_type", "") or "").strip().lower()
    landing_region = str(summary.get("landing_region", "") or "").strip().lower()
    landing_predicted = bool(summary.get("landing_predicted", False))

    if final_type in ("unknown", ""):
        dims["technique"] -= 15.0
        dims["timing"] -= 10.0
        dims["decision"] -= 8.0
    if event_type in ("unknown", ""):
        dims["decision"] -= 5.0
    if landing_predicted:
        dims["outcome"] -= 10.0
    if landing_region == "out":
        dims["outcome"] -= 22.0
        dims["decision"] -= 8.0
    elif "front" in landing_region:
        dims["timing"] += 2.0
    elif "back" in landing_region:
        dims["outcome"] += 2.0

    pose = summary.get("pose_features")
    if isinstance(pose, dict):
        elbow = _clamp_score(pose.get("right_elbow"))
        shoulder = _clamp_score(pose.get("right_shoulder"))
        trunk = _clamp_score(pose.get("trunk_angle"))
        if elbow is not None:
            if elbow < 120:
                dims["technique"] -= 8.0
            elif elbow > 175:
                dims["technique"] -= 5.0
        if shoulder is not None and shoulder < 60:
            dims["technique"] -= 6.0
        if trunk is not None and trunk < 10:
            dims["footwork"] -= 4.0

    for k in dims.keys():
        dims[k] = float(max(0.0, min(100.0, round(dims[k], 1))))
    return dims


def _fallback_text(summary: Dict[str, Any]) -> Tuple[str, str]:
    landing_region = str(summary.get("landing_region", "") or "").lower()
    final_type = str(summary.get("final_type", "") or "").lower()
    parts = []
    adv = []
    if landing_region == "out":
        parts.append("Landing control is unstable.")
        adv.append("Prioritize margin over power and target a safer central lane.")
    if final_type in ("unknown", ""):
        parts.append("Stroke identity is uncertain from current evidence.")
        adv.append("Record cleaner side-view clips and keep the hitter fully visible.")
    if not parts:
        parts.append("Execution is playable but not consistently repeatable.")
    if not adv:
        adv.append("Use split-step before contact and recover to base earlier.")
        adv.append("Repeat the same stroke pattern for 8-12 reps to improve consistency.")
    return " ".join(parts), " ".join(adv[:2])


def _extract_confidence(parsed: Optional[Dict[str, Any]], score_source: str, summary: Dict[str, Any]) -> float:
    c = None
    if isinstance(parsed, dict):
        try:
            c = float(parsed.get("confidence"))
        except Exception:
            c = None
    if c is None or (not np.isfinite(c)):
        c = 0.75 if score_source == "rubric" else 0.65 if score_source == "declared" else 0.45
    c = float(max(0.0, min(1.0, c)))
    if bool(summary.get("landing_predicted", False)):
        c = max(0.0, c - 0.10)
    return float(round(c, 3))


def normalize_llm_score_output(text: str, summary: Dict[str, Any]) -> Dict[str, Any]:
    parsed, raw = _extract_json_dict(text)
    declared_score = _extract_declared_score(parsed)
    dim_scores = _extract_dim_scores(parsed)
    rubric_score = _weighted_score(dim_scores)

    fallback_scores = _fallback_dim_scores(summary)
    fallback_score = _weighted_score(fallback_scores)
    if fallback_score is None:
        fallback_score = 65.0

    if rubric_score is not None:
        score = rubric_score
        score_source = "rubric"
    elif declared_score is not None:
        score = declared_score
        score_source = "declared"
    else:
        score = fallback_score
        score_source = "fallback"

    if len(dim_scores) < 3:
        dim_scores = fallback_scores
    else:
        # Keep score spread but avoid impossible mismatch vs weighted score.
        if rubric_score is not None and abs(float(score) - float(rubric_score)) > 12.0:
            score = rubric_score
            score_source = "rubric_recalibrated"

    mistake = None
    advice = None
    if isinstance(parsed, dict):
        mk = parsed.get("mistake")
        ad = parsed.get("advice")
        if isinstance(mk, str) and mk.strip():
            mistake = mk.strip()
        if isinstance(ad, str) and ad.strip():
            advice = ad.strip()
    if not mistake or not advice:
        fallback_mistake, fallback_advice = _fallback_text(summary)
        mistake = mistake or fallback_mistake
        advice = advice or fallback_advice

    confidence = _extract_confidence(parsed, score_source, summary)
    return {
        "raw": raw,
        "parsed": parsed,
        "score": float(max(0.0, min(100.0, round(float(score), 1)))),
        "score_source": score_source,
        "declared_score": declared_score,
        "rubric_score": rubric_score,
        "fallback_score": fallback_score,
        "scores": dim_scores,
        "mistake": mistake,
        "advice": advice,
        "confidence": confidence,
    }
