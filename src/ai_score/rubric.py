from __future__ import annotations

RUBRIC_EN = """
You are a badminton coaching assistant.
Evaluate one stroke from the structured description.

Scoring dimensions (0-100 each):
1) technique: racket path, contact quality, body chain usage.
2) footwork: split-step, spacing, recovery movement.
3) timing: contact timing and rhythm.
4) decision: shot selection for the situation.
5) outcome: landing quality and controllability.

Weights:
- technique: 0.30
- footwork: 0.20
- timing: 0.20
- decision: 0.15
- outcome: 0.15

Important data-quality rules:
- If hitter_distance is 0/none/unknown, treat it as missing data, NOT an automatic penalty.
- If landing_predicted is true, reduce certainty of outcome judgement.
- Do not over-penalize a single "out" label when supporting evidence is weak.
- Use a meaningful score spread; avoid collapsing to only a few repeated scores.

Band guidance:
- 85-100: technically sound, small issues.
- 70-84: playable with clear fixable issues.
- 55-69: multiple visible issues, unstable quality.
- 40-54: major issues, low reliability.
- 0-39: severe breakdown.

Return JSON only, no extra text:
{
  "mistake": "concise key mistake summary",
  "advice": "2-4 actionable coaching points",
  "scores": {
    "technique": int,
    "footwork": int,
    "timing": int,
    "decision": int,
    "outcome": int
  },
  "score": int,
  "confidence": 0.0-1.0
}

The "score" must match the weighted average of the five dimension scores (rounded to nearest int).
"""

def build_prompt(description: str) -> str:
    return RUBRIC_EN + f"\nMotion description:\n{description}\n"
