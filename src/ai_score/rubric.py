from __future__ import annotations

RUBRIC_EN = """
You are a badminton coaching assistant. Given a motion description, evaluate:
1) Mistake analysis (technique, footwork, timing).
2) Correction advice.
3) Score (0-100).
Return JSON: {"mistake": "...", "advice": "...", "score": int}
"""

def build_prompt(description: str) -> str:
    return RUBRIC_EN + f"\nMotion description:\n{description}\n"
