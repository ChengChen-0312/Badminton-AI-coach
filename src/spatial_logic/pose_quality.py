from __future__ import annotations

from typing import Dict, Optional

from .pose_features import extract_pose_features


def pose_features_to_prompt(pose_landmarks: Optional[Dict]) -> str:
    """
    Convert pose landmarks into a natural-language hint highlighting
    key biomechanical angles. Intended to be embedded into LLM prompts.
    """
    if pose_landmarks is None:
        return "No reliable pose landmarks were detected for this stroke."

    features = extract_pose_features(pose_landmarks)
    if features is None:
        return "Pose landmarks could not be converted into meaningful angles."

    def fmt(x: Optional[float]) -> str:
        return "N/A" if x is None else f"{x:.1f}°"

    text = [
        "Key biomechanical features for this stroke:",
        f"- Right elbow angle: {fmt(features.get('right_elbow'))}",
        f"- Right shoulder angle: {fmt(features.get('right_shoulder'))}",
        f"- Trunk rotation angle: {fmt(features.get('trunk_angle'))}",
        f"- Racket (wrist-index) angle: {fmt(features.get('racket_angle'))}",
        "",
        "When evaluating the technique, pay special attention to:",
        "1) Whether the hitting arm is sufficiently extended (elbow angle).",
        "2) Whether the shoulder is opened enough to generate power.",
        "3) How much the trunk rotates to transfer weight.",
        "4) Whether the racket face angle is appropriate for the intended stroke.",
    ]
    return "\n".join(text)
