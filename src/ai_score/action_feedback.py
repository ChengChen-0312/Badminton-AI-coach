from __future__ import annotations

from typing import Dict, Optional

from .rubric import build_prompt
from .student_mlx import MLXStudent
from .teacher_mlx import MLXTeacher


class ActionFeedback:
    """High-level interface to run teacher/student for scoring and advice."""

    def __init__(
        self,
        mode: str = "student_mlx",  # "student_mlx" | "teacher_mlx"
        teacher_mlx_path: Optional[str] = None,
        student_mlx_path: Optional[str] = None,
    ) -> None:
        self.mode = mode
        if mode == "student_mlx":
            self.backend = MLXStudent(model_path=student_mlx_path or "/Users/chencheng/llm/qwen3-4b")
        elif mode == "teacher_mlx":
            self.backend = MLXTeacher(model_path=teacher_mlx_path or "/Users/chencheng/llm/qwen3-30b")
        else:
            raise ValueError("mode must be student_mlx | teacher_mlx")

    def score_motion(self, description: str) -> str | Dict:
        prompt = build_prompt(description)
        if isinstance(self.backend, MLXTeacher):
            return self.backend.analyse_motion(description)
        return self.backend.generate_feedback(description)
