from __future__ import annotations

from typing import Dict, Optional

from .rubric import build_prompt
from .student_7b import Student7B
from .student_mlx import MLXStudent
from .teacher_local_32b import LocalTeacher32B
from .teacher_cloud_78b import CloudTeacher78B
from .teacher_mlx import MLXTeacher


class ActionFeedback:
    """High-level interface to run teacher/student for scoring and advice."""

    def __init__(
        self,
        mode: str = "student",  # "student" | "student_mlx" | "teacher32b" | "teacher78b" | "teacher_mlx"
        teacher_api: Optional[str] = None,
        teacher_mlx_path: Optional[str] = None,
        student_mlx_path: Optional[str] = None,
    ) -> None:
        self.mode = mode
        if mode == "student":
            self.backend = Student7B()
        elif mode == "student_mlx":
            self.backend = MLXStudent(model_path=student_mlx_path or "/Users/chencheng/llm/qwen3-4b")
        elif mode == "teacher32b":
            self.backend = LocalTeacher32B()
        elif mode == "teacher78b":
            self.backend = CloudTeacher78B(api_url=teacher_api or "http://localhost:8000/v1/inference")
        elif mode == "teacher_mlx":
            self.backend = MLXTeacher(model_path=teacher_mlx_path or "/Users/chencheng/llm/qwen3-30b")
        else:
            raise ValueError("mode must be student | student_mlx | teacher32b | teacher78b | teacher_mlx")

    def score_motion(self, description: str) -> str | Dict:
        prompt = build_prompt(description)
        if isinstance(self.backend, (CloudTeacher78B, LocalTeacher32B, MLXTeacher)):
            return self.backend.analyse_motion(description)
        return self.backend.generate_feedback(description)
