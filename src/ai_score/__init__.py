"""AI scoring and feedback package (MLX teacher/student only)."""

# Keep package import side-effect free.
# MLX backends are loaded lazily via __getattr__ to avoid hard crashes when users
# only need lightweight helpers (e.g., scoring_engine) without MLX runtime.
__all__ = ["ActionFeedback", "MLXStudent", "MLXTeacher"]


def __getattr__(name: str):
    if name == "ActionFeedback":
        from .action_feedback import ActionFeedback

        return ActionFeedback
    if name == "MLXStudent":
        from .student_mlx import MLXStudent

        return MLXStudent
    if name == "MLXTeacher":
        from .teacher_mlx import MLXTeacher

        return MLXTeacher
    raise AttributeError(name)
