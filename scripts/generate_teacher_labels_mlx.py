#!/usr/bin/env python3
"""
Generate teacher labels using local MLX teacher model (Qwen3-VL-30B 4bit).

Pipeline:
  video (.mp4)
    -> analyse_video(...)
    -> stroke summaries
    -> prompt (from summary)
    -> teacher_mlx -> score + comments
    -> JSONL line: {video_path, summary, prompt, teacher_output}
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import yaml

# Ensure repo root on path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.pipeline.analyse_video import analyse_video
from src.pipeline.extract_strokes import summarise_strokes_from_analysis, stroke_summaries_to_dicts
from src.ai_score.action_feedback import ActionFeedback
from src.spatial_logic.pose_quality import pose_features_to_prompt


def build_prompt_from_summary(summary: dict) -> str:
    """Convert a stroke summary into a coaching prompt, including pose hints if available."""
    final_type = summary.get("final_type", "unknown")
    hitter_role = summary.get("hitter_role", "unknown")
    landing_region = summary.get("landing_region", "unknown")
    event_type = summary.get("event_type", "unknown")
    frame_range = summary.get("frame_range", (0, 0))
    contact_frame = summary.get("contact_frame", None)
    landing_frame = summary.get("landing_frame", None)
    pose_landmarks = summary.get("pose_landmarks")

    pose_text = pose_features_to_prompt(pose_landmarks)

    prompt = f"""
You are a professional badminton coach.

Stroke info:
- Stroke type (model classification): {final_type}
- Event type (trajectory-based): {event_type}
- Hitter role (relative to camera): {hitter_role}
- Landing region on the court: {landing_region}
- Stroke frame range: {frame_range}
- Contact frame (approx.): {contact_frame}
- Landing frame (approx.): {landing_frame}

Pose/biomechanics (if available):
{pose_text}

Tasks:
1) Give an overall score from 0 to 100 for this stroke (higher = better technique).
2) List key mistakes or risks in the player's technique.
3) Give 2–3 concrete suggestions to improve this stroke.

Return your answer as compact JSON with fields:
- score: integer (0–100)
- mistakes: list of short strings
- suggestions: list of short strings
""".strip()
    return prompt


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate teacher labels (MLX Qwen3-30B) for badminton strokes.")
    parser.add_argument("--config", type=str, default="src/config/v3_realtime.yaml", help="YAML config for analyse_video")
    parser.add_argument("--videos-glob", type=str, default="archive/**/*.mp4", help="Glob pattern for videos")
    parser.add_argument("--output", type=str, default="data/distill/teacher_labels.jsonl", help="Output JSONL path")
    parser.add_argument("--max-videos", type=int, default=None, help="Limit number of videos for quick tests")
    parser.add_argument(
        "--teacher-model-path",
        type=str,
        default="/Users/chencheng/llm/qwen3-30b",
        help="Local MLX teacher model path (Qwen3-VL-30B)",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config, "r"))
    feedback = ActionFeedback(mode="teacher_mlx", teacher_mlx_path=args.teacher_model_path)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    videos = sorted(glob.glob(args.videos_glob, recursive=True))
    if args.max_videos is not None:
        videos = videos[: args.max_videos]
    print(f"[INFO] Found {len(videos)} videos")

    num_items = 0
    with out_path.open("w", encoding="utf-8") as f:
        for vid_idx, video in enumerate(videos):
            print(f"[INFO] [{vid_idx+1}/{len(videos)}] Analysing {video}")
            analysis = analyse_video(video, config=cfg)
            summaries = summarise_strokes_from_analysis(analysis)
            summary_dicts = stroke_summaries_to_dicts(summaries)

            for s in summary_dicts:
                prompt = build_prompt_from_summary(s)
                teacher_raw = feedback.score_motion(prompt)
                # Normalize teacher output to JSON-serializable text
                if hasattr(teacher_raw, "text"):
                    teacher_output = teacher_raw.text
                else:
                    teacher_output = str(teacher_raw)
                item = {
                    "video_path": video,
                    "summary": s,
                    "prompt": prompt,
                    "teacher_output": teacher_output,
                }
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                num_items += 1

    print(f"[DONE] Wrote {num_items} items to {out_path}")


if __name__ == "__main__":
    main()
