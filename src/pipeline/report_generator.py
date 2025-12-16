from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.analysis.tactical_stats import summarize_tactics
from src.analysis.heatmap import create_hitter_heatmap, create_landing_heatmap
from src.analysis.timeline import create_stroke_timeline


def generate_markdown_report(
    match_name: str,
    stroke_summaries: List[Dict[str, Any]],
    output_path: str,
    tactical_summary: Optional[Dict[str, Any]] = None,
    heatmap_path: Optional[str] = None,
    hitter_heatmap_path: Optional[str] = None,
    timeline_path: Optional[str] = None,
) -> str:
    """Generate a Markdown tactical match report."""
    lines: List[str] = []
    lines.append(f"# 🏸 Match Report — {match_name}\n")
    lines.append(f"Total strokes: **{len(stroke_summaries)}**\n")
    lines.append("---\n")

    # Summary stats (event-level)
    clear_cnt = sum(1 for s in stroke_summaries if s.get("event_type") == "clear")
    lift_cnt = sum(1 for s in stroke_summaries if s.get("event_type") == "lift")
    net_cnt = sum(1 for s in stroke_summaries if s.get("event_type") == "net_shot")

    lines.append("## 📊 Summary Statistics\n")
    lines.append(f"- Clears: **{clear_cnt}**")
    lines.append(f"- Lifts: **{lift_cnt}**")
    lines.append(f"- Net Shots: **{net_cnt}**\n")

    # Tactical insights
    if tactical_summary is not None:
        lines.append("---\n")
        lines.append("## 🎯 Tactical Insights\n")

        global_stats = tactical_summary.get("global", {})
        per_player = tactical_summary.get("per_player", {})

        lines.append(f"- Total strokes: **{global_stats.get('total_strokes', 0)}**")
        phase_counts = global_stats.get("phase_counts", {})
        if phase_counts:
            lines.append(
                f"- Phases (offense/defense/neutral): "
                f"{phase_counts.get('offense', 0)}/"
                f"{phase_counts.get('defense', 0)}/"
                f"{phase_counts.get('neutral', 0)}"
            )

        control_index = global_stats.get("control_index", {})
        if control_index:
            lines.append("- Control Index (higher → more offensive):")
            for role, score in control_index.items():
                lines.append(f"  - {role}: **{score:.2f}**")

        if per_player:
            lines.append("\n### Per-player breakdown\n")
            for role, stats in per_player.items():
                lines.append(f"#### Player: {role}")
                lines.append(f"- Total strokes: {stats.get('total_strokes', 0)}")
                phase = stats.get("phase_counts", {})
                lines.append(
                    f"- Phases: offense={phase.get('offense', 0)}, "
                    f"defense={phase.get('defense', 0)}, "
                    f"neutral={phase.get('neutral', 0)}"
                )
                region = stats.get("by_landing_region", {})
                if region:
                    lines.append(
                        f"- Landing regions: "
                        f"{', '.join([f'{k}={v}' for k, v in region.items()])}"
                    )
                lines.append("")

    # Visualizations
    if heatmap_path or hitter_heatmap_path or timeline_path:
        lines.append("---\n")
        lines.append("## 🎨 Visualizations\n")
        if heatmap_path:
            lines.append(f"![Landing Heatmap]({Path(heatmap_path).name})")
        if hitter_heatmap_path:
            lines.append(f"![Hitter (Contact) Heatmap]({Path(hitter_heatmap_path).name})")
        if timeline_path:
            lines.append(f"![Timeline]({Path(timeline_path).name})")
        lines.append("")

    lines.append("---\n")
    lines.append("## 📝 Stroke-by-Stroke Details\n")

    for i, s in enumerate(stroke_summaries):
        lines.append(f"### Stroke {i + 1}")
        lines.append(f"- Predicted Type: **{s.get('final_type')}**")
        lines.append(f"- Classifier Label: {s.get('classifier_label')}")
        lines.append(f"- Event Type (logic): {s.get('event_type')}")
        lines.append(
            f"- Hitter: **{s.get('hitter_role')}** (track_id={s.get('hitter_track_id')})"
        )
        lines.append(f"- Landing Region: {s.get('landing_region')}")
        lines.append(f"- Hitter Region: {s.get('contact_region')}")
        lines.append(f"- Contact Frame: {s.get('contact_frame')}")
        lines.append(f"- Landing Frame: {s.get('landing_frame')}")
        conf = s.get("confidence")
        if conf is not None:
            lines.append(f"- Confidence: {float(conf):.2f}")
        lines.append("")

    Path(output_path).write_text("\n".join(lines), encoding="utf-8")
    return output_path


def generate_json_report(stroke_summaries: List[Dict[str, Any]], output_path: str) -> str:
    Path(output_path).write_text(json.dumps(stroke_summaries, indent=4), encoding="utf-8")
    return output_path


def generate_csv_report(stroke_summaries: List[Dict[str, Any]], output_path: str) -> str:
    if len(stroke_summaries) == 0:
        raise ValueError("No stroke summaries to output.")

    keys = list(stroke_summaries[0].keys())
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(stroke_summaries)
    return output_path


def generate_match_report(
    match_name: str,
    stroke_summaries: List[Dict[str, Any]],
    out_dir: str,
    enable_heatmap: bool = True,
    enable_timeline: bool = True,
    heatmap_bins: int = 32,
) -> Dict[str, Any]:
    """Generate Markdown + JSON + CSV reports (with tactical summary and visualizations)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    md_path = out / f"{match_name}_report.md"
    json_path = out / f"{match_name}_report.json"
    csv_path = out / f"{match_name}_report.csv"
    heatmap_path = out / f"{match_name}_heatmap.png"
    hitter_heatmap_path = out / f"{match_name}_hitter_heatmap.png"
    timeline_path = out / f"{match_name}_timeline.png"

    tactical_summary = summarize_tactics(stroke_summaries)

    # Visualizations
    heatmap_file = None
    hitter_heatmap_file = None
    timeline_file = None
    if enable_heatmap:
        heatmap_file = create_landing_heatmap(
            stroke_summaries,
            str(heatmap_path),
            bins=heatmap_bins,
        )
        hitter_heatmap_file = create_hitter_heatmap(
            stroke_summaries,
            str(hitter_heatmap_path),
            bins=heatmap_bins,
        )
    if enable_timeline:
        timeline_file = create_stroke_timeline(
            stroke_summaries,
            str(timeline_path),
        )

    generate_markdown_report(
        match_name,
        stroke_summaries,
        str(md_path),
        tactical_summary=tactical_summary,
        heatmap_path=heatmap_file,
        hitter_heatmap_path=hitter_heatmap_file,
        timeline_path=timeline_file,
    )
    generate_json_report(stroke_summaries, str(json_path))
    generate_csv_report(stroke_summaries, str(csv_path))

    result = {
        "markdown": str(md_path),
        "json": str(json_path),
        "csv": str(csv_path),
        "tactical_summary": tactical_summary,
        "heatmap": heatmap_file,
        "hitter_heatmap": hitter_heatmap_file,
        "timeline": timeline_file,
    }

    # Optional inline content
    try:
        result["markdown_content"] = Path(md_path).read_text(encoding="utf-8")
    except Exception:
        result["markdown_content"] = None
    try:
        result["json_content"] = Path(json_path).read_text(encoding="utf-8")
    except Exception:
        result["json_content"] = None
    try:
        result["csv_content"] = Path(csv_path).read_text(encoding="utf-8")
    except Exception:
        result["csv_content"] = None

    return result


def generate_report(
    match_name: str,
    stroke_summaries: List[Dict[str, Any]],
    out_dir: str | None = None,
    fmt: str = "markdown",
    to_stdout: bool = False,
) -> str:
    """
    Generate a single-format report or return content directly.

    Args:
        match_name: report name prefix.
        stroke_summaries: list of stroke dicts.
        out_dir: directory to write file; required if to_stdout is False.
        fmt: one of {"markdown", "json", "csv"}.
        to_stdout: when True, return the report content as a string (and do not write files).
    """
    fmt = fmt.lower()
    if fmt not in {"markdown", "json", "csv"}:
        raise ValueError("fmt must be one of: markdown, json, csv")

    if to_stdout:
        if fmt == "markdown":
            lines = []
            lines.append(f"# 🏸 Match Report — {match_name}\n")
            lines.append(f"Total strokes: **{len(stroke_summaries)}**\n")
            for i, s in enumerate(stroke_summaries):
                lines.append(f"### Stroke {i + 1}")
                lines.append(f"- Predicted Type: **{s.get('final_type')}**")
                lines.append(f"- Hitter: **{s.get('hitter_role')}**")
                lines.append(f"- Landing Region: {s.get('landing_region')}")
            return "\n".join(lines)
        if fmt == "json":
            return json.dumps(stroke_summaries, indent=4)
        if fmt == "csv":
            if not stroke_summaries:
                return ""
            keys = list(stroke_summaries[0].keys())
            import io

            buf = io.StringIO()
            writer = csv.DictWriter(buf, fieldnames=keys)
            writer.writeheader()
            writer.writerows(stroke_summaries)
            return buf.getvalue()

    if out_dir is None:
        raise ValueError("out_dir is required when to_stdout is False")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if fmt == "markdown":
        path = out / f"{match_name}_report.md"
        generate_markdown_report(match_name, stroke_summaries, str(path))
        return str(path)
    if fmt == "json":
        path = out / f"{match_name}_report.json"
        generate_json_report(stroke_summaries, str(path))
        return str(path)
    path = out / f"{match_name}_report.csv"
    generate_csv_report(stroke_summaries, str(path))
    return str(path)
