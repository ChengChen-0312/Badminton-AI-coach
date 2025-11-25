from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List


def generate_markdown_report(
    match_name: str, stroke_summaries: List[Dict], output_path: str
) -> str:
    """Generate a Markdown tactical match report."""
    lines: List[str] = []
    lines.append(f"# 🏸 Match Report — {match_name}\n")
    lines.append(f"Total strokes: **{len(stroke_summaries)}**\n")
    lines.append("---\n")

    # Summary stats
    clear_cnt = sum(1 for s in stroke_summaries if s.get("event_type") == "clear")
    lift_cnt = sum(1 for s in stroke_summaries if s.get("event_type") == "lift")
    net_cnt = sum(1 for s in stroke_summaries if s.get("event_type") == "net_shot")

    lines.append("## 📊 Summary Statistics\n")
    lines.append(f"- Clears: **{clear_cnt}**")
    lines.append(f"- Lifts: **{lift_cnt}**")
    lines.append(f"- Net Shots: **{net_cnt}**\n")

    # Stroke details
    lines.append("---\n")
    lines.append("## 🎯 Stroke-by-Stroke Details\n")

    for i, s in enumerate(stroke_summaries):
        lines.append(f"### Stroke {i + 1}")
        lines.append(f"- Predicted Type: **{s.get('final_type')}**")
        lines.append(f"- Classifier Label: {s.get('classifier_label')}")
        lines.append(f"- Event Type (logic): {s.get('event_type')}")
        lines.append(f"- Hitter: **{s.get('hitter_role')}** (track_id={s.get('hitter_track_id')})")
        lines.append(f"- Landing Region: {s.get('landing_region')}")
        lines.append(f"- Contact Frame: {s.get('contact_frame')}")
        lines.append(f"- Landing Frame: {s.get('landing_frame')}")
        lines.append(f"- Confidence: {s.get('confidence'):.2f}" if isinstance(s.get("confidence"), (int, float)) else "- Confidence: n/a")
        lines.append("")

    Path(output_path).write_text("\n".join(lines), encoding="utf-8")
    return output_path


def generate_json_report(stroke_summaries: List[Dict], output_path: str) -> str:
    Path(output_path).write_text(json.dumps(stroke_summaries, indent=4), encoding="utf-8")
    return output_path


def generate_csv_report(stroke_summaries: List[Dict], output_path: str) -> str:
    if len(stroke_summaries) == 0:
        raise ValueError("No stroke summaries to output.")

    keys = list(stroke_summaries[0].keys())
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(stroke_summaries)
    return output_path


def generate_match_report(match_name: str, stroke_summaries: List[Dict], out_dir: str) -> Dict[str, str]:
    """Generate Markdown + JSON + CSV reports in the output directory."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    md_path = out / f"{match_name}_report.md"
    json_path = out / f"{match_name}_report.json"
    csv_path = out / f"{match_name}_report.csv"

    generate_markdown_report(match_name, stroke_summaries, str(md_path))
    generate_json_report(stroke_summaries, str(json_path))
    generate_csv_report(stroke_summaries, str(csv_path))

    return {
        "markdown": str(md_path),
        "json": str(json_path),
        "csv": str(csv_path),
    }


def generate_report(
    match_name: str,
    stroke_summaries: List[Dict],
    out_dir: str | None = None,
    fmt: str = "markdown",
    to_stdout: bool = False,
) -> str:
    """
    Generate a single-format report or print it directly.

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
            tmp_path = Path("_tmp_report.md")
            generate_markdown_report(match_name, stroke_summaries, str(tmp_path))
            content = tmp_path.read_text(encoding="utf-8")
            tmp_path.unlink(missing_ok=True)
            return content
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
