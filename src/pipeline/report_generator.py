"""Generate tactical reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def generate_report(analysis_result: Dict[str, Any], output_path: str | None = None) -> Dict[str, Any]:
    """Return a dict summary and optionally write to disk."""
    report = {"summary": "Analysis report", "details": analysis_result}
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2))
    return report
