#!/usr/bin/env python3
"""Export court calibration map (placeholder)."""

from __future__ import annotations

from pathlib import Path


def main() -> None:
    out = Path("runs/court_map.txt")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("Court map placeholder")
    print(f"Wrote court map placeholder to {out}")


if __name__ == "__main__":
    main()
