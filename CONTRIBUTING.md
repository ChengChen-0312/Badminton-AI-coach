# Contributing Guide

Thanks for contributing to Badminton AI Coach.

## Development Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Quick Quality Check (Required)

Run before opening a PR:

```bash
python scripts/run_regression_suite.py --report tests/assets/sample_report.json
```

Expected result:
- exit code `0`
- log contains `[OK] regression suite passed`

## Branch and PR Rules

- Use a focused branch per change.
- Keep PR scope small and reviewable.
- Include:
  - what changed
  - why it changed
  - how it was validated
- For pipeline changes, include one reproducible command and output path.

## Commit Message Suggestion

Use clear prefixes:
- `feat:`
- `fix:`
- `docs:`
- `refactor:`
- `test:`
- `chore:`

Example:

`feat: add CI workflow for regression suite`

## Reproducibility Notes

- Do not commit private videos.
- Do not commit large generated artifacts under `reports/`, `runs/`, `outputs/`.
- Keep court corner order consistent as `LB, RB, RT, LT`.

