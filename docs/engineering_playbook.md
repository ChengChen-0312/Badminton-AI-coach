# Engineering Playbook (MVP)

## 1) System Checks
Run local regression suite before commit:

```bash
python scripts/run_regression_suite.py --report tests/assets/sample_report.json
```

This runs:
1. `compileall` for `src/` and `scripts/`
2. unit tests under `tests/test_*.py`
3. report-quality monitor with strict thresholds

## 2) Regression Baseline
- Baseline fixture: `tests/assets/sample_report.json`
- Quality monitor: `scripts/monitor_report_quality.py`
- Court lock benchmark: `scripts/benchmark_court_lock.py`

Example:

```bash
python scripts/monitor_report_quality.py \
  --report tests/assets/sample_report.json \
  --strict \
  --max-unknown-ratio 0.80 \
  --min-court-confidence 0.20

# optional (requires local GT/video assets)
python scripts/benchmark_court_lock.py \
  --batch-gt archive/gt_corners_batch.json \
  --method hough_orient \
  --env-file src/config/court_env_profiles/hough_orient_stable.env \
  --strict
```

## 3) CI Gate
- Workflow: `.github/workflows/ci_regression.yml`
- Trigger: push + pull request
- Gate rules: compile + unit tests + report monitor

## 4) Release Gate
- Workflow: `.github/workflows/release_gate.yml`
- Trigger: manual (`workflow_dispatch`)
- Outputs:
  - report quality JSON
  - iOS MVP project snapshot
  - architecture + label-space docs

## 5) Runtime Monitoring Suggestions
- Persist `court_detection.confidence`, `court_detection.reason`, `label_space_version`, `stroke_count`, `unknown_ratio`.
- Alert conditions (initial thresholds):
  - `stroke_count == 0`
  - `unknown_ratio > 0.8`
  - `court_detection.confidence < 0.2` with reason not `manual`
