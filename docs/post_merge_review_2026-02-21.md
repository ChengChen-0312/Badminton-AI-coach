# Post-Merge Review Record (2026-02-21)

## 1) Context

This document backfills the missing PR review narrative for the already merged branch:
- merged commit: `4d218f017ea24e21e64e67a930703f4113b3afd6`
- merge subject: `Merge pull request #1 from ChengChen-0312/feat/instance-aware-court-detector`
- merge date: February 21, 2026

Reference links:
- merged PR: <https://github.com/ChengChen-0312/Badminton-AI-coach/pull/1>
- compare scope: <https://github.com/ChengChen-0312/Badminton-AI-coach/compare/35ed739...d967cac>

## 2) Scope Summary

Diff scope (`35ed739..d967cac`):
- files changed: `927`
- added: `61`
- modified: `15`
- deleted: `851`
- line delta: `+25997 / -4243`

Key point:
- the `851` deletions are mostly large tracked artifacts (`archive/`, `data/`, `runs/`, `outputs/`, demo images), intentionally removed from Git tracking.

## 3) Major Functional Changes

### A. Court detection and homography robustness
- added/updated core modules:
  - `src/vision/court_fit_homography.py`
  - `src/vision/court_detector.py`
  - `src/vision/court_env_profile.py`
  - `src/config/court_env_profiles/hough_orient_stable.env`
- pipeline integration:
  - `src/pipeline/analyse_video.py`
  - `src/pipeline/extract_strokes.py`
  - `src/pipeline/report_generator.py`

### B. Stroke/report schema hardening
- new/updated pipeline helpers:
  - `src/pipeline/label_space.py`
  - `src/pipeline/stroke_classifier_runtime.py`
- scoring consistency:
  - `src/ai_score/scoring_engine.py`

### C. API + iOS MVP path
- local API server:
  - `src/server/analysis_api.py`
  - `scripts/run_analysis_api.py`
- iOS scaffold:
  - `ios/BadmintonAICoach/*`

### D. Reproducibility and project governance
- CI and templates:
  - `.github/workflows/ci.yml`
  - `.github/ISSUE_TEMPLATE/*`
  - `.github/pull_request_template.md`
- repository standards:
  - `README.md`
  - `CONTRIBUTING.md`
  - `SECURITY.md`
  - `CODE_OF_CONDUCT.md`
  - `Makefile`
  - `requirements.lock.txt`
  - `.editorconfig`
  - `.gitattributes`
  - `.python-version`

### E. Artifact tracking cleanup
- `.gitignore` updated to prevent re-committing local media/generated files.
- heavy tracked files removed from index:
  - `archive/`
  - `data/distill/`, `data/mlx_train/`
  - `runs/`, selected `outputs/`
  - root demo images

## 4) Verification Evidence

Executed on February 21, 2026:

```bash
python scripts/run_regression_suite.py --report tests/assets/sample_report.json
```

Observed results:
- `15` tests passed
- final status: `[OK] regression suite passed`
- sample metrics:
  - `stroke_count: 2`
  - `unknown_ratio: 0.0`
  - `court_confidence: 0.76`
  - `label_space_version: stroke_6class_v1`

Repository hygiene checks:

```bash
tracked_mp4=0
tracked_archive=0
tracked_runs=0
tracked_outputs=0
tracked_data=0
```

## 5) Known Risks and Follow-ups

1. Repository history size is still large because historical blobs still exist in old commits.
2. If full history shrink is required, run one-time rewrite described in `docs/repository_hardening.md`.
3. Downstream users must provide their own local video input because media is no longer tracked.

## 6) Backout Plan

If any regression is detected after merge:
1. Revert merge commit `4d218f0` on `main`.
2. Re-introduce changes in smaller PRs:
   - repo hygiene and artifact policy
   - CI/governance docs
   - vision/pipeline functional changes

