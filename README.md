# Badminton AI Coach

End-to-end badminton video analytics and coaching pipeline:
- court detection + homography alignment
- player/shuttle detection and tracking
- stroke-level spatial-temporal reasoning
- report generation (`.md` / `.json` / `.csv` + heatmap / timeline)
- optional teacher-student feedback workflow

## 1) Quick Start (Reproducible)

### 1.1 Environment

Recommended (strict versions):

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.lock.txt
```

Flexible (development):

```bash
pip install -r requirements.txt
```

### 1.2 Baseline regression check (no private video required)

```bash
python scripts/run_regression_suite.py --report tests/assets/sample_report.json
```

Expected:
- exit code `0`
- terminal includes `[OK] regression suite passed`

### 1.3 End-to-end demo run

```bash
python scripts/demo_friend.py \
  --video /absolute/path/to/your_video.mp4 \
  --config src/config/v3_realtime.yaml \
  --match-name reproducible_demo \
  --out-dir reports/reproducible_demo \
  --no-llm
```

Expected artifacts in `reports/reproducible_demo/`:
- `reproducible_demo_report.md`
- `reproducible_demo_report.json`
- `reproducible_demo_report.csv`
- `reproducible_demo_heatmap.png` (if enabled)
- `reproducible_demo_timeline.png` (if enabled)
- `reproducible_demo_court_detect_debug.jpg` (if enabled)

### 1.4 Current court-fit reproduction run

Use this command to reproduce the current local `demo_friend` court-fit result
that was committed with the restored automatic court detection:

```bash
PYTHONPATH=. python scripts/demo_friend.py \
  --video archive/demo.mp4 \
  --config src/config/v3_realtime.yaml \
  --match-name demo_friend_iter9_gate_area_select_auto \
  --out-dir reports/demo_friend_iter9_gate_area_select_auto \
  --court-detect-once auto \
  --court-detect-frame-idx 0 \
  --pose auto \
  --no-llm
```

Expected court result:
- report JSON: `reports/demo_friend_iter9_gate_area_select_auto/demo_friend_iter9_gate_area_select_auto_report.json`
- debug image: `reports/demo_friend_iter9_gate_area_select_auto/demo_friend_iter9_gate_area_select_auto_court_detect_debug.jpg`
- `court_detection.source=auto`
- `court_detection.reason=OK`
- `court_detection.frame_idx=0`
- `court_detection.fit_backend=legacy`
- `strokes` count: `2`

For the short two-second court-only smoke clip, keep the same config and run:

```bash
PYTHONPATH=. python scripts/demo_friend.py \
  --video archive/demo3.mp4 \
  --config src/config/v3_realtime.yaml \
  --match-name demo3_court_smoke \
  --out-dir reports/demo3_court_smoke \
  --court-detect-once auto \
  --court-detect-frame-idx 0 \
  --pose off \
  --no-llm
```

The current 35-case visual audit output is committed under:

```text
reports/court_precision_iter9_gate_area_select_35/
```

To print the audited pass rate and failed cases:

```bash
python - <<'PY'
import json
from pathlib import Path

p = Path("reports/court_precision_iter9_gate_area_select_35/visual_review_relaxed_gate.json")
data = json.loads(p.read_text(encoding="utf-8"))
print(f"pass_rate={data['pass_count']}/{data['total']} ({data['pass_rate'] * 100:.1f}%)")
for row in data["rows"]:
    if not row["pass"]:
        print(f"- {row['case']}: {row['visual_note']}")
PY
```

Expected visual audit result: `30/35 = 85.7%`.

These are the key court-fit parameters for the current result in
`src/config/v3_realtime.yaml`:

```yaml
vision:
  court_env_file: "court_env_profiles/hough_orient_stable.env"
  court_detect_once: true
  court_detect_frame_idx: 0
  court_detect_fallback_on_fail: true
  court_detect_fallback_samples: 24
  court_detect_fallback_stride: 5
  court_detect_precision_max_p90: 18.0
  court_detect_precision_max_mean: 4.5
  court_detect_precision_joint_mean_min: 0.0
  court_detect_precision_joint_p90_min: 0.0
  court_detect_precision_max_far_gap_err_m: 0.25
  court_detect_hybrid_modern: true
  court_detect_hybrid_always_scan: true
  court_detect_profile_retry_on_fail: true
  court_detect_profile_retry_file: "court_env_profiles/hough_orient_stable.env"
  court_detect_temporal_refine: false
```

## 2) Main Runtime Path

1. `scripts/demo_friend.py`
2. `src/pipeline/analyse_video.py`
3. `src/pipeline/extract_strokes.py`
4. `src/pipeline/report_generator.py`
5. Optional: `src/ai_score/action_feedback.py`

High-impact court modules:
- `scripts/demo_friend.py`
- `src/vision/court_fit_homography.py`
- `src/vision/court_detector.py`
- `src/config/court_env_profiles/hough_orient_stable.env`

Corner order convention:
- `LB, RB, RT, LT`

## 3) Frequently Used Commands

Use `Makefile` shortcuts:

```bash
make setup
make test
make regression
make demo-help
```

Or run directly:

```bash
python scripts/calibrate_court_corners.py --video /absolute/path/to/your_video.mp4 --print-yaml
PYTHONPATH=. python scripts/debug_court_candidates.py --video /absolute/path/to/your_video.mp4 --frame 0 --config src/config/v3_realtime.yaml --topk 5
python scripts/run_analysis_api.py --host 127.0.0.1 --port 8765
```

API endpoints:
- `POST /v1/analysis/jobs`
- `GET /v1/analysis/jobs/{job_id}`
- `GET /v1/analysis/jobs/{job_id}/report`

## 4) Config Entry Points

- `src/config/v3_realtime.yaml`: practical realtime config
- `src/config/v3_ai_score.yaml`: realtime + AI-score options
- `src/config/v3.yaml`: generic v3 example

Training-related:
- `src/config/default.yaml`
- `src/config/v2_highcap.yaml`
- `src/config/v2_highcap_stable.yaml`

For stable court behavior:

```yaml
vision:
  court_env_file: "src/config/court_env_profiles/hough_orient_stable.env"
```

## 5) Repository Layout

- `src/vision`: court and detection modules
- `src/tracking`: player + shuttle tracking
- `src/geometry`: homography and coordinate transforms
- `src/spatial_logic`: event, landing, hitter, stroke reasoning
- `src/pipeline`: end-to-end orchestration and report outputs
- `src/analysis`: tactical and visualization analytics
- `src/ai_score`: teacher/student scoring logic
- `scripts`: runnable CLIs
- `tests`: unit and regression-related tests
- `ios/BadmintonAICoach`: iOS MVP scaffold
- `docs`: engineering docs and presentation-oriented material

## 6) CI and Collaboration

- CI workflow: `.github/workflows/ci.yml`
- Contribution guide: `CONTRIBUTING.md`
- Security policy: `SECURITY.md`
- Code of conduct: `CODE_OF_CONDUCT.md`
- Issue templates: `.github/ISSUE_TEMPLATE/`
- PR template: `.github/pull_request_template.md`

## 7) Reproducibility and Data Policy

- Large local videos and generated artifacts are intentionally ignored.
- Use your own local `.mp4` for runs.
- Historical large-file cleanup guide: `docs/repository_hardening.md`

Notes:
- If `mediapipe` is missing, use `--pose off` (or keep `--pose auto`).
- Court detection is scene-sensitive; for strict comparisons use fixed video + fixed config + fixed court env profile.

## 8) Additional Docs

- Interview/demo evidence guide: `docs/interview_demo.md`
- Engineering playbook: `docs/engineering_playbook.md`
- Court week plan: `docs/court_detection_week_plan.md`
- Stroke label contract: `docs/stroke_6class_contract.md`
