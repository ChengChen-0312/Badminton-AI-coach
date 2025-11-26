<!-- Auto-generated guidance for coding agents working on Badminton-AI-coach -->
# Copilot / Code Agent Instructions — Badminton-AI-coach

Purpose: give a compact, actionable orientation so an AI coding agent can be productive
immediately in this repository: architecture, workflows, conventions, and concrete
examples to run/extend the code.

**Big Picture**
- **Pipeline-centric**: The repository is organized around a unified analysis pipeline in `src/pipeline`.
  - `analyse_video` (in `src/pipeline/analyse_video.py`) integrates detectors, trackers, court ROI,
    and produces an `AnalyseResult` (frame-level `FrameResult`s with player/ball states).
  - `extract_strokes.py` converts pipeline outputs into stroke summaries; `report_generator.py`
    consumes those summaries and emits Markdown/JSON/CSV + visualizations.
- **Three main stacks**: vision (YOLO detections), tracking (ByteTrack + single-ball tracker),
  and spatial_logic (landing detection, hitter inference, stroke reasoning).

**Key Files & Directories (jump-to)**
- `src/pipeline/analyse_video.py`: primary entry for video analysis; returns `AnalyseResult`.
- `src/pipeline/extract_strokes.py`: converts analysis into stroke summaries used by reports.
- `src/pipeline/report_generator.py`: creates MD/JSON/CSV reports and calls heatmap/timeline builders.
- `src/vision/` and `src/tracking/`: detection and tracking implementations (YOLO wrappers and ByteTrack)
- `src/geometry/` and `src/spatial_logic/`: homography, region definitions, landing and event inference.
- `scripts/`: quick CLI scripts used for training and demos (e.g. `train_strokes_v2.py`, `analyse_match.py`).
- `src/config/`: YAML profiles (e.g. `v3_realtime.yaml`) contain runtime parameters for devices, models,
  tracking, visualization, and realtime settings. Use these as canonical configuration examples.

**Concrete Developer Workflows**
- Install: `pip install -r requirements.txt` (project expects ultralytics/torch + standard analysis libs).
- Run a quick analysis + report (example used in repo context):
  ```python
  from src.pipeline.analyse_video import analyse_video
  from src.pipeline.extract_strokes import summarise_strokes_from_analysis, stroke_summaries_to_dicts
  from src.pipeline.report_generator import generate_match_report
  import yaml
  cfg = yaml.safe_load(open("src/config/v3_realtime.yaml"))
  res = analyse_video("archive/backhand_drive/004.mp4", config=cfg)
  summaries = stroke_summaries_to_dicts(
      summarise_strokes_from_analysis(res, classifier_labels=["backhand_drive"]) 
  )
  paths = generate_match_report("demo_v37", summaries, "reports",
                                enable_heatmap=cfg.get("visualization", {}).get("enable_heatmap", True),
                                enable_timeline=cfg.get("visualization", {}).get("enable_timeline", True),
                                heatmap_bins=cfg.get("visualization", {}).get("heatmap_bins", 32))
  print(paths)
  ```
- Train stroke classifier (examples in `README.md` and `scripts/train_strokes_v2.py`):
  - `python scripts/train_strokes_v2.py --config src/config/default.yaml`
  - Eval: `python -m src.training.eval --config <cfg> --checkpoint runs/.../best_model.pt`

**Project-specific Conventions & Patterns**
- Config-first: behavior is driven by YAML files in `src/config`; prefer reading/writing configuration
  rather than hard-coding parameters. Example keys: `vision.yolo_model`, `tracking.detect_stride`,
  `realtime.batch_size`, `visualization.enable_heatmap`.
- Lightweight scaffolding: many modules are intentionally small, well-factored, and designed for
  code agents to extend (e.g., `src/vision` expects an Ultralyics wrapper with `.detect()` returning
  per-frame detections).
- Data layout expectations:
  - `archive/` contains source videos (folder-per-class for dataset videos).
  - `runs/` stores model outputs and trained checkpoints.
  - `reports/` is where `generate_match_report` places outputs.
- API surface examples to follow:
  - `AnalyseResult` from `analyse_video` contains `fps`, `frame_results`, `court_corners`, `player_tracks`.
  - A stroke summary is a dict with keys like `final_type`, `classifier_label`, `event_type`, `hitter_role`,
    `hitter_track_id`, `contact_frame`, `landing_frame`, `landing_region`, and `confidence`.

**Integration points & dependencies to watch for**
- Ultralytics / YOLO: `src/vision` relies on a YOLO model (the repo ships `yolov8n.pt`).
- Tracking: `src/tracking/bytetrack.py` is the project ByteTrack variant; trackers expose `.update()` and
  `.predict_only()` methods used by `analyse_video`.
- Device considerations on macOS: `src/config/v3_realtime.yaml` defaults show `yolo_device: cpu` and
  `classifier_device: mps` — prefer these to avoid cross-device issues on M1/M2 Macs.

**What to change or extend (guidance for agents)**
- When adding detectors, implement `detect(frames: List[np.ndarray]) -> List[detections]` to match usage
  in `analyse_video` (batch-wise detection). Look at `PlayerDetector` and `BallDetector` wrappers for shape.
- Keep `AnalyseResult` and `FrameResult` stable; downstream modules (`extract_strokes.py`, `report_generator.py`)
  expect the fields shown in `src/pipeline/*.py`.
- Visual assets: `report_generator` expects heatmap/timeline functions in `src/analysis/` that return file paths; match
  the API (`create_landing_heatmap(strokes, out_path, bins)` and `create_stroke_timeline(strokes, out_path)`).

**Quick Sanity Checks an agent should run after edits**
- Linting: run `python -m pip install -r requirements.txt` then run small smoke script (example above) on a short
  video from `archive/` to verify no runtime errors.
- Run `scripts/test_realtime_fps.py` when modifying batching/tracking to ensure expected FPS behavior.

If anything in these instructions is unclear or you'd like more detail about a particular module
(e.g., exact detection output schema, tracker internals, or stroke-summary fields), tell me which
file to inspect further and I will extend this doc with concrete examples and tests.
