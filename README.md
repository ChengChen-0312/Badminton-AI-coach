# Badminton AI Coach (v3 Blueprint)

End-to-end badminton analytics and coaching system. Includes stroke classification (PyTorch), vision/tracking/court geometry, spatial logic, and unified pipelines. Designed for code agents to extend or generate code file-by-file.

## Project Layout
- `src/data`, `src/models`, `src/training`, `src/config`: stroke-classification stack.
- `src/vision` **(finished)**: YOLO-based detection (Ultralytics YOLOv8), ball detector with class filtering, optional MediaPipe pose, simple court detector (Canny+Hough fallback).
- `src/tracking` **(finished)**: ByteTrack-style multi-object tracker, single-ball tracker with smoothing, role assignment (near/far) heuristics.
- `src/geometry` **(finished)**: homography, coordinate mapping, region definitions, metrics.
- `src/spatial_logic` **(finished)**: relations, events (velocity-drop hit detector), temporal smoothing/segment slicing, stroke_reasoner (zone→stroke heuristic).
- `src/pipeline` **(finished)**: unified analysis (ball traj + reasoning), stroke extraction, overlay rendering (players/ball), report generator (JSON).
- `scripts` **(finished)**: training/eval plus `analyse_match.py`, `export_court_map.py`, `test_realtime_fps.py` (FPS benchmark for realtime pipeline).
- `archive/`: dataset (folder-per-class videos, untouched).

## Stroke Classification (V2 / V2.1)
- Dataset auto-discovers classes; supports sampling modes (uniform/rand_uniform/strided) and court-focus cropping (`full`/`far`/`near`).
- Augmentations: mild/strong; channel-last for MPS; ImageNet normalization.
- Models: resnet18/50 (2D + temporal pool), mc3_18, r3d_18, r3d_34, optional x3d_s/m (requires newer torchvision; falls back to r3d_34), slowfast fallback handled.
- Training: AMP, cosine/onecycle schedulers, label smoothing, class weighting, optional weighted sampler (class boost), grad clipping, early stopping, per-class metrics; channels_last and pin_memory disabled for MPS.
- Configs: `src/config/default.yaml` (balanced), `src/config/v2_highcap.yaml` (higher res/frames, far-court focus), `src/config/v3.yaml` (vision/tracking pipeline example).
- CLI: `python scripts/train_strokes_v2.py --config src/config/default.yaml` (or `v2_highcap`). Eval: `python -m src.training.eval --config ... --checkpoint runs/.../best_model.pt`.

## Vision / Tracking / Geometry (Blueprint Stubs)
- `src/vision/detectors.py`: YOLO wrapper (players/ball); plug in `ultralytics`.
- `src/vision/pose_estimator.py`, `court_detector.py`, `ball_detector.py`, `utils.py`: placeholders for keypoints/court/ball processing.
- `src/tracking/bytetrack.py`, `ball_track.py`, `id_assign.py`, `utils.py`: ByteTrack/Kalman + ID mapping stubs.
- `src/geometry/homography.py`, `coordinate_mapper.py`, `region_definitions.py`, `metrics.py`: homography and court metrics.
- `src/spatial_logic/*`: relations, events, temporal smoothing, stroke_reasoner.
- `src/pipeline/analyse_video.py`: unified pipeline stub; `extract_strokes.py`, `render_overlay.py`, `report_generator.py` scaffolding.
- `scripts/analyse_match.py`: CLI for full pipeline using `src/config/v3.yaml`; `scripts/export_court_map.py` placeholder.

## Config v3 Example (`src/config/v3.yaml`)
```yaml
vision:
  yolo_model: "yolov8n.pt"
  court_model: "court_segmenter.pt"
  pose_model: "yolo-pose-s"
  device: "mps"
tracking:
  player: "bytetrack"
  ball: "kalman"
geometry:
  homography: true
spatial_logic:
  zones: ["front", "mid", "back"]
  detect_events: true
output:
  save_overlay: true
  save_report: true
```

### v3.3 – Landing detection + spatial logic
- `src/spatial_logic/landing_detector.py`: landing estimation from smoothed ball tracks with optional homography + region classification.
- `src/geometry/homography.py`, `src/geometry/region_definitions.py`: map image coords to court meters and classify front/mid/back.
- `src/spatial_logic/events.py`: heuristic stroke event inference (clear/lift/net_shot/drive/smash) from trajectory + landing.
- `src/spatial_logic/stroke_reasoner.py`: fuse classifier labels with spatial events into a final stroke type and confidence.
- `src/pipeline/extract_strokes.py`: turn `analyse_video` outputs into stroke summaries (frame range, type, landing region) for reporting.

## Quickstart (Stroke Training)
```bash
pip install -r requirements.txt
# train (balanced)
python scripts/train_strokes_v2.py --config src/config/default.yaml
# train (high-capacity)
python scripts/train_strokes_v2.py --config src/config/v2_highcap.yaml
# evaluate
python -m src.training.eval --config src/config/default.yaml --checkpoint runs/stroke_baseline/best_model.pt
```

## Notes
- Court-focus: set `data.focus_court: far` to prioritize far-side player.
- x3d backbones need newer torchvision; if unavailable, code falls back to `r3d_34`.
- Vision/tracking/geometry/pipeline files are scaffolds for code agents to fill with real models and logic.
