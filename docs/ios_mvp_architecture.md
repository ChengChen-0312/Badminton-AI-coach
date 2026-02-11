# iOS MVP Architecture (Badminton AI Coach)

## 1. Product Goal
Deliver an iOS MVP that can:
- record/import badminton video,
- run core analysis (court lock + stroke extraction + scoring),
- return structured feedback with stable fallback behavior.
- use current production classifier label space `stroke_6class_v1` (6 classes).

## 2. Recommended Architecture
Use a **hybrid split** for MVP speed and reliability:
- On-device (iOS): camera capture, court lock UX, lightweight realtime overlay, report viewer.
- Cloud/API: heavy video analysis job, optional LLM coaching generation, historical analytics.

This keeps iOS app responsive while preserving upgrade flexibility for model iterations.

## 3. Module Split
### 3.1 iOS Client (Swift/SwiftUI)
- `CaptureModule`: AVFoundation capture/import.
- `SessionModule`: local metadata, upload queue, retry/resume.
- `InferencePreviewModule`: optional low-rate local hints (not final scoring).
- `ReportModule`: render timeline/heatmap/score cards from JSON.
- `AuthModule`: token lifecycle and secure storage (Keychain).
- Repo scaffold: `ios/BadmintonAICoach/BadmintonAICoach.xcodeproj`

### 3.2 Backend/API
- `UploadService`: signed URL / multipart upload.
- `AnalysisService`: trigger Python pipeline (`analyse_video` -> `extract_strokes` -> scoring).
- `ResultService`: serve normalized report JSON + assets (heatmap/timeline images).
- `ModelService` (optional): model registry/version routing for classifier and LLM variants.

## 4. API Contract (MVP)
Reference local server implementation:
- `scripts/run_analysis_api.py`
- `src/server/analysis_api.py`
### 4.1 Create Analysis Job
- `POST /v1/analysis/jobs`
- Request:
```json
{
  "video_url": "s3://.../upload.mp4",
  "video_path": "/absolute/local/path.mp4",
  "config_profile": "v3_realtime_ios",
  "enable_llm": true
}
```
- For local desktop/simulator bridge, prefer `video_path` or `file://` URL.
- Response:
```json
{
  "job_id": "job_123",
  "status": "queued"
}
```

### 4.2 Poll Job Status
- `GET /v1/analysis/jobs/{job_id}`
- Response:
```json
{
  "job_id": "job_123",
  "status": "running",
  "progress": 0.65
}
```

### 4.3 Get Final Report
- `GET /v1/analysis/jobs/{job_id}/report`
- Response:
```json
{
  "court_detection": {},
  "stroke_classifier": {},
  "stroke_count": 0,
  "strokes": [],
  "summary": {
    "overall_score": 0,
    "confidence": 0.0
  },
  "assets": {
    "heatmap_url": "",
    "timeline_url": ""
  }
}
```

## 5. Runtime Flow
1. iOS captures/imports video.
2. iOS uploads media and creates analysis job.
3. Backend runs current Python pipeline.
4. Backend stores JSON + assets and returns report.
5. iOS renders report and caches key metrics locally.

## 6. Versioning and Compatibility
- Include `pipeline_version`, `classifier_checkpoint`, `rubric_version` in report metadata.
- iOS should parse unknown fields safely (forward-compatible JSON parsing).
- Keep `score_source` (`rubric`, `declared`, `fallback`) visible for auditability.

## 7. MVP Non-Goals
- Full on-device 18-class inference parity with cloud.
- Realtime multi-angle tracking.
- Advanced personalization loop.

## 8. Phased Delivery
### Phase A (2 weeks)
- iOS capture/upload + job polling + static report rendering.

### Phase B (2-3 weeks)
- Integrate stroke classifier metadata and calibrated scoring cards.

### Phase C (2 weeks)
- Add local realtime preview and offline queue retry.

## 9. Key Risks and Mitigations
- Latency variability: async jobs + progress polling + cached previous reports.
- Model drift: pin checkpoint/version in response metadata.
- API instability: schema tests for report JSON before deployment.
