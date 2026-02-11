# BadmintonAICoach iOS MVP

This is a SwiftUI MVP scaffold for the badminton analysis app.

## What It Does
- accepts API base URL + video URL
- supports local `video_path` input for simulator/desktop bridge
- creates an analysis job (`POST /v1/analysis/jobs`)
- polls status (`GET /v1/analysis/jobs/{job_id}`)
- fetches final report (`GET /v1/analysis/jobs/{job_id}/report`)

## Project
- Xcode project: `ios/BadmintonAICoach/BadmintonAICoach.xcodeproj`
- Swift sources: `ios/BadmintonAICoach/BadmintonAICoach/*.swift`

## Local Backend
Run local backend first:

```bash
python scripts/run_analysis_api.py --host 127.0.0.1 --port 8765
```
