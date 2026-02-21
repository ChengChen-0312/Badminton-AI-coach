# Interview Demo: Minimal Reproducible Evidence

Use this page when you need to show run-proof to a reviewer, teacher, or interviewer.

## 1) One-command evidence run

```bash
python scripts/demo_friend.py \
  --video /absolute/path/to/your_video.mp4 \
  --config src/config/v3_realtime.yaml \
  --match-name proof_demo \
  --out-dir reports/proof_demo \
  --no-llm
```

## 2) Screenshot-friendly proof files

Capture from:
- `reports/proof_demo/proof_demo_court_detect_debug.jpg`
- `reports/proof_demo/proof_demo_heatmap.png`
- `reports/proof_demo/proof_demo_timeline.png`
- `reports/proof_demo/proof_demo_report.md`

## 3) JSON field proof

Open:
- `reports/proof_demo/proof_demo_report.json`

Minimum expected structure:

```json
{
  "court_detection": {
    "source": "manual|auto|auto_failed",
    "corners": [[0, 0], [0, 0], [0, 0], [0, 0]],
    "confidence": 0.0,
    "reason": "OK|R_*",
    "frame_idx": 0
  },
  "stroke_classifier": {
    "enabled": true,
    "mode": "auto|on|off",
    "label_space": {
      "version": "stroke_6class_v1"
    }
  },
  "strokes": [
    {
      "final_type": "string",
      "hitter_role": "near|far|unknown",
      "landing_region": "string",
      "contact_frame": 0,
      "confidence": 0.0
    }
  ]
}
```

## 4) Quick JSON verification command

```bash
python - <<'PY'
import json, pathlib
p = pathlib.Path("reports/proof_demo/proof_demo_report.json")
d = json.loads(p.read_text(encoding="utf-8"))
print("has_court_detection:", "court_detection" in d)
print("has_stroke_classifier:", "stroke_classifier" in d)
strokes = d.get("strokes", [])
print("stroke_count:", len(strokes))
if strokes:
    s0 = strokes[0]
    print("sample_keys:", [k for k in ("final_type","hitter_role","landing_region","contact_frame","confidence") if k in s0])
PY
```

## 5) 60-second demo order

1. Show court-lock image: `reports/proof_demo/proof_demo_court_detect_debug.jpg`
2. Show JSON runtime evidence: `reports/proof_demo/proof_demo_report.json`
3. Show analytics artifacts:
   - `reports/proof_demo/proof_demo_heatmap.png`
   - `reports/proof_demo/proof_demo_timeline.png`
4. Show human-readable report: `reports/proof_demo/proof_demo_report.md`

Suggested closing line:
- Single command produced machine-readable JSON, visual analytics, and readable report with court-detection diagnostics for reproducibility.

