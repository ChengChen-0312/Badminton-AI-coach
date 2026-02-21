# Stroke Label Space Contract (`stroke_6class_v1`)

## Scope
Current production stroke classifier in this repository is a **6-class** model:

1. `backhand_drive`
2. `backhand_net_shot`
3. `forehand_clear`
4. `forehand_drive`
5. `forehand_lift`
6. `forehand_net_shot`

## Runtime Rules
- `classifier.min_confidence` gates classifier fusion.
- If classifier confidence is below threshold, classifier label is rejected and spatial event logic is used.
- Report metadata includes `stroke_classifier.label_space` and top-level `label_space_version`.

## Config Fields
`src/config/v3_realtime.yaml` and `src/config/v3_ai_score.yaml` expose:
- `classifier.label_space_version`
- `classifier.expected_num_classes`
- `classifier.class_names`
- `classifier.min_confidence`

## Notes
- `18-class` target is not removed, but should be treated as a future data/model milestone.
- Any non-6-class checkpoint should be surfaced as `label_space.mismatch=true` in report metadata.
