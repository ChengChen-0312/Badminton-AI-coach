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

### v3.4 – Hitter Identity (Who Hits the Shuttle?)
- `src/tracking/player_track.py`: maintains two long-lived player tracks, assigns near/far roles via IoU + vertical position.
- `src/pipeline/analyse_video.py`: integrates PlayerTracker so each FrameResult carries player_states; AnalyseResult exposes player_tracks.
- `src/spatial_logic/hitter_detector.py`: picks the closest player to the ball at contact_frame (role + track_id + distance).
- `src/pipeline/extract_strokes.py`: stroke summaries now include `hitter_track_id` and `hitter_distance`, combining classifier label, event type, hitter, and landing info.

### v3.5 – Match Report Generation
- `src/pipeline/report_generator.py`: generate match reports in Markdown, JSON, and CSV. Summaries include hitter (near/far + track_id), stroke type, landing region, contact/landing frames, and confidence. Summary stats included in Markdown.

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

## MLX Teacher/Student Distillation (Cleaned)
- Active backends: `teacher_mlx` (Qwen3-VL-30B) and `student_mlx` (Qwen3-VL-4B). All legacy HF/HTTP backends have been removed.
- Generate labels from all videos:
  ```bash
  python scripts/generate_teacher_labels_mlx.py --videos-glob "archive/**/*.mp4" \
    --output data/distill/teacher_labels.jsonl --config src/config/v3_realtime.yaml \
    --teacher-model-path /Users/chencheng/llm/qwen3-30b
  ```
- Prepare chat + MLX text data:
  ```bash
  python scripts/prepare_distill_dataset.py \
    --input data/distill/teacher_labels.jsonl \
    --output data/distill/distill_data_chat.jsonl
  python scripts/prepare_mlx_data.py  # builds data/mlx_train/{train,valid}.jsonl
  ```
- Train LoRA (example, tweak iters/LR/layers as needed):
  ```bash
  python -m mlx_lm lora --model /Users/chencheng/llm/qwen3-4b --train --data data/mlx_train \
    --batch-size 1 --num-layers 16 --iters 500 --learning-rate 5e-5 \
    --steps-per-eval 50 --adapter-path outputs/lora_adapters
  python -m mlx_lm fuse --model /Users/chencheng/llm/qwen3-4b \
    --adapter-path outputs/lora_adapters --save-path outputs/fused_student_model
  ```

## Notes
- Court-focus: set `data.focus_court: far` to prioritize far-side player.
- x3d backbones need newer torchvision; if unavailable, code falls back to `r3d_34`.
- Vision/tracking/geometry/pipeline files are scaffolds for code agents to fill with real models and logic.

# 开发日志 / 操作手册（v1.0 → v3.7）

## v1.0 – 基础视频分类管线
- 改动：folder-per-class 自动发现；固定帧采样+224×224+ImageNet 归一化；train/val 分离增强；模型 2D ResNet18（逐帧+时序平均）或 3D r3d_18，动态类别输出；训练/评估保存最佳模型。
- 命令：
  ```bash
  python scripts/train_strokes.py --config src/config/default.yaml
  python -m src.training.eval --config src/config/default.yaml \
    --checkpoint runs/stroke_baseline/best_model.pt
  ```
- 结果：训练/评估可跑通，输出整体/类别精度。

## v2.0 – 准确率提升 & MPS 优化
- 改动：新增 mc3_18, r3d_34, x3d_s/x3d_m；采样 uniform/rand_uniform/strided；增强 HFlip/Rotation/ColorJitter/RRCrop；class weights + weighted sampler；Cosine/OneCycle LR，AMP/grad clip/label smoothing/早停；MPS pin_memory=False + channels_last；配置 default.yaml / v2_highcap.yaml；CLI train_strokes_v2.py。
- 命令：
  ```bash
  python scripts/train_strokes_v2.py --config src/config/v2_highcap.yaml
  python -m src.training.eval --config src/config/v2_highcap.yaml \
    --checkpoint runs/v2_high_accuracy/best_model.pt
  ```
- 结果：高容量配置可显著提升准确率（x3d 需新版 torchvision）。

## v2.1 – Court focus & 增强开关
- 改动：data.focus_court full/far/near 预裁剪；data.strong_aug 轻/重增强；v2_highcap_stable 下调分辨率、关闭 class weights/weighted sampler/LS。
- 验证：切换配置即可观察裁剪/增强效果。

## v2.2 – 性能调优（DataLoader/MPS）
- 改动：persistent_workers、prefetch_factor、num_workers 适配 macOS；pin_memory=False；set_num_threads；channels_last。
- 结果：迭代耗时下降（瓶颈仍在视频解码）。

## v2.3 – 缓存方案（建议型）
- 提出 frame cache 减少 mp4 解码开销（未默认开启，可按需用）。

## v3.0 – 视觉+跟踪架构雏形
- 目录：vision（YOLO 检测/pose stub/court detector）、tracking（简化 ByteTrack、ball tracker）、geometry、spatial_logic、pipeline。
- analyse_video 原型：检测+跟踪+ near/far 角色分配；README blueprint 描述整体体系。

## v3.1 – 实时化（detect_stride/ROI/批量 YOLO）
- 改动：detect_stride=3 关键帧检测，非关键帧 tracker 预测；YOLO batch imgsz=384，CPU 推理，court ROI 裁剪；配置 v3_realtime.yaml。
- 验证：`python scripts/test_realtime_fps.py`，64 帧短视频可生成 frame_results，显示 FPS。

## v3.2 – 落点与空间逻辑初版
- 改动：landing_detector 轨迹平滑+缺失窗口推落点（支持 homography/区域分类）；geometry/homography+region_definitions 前/中/后；events 启发式 clear/lift/net_shot/drive/smash；stroke_reasoner 融合分类标签与事件；extract_strokes 生成 summaries。
- 验证示例：
  ```bash
  python - <<'PY'
  from src.pipeline.analyse_video import analyse_video
  from src.pipeline.extract_strokes import summarise_strokes_from_analysis, stroke_summaries_to_dicts
  import yaml
  cfg = yaml.safe_load(open('src/config/v3_realtime.yaml'))
  res = analyse_video('archive/forehand_drive/048.mp4', config=cfg)
  summaries = summarise_strokes_from_analysis(res, classifier_labels=['forehand_drive'])
  print(stroke_summaries_to_dicts(summaries))
  PY
  ```

## v3.3 – 报告与可视化
- 改动：report_generator Markdown/JSON/CSV + tactical summary；heatmap 绿色底白线红点，无方框；timeline 角色散点；visualization 开关。
- 验证：
  ```bash
  python - <<'PY'
  from src.pipeline.analyse_video import analyse_video
  from src.pipeline.extract_strokes import summarise_strokes_from_analysis, stroke_summaries_to_dicts
  from src.pipeline.report_generator import generate_match_report
  import yaml
  cfg = yaml.safe_load(open('src/config/v3_realtime.yaml'))
  res = analyse_video('archive/backhand_drive/004.mp4', config=cfg)
  summaries = stroke_summaries_to_dicts(
      summarise_strokes_from_analysis(res, classifier_labels=['backhand_drive'])
  )
  paths = generate_match_report('demo_v37', summaries, 'reports',
    enable_heatmap=cfg.get('visualization',{}).get('enable_heatmap',True),
    enable_timeline=cfg.get('visualization',{}).get('enable_timeline',True),
    heatmap_bins=cfg.get('visualization',{}).get('heatmap_bins',32))
  print(paths)
  PY
  ```

## v3.4 – 击球者身份
- 改动：player_track 稳定 near/far；hitter_detector 球-人距离判定；summaries 增 hitter_track_id/hitter_distance/contact_frame。
- 验证：summaries 中含 hitter_role/hitter_distance 字段。

## v3.5 – Match Report
- 改动：report_generator 增击球者、落点、事件、置信度等明细；Markdown/JSON/CSV 输出。

## v3.6 – 战术统计
- 改动：analysis/tactical_stats offense/defense/neutral，control_index，per-player 分布；report_generator 嵌入 “Tactical Insights”。

## v3.7 – 可视化完善
- 改动：heatmap 浅绿地板白线红点，去方框；可选热度栅格；timeline 角色散点；report_generator 嵌入最新 heatmap/timeline。
- 验证：demo_v37_report 生成 PNG；analyse_video + summarise + report 可产出可视化。

---

# MLX 教师/学生蒸馏（基于 v3.7 后清理）

## 清理与后端
- 仅保留 MLXTeacher（Qwen3-VL-30B）/ MLXStudent（Qwen3-VL-4B）；删除 HF/HTTP/student_7b 相关分支与脚本。
- action_feedback 仅支持 teacher_mlx / student_mlx。

## 生成标签（全量 835 视频）
```bash
python scripts/generate_teacher_labels_mlx.py \
  --videos-glob "archive/**/*.mp4" \
  --output data/distill/teacher_labels.jsonl \
  --config src/config/v3_realtime.yaml \
  --teacher-model-path /Users/chencheng/llm/qwen3-30b
```
- 结果：teacher_labels.jsonl 共 835 条。

## 转换数据
```bash
python scripts/prepare_distill_dataset.py \
  --input data/distill/teacher_labels.jsonl \
  --output data/distill/distill_data_chat.jsonl

python scripts/prepare_mlx_data.py  # 生成 data/mlx_train/train(751)/valid(84).jsonl，格式 {"text": "..."}
```

## LoRA 训练（4B 基座示例）
```bash
python -m mlx_lm lora \
  --model /Users/chencheng/llm/qwen3-4b \
  --train \
  --data data/mlx_train \
  --batch-size 1 \
  --num-layers 16 \
  --iters 500 \
  --learning-rate 5e-5 \
  --steps-per-eval 50 \
  --adapter-path outputs/lora_adapters
```
- 结果：Val loss ~0.14–0.20；生成 outputs/lora_adapters/adapters.safetensors。

## Fuse（LLM-only）
```bash
python -m mlx_lm fuse \
  --model /Users/chencheng/llm/qwen3-4b \
  --adapter-path outputs/lora_adapters \
  --save-path outputs/fused_student_model
```
- 提示：fuse 结果不含 vision_tower，不能直接在 VLM 管线加载；可用于纯文本。

## 教师/学生对比示例
```bash
python - <<'PY'
import yaml
from src.ai_score.action_feedback import ActionFeedback
from src.pipeline.analyse_video import analyse_video
from src.pipeline.extract_strokes import summarise_strokes_from_analysis, stroke_summaries_to_dicts

video = "archive/forehand_net_shot/099.mp4"
cfg = yaml.safe_load(open("src/config/v3_ai_score.yaml"))
analysis = analyse_video(video, config=cfg)
s_dicts = stroke_summaries_to_dicts(
    summarise_strokes_from_analysis(
        analysis,
        enable_hitter_inference=True,
        hitter_distance_max=cfg.get("spatial_logic", {}).get("hitter_distance_max", 200.0),
    )
)

# Teacher 30B
teacher = ActionFeedback(mode="teacher_mlx", teacher_mlx_path="/Users/chencheng/llm/qwen3-30b")
print("---- Teacher 30B ----")
for s in s_dicts:
    desc = f"stroke: {s.get('final_type')}, hitter: {s.get('hitter_role')}, landing_region: {s.get('landing_region')}, contact_region: {s.get('contact_region')}"
    print(desc)
    print(teacher.score_motion(desc))

# Student 4B (基座 VLM)
student = ActionFeedback(mode="student_mlx", student_mlx_path="/Users/chencheng/llm/qwen3-4b")
print("---- Student 4B (base VLM) ----")
for s in s_dicts:
    desc = f"stroke: {s.get('final_type')}, hitter: {s.get('hitter_role')}, landing_region: {s.get('landing_region')}, contact_region: {s.get('contact_region')}"
    print(desc)
    print(student.score_motion(desc))
PY
```
- 示例输出：
  - 教师 30B：score≈55，指出击球点过低/偏后，建议更高点位发力。
  - 学生 4B 基座：score≈75，通用建议（站姿/随挥/落点控制）。
  - fused_student_model 为 LLM-only，不含 vision_tower，当前 pipeline 不可直接加载；可用于纯文本模式或等待 VLM LoRA 支持。

## 训练/推理配置提示
- MPS：device=mps，pin_memory=False，可选 channels_last。
- 性能：LoRA 训练峰值内存 ~4.3 GB（M4 Max，batch=1，layers=16，iters=500）。
- 提升：增加 iters（800–1000）、调 LR（1e-4）、增 layers；保持 batch=1 以稳内存。

LoRA 蒸馏与对比（高容量版）
思路

保留完整的视觉塔：使用原始 VLM (Qwen3-VL-4B) 作为基座，LoRA 只作用于语言模型部分，strict=False。
蒸馏数据：835 个视频生成 teacher_labels → distill_data_chat → MLX 训练集 (train=751 / valid=84)。
高容量训练：batch=2、num_layers=32、iters=1200、lr=5e-5，生成适配器 outputs/lora_adapters_highcap（加载时可见 Applied 448 adapter weights）。
推理方式：student_mlx 传入 adapter_path，保持图像输入能力（无需 fuse）。
对比结果（示例）

视频：archive/forehand_net_shot/099.mp4, archive/backhand_drive/004.mp4, archive/forehand_clear/017.mp4
Teacher 30B：score 55–65，重点在击球点/时机/发力链。
Student 4B 基座：给出通用建议，score 多在 55–65。
Student 4B + LoRA highcap：输出在措辞和侧重点上更接近教师，对脚步、时机、击球点、姿态等细节有改进；score 保持在 55–65 区间，体现蒸馏效果。
测试命令（对同组视频对比三种模型）

python - <<'PY'
import yaml
from src.ai_score.action_feedback import ActionFeedback
from src.pipeline.analyse_video import analyse_video
from src.pipeline.extract_strokes import summarise_strokes_from_analysis, stroke_summaries_to_dicts

videos = [
    "archive/forehand_net_shot/099.mp4",
    "archive/backhand_drive/004.mp4",
    "archive/forehand_clear/017.mp4",
]
cfg = yaml.safe_load(open("src/config/v3_ai_score.yaml"))

teacher = ActionFeedback(mode="teacher_mlx", teacher_mlx_path="/Users/chencheng/llm/qwen3-30b")
student_base = ActionFeedback(mode="student_mlx", student_mlx_path="/Users/chencheng/llm/qwen3-4b")
student_lora = ActionFeedback(
    mode="student_mlx",
    student_mlx_path="/Users/chencheng/llm/qwen3-4b",
    adapter_path="outputs/lora_adapters_highcap",
)

for video in videos:
    analysis = analyse_video(video, config=cfg)
    s_dicts = stroke_summaries_to_dicts(
        summarise_strokes_from_analysis(
            analysis,
            enable_hitter_inference=True,
            hitter_distance_max=cfg.get("spatial_logic", {}).get("hitter_distance_max", 200.0),
        )
    )
    print(f"\n=== Video: {video} ===")
    print("---- Teacher 30B ----")
    for s in s_dicts:
        desc = f"stroke: {s.get('final_type')}, hitter: {s.get('hitter_role')}, landing_region: {s.get('landing_region')}, contact_region: {s.get('contact_region')}"
        print(desc)
        print(teacher.score_motion(desc))
    print("---- Student 4B (base) ----")
    for s in s_dicts:
        desc = f"stroke: {s.get('final_type')}, hitter: {s.get('hitter_role')}, landing_region: {s.get('landing_region')}, contact_region: {s.get('contact_region')}"
        print(desc)
        print(student_base.score_motion(desc))
    print("---- Student 4B + LoRA (highcap) ----")
    for s in s_dicts:
        desc = f"stroke: {s.get('final_type')}, hitter: {s.get('hitter_role')}, landing_region: {s.get('landing_region')}, contact_region: {s.get('contact_region')}"
        print(desc)
        print(student_lora.score_motion(desc))
PY
加载提示

使用 LoRA 时传入 adapter_path="outputs/lora_adapters_highcap"，不需 fuse，保持视觉输入能力。
若想更强：可在内存允许下进一步加大 iters（1500–2000）或层数（32→更高），不够则降批次/降层数。