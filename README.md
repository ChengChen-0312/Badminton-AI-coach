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
cd /Users/chencheng/Documents/GitHub/Badminton-AI-coach

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
注意：蒸馏参数为：
python -m mlx_lm lora --model /Users/chencheng/llm/qwen3-4b --train --data data/mlx_train --batch-size 2 --num-layers 32 --iters 1200 --learning-rate 5e-5 --steps-per-eval 50 --adapter-path outputs/lora_adapters_highcap

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

蒸馏报告：
覆盖“未蒸馏基座 → 早期 LoRA（500 iters）→ 高容量 LoRA（1200 iters, 32 层）”的三档效果，重点关注输出质量与教师风格贴合度。
1) 训练与模型设置
教师：Qwen3-VL-30B（MLX，/Users/chencheng/llm/qwen3-30b）
学生基座：Qwen3-VL-4B（MLX，/Users/chencheng/llm/qwen3-4b）
数据：archive 全量 835 段视频 → teacher_labels.jsonl → distill_data_chat.jsonl → MLX 训练集 (train=751, valid=84)
LoRA（高容量版）：
batch=2，num_layers=32，iters=1200，lr=5e-5，adapter_path=outputs/lora_adapters_highcap
视觉塔保留，LoRA 仅作用于 language_model，strict=False
推理：在 student_mlx 传入 adapter_path 方式加载（不用 fuse，以保持视觉能力）
2) 对比样本与输出（摘要）
测试视频：

A: archive/forehand_net_shot/099.mp4
B: archive/backhand_drive/004.mp4
C: archive/forehand_clear/017.mp4
模型档位：

Teacher 30B
Student 4B（未蒸馏）
Student 4B + LoRA early（500 iters，旧 adapter）
Student 4B + LoRA highcap（1200 iters, 32 层，outputs/lora_adapters_highcap）
样本 A（forehand_net_shot/099）
Teacher：score≈65，强调击球点需更高更前；发力链/时机建议清晰。
Student base：score≈55，通用脚步/击球点建议，风格较泛。
LoRA early：score≈55，稍贴近教师，但措辞略泛。
LoRA highcap：score≈55，错误/建议更贴近教师（脚步、时机、姿态）且措辞更紧凑。
样本 B（backhand_drive/004，hitter=near）
Teacher：score≈55，指出靠网、击球点偏低；建议后撤、抬高击球点、前送发力。
Student base：score≈55，建议前脚发力、提高接触点，偏通用。
LoRA early：score≈65，部分对齐教师侧重点（脚步/时机）。
LoRA highcap：score≈65，描述更接近教师（前冲/低重心/高接触点、全身旋转）。
样本 C（forehand_clear/017，net_shot near）
Teacher：score≈65，强调网前控制、触点更低更前、减小力量。
Student base：score≈55，提示站位过近、控制不足，较泛。
LoRA early：score≈55，内容与 base 接近。
LoRA highcap：score≈65，输出与教师关注点一致（脚步后撤、控制力量、触点前/低，稳定摆动）。
3) 结论与效果
视觉能力保持：基于原始 VLM + adapter，支持图像输入；加载日志显示 Applied 448 adapter weights。
贴合度：高容量 LoRA（1200 iters, 32 层, batch=2）在多个样本上输出更接近教师，得分区间与教师一致（55–65），错误/建议的侧重点向教师靠拢。
推理性能：基于先前测试，单样本 <1s（M4 Max），新增高容量 adapter 对推理时间影响轻微。
建议默认推理模式：student_mlx + adapter_path=outputs/lora_adapters_highcap（保留视觉塔，不用 fuse）。
4) 后续优化建议（若继续提升）
数据：增加样本或重复高质量样本；优化 teacher prompt，确保 JSON 结构稳定。
训练：可试 iters=1500–2000（时间换效果）；若 OOM 将 batch=1 或 num_layers 降到 24/16。
对比评估：固定一组测试视频，定期比对教师/学生/学生+LoRA 输出，观察建议的具体性与风格一致性。
可将上述内容直接放入 README 的“开发日志/蒸馏报告”章节，方便后续追踪与对外说明。

我理解你要做的 v3.9 升级：在现有“视觉+蒸馏”框架里，正式把姿态特征（肘角/肩角/躯干/拍面）纳入分析和提示，替换掉早期的 if-else 打分，形成完整闭环：pose → 角度特征 → 提示 → Teacher/Student LLM 评估。

下面给出可落地的实施路线（包含关键改动点、验收标准与建议命令），你可以按需提交为 v3.9 PR。

# v3.9 目标 在 analyse_video 中可选启用 pose（MediaPipe），对击球帧附近 ±N 帧计算姿态。
提取肘角/肩角/躯干角/拍面角，存入 stroke summary。
Teacher 标签生成时，将这些角度以自然语言提示注入 prompt；Student 蒸馏与在线反馈沿用同一提示格式。
产出 JSON 里能看到 pose 特征，反馈能引用关节级建议（“肘部未打开”等）。
建议改动点
pose 特征整理（替换硬规则）
新增/更新 src/spatial_logic/pose_quality.py：
pose_features_to_prompt(pose_landmarks) -> str：把提取的角度变成 LLM 提示文字。
保留数值提取（肘、肩、躯干、拍面角），但不在这里打分。
在 stroke summary 保存姿态
src/pipeline/extract_strokes.py：
在生成 StrokeSummary 时，针对 contact_frame（或附近帧）取 pose_landmarks。
调用 extract_pose_features 得到角度字典，放入 pose_landmarks / pose_features 字段（需在 dataclass 上新增字段）。
Teacher 标签 prompt 注入姿态
scripts/generate_teacher_labels_mlx.py：
引入 pose_features_to_prompt，在构造 user_prompt 时追加“关键点信息”段落。
系统提示强调“关注肘角/肩角/躯干旋转/拍面角度”。
在线反馈也走同一提示
src/ai_score/action_feedback.py（或对应 student 调用处）：
构造 student prompt 时，追加 pose_features_to_prompt(summary.pose_landmarks) 结果。
让学生模型输出 score/comment/tips JSON。
analyse_video 启用 pose（可选）
在 analyse_video 增加 enable_pose 参数，默认 False；开启时对击球帧附近 ±N 帧跑 pose_estimator（MediaPipe）。
将 pose 结果挂到 analysis_result（例如 analysis_result.pose_results[frame_idx] = landmarks）。
验收标准
结果 JSON 中可看到 pose_features 字段（角度数值或缺失说明）。
反馈文本能引用关节级问题（如“肘部未打开”“拍面角度偏平”）。
Pose 计算范围仅在击球帧附近 ±N 帧，成本可控。
参考命令与流程
生成标签（启用 pose 后重新跑）：
python scripts/generate_teacher_labels_mlx.py --videos-glob "archive/**/*.mp4" \
  --output data/distill/teacher_labels_pose.jsonl \
  --config src/config/v3_realtime.yaml \
  --teacher-model-path /Users/chencheng/llm/qwen3-30b
转数据：
python scripts/prepare_distill_dataset.py --input data/distill/teacher_labels_pose.jsonl \
  --output data/distill/distill_data_chat_pose.jsonl
python scripts/prepare_mlx_data.py  # 生成 data/mlx_train 带 pose 描述的纯 text
训练 LoRA（可沿用 highcap 参数）：
python -m mlx_lm lora --model /Users/chencheng/llm/qwen3-4b --train --data data/mlx_train \
  --batch-size 2 --num-layers 32 --iters 1200 --learning-rate 5e-5 \
  --steps-per-eval 50 --adapter-path outputs/lora_adapters_pose
推理对比（teacher / base / LoRA_pose）：
在 ActionFeedback 里传 adapter_path="outputs/lora_adapters_pose"，观察反馈中是否出现关节级建议。
提交/PR 描述示例
v3.9 – Pose-aware prompts for teacher/student LLM feedback

- Add pose_features_to_prompt to convert elbow/shoulder/trunk/racket angles into LLM-friendly hints.
- Stroke summaries now store pose_landmarks/pose_features for contact frames.
- Teacher label generation and online ActionFeedback inject pose hints into prompts, removing hard-coded scoring rules.
- Optional pose estimation toggle in analyse_video to compute pose only around contact frames for cost control.
- LoRA distillation re-run with pose-aware prompts (adapter: outputs/lora_adapters_pose).

Below is a single Python one-liner you can run in the repo root to compare the four variants on the same three videos (archive/forehand_net_shot/099.mp4, archive/backhand_drive/004.mp4, archive/forehand_clear/017.mp4):

Teacher 30B (Qwen3-VL-30B at /Users/chencheng/llm/qwen3-30b)
Student 4B base (no adapter)
Student 4B + old LoRA (e.g., outputs/lora_adapters_highcap)
Student 4B + new pose-aware LoRA (e.g., outputs/lora_adapters_pose)
Adjust adapter_old/adapter_pose if your paths differ.
python - <<'PY'
import yaml
from src.pipeline.analyse_video import analyse_video
from src.pipeline.extract_strokes import summarise_strokes_from_analysis, stroke_summaries_to_dicts
from src.ai_score.action_feedback import ActionFeedback

cfg = yaml.safe_load(open("src/config/v3_realtime.yaml"))
videos = [
    "archive/forehand_net_shot/099.mp4",
    "archive/backhand_drive/004.mp4",
    "archive/forehand_clear/017.mp4",
]

teacher_path = "/Users/chencheng/llm/qwen3-30b"
student_base = "/Users/chencheng/llm/qwen3-4b"
adapter_old = "outputs/lora_adapters_highcap"   # old distill (no pose)
adapter_pose = "outputs/lora_adapters_pose"     # pose-aware distill

teacher = ActionFeedback(mode="teacher_mlx", teacher_mlx_path=teacher_path)
student_base_engine = ActionFeedback(mode="student_mlx", student_mlx_path=student_base)
student_old_lora = ActionFeedback(mode="student_mlx", student_mlx_path=student_base, adapter_path=adapter_old)
student_pose_lora = ActionFeedback(mode="student_mlx", student_mlx_path=student_base, adapter_path=adapter_pose)

def run_video(path):
    print(f"\n=== Video: {path} ===")
    res = analyse_video(path, config=cfg)
    s_dicts = stroke_summaries_to_dicts(
        summarise_strokes_from_analysis(
            res,
            enable_hitter_inference=True,
            pose_window=cfg.get("pose", {}).get("window", 3),
        )
    )
    for s in s_dicts:
        desc = f"stroke: {s.get('final_type')}, hitter: {s.get('hitter_role')}, landing_region: {s.get('landing_region')}, contact_region: {s.get('contact_region')}"
        print("Summary:", desc)
        print("Teacher:", teacher.score_motion(desc))
        print("Student 4B base:", student_base_engine.score_motion(desc))
        print("Student 4B + old LoRA:", student_old_lora.score_motion(desc))
        print("Student 4B + pose LoRA:", student_pose_lora.score_motion(desc))

for v in videos:
    run_video(v)
PY
v3.9 – Pose-Aware Distillation (Qwen3-VL Teacher → Student)
新增能力

在 analyse_video 中接入 MediaPipe Pose（可配置 pose.enable/stride/window），输出 pose_landmarks 与关节角度特征（肘角、肩角、躯干角、拍面角）。
pose_quality.py 将关节角度转成 LLM 提示文本；extract_strokes 为每个击球挂上 pose_features（缺帧时向前后窗口找最近姿态）。
Teacher/Student 提示统一加入姿态关注点，Teacher 标签与在线反馈都能引用肘/肩/躯干/拍面细节。
LoRA 训练保持视觉塔：使用基座 VLM + LoRA adapter（不使用 fused 模型）。
蒸馏与模型效果（示例视频：forehand_net_shot/099.mp4, backhand_drive/004.mp4, forehand_clear/017.mp4）

Teacher 30B：score 通常 55–65；反馈聚焦击球点高度、前移、发力链、时机。
Student 4B 基座：给出通用建议，score 多在 55–65。
Student 4B + LoRA (pose-aware, 32 layers, 1200 iters, lr=5e-5)：用词、侧重点更接近 Teacher（脚步/时机/击球点/姿态），score 仍在 55–65 区间，显示蒸馏在关注点对齐上有效，但分数尚未明显上移。
当前使用方式

Teacher: ActionFeedback(mode="teacher_mlx", teacher_mlx_path="/Users/chencheng/llm/qwen3-30b")
Student 4B 基座: ActionFeedback(mode="student_mlx", student_mlx_path="/Users/chencheng/llm/qwen3-4b")
Student 4B + LoRA (旧版无姿态): adapter_path="outputs/lora_adapters_highcap"
Student 4B + LoRA (姿态版): adapter_path="outputs/lora_adapters_pose"（推荐）
关键改动点

Pose 特征→LLM 提示：统一在 Teacher/Student prompt 中强调肘/肩/躯干/拍面四个角度。
LoRA 仅作用于语言头，视觉塔保持原始能力，避免丢失图像理解。
数据/脚本：generate_teacher_labels_mlx.py（可带 pose）、prepare_distill_dataset.py、prepare_mlx_data.py、mlx_lm lora 训练，适配器目录示例：outputs/lora_adapters_pose/.
后续提升建议

若需更显著的分数差异：在 Teacher 标签或 prompt 中加入评分准则（优质动作 80–95；明显错误 40–60），或对标签分布做拉宽，再跑一轮 LoRA。
可增加训练迭代数或微调学习率/调度（如 cosine + warmup），继续用 pose-aware 数据。
如需更强容量，可尝试更高 r/alpha 或 DoRA（如果 CLI 支持），注意内存/速度平衡（M4 Max 36GB 目前 32 层、batch=1–2、iters 1200 已验证可跑，峰值 ~8GB）。