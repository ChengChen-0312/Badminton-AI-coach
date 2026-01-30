# Court Detection 一周完善计划（3 人）

目标：把当前 court detection 从“能跑”提升到“可靠可用”，重点解决：
- 误把内场/发球区当外侧双打边界
- roof/wall 误吸附导致 homography 崩坏
- confidence/reason/debug 图不够可解释

## 本周验收（Done Definition）

1. 在测试视频集（建议 20 段，覆盖不同馆/角度/遮挡）上：
   - 自动检测成功率 ≥ 80%（能输出 corners + homography 不崩）
   - 误选发球区/内场矩形 ≤ 5%
   - roof/wall 误吸附 ≤ 5%
2. 每次运行 demo 输出 `reports/**/court_detect_debug.jpg`，标题可读且包含：
   - `source/conf/reason/method`
   - `span_x/span_y/area_floor/rel_top/tpl_f1/edge_support/mask_density/floor_bbox_is_floor`
3. 失败时有明确回退路径：
   - `auto_failed` → 提示用 `python scripts/calibrate_court_corners.py --video <path> --print-yaml` 手动标定

## 角色分工（固定角色）

- A（集成&调参负责人）：主线合并、参数打包、最终验收、回归测试
- B（语义/规则验证负责人）：专治“内场小矩形高置信度通过”
- C（Debug/数据集负责人）：测试集、失败分类、可视化与指标输出、评估脚本

## 每人需要改哪些代码（文件责任表）

### A：集成/调参/多帧稳定
- `src/pipeline/analyse_video.py`：多帧选最佳角点（前 N 帧）与回退策略接入
- `scripts/demo_friend.py`：demo 输出 court debug 图、标题字段补齐
- `src/vision/court_detector.py`：只做“参数/权重/阈值”整合（不要在这里写大量新规则逻辑）
- `README.md` 或 `docs/*`：补充使用手册（auto_failed 的处理、手动标定流程）

### B：规则层（语义约束）
- `src/vision/court_rules.py`（新建）：实现 `validate_by_rules()`、`corner_cross_score()`、`interior_line_density()` 等
- `src/vision/court_detector.py`：在 `_validate_and_score()` 中调用规则层；新增 reject reason；调整 hard/soft 逻辑

### C：评估/可视化/数据集
- `scripts/eval_court_detector.py`（新建）：读取 `manifest.csv`，批量跑检测，输出 summary CSV
- `tests/court_eval_set/manifest.csv`（新建，或改放 `data/court_eval_set/manifest.csv`）：视频清单与场景标签
- `reports/court_eval_summary.csv`、`reports/*`：回归对比报告与典型案例图册（输出文件无需进 git，可只保留模板/README）

## 计划表（按天拆解）

### Day 1（基线与数据集）
- A
  - 锁定当前基线（打 tag 或记录 commit hash）
  - 统一落盘 `court_metrics.json`（每段视频一条）
- B
  - 定义失败类型：`ROOF_WALL / INNER_RECT / PARTIAL`
  - 输出规则列表草案：`docs/court_rules.md`
- C
  - 准备 20 段测试集并分类，产出 `manifest.csv`

### Day 2（增强 Debug 可解释性）
- A
  - debug overlay 标题固定字段补齐（见验收标准）
- B
  - 固化接口：`validate_by_rules(metrics)->(ok, reason)`
  - 新建 `src/vision/court_rules.py`（先签名+空实现）
- C
  - 一键评估脚本：跑 manifest 并输出汇总

### Day 3（解决“内场小矩形误通过”）
- B（核心）
  - 强约束：`span_y`（仅在 `floor_bbox_is_floor=True` 时）更严格（建议 0.85~0.90 起）
  - 强约束：`rel_top` 太大（上边掉到中场）→ `R_top_edge_too_low`
  - 强约束：`top_edge` 白线支持度太低 → reject
- A
  - 调整置信度权重：降低纯 `edge_support` 权重，提高 `span/area/tpl_f1`
- C
  - 跑回归并输出 before/after 报告

### Day 4（角点必须是真交叉点）
- B
  - 实现 `corner_cross_score()`（Sobel + 方向双峰）
  - 加硬阈值：平均 cross_score < 0.45 → `R_corner_not_intersection`
- A
  - 把 cross_score 纳入 debug overlay + metrics
- C
  - 专测 roof/wall 类视频，确认误检率下降

### Day 5（模板匹配升级：tpl_f1 真正有效）
- A（核心）
  - 让 `_template_f1_score()` 使用更“线性”的 line mask（`_compute_line_mask()` 输出）或同时算两种 `tpl_f1`
  - 提高门槛（先保守）：`tpl_f1 < 0.08` → reject（按数据集调）
- B
  - 为 `R4_template_mismatch` 增加子原因/统计，便于阈值调参
- C
  - summary.csv 增加列：`tpl_f1_line/tpl_f1_white/cross_score/rel_top`

### Day 6（多帧稳定选择 + 防抖）
- A（核心）
  - 在 `analyse_video.py`：前 N 帧跑 detector，选 confidence 最大的帧作为最终 corners
  - 可选：中位数融合（防单帧异常）
- B
  - 可选：temporal consistency（与历史最好角点差太大扣分）
- C
  - 全量重跑测试集，输出最终指标

### Day 7（打包交付与使用手册）
- A
  - 参数整理（配置/默认阈值）、写 playbook
- B
  - 维护 reject reason 对照表：Reason → 可能原因 → 修复/标定建议
- C
  - 典型案例图册（成功/失败各 2 张/类）

## 备注（建议约定）

- 任何输出路径尽量用相对路径（不要写绝对路径到文档/代码里）。
- “宁可失败也不要给错误 homography”：auto_failed 时跳过 heatmap/region，避免误导结果。
