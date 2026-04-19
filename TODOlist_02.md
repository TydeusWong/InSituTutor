# InSituTutor 重构 TODO List v0.2（TODOlist_02）

## 0. 重构目标与边界（先冻结）

- [x] 冻结新主线：`示教视频 -> section/动作单元解析 -> 判定标准训练 -> 实时教学反馈`
- [x] 明确阶段边界：
  - 离线阶段允许慢（分钟级~小时级）
  - 在线阶段要求高实时（端到端低延迟事件流）
- [x] 明确分类闭集：最小动作单元仅允许 `step | error | not_related`
- [x] 明确在线行为：
  - 命中当前 step 判定标准 -> 进入下一 step
  - 命中本 section error -> 进入纠错
  - 两者都不命中 -> 调 Omni 生成纠错并沉淀 error 库
- [x] 冻结文档：`docs/architecture_v2.md`、`docs/data_schema_v2.md`、`docs/realtime_contract_v2.md`

---

## 1. 新架构设计（模块重组）

### 1.1 目录与服务重构
- [x] 设计并落地 v2 目录（建议）：
  - `services/video-ingest`（保留）
  - `services/teaching-segmentation`（Omni 章节+动作单元解析）
  - `services/strategy-builder`（教学策略构建）
  - `services/criteria-trainer`（小模型训练与判定标准生成）
  - `services/realtime-orchestrator`（实时章节/步骤调度）
  - `services/error-memory`（在线新增 error 沉淀）
  - `apps/student-web`（升级为实时事件驱动）
- [x] 梳理老模块映射关系与弃用计划（`teaching-analysis`、`knowledge-builder` 的兼容层）
- [x] 统一事件总线接口（先本地队列/Redis，后可替换）

### 1.2 数据契约 v2（核心）
- [x] 定义 `section` 结构：
  - `section_id`, `section_name`, `section_goal`, `expected_section_state`, `time_range`
- [x] 定义 `atomic_unit`（最小有意义动作单元）结构：
  - `unit_id`, `time_range`, `evidence(audio/vision)`, `class(step|error|not_related)`
- [x] 定义 `step` 结构（从 unit 提炼）：
  - `step_id`, `prompt`, `focus_points`, `common_mistakes`, `expected_post_state`
- [x] 定义 `error` 结构：
  - `error_id`, `trigger_signature`, `correction_prompt`, `recovery_actions`
- [x] 定义 `detector_plan` 结构（程序化判定语言）：
  - `models_required`, `features`, `constraints`, `score_fn`, `pass_threshold`
- [x] 定义 `runtime_state`：
  - `current_section`, `current_step`, `active_error`, `last_passed_step`, `latency_ms`
- [x] 定义 `error_memory`：
  - `discovered_error`, `context_clip`, `omni_fix`, `review_status`, `promoted_to_catalog`

---

## 2. Pipeline A：视频理解与教学策略生成（离线）

### 2.1 视频预处理（沿用）
- [x] 保持 `video-ingest` 逻辑不变（压缩、时长、ingest manifest）
- [x] 增加 v2 字段：`ingest_fingerprint`, `source_audio_quality`, `scene_tags`

### 2.2 Omni 章节切分与动作单元抽取
- [x] 输入压缩视频到 Omni，输出 section 列表
- [x] 为每个 section 生成：
  - 本章内容摘要
  - 本章预期效果（最终状态描述）
  - 时间范围
- [x] 将 section 内行为切分为最小有意义动作单元（unit）
- [x] 为每个 unit 打标 `step|error|not_related`（融合语音+视觉证据）
- [x] 丢弃 `not_related`，保留 `step/error`

### 2.3 Step/Error 教学策略构建
- [x] 为每个 step 生成：
  - 执行提示语
  - 重点观察点
  - 易错点
  - 完成后预期状态
- [x] 为每个 error 生成：
  - 触发描述
  - 纠错提示
  - 恢复动作

### 2.4 小模型列表与运行环境（先于 detector_plan）
- [x] 冻结小模型列表（v1）：
  - 姿态识别：`MediaPipe Pose Landmarker`
  - 手部识别：`MediaPipe Hand Landmarker`
  - 物体位置识别：`Grounding DINO` / `YOLO`
  - 物体颜色识别：`OpenCV`
- [x] 设计并落地小模型目录结构（保持架构整齐、可读）：
  - `models/small-models/pose/mediapipe-pose-landmarker/`
  - `models/small-models/hand/mediapipe-hand-landmarker/`
  - `models/small-models/object/grounding-dino/`
  - `models/small-models/object/yolo/`
  - `models/small-models/color/opencv/`
  - `services/criteria-trainer/adapters/`（统一推理适配层）
  - `services/criteria-trainer/configs/`（模型配置与阈值）
- [x] 完成环境初始化与依赖校验：
  - 模型依赖安装脚本
  - GPU/CPU 运行能力检测
  - 统一 I/O 接口约定（输入帧、输出特征、置信度）
  - `healthcheck` 脚本（逐模型可用性）

### 2.5 按 step/error 时间切片
- [x] 基于 `sections_units.json` 与 `teaching_strategy_v2.json`，按每个 step/error 的 `time_range` 切片
- [x] 输出切片数据集（建议）：
  - `data/<case_id>/v2/slices/<section_id>/<step_or_error_id>/clip.mp4`
  - `data/<case_id>/v2/slices/index.json`
- [x] 每个切片附带上下文元信息：
  - 所属 section 基本信息
  - 该 step/error 基本信息
  - 全视频 summary（来自 `video_overview.summary`）

---

## 3. Pipeline B：判定标准训练与校验（离线）

### 3.1 Omni 检测策略决策（先只尝试发送第一个step步骤，避免浪费token）
- [x] 将“单个切片 + 全局 summary + section 信息 + step/error 信息 + 小模型列表”发送给 Omni
- [x] 让 Omni 产出该片段的模型选择结果：
  - 应使用哪些小模型（可多选）
  - 每个模型检测什么目标/特征
  - 检测顺序与并行策略
- [x] 让 Omni 产出具体检测策略（结构化）：
  - 判定条件代码（`judgement_conditions[*].code`）
- [x] 输出文件：
  - `data/<case_id>/v2/detector-plans/detector_plan_v2.json`
  - （可选）`data/<case_id>/v2/detector-plans/<step_or_error_id>.json`
- [x] 预定义 Grounding DINO 判定原语并固化文档（供后续直接引用代码）：
  - 绝对位置：`abs_center_xy`、`abs_pos_distance`
  - 相对位置：`rel_x`、`rel_y`、`rel_distance`
  - Hand + DINO 关系：`hand_to_bbox_distance`、`hand_in_bbox`
  - 配置文件：`services/criteria-trainer/configs/detection_condition_primitives_v1.json`
- [x] detector_plan 输出规范收敛：
  - `model_selection` 不含 `reason`
  - 不输出 `pass_threshold`
  - `features + constraints` 合并为 `judgement_conditions`
  - `judgement_conditions[*].code` 必须为可直接执行的判定条件代码

### 3.2 回放校验（先仅验证第一个 step，避免浪费时间）
- [x] 使用该 step 的判定策略回放检测原示教视频（压缩后的）；一旦判定成功，立即结束程序，并返回“首次判定成功”的视频时间点（秒）。
- [x] 考虑 DINO 推理较慢，回放检测时按“每秒采样 2 帧”执行。  
      采样规则：若视频为 2n FPS，则取每秒第 1 帧和第 n+1 帧。

### 3.3 对所有步骤执行Omni 检测策略决策
- [x] 遍历 `data/<case_id>/v2/slices/index.json` 中所有 `slice_type=step` 的切片，逐个执行 2.6 的 Omni 决策流程（不再只限第一个 step）。
- [x] 逐 step 执行时沿用同一输出规范：
  - `model_selection` 不含 `reason`
  - 不输出 `pass_threshold`
  - 输出 `judgement_conditions[*].code`，且仅可调用 `detection_condition_primitives_v1.json` 中已定义原语
  - 采用“最小可判定模型集”原则（非必要不增模型/实体）
- [x] 执行策略：
  - 默认串行（避免 token/并发开销过高）
  - 支持断点续跑：已存在且合法的 `<step_id>.json` 可跳过
  - 任一步失败时记录错误并继续后续 step，最终输出失败清单
- [x] 输出文件：
  - `data/<case_id>/v2/detector-plans/<step_id>.json`（每个 step 一个）
  - `data/<case_id>/v2/detector-plans/detector_plan_v2.json`（聚合索引，含成功/失败统计）
- [x] 完成后做一致性校验：
  - `step` 总数 == 产出的 `<step_id>.json` 总数（减去失败清单）
  - 每个结果文件均可被 `sanitize_detector_plan` 通过
  - `judgement_conditions[*].code` 可被回放校验脚本直接引用


---

## 4. Pipeline C：高实时教学反馈（在线）

### 4.1 实时输入与状态机
- [ ] 接入实时流式视频（WebRTC/WebSocket）
- [ ] section 级状态机：
  - `SECTION_ENTERED -> STEP_RUNNING -> SECTION_VALIDATING -> SECTION_DONE`
- [ ] step 级状态机：
  - `WAITING -> RUNNING -> PASSED | ERROR_HIT | UNKNOWN`

### 4.2 实时判定与调度
- [ ] RUNNING 时仅发布当前 step 提示，抑制多余动作
- [ ] 命中当前 step 判定标准 -> `next_step`
- [ ] 命中 section 内 error 判定标准 -> 进入纠错流程
- [ ] UNKNOWN（都不命中） -> 调 Omni 生成纠错建议 + 写入 error memory

### 4.3 纠错闭环
- [ ] 纠错后持续检测，直到：
  - 命中当前 step 的动作或预期状态
  - 或升级为人工介入
- [ ] section 结束时比对 `expected_section_state`
- [ ] 一致才允许进入下一 section

### 4.4 延迟与稳定性
- [ ] 端到端延迟监控：采集、推理、规则判定、下发提示
- [ ] 目标指标（先设默认值，审阅后再冻结）：
  - `P95 step 判定延迟 < 300ms`
  - `P95 提示下发延迟 < 500ms`
- [ ] 建立降级策略（模型超时、单模型失效、丢帧）

---

## 5. 重构执行顺序（建议）

- [ ] M1：冻结 v2 契约与目录（只做文档与接口，不动旧链路）
- [ ] M2：实现 `teaching-segmentation + strategy-builder`，产出 `teaching_strategy_v2.json`
- [ ] M3：实现 `criteria-trainer`，产出 `criteria_bundle_v2`
- [ ] M4：实现 `realtime-orchestrator` 的 section/step 状态机
- [ ] M5：接入 `error-memory` 与 Omni UNKNOWN 纠错
- [ ] M6：升级 `student-web` 为事件驱动 UI（section/step/error）
- [ ] M7：灰度替换老链路，保留回退开关

---

## 6. 验收口径（Definition of Done）

- [ ] 至少 1 个完整任务可跑通 section->step->error->纠错->section 完成
- [ ] 每个 section 至少 3 个 step，且有明确 `expected_section_state`
- [ ] 每个 step/error 均有可执行 `detector_plan` 与对应模型引用
- [ ] 离线回放达到阈值（阈值待审阅后冻结）
- [ ] 在线演示可稳定推进，不因 UNKNOWN 卡死
- [ ] error memory 可新增、复盘、并可人工审核后入库

---

## 7. 风险与预案（v2）

- [ ] 风险：Omni 对最小动作单元切分不稳定  
  预案：增加二次切分规则与人工抽检入口
- [ ] 风险：DINO 自动标注噪声影响 YOLO 训练  
  预案：样本质检 + 小规模人工修标
- [ ] 风险：在线多模型并行导致延迟抖动  
  预案：按 step 动态启停模型，做模型裁剪与缓存
- [ ] 风险：UNKNOWN 过多导致 error 库膨胀  
  预案：设定去重与晋升规则（置信度+出现频次+人工审核）

---

## 8. 本周可立即开工项（第一批）

- [ ] 建立 `docs/data_schema_v2.md` 初稿（section/unit/plan DSL）
- [ ] 新建 `services/teaching-segmentation` 骨架与 I/O 契约
- [ ] 新建 `services/strategy-builder`，先实现 JSON 拼装与校验
- [ ] 设计 `detector_plan` DSL 的最小可用语法与示例
- [ ] 准备一段示教视频做端到端离线小样验证



