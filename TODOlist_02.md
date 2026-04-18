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
- [x] 生成 section 内 step/error 顺序图与依赖关系

### 2.4 检测器规划（从自然语言到程序化）
- [ ] 为每个 step/error 生成 `detector_plan`：
  - 需要哪些小模型（MediaPipe / Hand Landmark / Grounding DINO/yolo11n / OpenCV）
  - 每个模型输出哪些特征
  - 如何组合成数学/逻辑判定
- [ ] 定义统一 DSL（建议）：
  - 时序谓词：`before/after/within`
  - 空间谓词：`overlap/iou/above/left_of/distance`
  - 动作谓词：`grasp/release/move_to`
  - 置信度聚合：`weighted_score >= threshold`
- [ ] 产出文件：`teaching_strategy_v2.json`

---

## 3. Pipeline B：判定标准训练与校验（离线）

### 3.1 训练样本自动构建
- [ ] 根据 step/error 的时间范围切出视频片段
- [ ] 对每个目标对象调用 Grounding DINO 自动标注
- [ ] 每类对象抽样 25-30 张高质量帧做训练集
- [ ] 建立样本质检规则（框质量、遮挡、运动模糊）

### 3.2 小模型训练
- [ ] YOLO 训练（按对象类别/场景版本组织）
- [ ] 姿态与手部特征抽取流程固定化（MediaPipe）
- [ ] 颜色识别流程固定化（OpenCV，颜色空间/阈值配置）
- [ ] 形成 `detector_registry`（模型版本、适用 step/error、输入输出）

### 3.3 判定标准编译
- [ ] 将 `detector_plan` 编译为可执行规则图（RuleGraph）
- [ ] 输出每个 step/error 的 `pass/fail` 判定器
- [ ] 支持规则解释（失败原因、命中证据、阈值明细）

### 3.4 离线回放校验（必须）
- [ ] 用原示教视频回放每个 step/error
- [ ] 校验项：
  - step 通过率
  - error 召回率
  - 误报率
  - 时间对齐误差
- [ ] 达标后发布 `criteria_bundle_v2`

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
