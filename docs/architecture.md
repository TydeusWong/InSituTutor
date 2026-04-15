# InSituTutor MVP Architecture

## 1. 目标与原则

- 目标：在智慧空间中跑通“示教分析 -> 知识构建 -> 学生实时提示 -> 偏差纠错”的最小闭环。
- 原则：先稳定闭环，再提升精度；先规则后模型；先学生端网页示意，后接投影。

## 2. 业务闭环（1.1）

```text
老师示教（离线视频）
  -> Qwen3.5-Omni 分析长音视频
  -> 结构化教学知识（steps/actions/attention/errors/interventions）
  -> 学生练习实时感知（摄像头）
  -> Step Tracker 判断当前步骤与偏差
  -> Intervention Engine 触发提示/纠偏
  -> 日志回流（用于复盘与规则更新）
```

## 3. 环节 I/O、依赖、兜底（1.1）

| 环节 | 输入 | 输出 | 依赖 | 失败兜底 |
|---|---|---|---|---|
| 示教接入 `ingest` | 老师示教视频 | 标准化视频元数据、存储路径 | 本地/对象存储 | 文件不可读时记录错误并跳过 |
| 视频分析 `analysis` | 长音视频 + Prompt | 结构化 JSON 草稿 | Qwen3.5-Omni API | 超时重试；失败回退到分段分析 |
| 知识构建 `knowledge` | 多次示教 JSON | 课程级知识库 | Schema 校验器 | 不合法字段进入待人工复核队列 |
| 实时跟踪 `realtime` | 摄像头帧、知识库 | 当前 step、偏差事件 | 姿态/物体检测模块 | 检测低置信度时保持上一步并降频提示 |
| 纠偏触发 `intervention` | step 状态、偏差类型 | 文本提示事件 | 规则引擎、模板库 | 限流与冷却，避免提示轰炸 |
| 评估回流 `evaluation` | 事件日志 | 指标看板输入、复盘数据 | 日志存储 | 关键字段缺失时打 warning 并补默认值 |

## 4. Step 状态机定义（1.1）

状态集合：

- `NOT_STARTED`：步骤尚未开始
- `IN_PROGRESS`：步骤进行中
- `COMPLETED`：步骤完成
- `DEVIATED`：检测到偏差，待纠偏

状态迁移：

1. `NOT_STARTED -> IN_PROGRESS`：命中步骤起始观测点
2. `IN_PROGRESS -> COMPLETED`：命中步骤达成条件
3. `IN_PROGRESS -> DEVIATED`：命中偏差规则
4. `DEVIATED -> IN_PROGRESS`：纠偏后重新满足进行条件
5. `DEVIATED -> NOT_STARTED`：严重错误，回退并重做本步

## 5. 模块解耦（1.2）

### `ingest`（视频接入）
- 职责：接入老师示教视频，生成元数据并落盘。
- 边界：不做语义理解，只做可用性与格式化。

### `analysis`（示教分析）
- 职责：调用 `Qwen3.5-Omni` 抽取步骤、关键动作、观测点、错误、纠偏。
- 边界：输出“结构化草稿”，不负责最终在线判定。

### `knowledge`（知识构建）
- 职责：聚合多次示教，形成稳定课程知识结构。
- 边界：不处理实时帧，专注离线数据治理。

### `realtime`（实时状态估计）
- 职责：基于摄像头观测与规则判断当前步骤与偏差。
- 边界：不直接出文案，只产出事件。

### `intervention`（提示与纠偏）
- 职责：把事件映射成可执行提示文本并限流输出。
- 边界：不负责检测，消费检测结果。

### `api`（接口层）
- 职责：聚合服务并对学生端网页提供统一接口。
- 边界：不承载核心业务逻辑。

### `ui`（学生端）
- 职责：展示当前步骤、下一步建议、偏差告警。
- 边界：MVP 仅网页示意，不做教师端。

### `evaluation`（评估）
- 职责：记录日志并计算闭环指标。
- 边界：不参与实时决策，只做观测和复盘。

## 6. 数据流图（1.4）

```text
Teacher Demo Video
  -> ingest
  -> analysis (Qwen3.5-Omni)
  -> knowledge store (JSON/DB)
  -> realtime coach (load task config)
  -> intervention engine
  -> student-web UI
  -> event logs
  -> evaluation
```

## 7. 时序图（1.4）

```text
学生开始练习
  -> 摄像头持续采样
  -> realtime: 计算观测点与当前 step
  -> 若正常推进: 触发 next_step_hint
  -> 若偏差命中: 触发 intervention_text
  -> UI 实时展示提示
  -> evaluation 记录事件（step/deviation/intervention/outcome）
```

## 8. 目录结构 v0.1（1.3）

```text
InSituTutor/
  README.md
  TODOlist.md
  docs/
    mvp_goal.md
    architecture.md
    mvp_scope.md
    data_schema.md
    prompt_design.md
    evaluation_plan.md
  apps/
    api/
    student-web/
  services/
    video-ingest/
    teaching-analysis/
    knowledge-builder/
    realtime-coach/
    intervention-engine/
  models/
    prompts/
    schemas/
  data/
    raw/
    processed/
    sessions/
  scripts/
  tests/
    unit/
    integration/
  infra/
    docker/
    compose/
```
