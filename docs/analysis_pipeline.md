# Step 2 执行说明（示教分析）

## 1. 视频命名规范

统一使用：

`task__scene__teacher__v01`

示例：

- `task-demo__space-a__teacher01__v01`
- `task-demo__space-a__teacher01__v02`

## 2. 两阶段输入规范

1) Step-1（video-ingest）读取原始 `demo_manifest.json`，输出 `ingest_manifest.json`  
2) Step-2（teaching-analysis）只读取 `ingest_manifest.json`

Step-2 demo 关键字段：

- `ingest_video_path` 或 `ingest_video_uri`
- `ingest_duration_sec`

## 3. 分析策略

1. 优先整段长视频分析（full_video）
2. 失败则降级为时间段分析（segmented）
3. 将分段结果聚合为统一教学知识

## 4. 质量抽检流程

- 抽检比例：每个任务至少抽检 20% 示例（最少 1 条）
- 抽检维度：
  - step 顺序是否合理
  - 关键动作是否可执行
  - 观测点是否可检测
  - 错误与纠偏是否一一对应
- 抽检结论写入 `meta.review_status`：
  - `draft`
  - `needs_human_review`
  - `approved`

## 5. 可解释性与追溯

- 每条结果必须带 `meta.source_videos`
- 输出按视频单独保存，再输出聚合文件
- 聚合文件保留 `meta.aggregation_notes`
