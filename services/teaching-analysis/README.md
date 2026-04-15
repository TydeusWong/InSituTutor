# Teaching Analysis Service (Step 2)

本服务完成“示教视频 -> 结构化教学知识”的分析流程。

## 文件说明

- `run_analysis.py`：主脚本（调用 Qwen、修复校验、聚合、输出）
- `../../config.py`：全局配置（常量 + `.env` 读取 + API Key 解析）
- `../../models/prompts/*.md`：提示词统一管理
- `../../models/schemas/teaching_knowledge.schema.json`：结构化 schema
- `../../data/processed/ingest_manifest.json`：由 Step-1 产出的输入清单

## 输入规范（manifest）

每条 demo 建议字段：

- `task_id`
- `task_name`
- `environment`
- `video_id`（命名规范：`task__scene__teacher__v01`）
- `ingest_video_path`（本地已标准化视频路径，可选）
- `ingest_video_uri`（远端视频 URL，可选）
- `ingest_duration_sec`
- `fps`

说明：`ingest_video_path` 与 `ingest_video_uri` 至少提供一个。

## 与视频预处理模块的关系

- 本服务不再执行视频压缩/预处理
- 必须先运行 `video-ingest` 生成 ingest manifest，再运行本服务
- 职责边界：
  - `video-ingest`：视频输入、压缩、ingest manifest 生成
  - `teaching-analysis`：教学知识分析与结构化输出

## 配置方式

1) 在项目根 `.env` 填写 API Key：

```dotenv
DASHSCOPE_API_KEY=你的阿里云DashScope Key
```

2) 在 `config.py` 修改常量（模型、超时、阈值、路径等）

## 运行方式

真实调用：

```powershell
python services/teaching-analysis/run_analysis.py --manifest data/processed/ingest_manifest.json
```

本地 mock：

```powershell
python services/teaching-analysis/run_analysis.py --manifest data/processed/ingest_manifest.json --mock
```

## 输出文件

默认输出到 `data/processed/analysis/`：

- `{video_id}.json`
- `structured_teaching_knowledge.json`
- `quality_report.json`
