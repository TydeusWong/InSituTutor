# Video Ingest Service (Step 1)

本服务负责视频输入侧标准化，并产出给 Step-2 使用的 ingest manifest。

## 文件说明

- `preprocess_video.py`：视频预处理与 ingest manifest 生成脚本
- `../../config.py`：全局配置（`VI_VIDEO_MAX_MB`、`VI_PREPROCESSED_VIDEO_DIR`、manifest 路径）

## 预处理策略

- 若输入视频大小 <= `VI_VIDEO_MAX_MB`，直接透传
- 若超阈值，执行压缩：
  - 仅压缩视频流（`-c:v libx264`）
  - 音频流保持不变（`-c:a copy`）
- 采用多轮码率下降策略确保尽量落到目标体积

## 运行方式

批处理（推荐，产出 `ingest_manifest.json`）：

```powershell
python services/video-ingest/preprocess_video.py --manifest data/raw/demo_manifest.json --output-manifest data/processed/ingest_manifest.json
```

单文件模式（仅输出预处理后视频路径）：

```powershell
python services/video-ingest/preprocess_video.py --video "data/raw/your_demo.mp4"
```

单文件模式现在会同时自动生成 `ingest_manifest.json`（默认路径由 `--output-manifest` 控制）。

可选参数：

- `--max-mb`：目标最大体积（默认读取 `VI_VIDEO_MAX_MB`）
- `--output-dir`：输出目录（默认读取 `VI_PREPROCESSED_VIDEO_DIR`）
- `--manifest`：输入原始 manifest（默认 `VI_INPUT_MANIFEST_PATH`）
- `--output-manifest`：输出 ingest manifest（默认 `VI_OUTPUT_MANIFEST_PATH`）
- `--task-id` / `--task-name` / `--environment`：单文件模式下用于补齐 demo 元信息
- `--video-id`：单文件模式可手动指定；留空自动生成合规 ID

## ingest manifest 关键字段

每条 demo 在 ingest 后会补充：

- `ingest_video_path`：可直接给 Step-2 使用的本地视频路径（相对项目根或绝对路径）
- `ingest_duration_sec`：预处理后时长
- `ingest_status`：`ready` 或 `ready_remote`
- `source_video_path`：原始输入路径（仅追溯用）

## 依赖

- 需安装 `ffmpeg` 与 `ffprobe` 并加入 PATH
