# Video Ingest Service (Step 1)

本服务负责视频输入侧标准化，并产出给 Step-2 使用的 ingest manifest。

## 文件说明

- `preprocess_video.py`：视频预处理与 ingest manifest 生成脚本
- `../../config.py`：全局配置（`VI_VIDEO_MAX_MB`、`VI_PREPROCESSED_VIDEO_DIR`、manifest 路径）

## 预处理策略

两阶段固定流程：

1. 先统一降到 `10fps`
2. 如果结果仍大于 `VI_VIDEO_MAX_MB`，再做画质/码率压缩

压缩阶段：

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
- `--video-id`：单文件模式可手动指定；留空自动生成合规 ID
- `--case-id`：指定输出 case 目录名（默认由视频文件名自动推断）

## ingest manifest 关键字段

每条 demo 在 ingest 后会补充：

- `ingest_video_path`：可直接给 Step-2 使用的本地视频路径（相对项目根或绝对路径）
- `ingest_duration_sec`：预处理后时长
- `ingest_status`：`ready` 或 `ready_remote`
- `source_video_path`：原始输入路径（仅追溯用）
- `fps`：预处理后的视频帧率（本地视频会自动写入，通常为 `10.0`）

说明：

- `task_id` 单文件模式下自动使用视频文件名（例如 `test_cake.mp4 -> task_id=test_cake`）
- `task_name` 与 `environment` 在 ingest 阶段不再产出（留给后续 Omni 阶段推断）

## 依赖

- 需安装 `ffmpeg` 与 `ffprobe` 并加入 PATH

## v2 输出隔离（避免覆盖）

推荐单视频直接运行，不显式传 `--output-manifest`，系统会按视频名自动分目录：

```powershell
python services/video-ingest/preprocess_video.py --video "tests/videos/test_cake.mp4"
```

默认会输出到：

- `data/test_cake/ingest_manifest.json`
- `data/test_cake/processed/preprocessed_videos/...`

可用 `--case-id` 显式指定目录名，例如：

```powershell
python services/video-ingest/preprocess_video.py --video "tests/videos/test_cake.mp4" --case-id test_cake
```
