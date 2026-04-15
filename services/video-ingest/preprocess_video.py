import argparse
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import (
    VI_INPUT_MANIFEST_PATH,
    VI_OUTPUT_MANIFEST_PATH,
    VI_PREPROCESSED_VIDEO_DIR,
    VI_VIDEO_MAX_MB,
)


def get_file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def to_rel_or_abs(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def resolve_under_root(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p.resolve()
    return (ROOT / p).resolve()


def normalize_token(raw: str, fallback: str) -> str:
    token = raw.strip().lower().replace("_", "-")
    token = re.sub(r"[^a-z0-9\-]+", "-", token)
    token = re.sub(r"-{2,}", "-", token).strip("-")
    return token or fallback


def probe_duration_sec(path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return 0.0
    try:
        return float(proc.stdout.strip())
    except ValueError:
        return 0.0


def run_ffmpeg_compress_video_only(src: Path, dst: Path, target_mb: int, ratio: float = 0.95) -> None:
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise RuntimeError("未检测到 ffmpeg/ffprobe，请先安装并加入 PATH。")
    dst.parent.mkdir(parents=True, exist_ok=True)
    duration_sec = probe_duration_sec(src)
    if duration_sec <= 0:
        raise RuntimeError(f"无法读取视频时长: {src}")
    target_total_kbps = int((target_mb * 8192) / duration_sec)
    # 音频不压缩（copy），因此将主要预算给视频码率，同时预留容器开销
    video_kbps = max(250, int(target_total_kbps * ratio))
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src),
        "-c:v",
        "libx264",
        "-b:v",
        f"{video_kbps}k",
        "-maxrate",
        f"{int(video_kbps * 1.2)}k",
        "-bufsize",
        f"{int(video_kbps * 2)}k",
        "-preset",
        "medium",
        "-c:a",
        "copy",
        str(dst),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg 压缩失败: {proc.stderr.strip()[:800]}")


def preprocess_video_if_needed(video_path: Path, max_mb: Optional[int] = None, out_dir: Optional[Path] = None) -> Path:
    max_mb = max_mb if max_mb is not None else VI_VIDEO_MAX_MB
    out_dir = out_dir if out_dir is not None else VI_PREPROCESSED_VIDEO_DIR
    if not video_path.exists():
        raise FileNotFoundError(f"视频不存在: {video_path}")
    size_mb = get_file_size_mb(video_path)
    if size_mb <= max_mb:
        return video_path
    out_path = out_dir / f"{video_path.stem}_compressed_{max_mb}mb.mp4"
    # 逐轮降低视频码率，确保目标体积可达（音频始终 copy）
    ratios = [0.95, 0.85, 0.75, 0.65, 0.55]
    new_size = size_mb
    for ratio in ratios:
        run_ffmpeg_compress_video_only(video_path, out_path, max_mb, ratio=ratio)
        new_size = get_file_size_mb(out_path)
        if new_size <= max_mb:
            return out_path
    raise RuntimeError(f"压缩后仍超过 {max_mb}MB，当前 {new_size:.2f}MB: {out_path}")


def build_ingest_item(item: Dict[str, Any], max_mb: int, out_dir: Path) -> Dict[str, Any]:
    out_item = dict(item)
    video_uri = str(item.get("video_uri", "") or "")
    video_path = str(item.get("video_path", "") or "")

    if video_path:
        raw_path = resolve_under_root(video_path)
        processed_path = preprocess_video_if_needed(raw_path, max_mb=max_mb, out_dir=out_dir)
        out_item["ingest_video_path"] = to_rel_or_abs(processed_path)
        out_item["ingest_duration_sec"] = probe_duration_sec(processed_path)
        out_item["source_video_path"] = video_path
        out_item["ingest_status"] = "ready"
        # 严格两阶段：下游只消费 ingest_* 字段，不再读取原始 video_path
        out_item.pop("video_path", None)
        return out_item

    if video_uri:
        out_item["ingest_status"] = "ready_remote"
        out_item["ingest_video_uri"] = video_uri
        return out_item

    video_id = item.get("video_id", "unknown_video")
    raise ValueError(f"{video_id} 缺少 video_path/video_uri，无法进行 ingest。")


def build_ingest_manifest(manifest: Dict[str, Any], max_mb: int, out_dir: Path) -> Dict[str, Any]:
    demos = manifest.get("demos", [])
    processed = [build_ingest_item(item, max_mb=max_mb, out_dir=out_dir) for item in demos]
    return {
        "pipeline_stage": "video-ingest",
        "generated_at": utc_now_iso(),
        "demos": processed,
    }


def build_single_video_manifest(
    video_path: Path,
    processed_path: Path,
    task_id: str,
    task_name: str,
    environment: str,
    video_id: str,
    fps: str,
) -> Dict[str, Any]:
    return {
        "pipeline_stage": "video-ingest",
        "generated_at": utc_now_iso(),
        "demos": [
            {
                "task_id": task_id,
                "task_name": task_name,
                "environment": environment,
                "video_id": video_id,
                "fps": fps,
                "ingest_video_path": to_rel_or_abs(processed_path),
                "ingest_duration_sec": probe_duration_sec(processed_path),
                "ingest_status": "ready",
                "source_video_path": to_rel_or_abs(video_path),
            }
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Step-1: 视频预处理（按需压缩）")
    parser.add_argument("--video", help="输入视频路径（单文件模式）")
    parser.add_argument("--task-id", default="task-demo", help="单文件模式下的 task_id")
    parser.add_argument("--task-name", default="demo task", help="单文件模式下的 task_name")
    parser.add_argument("--environment", default="default_space", help="单文件模式下的 environment")
    parser.add_argument("--video-id", help="单文件模式下的 video_id；留空时自动生成")
    parser.add_argument("--fps", default="unknown", help="单文件模式下的 fps 元信息")
    parser.add_argument("--manifest", default=str(VI_INPUT_MANIFEST_PATH), help="输入原始 manifest（批处理模式）")
    parser.add_argument(
        "--output-manifest",
        default=str(VI_OUTPUT_MANIFEST_PATH),
        help="输出 ingest manifest（批处理模式）",
    )
    parser.add_argument("--max-mb", type=int, default=VI_VIDEO_MAX_MB, help="目标最大体积（MB）")
    parser.add_argument(
        "--output-dir",
        default=str(VI_PREPROCESSED_VIDEO_DIR),
        help="压缩后输出目录",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir).resolve()
    if args.video:
        video_path = resolve_under_root(args.video)
        result = preprocess_video_if_needed(video_path, max_mb=args.max_mb, out_dir=out_dir)
        auto_video_id = args.video_id
        if not auto_video_id:
            token = normalize_token(video_path.stem, "demo")
            auto_video_id = f"{token}__default-scene__teacher01__v01"
        manifest_obj = build_single_video_manifest(
            video_path=video_path,
            processed_path=result,
            task_id=args.task_id,
            task_name=args.task_name,
            environment=args.environment,
            video_id=auto_video_id,
            fps=args.fps,
        )
        output_manifest_path = resolve_under_root(args.output_manifest)
        write_json(output_manifest_path, manifest_obj)
        print(f"[OK] 预处理后视频: {result}")
        print(f"[OK] ingest manifest: {output_manifest_path}")
        return

    manifest_path = resolve_under_root(args.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"未找到 manifest: {manifest_path}\n"
            "可选方案：\n"
            "1) 提供 --manifest 指向现有清单；\n"
            "2) 使用 --video 单文件模式自动生成 ingest manifest。"
        )
    manifest = read_json(manifest_path)
    ingest_manifest = build_ingest_manifest(manifest, max_mb=args.max_mb, out_dir=out_dir)
    output_manifest_path = resolve_under_root(args.output_manifest)
    write_json(output_manifest_path, ingest_manifest)
    print(str(output_manifest_path))


if __name__ == "__main__":
    main()
