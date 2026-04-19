import argparse
import hashlib
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


def normalize_case_id(raw: str, fallback: str) -> str:
    token = raw.strip().lower().replace("-", "_")
    token = re.sub(r"[^a-z0-9_\-]+", "_", token)
    token = re.sub(r"_{2,}", "_", token).strip("_")
    return token or fallback


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_scene_tags(item: Dict[str, Any], environment: str) -> list[str]:
    raw = item.get("scene_tags")
    if isinstance(raw, list):
        tags = [str(x).strip() for x in raw if str(x).strip()]
        if tags:
            return tags
    if isinstance(raw, str) and raw.strip():
        return [x.strip() for x in raw.split(",") if x.strip()]
    return [environment]


def infer_task_id(item: Dict[str, Any]) -> str:
    raw = str(item.get("task_id", "")).strip()
    if raw:
        return normalize_case_id(raw, "demo")
    if item.get("video_path"):
        return normalize_case_id(Path(str(item["video_path"])).stem, "demo")
    if item.get("video_uri"):
        return normalize_case_id(Path(str(item["video_uri"])).stem, "demo")
    if item.get("video_id"):
        return normalize_case_id(str(item["video_id"]).split("__")[0], "demo")
    return "demo"


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


def _parse_fraction(value: str) -> float:
    raw = (value or "").strip()
    if not raw:
        return 0.0
    if "/" in raw:
        left, right = raw.split("/", 1)
        try:
            num = float(left)
            den = float(right)
            if den == 0:
                return 0.0
            return num / den
        except ValueError:
            return 0.0
    try:
        return float(raw)
    except ValueError:
        return 0.0


def probe_fps(path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=avg_frame_rate,r_frame_rate",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return 0.0
    lines = [x.strip() for x in proc.stdout.splitlines() if x.strip()]
    if not lines:
        return 0.0
    avg = _parse_fraction(lines[0])
    if avg > 0:
        return avg
    if len(lines) > 1:
        return _parse_fraction(lines[1])
    return 0.0


def probe_resolution(path: Path) -> tuple[int, int]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "csv=p=0:s=x",
        str(path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return (0, 0)
    raw = proc.stdout.strip()
    if "x" not in raw:
        return (0, 0)
    left, right = raw.split("x", 1)
    try:
        w = int(left.strip())
        h = int(right.strip())
        return (w, h)
    except ValueError:
        return (0, 0)


def probe_video_bitrate_kbps(path: Path) -> float:
    # Prefer video stream bitrate; fallback to container bitrate if unavailable.
    stream_cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=bit_rate",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    proc = subprocess.run(stream_cmd, capture_output=True, text=True)
    if proc.returncode == 0:
        raw = proc.stdout.strip()
        if raw.isdigit():
            return round(int(raw) / 1000.0, 3)

    format_cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=bit_rate",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    proc2 = subprocess.run(format_cmd, capture_output=True, text=True)
    if proc2.returncode == 0:
        raw = proc2.stdout.strip()
        if raw.isdigit():
            return round(int(raw) / 1000.0, 3)
    return 0.0


def to_quality_label(width: int, height: int) -> str:
    if width <= 0 or height <= 0:
        return "unknown"
    h = min(width, height)
    if h >= 2160:
        return "2160p"
    if h >= 1440:
        return "1440p"
    if h >= 1080:
        return "1080p"
    if h >= 720:
        return "720p"
    if h >= 480:
        return "480p"
    return "unknown"


def run_ffmpeg_force_fps(src: Path, dst: Path, target_fps: int = 10) -> None:
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise RuntimeError("ffmpeg/ffprobe not found in PATH.")
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src),
        "-vf",
        f"fps={target_fps}",
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-preset",
        "medium",
        "-c:a",
        "copy",
        str(dst),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg force-fps failed: {proc.stderr.strip()[:800]}")


def run_ffmpeg_scale_to_720p(src: Path, dst: Path) -> None:
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise RuntimeError("ffmpeg/ffprobe not found in PATH.")
    dst.parent.mkdir(parents=True, exist_ok=True)
    # Keep aspect ratio and make dimensions even.
    scale_filter = "scale='if(gte(iw,ih),-2,720)':'if(gte(iw,ih),720,-2)'"
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src),
        "-vf",
        scale_filter,
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-preset",
        "medium",
        "-c:a",
        "copy",
        str(dst),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg 720p scaling failed: {proc.stderr.strip()[:800]}")


def run_ffmpeg_compress_video_only(src: Path, dst: Path, target_mb: int, ratio: float = 0.95) -> None:
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise RuntimeError("ffmpeg/ffprobe not found in PATH.")
    dst.parent.mkdir(parents=True, exist_ok=True)
    duration_sec = probe_duration_sec(src)
    if duration_sec <= 0:
        raise RuntimeError(f"Unable to read video duration: {src}")
    target_total_kbps = int((target_mb * 8192) / duration_sec)
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
        raise RuntimeError(f"ffmpeg compression failed: {proc.stderr.strip()[:800]}")


def preprocess_video_if_needed(video_path: Path, max_mb: Optional[int] = None, out_dir: Optional[Path] = None) -> Path:
    max_mb = max_mb if max_mb is not None else VI_VIDEO_MAX_MB
    out_dir = out_dir if out_dir is not None else VI_PREPROCESSED_VIDEO_DIR
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    out_dir.mkdir(parents=True, exist_ok=True)
    fps_path = out_dir / f"{video_path.stem}_fps10.mp4"
    # Stage 1: force output fps to 10
    run_ffmpeg_force_fps(video_path, fps_path, target_fps=10)

    # Stage 2: if higher than 720p, downscale to 720p
    w, h = probe_resolution(fps_path)
    quality_path = fps_path
    if min(w, h) > 720:
        scaled_path = out_dir / f"{video_path.stem}_fps10_720p.mp4"
        run_ffmpeg_scale_to_720p(fps_path, scaled_path)
        quality_path = scaled_path

    size_mb = get_file_size_mb(quality_path)
    if size_mb <= max_mb:
        return quality_path

    # Stage 3: if still too large, reduce video bitrate
    out_path = out_dir / f"{video_path.stem}_fps10_720p_compressed_{max_mb}mb.mp4"
    ratios = [0.95, 0.85, 0.75, 0.65, 0.55]
    new_size = size_mb
    for ratio in ratios:
        run_ffmpeg_compress_video_only(quality_path, out_path, max_mb, ratio=ratio)
        new_size = get_file_size_mb(out_path)
        if new_size <= max_mb:
            return out_path
    raise RuntimeError(f"Compressed video still exceeds {max_mb}MB: {new_size:.2f}MB ({out_path})")


def build_ingest_item(item: Dict[str, Any], max_mb: int, out_dir: Path) -> Dict[str, Any]:
    out_item = dict(item)
    video_uri = str(item.get("video_uri", "") or "")
    video_path = str(item.get("video_path", "") or "")
    environment = str(out_item.get("environment", "default_space"))
    out_item["task_id"] = infer_task_id(out_item)

    if video_path:
        raw_path = resolve_under_root(video_path)
        processed_path = preprocess_video_if_needed(raw_path, max_mb=max_mb, out_dir=out_dir)
        out_item["ingest_video_path"] = to_rel_or_abs(processed_path)
        out_item["ingest_duration_sec"] = probe_duration_sec(processed_path)
        measured_fps = probe_fps(processed_path)
        measured_bitrate_kbps = probe_video_bitrate_kbps(processed_path)
        width, height = probe_resolution(processed_path)
        out_item["fps"] = round(measured_fps, 3) if measured_fps > 0 else "unknown"
        out_item["video_bitrate_kbps"] = measured_bitrate_kbps if measured_bitrate_kbps > 0 else "unknown"
        out_item["resolution"] = f"{width}x{height}" if width > 0 and height > 0 else "unknown"
        out_item["video_quality"] = to_quality_label(width, height)
        out_item["source_video_path"] = video_path
        out_item["ingest_status"] = "ready"
        out_item["ingest_fingerprint"] = sha256_file(processed_path)
        out_item["source_audio_quality"] = str(out_item.get("source_audio_quality") or "unknown")
        out_item["scene_tags"] = build_scene_tags(out_item, environment)
        out_item.pop("task_name", None)
        out_item.pop("environment", None)
        out_item.pop("video_path", None)
        return out_item

    if video_uri:
        out_item["ingest_status"] = "ready_remote"
        out_item["ingest_video_uri"] = video_uri
        out_item["fps"] = out_item.get("fps", "unknown")
        out_item["video_bitrate_kbps"] = out_item.get("video_bitrate_kbps", "unknown")
        out_item["resolution"] = out_item.get("resolution", "unknown")
        out_item["video_quality"] = out_item.get("video_quality", "unknown")
        out_item["ingest_fingerprint"] = hashlib.sha256(video_uri.encode("utf-8")).hexdigest()
        out_item["source_audio_quality"] = str(out_item.get("source_audio_quality") or "unknown")
        out_item["scene_tags"] = build_scene_tags(out_item, environment)
        out_item.pop("task_name", None)
        out_item.pop("environment", None)
        return out_item

    video_id = item.get("video_id", "unknown_video")
    raise ValueError(f"{video_id} missing video_path/video_uri for ingest.")


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
    video_id: str,
) -> Dict[str, Any]:
    measured_fps = probe_fps(processed_path)
    measured_bitrate_kbps = probe_video_bitrate_kbps(processed_path)
    width, height = probe_resolution(processed_path)
    return {
        "pipeline_stage": "video-ingest",
        "generated_at": utc_now_iso(),
        "demos": [
            {
                "task_id": task_id,
                "video_id": video_id,
                "fps": round(measured_fps, 3) if measured_fps > 0 else "unknown",
                "video_bitrate_kbps": measured_bitrate_kbps if measured_bitrate_kbps > 0 else "unknown",
                "resolution": f"{width}x{height}" if width > 0 and height > 0 else "unknown",
                "video_quality": to_quality_label(width, height),
                "ingest_video_path": to_rel_or_abs(processed_path),
                "ingest_duration_sec": probe_duration_sec(processed_path),
                "ingest_status": "ready",
                "ingest_fingerprint": sha256_file(processed_path),
                "source_audio_quality": "unknown",
                "scene_tags": ["default_space"],
                "source_video_path": to_rel_or_abs(video_path),
            }
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Step-1: video preprocessing (compress if needed)")
    parser.add_argument("--video", help="input video path (single file mode)")
    parser.add_argument("--video-id", help="video_id in single file mode; auto-generate if empty")
    parser.add_argument("--manifest", default=str(VI_INPUT_MANIFEST_PATH), help="raw input manifest path")
    parser.add_argument("--case-id", help="output case folder under data/, e.g. test_cake")
    parser.add_argument(
        "--output-manifest",
        default=None,
        help="output ingest manifest path",
    )
    parser.add_argument("--max-mb", type=int, default=VI_VIDEO_MAX_MB, help="target max size in MB")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="output directory for compressed videos",
    )
    args = parser.parse_args()
    if args.video:
        video_path = resolve_under_root(args.video)
        case_id = args.case_id or normalize_case_id(video_path.stem, "demo")
        out_dir = Path(args.output_dir).resolve() if args.output_dir else (ROOT / "data" / case_id / "processed" / "preprocessed_videos")
        result = preprocess_video_if_needed(video_path, max_mb=args.max_mb, out_dir=out_dir)
        auto_video_id = args.video_id
        if not auto_video_id:
            token = normalize_token(video_path.stem, "demo")
            auto_video_id = f"{token}__default-scene__teacher01__v01"
        auto_task_id = normalize_case_id(video_path.stem, "demo")
        manifest_obj = build_single_video_manifest(
            video_path=video_path,
            processed_path=result,
            task_id=auto_task_id,
            video_id=auto_video_id,
        )
        output_manifest_path = resolve_under_root(args.output_manifest) if args.output_manifest else (ROOT / "data" / case_id / "ingest_manifest.json")
        write_json(output_manifest_path, manifest_obj)
        print(f"[OK] preprocessed video: {result}")
        print(f"[OK] ingest manifest: {output_manifest_path}")
        return

    manifest_path = resolve_under_root(args.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"manifest not found: {manifest_path}\n"
            "options:\n"
            "1) provide --manifest to an existing file;\n"
            "2) use --video single file mode to auto-generate ingest manifest."
        )
    manifest = read_json(manifest_path)
    case_id = args.case_id or normalize_case_id(manifest_path.stem, "batch")
    out_dir = Path(args.output_dir).resolve() if args.output_dir else (ROOT / "data" / case_id / "processed" / "preprocessed_videos")
    ingest_manifest = build_ingest_manifest(manifest, max_mb=args.max_mb, out_dir=out_dir)
    output_manifest_path = resolve_under_root(args.output_manifest) if args.output_manifest else (ROOT / "data" / case_id / "ingest_manifest.json")
    write_json(output_manifest_path, ingest_manifest)
    print(str(output_manifest_path))


if __name__ == "__main__":
    main()
