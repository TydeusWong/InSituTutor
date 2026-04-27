import argparse
import subprocess
from pathlib import Path

from common import read_json, rel_path, safe_float, session_root, utc_now_iso, write_json


ROOT = Path(__file__).resolve().parents[2]


def cut_clip(src: Path, start: float, end: float, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{max(0.0, start):.3f}",
        "-to",
        f"{max(0.0, end):.3f}",
        "-i",
        str(src),
        "-c",
        "copy",
        str(dst),
    ]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            f"{max(0.0, start):.3f}",
            "-to",
            f"{max(0.0, end):.3f}",
            "-i",
            str(src),
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            str(dst),
        ]
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg error slice failed: {proc.stderr.strip()[:500]}")


def build_slices(case_id: str, session_id: str, pad_sec: float) -> Path:
    base = session_root(case_id, session_id)
    manifest = read_json(base / "session_manifest.json") if (base / "session_manifest.json").exists() else {}
    video_value = manifest.get("review_media_path") or manifest.get("normalized_video_path") or (base / "raw" / "teaching_session_review.mp4")
    video_path = Path(str(video_value))
    if not video_path.is_absolute():
        video_path = ROOT / video_path
    errors_path = base / "reflection" / "error_events_v1.json"
    errors = read_json(errors_path) if errors_path.exists() else []
    if not isinstance(errors, list):
        errors = []
    items = []
    for idx, err in enumerate(errors, start=1):
        if not isinstance(err, dict):
            continue
        error_id = str(err.get("error_id") or f"err_{idx:04d}")
        tr = err.get("time_range_sec") if isinstance(err.get("time_range_sec"), dict) else {}
        start = max(0.0, safe_float(tr.get("start"), safe_float(tr.get("start_sec"), 0.0)) - pad_sec)
        end = max(start + 0.1, safe_float(tr.get("end"), safe_float(tr.get("end_sec"), start + 0.1)) + pad_sec)
        clip_path = base / "error-slices" / error_id / "clip.mp4"
        if video_path.exists():
            cut_clip(video_path, start, end, clip_path)
        items.append(
            {
                "error_id": error_id,
                "time_range_sec": {"start": start, "end": end},
                "clip_path": rel_path(clip_path),
                "source_video_path": rel_path(video_path),
                "scope": err.get("scope", {}),
            }
        )
    index = {
        "pipeline_stage": "self-evolution:error-slices",
        "generated_at": utc_now_iso(),
        "case_id": case_id,
        "session_id": session_id,
        "slice_count": len(items),
        "slices": items,
    }
    index_path = base / "error-slices" / "index.json"
    write_json(index_path, index)
    return index_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build video slices for reflected error events.")
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--pad-sec", type=float, default=1.5)
    args = parser.parse_args()
    out = build_slices(args.case_id, args.session_id, args.pad_sec)
    print(f"[OK] error slice index: {out}")


if __name__ == "__main__":
    main()
