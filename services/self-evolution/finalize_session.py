import argparse
from pathlib import Path

from common import ensure_session_dirs, read_json, read_jsonl, session_root, utc_now_iso, write_json


def finalize_session(case_id: str, session_id: str) -> Path:
    base = ensure_session_dirs(case_id, session_id)
    raw_dir = base / "raw"
    logs_dir = base / "logs"
    ipcam_video_path = raw_dir / "teaching_session_ipcam.mp4"
    review_video_path = raw_dir / "teaching_session_review.mp4"
    webm_path = raw_dir / "teaching_session.webm"
    mp4_path = raw_dir / "teaching_session.mp4"
    source_video = ipcam_video_path if ipcam_video_path.exists() else (webm_path if webm_path.exists() else mp4_path)
    if not source_video.exists():
        raise FileNotFoundError(f"missing raw session video: {ipcam_video_path}")
    audio_candidates = [raw_dir / "teaching_audio.webm", raw_dir / "teaching_audio.wav", raw_dir / "teaching_audio.mp3"]
    source_audio = next((p for p in audio_candidates if p.exists()), None)

    # M1 will add ffmpeg normalization. For the 0-4 scaffold, keep the original
    # media and expose an mp4 path only when it already exists.
    normalized_video = review_video_path if review_video_path.exists() else (mp4_path if mp4_path.exists() else source_video)
    meta_path = logs_dir / "session_meta.json"
    meta = read_json(meta_path) if meta_path.exists() else {}
    manifest = {
        "case_id": case_id,
        "session_id": session_id,
        "generated_at": utc_now_iso(),
        "raw_video_path": str(source_video),
        "raw_audio_path": str(source_audio) if source_audio else "",
        "review_media_path": str(review_video_path) if review_video_path.exists() else "",
        "normalized_video_path": str(normalized_video),
        "session_meta_path": str(meta_path),
        "events_path": str(logs_dir / "events.jsonl"),
        "step_trace_path": str(logs_dir / "step_trace.jsonl"),
        "system_prompts_path": str(logs_dir / "system_prompts.jsonl"),
        "teacher_interventions_path": str(logs_dir / "teacher_interventions.jsonl"),
        "events_count": len(read_jsonl(logs_dir / "events.jsonl")),
        "step_trace_count": len(read_jsonl(logs_dir / "step_trace.jsonl")),
        "capture": {
            "target_width": meta.get("target_width"),
            "target_height": meta.get("target_height"),
            "target_fps": meta.get("target_fps"),
            "actual_width": meta.get("actual_width"),
            "actual_height": meta.get("actual_height"),
            "actual_fps": meta.get("actual_fps"),
            "capture_degraded": meta.get("capture_degraded", False),
            "degrade_reason": meta.get("degrade_reason", ""),
        },
    }
    out_path = base / "session_manifest.json"
    write_json(out_path, manifest)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Finalize one self-evolution session.")
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--session-id", required=True)
    args = parser.parse_args()
    out_path = finalize_session(args.case_id, args.session_id)
    print(f"[OK] session manifest: {out_path}")


if __name__ == "__main__":
    main()
