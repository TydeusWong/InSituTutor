import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[2]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def resolve_video_path(ingest_manifest: Dict[str, Any]) -> Path:
    demos = ingest_manifest.get("demos", [])
    if not demos:
        raise ValueError("ingest manifest has no demos")
    video_path = demos[0].get("ingest_video_path")
    if not video_path:
        raise ValueError("ingest_video_path missing")
    p = Path(str(video_path))
    if not p.is_absolute():
        p = (ROOT / p).resolve()
    return p


def normalize_range(time_range: Dict[str, Any]) -> Tuple[float, float]:
    start = float(time_range.get("start_sec", 0.0) or 0.0)
    end = float(time_range.get("end_sec", start) or start)
    if end < start:
        end = start
    return start, end


def run_ffmpeg_cut(video_path: Path, start: float, end: float, out_file: Path) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start:.3f}",
        "-to",
        f"{end:.3f}",
        "-i",
        str(video_path),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-c:a",
        "aac",
        str(out_file),
    ]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {out_file}: {proc.stderr.strip()[:300]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build step/error clip slices from strategy time ranges")
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--sections-units", default=None)
    parser.add_argument("--strategy", default=None)
    parser.add_argument("--ingest-manifest", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    case_dir = ROOT / "data" / args.case_id / "v2"
    sections_units_path = Path(args.sections_units) if args.sections_units else (case_dir / "segmentation" / "sections_units.json")
    strategy_path = Path(args.strategy) if args.strategy else (case_dir / "strategy" / "teaching_strategy_v2.json")
    ingest_manifest_path = Path(args.ingest_manifest) if args.ingest_manifest else (ROOT / "data" / args.case_id / "ingest_manifest.json")
    output_dir = Path(args.output_dir) if args.output_dir else (case_dir / "slices")

    sections_units = read_json(sections_units_path)
    strategy = read_json(strategy_path)
    ingest_manifest = read_json(ingest_manifest_path)

    video_path = resolve_video_path(ingest_manifest)
    demo = (sections_units.get("demos") or [{}])[0]
    video_overview = demo.get("video_overview", {})
    global_summary = str(video_overview.get("summary", "")).strip()

    slices: List[Dict[str, Any]] = []
    for section in strategy.get("sections", []):
        section_id = str(section.get("section_id", "section_unknown"))
        section_meta = {
            "section_id": section_id,
            "section_name": section.get("section_name"),
            "section_goal": section.get("section_goal"),
            "expected_section_state": section.get("expected_section_state"),
            "time_range": section.get("time_range"),
        }

        for step in section.get("steps", []):
            step_id = str(step.get("step_id"))
            start, end = normalize_range(step.get("time_range", {}))
            clip_rel = Path(args.case_id) / "v2" / "slices" / section_id / step_id / "clip.mp4"
            clip_abs = ROOT / "data" / clip_rel
            run_ffmpeg_cut(video_path, start, end, clip_abs)
            slices.append(
                {
                    "slice_id": step_id,
                    "slice_type": "step",
                    "section_id": section_id,
                    "time_range": {"start_sec": start, "end_sec": end},
                    "clip_path": str(clip_rel).replace("\\", "/"),
                    "section": section_meta,
                    "target": {
                        "step_id": step_id,
                        "step_order": step.get("step_order"),
                        "prompt": step.get("prompt"),
                        "focus_points": step.get("focus_points", []),
                        "common_mistakes": step.get("common_mistakes", []),
                    },
                    "video_overview_summary": global_summary,
                }
            )

        for error in section.get("errors", []):
            error_id = str(error.get("error_id"))
            start, end = normalize_range(error.get("time_range", {}))
            clip_rel = Path(args.case_id) / "v2" / "slices" / section_id / error_id / "clip.mp4"
            clip_abs = ROOT / "data" / clip_rel
            run_ffmpeg_cut(video_path, start, end, clip_abs)
            slices.append(
                {
                    "slice_id": error_id,
                    "slice_type": "error",
                    "section_id": section_id,
                    "time_range": {"start_sec": start, "end_sec": end},
                    "clip_path": str(clip_rel).replace("\\", "/"),
                    "section": section_meta,
                    "target": {
                        "error_id": error_id,
                        "trigger_signature": error.get("trigger_signature"),
                        "correction_prompt": error.get("correction_prompt"),
                        "recovery_actions": error.get("recovery_actions", []),
                    },
                    "video_overview_summary": global_summary,
                }
            )

    slices.sort(key=lambda x: float(x["time_range"]["start_sec"]))
    index = {
        "pipeline_stage": "criteria-trainer:slicing",
        "generated_at": utc_now_iso(),
        "case_id": args.case_id,
        "source_video_path": str(video_path),
        "video_overview_summary": global_summary,
        "slice_count": len(slices),
        "slices": slices,
    }
    write_json(output_dir / "index.json", index)
    print(f"[OK] output index: {output_dir / 'index.json'}")
    print(f"[OK] slice count: {len(slices)}")


if __name__ == "__main__":
    main()
