import argparse

from common import ROOT, read_json, read_jsonl, safe_float, session_root, utc_now_iso, write_json


def _load_turns(base):
    transcript_path = base / "asr" / "transcript_v1.json"
    if not transcript_path.exists():
        return []
    data = read_json(transcript_path)
    turns = data.get("segments", [])
    return turns if isinstance(turns, list) else []


def _event_elapsed(item):
    payload = item.get("payload") if isinstance(item.get("payload"), dict) else {}
    return safe_float(item.get("elapsed_sec", payload.get("start_sec", 0.0)), 0.0)


def align_session(case_id: str, session_id: str):
    base = session_root(case_id, session_id)
    logs_dir = base / "logs"
    turns = _load_turns(base)
    events = read_jsonl(logs_dir / "events.jsonl")
    prompts = read_jsonl(logs_dir / "system_prompts.jsonl")
    teacher_interventions = read_jsonl(logs_dir / "teacher_interventions.jsonl")
    strategy = read_json(ROOT / "data" / case_id / "v2" / "strategy" / "teaching_strategy_v2.json")

    step_starts = []
    for item in prompts:
        if not isinstance(item, dict):
            continue
        step_id = str(item.get("step_id", "")).strip()
        if not step_id:
            continue
        step_starts.append(
            {
                "step_id": step_id,
                "section_id": str(item.get("section_id", "")),
                "start_sec": safe_float(item.get("start_sec", item.get("elapsed_sec", 0.0))),
                "system_prompt": item.get("prompt", {}),
                "focus_points": item.get("focus_points", []),
                "common_mistakes": item.get("common_mistakes", []),
            }
        )
    step_starts.sort(key=lambda x: x["start_sec"])
    session_meta = read_json(logs_dir / "session_meta.json") if (logs_dir / "session_meta.json").exists() else {}
    duration = safe_float(session_meta.get("recorded_frame_count"), 0.0) / max(0.001, safe_float(session_meta.get("target_fps"), 10.0))

    windows = []
    for idx, step in enumerate(step_starts):
        end_sec = step_starts[idx + 1]["start_sec"] if idx + 1 < len(step_starts) else duration
        step_turns = [
            t
            for t in turns
            if safe_float(t.get("start_sec")) < end_sec and safe_float(t.get("end_sec", t.get("start_sec"))) >= step["start_sec"]
        ]
        windows.append({**step, "end_sec": round(max(end_sec, step["start_sec"]), 3), "asr_turns": step_turns})

    timeline = []
    for t in turns:
        sec = safe_float(t.get("start_sec"))
        matched_step = next((w for w in windows if w["start_sec"] <= sec <= w["end_sec"]), None)
        timeline.append({"type": "asr_turn", "time_sec": sec, "step_id": matched_step.get("step_id") if matched_step else None, "data": t})
    for e in events:
        timeline.append({"type": "runtime_event", "time_sec": _event_elapsed(e), "step_id": None, "data": e})
    for p in prompts:
        timeline.append({"type": "system_prompt", "time_sec": safe_float(p.get("start_sec", p.get("elapsed_sec", 0.0))), "step_id": p.get("step_id"), "data": p})
    for ti in teacher_interventions:
        timeline.append({"type": "teacher_intervention_candidate", "time_sec": _event_elapsed(ti), "step_id": None, "data": ti})
    timeline.sort(key=lambda x: safe_float(x.get("time_sec")))

    out_timeline = {
        "pipeline_stage": "self-evolution:alignment:timeline",
        "generated_at": utc_now_iso(),
        "case_id": case_id,
        "session_id": session_id,
        "timeline": timeline,
    }
    out_windows = {
        "pipeline_stage": "self-evolution:alignment:step-windows",
        "generated_at": utc_now_iso(),
        "case_id": case_id,
        "session_id": session_id,
        "strategy_ref": f"data/{case_id}/v2/strategy/teaching_strategy_v2.json",
        "step_windows": windows,
    }
    timeline_path = base / "alignment" / "timeline_v1.json"
    windows_path = base / "alignment" / "step_windows_v1.json"
    write_json(timeline_path, out_timeline)
    write_json(windows_path, out_windows)
    return timeline_path, windows_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Align ASR, step trace, prompts, and video time.")
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--session-id", required=True)
    args = parser.parse_args()
    timeline_path, windows_path = align_session(args.case_id, args.session_id)
    print(f"[OK] timeline: {timeline_path}")
    print(f"[OK] step windows: {windows_path}")


if __name__ == "__main__":
    main()
