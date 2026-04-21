import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_case_id(raw: str, fallback: str) -> str:
    import re

    token = raw.strip().lower().replace("-", "_")
    token = re.sub(r"[^a-z0-9_\-]+", "_", token)
    token = re.sub(r"_{2,}", "_", token).strip("_")
    return token or fallback


def infer_case_id_from_segmentation(src: Dict[str, Any]) -> str:
    demos = src.get("demos", [])
    if isinstance(demos, list) and demos:
        video_id = str(demos[0].get("video_id", "")).strip()
        if video_id:
            return normalize_case_id(video_id.split("__")[0], "default_case")
    case_id = src.get("case_id")
    if isinstance(case_id, str) and case_id.strip():
        return normalize_case_id(case_id, "default_case")
    return "default_case"


def normalize_time_range(time_range: Dict[str, Any]) -> Dict[str, float]:
    start = float(time_range.get("start_sec", 0.0) or 0.0)
    end = float(time_range.get("end_sec", start) or start)
    if end < start:
        end = start
    return {"start_sec": start, "end_sec": end}


def dedupe_str_list(values: List[Any]) -> List[str]:
    out: List[str] = []
    seen = set()
    for value in values:
        item = str(value).strip()
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def build_scene_entities_catalog(case_id: str, entities: List[str]) -> Dict[str, Any]:
    deduped = dedupe_str_list(entities)
    return {
        "version": "v1",
        "frozen_at": utc_now_iso(),
        "case_id": case_id,
        "source_stage": "strategy-builder",
        "entity_count": len(deduped),
        "entity_names": deduped,
    }


def build_step(section_id: str, order: int, unit: Dict[str, Any]) -> Dict[str, Any]:
    unit_id = str(unit.get("unit_id", f"unit_{section_id}_{order:03d}"))
    step_id = f"{section_id}_step_{order:02d}"
    sf = unit.get("step_fields") if isinstance(unit.get("step_fields"), dict) else {}

    prompt_value = sf.get("prompt")
    if isinstance(prompt_value, dict):
        prompt_obj = {
            "en": str(prompt_value.get("en", "")).strip(),
            "zh": str(prompt_value.get("zh", "")).strip(),
        }
    else:
        raw_prompt = str(prompt_value or unit.get("description") or "Please complete the current step.").strip()
        prompt_obj = {
            "en": raw_prompt,
            "zh": raw_prompt,
        }

    return {
        "step_id": step_id,
        "step_order": order,
        "unit_refs": [unit_id],
        "time_range": normalize_time_range(unit.get("time_range") if isinstance(unit.get("time_range"), dict) else {}),
        "prompt": prompt_obj,
        "focus_points": sf.get("focus_points") if isinstance(sf.get("focus_points"), list) else [],
        "common_mistakes": sf.get("common_mistakes") if isinstance(sf.get("common_mistakes"), list) else [],
    }


def build_error(section_id: str, order: int, unit: Dict[str, Any]) -> Dict[str, Any]:
    unit_id = str(unit.get("unit_id", f"unit_{section_id}_{order:03d}"))
    error_id = f"{section_id}_error_{order:02d}"
    ef = unit.get("error_fields") if isinstance(unit.get("error_fields"), dict) else {}

    return {
        "error_id": error_id,
        "unit_refs": [unit_id],
        "time_range": normalize_time_range(unit.get("time_range") if isinstance(unit.get("time_range"), dict) else {}),
        "trigger_signature": str(ef.get("trigger_signature") or f"{section_id}:error_{order:02d}"),
        "correction_prompt": str(ef.get("correction_prompt") or "检测到偏差，请回到当前步骤重新执行。"),
        "recovery_actions": ef.get("recovery_actions") if isinstance(ef.get("recovery_actions"), list) else [],
    }


def transform_demo(demo: Dict[str, Any]) -> Dict[str, Any]:
    sections_out: List[Dict[str, Any]] = []

    for section in demo.get("sections", []):
        section_id = str(section.get("section_id", "section_unknown"))
        section_name = str(section.get("section_name", section_id))
        section_goal = str(section.get("section_summary") or section.get("section_goal") or "完成本章节任务")

        if isinstance(section.get("expected_section_state"), str):
            expected_state: Any = section.get("expected_section_state")
        elif isinstance(section.get("expected_section_state"), dict):
            expected_state = section.get("expected_section_state")
        else:
            expected_state = "unknown"

        time_range = normalize_time_range(section.get("time_range") if isinstance(section.get("time_range"), dict) else {})

        raw_units = section.get("atomic_units", []) if isinstance(section.get("atomic_units"), list) else []
        kept_units = [u for u in raw_units if u.get("class") in ("step", "error")]

        steps: List[Dict[str, Any]] = []
        errors: List[Dict[str, Any]] = []
        step_order = 1
        error_order = 1

        for unit in kept_units:
            cls = str(unit.get("class"))
            if cls == "step":
                steps.append(build_step(section_id=section_id, order=step_order, unit=unit))
                step_order += 1
            elif cls == "error":
                errors.append(build_error(section_id=section_id, order=error_order, unit=unit))
                error_order += 1

        sections_out.append(
            {
                "section_id": section_id,
                "section_name": section_name,
                "section_goal": section_goal,
                "expected_section_state": expected_state,
                "time_range": time_range,
                "steps": steps,
                "errors": errors,
            }
        )

    overview = demo.get("video_overview") if isinstance(demo.get("video_overview"), dict) else {}
    camera_view = str(overview.get("camera_view") or "unknown")
    scene_entities = dedupe_str_list(overview.get("scene_entities") if isinstance(overview.get("scene_entities"), list) else [])

    return {
        "task": {
            "task_id": demo.get("task_id", "task-demo"),
            "camera_view": camera_view,
            "scene_entities": scene_entities,
            "source_audio_quality": demo.get("source_audio_quality", "unknown"),
        },
        "sections": sections_out,
        "meta": {
            "source_stage": "strategy-builder",
            "source_video_id": demo.get("video_id", "unknown_video"),
            "strategy_version": "v2.0.0",
            "generated_at": utc_now_iso(),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline A - strategy builder")
    parser.add_argument("--input", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--case-id", help="case folder under data/, e.g. test_cake")
    args = parser.parse_args()

    if args.input:
        input_path = Path(args.input)
    elif args.case_id:
        input_path = ROOT / "data" / normalize_case_id(args.case_id, "default_case") / "v2" / "segmentation" / "sections_units.json"
    else:
        input_path = ROOT / "data" / "processed" / "v2" / "segmentation" / "sections_units.json"

    src = read_json(input_path)
    demos = src.get("demos", []) if isinstance(src.get("demos"), list) else []
    if not demos:
        raise ValueError("no demos found in segmentation output")

    case_id = args.case_id or infer_case_id_from_segmentation(src)
    output_dir = Path(args.output_dir) if args.output_dir else (ROOT / "data" / case_id / "v2" / "strategy")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build global union for scene entities across demos, so top-level strategy can reuse a stable catalog.
    global_scene_entities: List[str] = []
    for demo in demos:
        overview = demo.get("video_overview") if isinstance(demo.get("video_overview"), dict) else {}
        if isinstance(overview.get("scene_entities"), list):
            global_scene_entities.extend(overview["scene_entities"])
    global_scene_entities = dedupe_str_list(global_scene_entities)

    built = []
    for demo in demos:
        strategy = transform_demo(demo)
        if global_scene_entities:
            strategy["task"]["scene_entities"] = global_scene_entities
        video_id = str(demo.get("video_id", "unknown_video"))
        out_file = output_dir / f"{video_id}_teaching_strategy_v2.json"
        write_json(out_file, strategy)
        built.append(strategy)

    write_json(output_dir / "teaching_strategy_v2.json", built[0] if len(built) == 1 else {"tasks": built, "generated_at": utc_now_iso()})
    write_json(output_dir / "scene_entities_v1.json", build_scene_entities_catalog(case_id=case_id, entities=global_scene_entities))
    print(f"[OK] built strategies: {len(built)}")
    print(f"[OK] output dir: {output_dir}")
    print(f"[OK] scene entities catalog: {output_dir / 'scene_entities_v1.json'}")


if __name__ == "__main__":
    main()
