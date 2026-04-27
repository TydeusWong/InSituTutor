import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import httpx
from openai import OpenAI

from common import dedupe_str, extract_json_from_text, read_json, session_root, utc_now_iso, write_json


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import TA_BASE_URL, TA_MAX_RETRIES, TA_MODEL, TA_REQUEST_TIMEOUT_SEC, get_api_key  # noqa: E402


def infer_targets(error: dict, allowed_entities: list[str]) -> list[str]:
    blob = json.dumps(error, ensure_ascii=False)
    targets = [x for x in allowed_entities if x and x in blob]
    if not targets:
        lowered = blob.lower()
        for entity in allowed_entities:
            parts = [p.strip().lower() for p in str(entity).split(".") if p.strip()]
            if any(part and part in lowered for part in parts):
                targets.append(entity)
    return dedupe_str(targets)


def infer_condition(error: dict, targets: list[str]) -> str:
    blob = json.dumps(error, ensure_ascii=False).lower()
    if len(targets) >= 2:
        a, b = targets[0], targets[1]
        if any(word in blob for word in ["inside", "opening", "塞进", "开口", "罐口"]):
            return f"iou('{a}', '{b}') >= 0.05 and rel_distance('{a}', '{b}') <= 0.20"
        if any(word in blob for word in ["off-center", "偏", "misalign", "center"]):
            return f"rel_distance('{a}', '{b}') > 0.15"
        if any(word in blob for word in ["cover", "压住", "遮挡"]):
            return f"iou('{a}', '{b}') >= 0.10"
        return f"rel_distance('{a}', '{b}') <= 0.15"
    if len(targets) == 1:
        return f"visible('{targets[0]}')"
    return "False"


def fallback_plan(error: dict, allowed_entities: list[str]) -> dict:
    error_id = str(error.get("error_id", "err_unknown"))
    scope = error.get("scope") if isinstance(error.get("scope"), dict) else {}
    message = error.get("correction_message") if isinstance(error.get("correction_message"), dict) else {}
    targets = infer_targets(error, allowed_entities)
    use_hand = any(word in json.dumps(error, ensure_ascii=False).lower() for word in ["hand", "finger", "pushing", "手", "按", "拿", "放"])
    models_required = ["yolo", *([] if not use_hand else ["mediapipe-hand-landmarker"])]
    model_selection = [{"model_id": "yolo", "detect_targets": targets}]
    if use_hand:
        model_selection.append({"model_id": "mediapipe-hand-landmarker", "detect_targets": ["hand"]})
    return {
        "slice_id": error_id,
        "slice_type": "error",
        "error_id": error_id,
        "models_required": models_required,
        "model_selection": model_selection,
        "execution_plan": {"order": models_required, "parallel_groups": [models_required]},
        "object_targets": targets,
        "scope": {
            "applies_to_sections": dedupe_str(scope.get("applies_to_sections", [])),
            "applies_to_steps": dedupe_str(scope.get("applies_to_steps", [])),
        },
        "judgement_conditions": [
            {
                "condition_id": f"{error_id}_is_happening",
                "when": "error",
                "code": infer_condition(error, targets),
            }
        ],
        "correction_message": {
            "zh": str(message.get("zh") or error.get("student_wrong_action") or "检测到可能的错误，请回到当前步骤检查。"),
            "en": str(message.get("en") or "A possible error was detected. Please check the current step."),
        },
        "source_error_event": error,
    }


def normalize_plan(plan: dict, error: dict, allowed_entities: list[str]) -> dict:
    fallback = fallback_plan(error, allowed_entities)
    out = dict(plan) if isinstance(plan, dict) else {}
    out.pop("detection_logic", None)
    out.pop("small_models", None)
    out.pop("models", None)
    out.setdefault("slice_id", str(error.get("error_id", fallback["slice_id"])))
    out.setdefault("slice_type", "error")
    out.setdefault("error_id", str(error.get("error_id", out.get("slice_id", ""))))
    out.setdefault("scope", fallback["scope"])
    out.setdefault("correction_message", fallback["correction_message"])

    models_required = out.get("models_required")
    if not isinstance(models_required, list) or not models_required:
        models_required = fallback["models_required"]
    out["models_required"] = [m for m in dedupe_str(models_required) if str(m).lower() != "dino"]

    model_selection = out.get("model_selection")
    if not isinstance(model_selection, list) or not model_selection:
        model_selection = fallback["model_selection"]
    out["model_selection"] = model_selection

    execution_plan = out.get("execution_plan")
    if not isinstance(execution_plan, dict):
        execution_plan = {"order": out["models_required"], "parallel_groups": [out["models_required"]]}
    out["execution_plan"] = execution_plan

    jcs = out.get("judgement_conditions")
    valid_jcs = []
    if isinstance(jcs, list):
        for idx, cond in enumerate(jcs, start=1):
            if not isinstance(cond, dict):
                continue
            code = str(cond.get("code", "")).strip()
            if not code:
                continue
            valid_jcs.append(
                {
                    "condition_id": str(cond.get("condition_id") or f"{out['error_id']}_condition_{idx:02d}"),
                    "when": str(cond.get("when") or "error"),
                    "code": code,
                }
            )
    if not valid_jcs:
        valid_jcs = fallback["judgement_conditions"]
    out["judgement_conditions"] = valid_jcs
    out.setdefault("object_targets", fallback["object_targets"])
    return out


def build_plans(case_id: str, session_id: str, model: str, allow_fallback: bool) -> tuple[Path, Path]:
    base = session_root(case_id, session_id)
    errors = read_json(base / "reflection" / "error_events_v1.json") if (base / "reflection" / "error_events_v1.json").exists() else []
    if not isinstance(errors, list):
        errors = []
    slice_index = read_json(base / "error-slices" / "index.json") if (base / "error-slices" / "index.json").exists() else {"slices": []}
    strategy = read_json(ROOT / "data" / case_id / "v2" / "strategy" / "teaching_strategy_v2.json")
    entities = read_json(ROOT / "data" / case_id / "v2" / "strategy" / "scene_entities_v1.json")
    class_map_path = ROOT / "data" / case_id / "v3" / "yolo-dataset" / "class_map.json"
    class_map = read_json(class_map_path) if class_map_path.exists() else {}
    allowed_entities = dedupe_str(entities.get("entity_names", []))
    system_prompt = (ROOT / "models" / "prompts" / "error_detector_plan_system_prompt_v1.md").read_text(encoding="utf-8")
    user_template = (ROOT / "models" / "prompts" / "error_detector_plan_user_prompt_template_v1.md").read_text(encoding="utf-8")
    api_key = get_api_key()
    plans = []
    for err in errors:
        if not isinstance(err, dict):
            continue
        plan: dict[str, Any] | None = None
        if api_key:
            payload = {
                "instructions": user_template,
                "error_event": err,
                "error_slices": slice_index.get("slices", []),
                "teaching_strategy": strategy,
                "allowed_scene_entities": allowed_entities,
                "yolo_class_map": class_map,
            }
            try:
                client = OpenAI(api_key=api_key, base_url=TA_BASE_URL, timeout=TA_REQUEST_TIMEOUT_SEC)
                for attempt in range(1, TA_MAX_RETRIES + 1):
                    try:
                        resp = client.chat.completions.create(
                            model=model,
                            temperature=0,
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
                            ],
                        )
                        plan = extract_json_from_text(resp.choices[0].message.content or "")
                        break
                    except Exception:
                        if attempt == TA_MAX_RETRIES:
                            raise
                        time.sleep(attempt * 1.2)
            except Exception:
                if not allow_fallback:
                    raise
        if not isinstance(plan, dict):
            plan = fallback_plan(err, allowed_entities)
        plan = normalize_plan(plan, err, allowed_entities)
        plan_path = base / "detector-plans" / "errors" / f"{plan.get('error_id')}.json"
        write_json(plan_path, plan)
        plans.append({"error_id": plan.get("error_id"), "plan_path": str(plan_path), "scope": plan.get("scope", {})})

    library = {
        "version": "v1",
        "generated_at": utc_now_iso(),
        "case_id": case_id,
        "session_id": session_id,
        "error_count": len(plans),
        "errors": plans,
    }
    library_path = ROOT / "data" / case_id / "v5" / "errors" / "error_library_v1.json"
    write_json(library_path, library)
    return base / "detector-plans" / "errors", library_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build detector plans for reflected error events.")
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--model", default=TA_MODEL)
    parser.add_argument("--allow-fallback", action="store_true")
    args = parser.parse_args()
    plan_dir, library_path = build_plans(args.case_id, args.session_id, args.model, args.allow_fallback)
    print(f"[OK] error detector plans: {plan_dir}")
    print(f"[OK] error library: {library_path}")


if __name__ == "__main__":
    main()
