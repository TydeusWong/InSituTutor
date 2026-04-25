import argparse
import ast
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import httpx
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[2]

import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import TA_BASE_URL, TA_MODEL, TA_REQUEST_TIMEOUT_SEC, get_api_key  # noqa: E402

PROMPT_DIR = ROOT / "models" / "prompts"
DETECTOR_SYSTEM_PROMPT_PATH = PROMPT_DIR / "detector_plan_system_prompt_v2.md"
DETECTOR_USER_PROMPT_TEMPLATE_PATH = PROMPT_DIR / "detector_plan_user_prompt_template_v2.md"
CONDITION_PRIMITIVES_PATH = ROOT / "services" / "criteria-trainer" / "configs" / "detection_condition_primitives_v1.json"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def extract_json_from_text(text: str) -> Dict[str, Any]:
    fenced = re.search(r"```(?:json)?\s*(\{[\s\S]*\})\s*```", text)
    if fenced:
        return json.loads(fenced.group(1))
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return json.loads(text[start : end + 1])
    raise ValueError("no json object in model output")


def build_user_prompt(template: str, payload: Dict[str, Any]) -> str:
    out = template
    for k, v in payload.items():
        out = out.replace("{{" + k + "}}", str(v))
    return out


def normalize_name(value: Any) -> str:
    return str(value).strip().lower()


def dedupe_str_list(values: Any) -> List[str]:
    if not isinstance(values, list):
        return []
    out: List[str] = []
    seen: Set[str] = set()
    for v in values:
        s = str(v).strip()
        if not s:
            continue
        key = normalize_name(s)
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


def get_allowed_object_targets(scene_entity_catalog: Dict[str, Any]) -> List[str]:
    # Source of truth: strategy scene entity catalog file.
    return dedupe_str_list(scene_entity_catalog.get("entity_names"))


def filter_small_models_for_detector_plan(small_models: Dict[str, Any]) -> Dict[str, Any]:
    filtered = dict(small_models)
    models = small_models.get("models", [])
    if isinstance(models, list):
        filtered["models"] = [
            m for m in models
            if isinstance(m, dict) and str(m.get("id", "")).strip() != "grounding-dino"
        ]
    return filtered


def load_yolo_entity_names(case_id: str, fallback_entities: List[str]) -> List[str]:
    class_map_path = ROOT / "data" / case_id / "v3" / "yolo-dataset" / "class_map.json"
    if class_map_path.exists():
        class_map = read_json(class_map_path)
        classes = class_map.get("classes", [])
        names: List[str] = []
        if isinstance(classes, list):
            for item in classes:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("name", "")).strip()
                if name and name not in names:
                    names.append(name)
        if names:
            return names
    return list(fallback_entities)


def build_detector_user_payload(
    target_slice: Dict[str, Any],
    small_models: Dict[str, Any],
    condition_primitives: Dict[str, Any],
    allowed_yolo_detect_targets: List[str],
    scene_entity_catalog: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "task": "build_detector_plan_for_single_slice",
        "policy": {
            "minimal_models_only": True,
            "minimal_entities_only": True,
            "description": "Use the minimum sufficient models to judge completion. Avoid unnecessary entities. Object detection must use the already-trained YOLO model, not Grounding DINO.",
            "object_detector": "yolo",
            "grounding_dino_is_not_available": True,
            "yolo_is_trained": True,
            "yolo_targets_must_come_from_scene_entities": True,
        },
        "scene_entity_catalog_ref": "data/<case_id>/v2/strategy/scene_entities_v1.json",
        "scene_entity_catalog": scene_entity_catalog,
        "condition_primitives_ref": "services/criteria-trainer/configs/detection_condition_primitives_v1.json",
        "condition_primitives": condition_primitives,
        "condition_primitive_runtime_note": "Primitive groups may contain legacy names such as grounding_dino, but object bbox primitives are executed against YOLO detections in the YOLO replay/realtime runtime.",
        "allowed_yolo_detect_targets": allowed_yolo_detect_targets,
        "yolo_capability": {
            "status": "trained_before_detector_plan_generation",
            "description": "YOLO has already been trained from entity-presence step clips and can localize every entity listed in allowed_yolo_detect_targets.",
            "detectable_entities": allowed_yolo_detect_targets,
            "class_map_ref": "data/<case_id>/v3/yolo-dataset/class_map.json",
            "registry_ref": "services/criteria-trainer/configs/yolo_registry_v1.json",
        },
        "slice": target_slice,
        "small_model_catalog": small_models,
        "output_schema": {
            "slice_id": "string",
            "slice_type": "step|error",
            "models_required": ["string"],
            "model_selection": [{"model_id": "string", "detect_targets": ["string"]}],
            "execution_plan": {"order": ["string"], "parallel_groups": [["string"]]},
            "judgement_conditions": [
                {
                    "condition_id": "string",
                    "when": "step|error|both",
                    "code": "string",
                }
            ],
            "code_rule": "code must only call predefined primitive function names in condition_primitives",
        },
    }


def sanitize_detector_plan(plan: Dict[str, Any], allowed_object_targets: List[str] | None = None) -> Dict[str, Any]:
    allowed = {
        "slice_id",
        "slice_type",
        "models_required",
        "model_selection",
        "execution_plan",
        "judgement_conditions",
    }
    cleaned = {k: v for k, v in plan.items() if k in allowed}
    allowed_map: Dict[str, str] = {}
    if isinstance(allowed_object_targets, list):
        for t in allowed_object_targets:
            ts = str(t).strip()
            if ts:
                allowed_map[normalize_name(ts)] = ts
    model_selection = cleaned.get("model_selection")
    if isinstance(model_selection, list):
        agg: Dict[str, List[str]] = {}
        for item in model_selection:
            if not isinstance(item, dict):
                continue
            model_id = str(item.get("model_id", "")).strip()
            if not model_id:
                continue
            if model_id == "grounding-dino":
                continue
            targets: List[str] = []
            if isinstance(item.get("detect_targets"), list):
                targets = [str(x).strip() for x in item["detect_targets"] if str(x).strip()]
            elif isinstance(item.get("detect_target"), str):
                raw = item["detect_target"]
                targets = [x.strip() for x in re.split(r",| and ", raw) if x.strip()]
            agg.setdefault(model_id, [])
            for t in targets:
                if model_id == "yolo" and allowed_map:
                    key = normalize_name(t)
                    if key not in allowed_map:
                        continue
                    t = allowed_map[key]
                if t not in agg[model_id]:
                    agg[model_id].append(t)
        cleaned["model_selection"] = [{"model_id": m, "detect_targets": ts} for m, ts in agg.items()]
    if isinstance(cleaned.get("models_required"), list):
        models_required: List[str] = []
        for model_id in cleaned["models_required"]:
            mid = str(model_id).strip()
            if not mid or mid == "grounding-dino":
                continue
            if mid not in models_required:
                models_required.append(mid)
        cleaned["models_required"] = models_required
    execution_plan = cleaned.get("execution_plan")
    if isinstance(execution_plan, dict):
        order = execution_plan.get("order")
        if isinstance(order, list):
            execution_plan["order"] = [
                str(mid).strip()
                for mid in order
                if str(mid).strip() and str(mid).strip() != "grounding-dino"
            ]
        groups = execution_plan.get("parallel_groups")
        if isinstance(groups, list):
            clean_groups: List[List[str]] = []
            for group in groups:
                if not isinstance(group, list):
                    continue
                clean_group = [
                    str(mid).strip()
                    for mid in group
                    if str(mid).strip() and str(mid).strip() != "grounding-dino"
                ]
                if clean_group:
                    clean_groups.append(clean_group)
            execution_plan["parallel_groups"] = clean_groups
    return cleaned


def extract_object_detector_targets(plan: Dict[str, Any]) -> List[str]:
    targets: List[str] = []
    model_selection = plan.get("model_selection", [])
    if not isinstance(model_selection, list):
        return targets
    for item in model_selection:
        if not isinstance(item, dict):
            continue
        if str(item.get("model_id", "")).strip() not in {"grounding-dino", "yolo"}:
            continue
        raw_targets = item.get("detect_targets")
        if isinstance(raw_targets, list):
            for x in raw_targets:
                t = str(x).strip()
                if t and t not in targets:
                    targets.append(t)
        elif isinstance(item.get("detect_target"), str):
            for x in re.split(r",| and ", item["detect_target"]):
                t = x.strip()
                if t and t not in targets:
                    targets.append(t)
    return targets


def extract_primitive_names(primitives: Dict[str, Any]) -> Set[str]:
    out: Set[str] = set()
    gd = primitives.get("grounding_dino", {})
    for group in ("absolute_position", "relative_position"):
        for item in gd.get(group, []):
            name = item.get("name")
            if isinstance(name, str) and name.strip():
                out.add(name.strip())
    for item in primitives.get("hand_dino_relation", []):
        name = item.get("name")
        if isinstance(name, str) and name.strip():
            out.add(name.strip())
    return out


def extract_called_functions(code: str) -> Set[str]:
    return set(re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", code))


def validate_judgement_code(code: str, primitive_names: Set[str]) -> Tuple[bool, str]:
    try:
        ast.parse(code, mode="eval")
    except Exception as exc:
        return False, f"invalid python expression: {exc}"
    called = extract_called_functions(code)
    illegal = sorted([fn for fn in called if fn not in primitive_names])
    if illegal:
        return False, f"undefined primitive calls: {', '.join(illegal)}"
    return True, ""


def validate_plan(plan: Dict[str, Any], primitive_names: Set[str], allowed_object_targets: List[str] | None = None) -> Tuple[bool, str]:
    required = {"slice_id", "slice_type", "models_required", "model_selection", "execution_plan", "judgement_conditions"}
    missing = sorted([k for k in required if k not in plan])
    if missing:
        return False, f"missing keys: {', '.join(missing)}"
    if not isinstance(plan.get("judgement_conditions"), list) or not plan["judgement_conditions"]:
        return False, "judgement_conditions must be non-empty list"
    if not isinstance(plan.get("model_selection"), list) or not plan["model_selection"]:
        return False, "model_selection must be non-empty list"
    if not isinstance(plan.get("models_required"), list) or not plan["models_required"]:
        return False, "models_required must be non-empty list"
    if any(str(model_id).strip() == "grounding-dino" for model_id in plan["models_required"]):
        return False, "grounding-dino is not available in models_required; use yolo for object localization"
    execution_plan = plan.get("execution_plan")
    if not isinstance(execution_plan, dict):
        return False, "execution_plan must be object"
    order = execution_plan.get("order", [])
    if isinstance(order, list) and any(str(model_id).strip() == "grounding-dino" for model_id in order):
        return False, "grounding-dino is not available in execution_plan.order; use yolo for object localization"
    groups = execution_plan.get("parallel_groups", [])
    if isinstance(groups, list):
        for group in groups:
            if isinstance(group, list) and any(str(model_id).strip() == "grounding-dino" for model_id in group):
                return False, "grounding-dino is not available in execution_plan.parallel_groups; use yolo for object localization"
    seen_model_ids: Set[str] = set()
    allowed_map: Dict[str, str] = {}
    if isinstance(allowed_object_targets, list):
        for t in allowed_object_targets:
            ts = str(t).strip()
            if ts:
                allowed_map[normalize_name(ts)] = ts
    for idx, item in enumerate(plan["model_selection"]):
        if not isinstance(item, dict):
            return False, f"model_selection[{idx}] must be object"
        model_id = str(item.get("model_id", "")).strip()
        if not model_id:
            return False, f"model_selection[{idx}].model_id is empty"
        if model_id == "grounding-dino":
            return False, "grounding-dino is not available for detector plans; use yolo for object localization"
        if model_id in seen_model_ids:
            return False, f"duplicated model_id in model_selection: {model_id}"
        seen_model_ids.add(model_id)
        targets = item.get("detect_targets")
        if not isinstance(targets, list) or not targets:
            return False, f"model_selection[{idx}].detect_targets must be non-empty array"
        dedup_targets = [str(t).strip() for t in targets if str(t).strip()]
        if len(dedup_targets) != len(set(dedup_targets)):
            return False, f"model_selection[{idx}].detect_targets contains duplicates"
        if model_id == "yolo" and allowed_map:
            illegal = [t for t in dedup_targets if normalize_name(t) not in allowed_map]
            if illegal:
                return False, f"model_selection[{idx}].detect_targets not in allowed YOLO entities: {', '.join(illegal)}"
    for idx, item in enumerate(plan["judgement_conditions"]):
        code = str(item.get("code", "")).strip() if isinstance(item, dict) else ""
        if not code:
            return False, f"judgement_conditions[{idx}].code is empty"
        ok, err = validate_judgement_code(code, primitive_names)
        if not ok:
            return False, f"judgement_conditions[{idx}] {err}"
    return True, ""


def upload_video_to_dashscope_oss(video_path: Path, api_key: str, model: str) -> str:
    direct_http = httpx.Client(transport=httpx.HTTPTransport(), timeout=300)
    policy_resp = direct_http.get(
        "https://dashscope.aliyuncs.com/api/v1/uploads",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        params={"action": "getPolicy", "model": model},
    )
    if policy_resp.status_code != 200:
        raise RuntimeError(f"get upload policy failed: {policy_resp.text}")
    data = policy_resp.json()["data"]
    upload_key = f"{data['upload_dir']}/{video_path.name}"
    with video_path.open("rb") as f:
        upload_resp = direct_http.post(
            data["upload_host"],
            files={
                "OSSAccessKeyId": (None, data["oss_access_key_id"]),
                "Signature": (None, data["signature"]),
                "policy": (None, data["policy"]),
                "x-oss-object-acl": (None, data["x_oss_object_acl"]),
                "x-oss-forbid-overwrite": (None, data["x_oss_forbid_overwrite"]),
                "key": (None, upload_key),
                "success_action_status": (None, "200"),
                "file": (video_path.name, f, "video/mp4"),
            },
        )
    if upload_resp.status_code != 200:
        raise RuntimeError(f"upload failed: {upload_resp.text}")
    return f"oss://{upload_key}"


def call_omni_for_detector_plan(
    api_key: str,
    model: str,
    base_url: str,
    request_timeout_sec: int,
    max_retries: int,
    system_prompt: str,
    user_prompt: str,
    video_uri: str,
) -> Dict[str, Any]:
    extra_headers: Dict[str, str] = {}
    if video_uri.startswith("oss://"):
        extra_headers["X-DashScope-OssResourceResolve"] = "enable"

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=request_timeout_sec,
        http_client=httpx.Client(transport=httpx.HTTPTransport(), timeout=request_timeout_sec),
    )

    for i in range(1, max_retries + 1):
        try:
            stream = client.chat.completions.create(
                model=model,
                temperature=0,
                modalities=["text"],
                stream=True,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_prompt},
                            {"type": "video_url", "video_url": {"url": video_uri}},
                        ],
                    },
                ],
                extra_headers=extra_headers,
            )
            parts: List[str] = []
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    parts.append(chunk.choices[0].delta.content)
            parsed = extract_json_from_text("".join(parts))
            return parsed
        except Exception:
            if i == max_retries:
                raise
            time.sleep(i * 1.2)
    raise RuntimeError("unreachable")


def load_strategy_unit_context(case_id: str, strategy_path: Path) -> Dict[str, Dict[str, Any]]:
    if not strategy_path.exists():
        return {}
    strategy = read_json(strategy_path)
    out: Dict[str, Dict[str, Any]] = {}
    sections = strategy.get("sections", [])
    if not isinstance(sections, list):
        return out
    for section in sections:
        if not isinstance(section, dict):
            continue
        section_id = str(section.get("section_id", "section_unknown"))
        section_meta = {
            "section_id": section_id,
            "section_name": section.get("section_name"),
            "section_goal": section.get("section_goal"),
            "expected_section_state": section.get("expected_section_state"),
            "time_range": section.get("time_range"),
        }
        for step in section.get("steps", []):
            if not isinstance(step, dict):
                continue
            step_id = str(step.get("step_id", "")).strip()
            unit_refs = step.get("unit_refs", [])
            if not step_id or not isinstance(unit_refs, list):
                continue
            for unit_ref in unit_refs:
                unit_id = str(unit_ref).strip()
                if not unit_id:
                    continue
                out[unit_id] = {
                    "slice_id": step_id,
                    "slice_type": "step",
                    "section_id": section_id,
                    "section": section_meta,
                    "target": {
                        "step_id": step_id,
                        "step_order": step.get("step_order"),
                        "prompt": step.get("prompt"),
                        "focus_points": step.get("focus_points", []),
                        "common_mistakes": step.get("common_mistakes", []),
                        "unit_refs": unit_refs,
                    },
                    "case_id": case_id,
                }
    return out


def get_step_slices(index_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    slices = index_data.get("slices", [])
    if not isinstance(slices, list):
        return []
    steps = [s for s in slices if s.get("slice_type") == "step"]
    steps.sort(key=lambda s: float((s.get("time_range") or {}).get("start_sec", 0.0)))
    return steps


def get_atomic_step_slices(index_data: Dict[str, Any], unit_context: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    slices = index_data.get("slices", [])
    if not isinstance(slices, list):
        return []
    out: List[Dict[str, Any]] = []
    for item in slices:
        if not isinstance(item, dict):
            continue
        if str(item.get("unit_class", "")).strip() != "step":
            continue
        unit_id = str(item.get("unit_id", "")).strip()
        ctx = unit_context.get(unit_id, {})
        step_id = str(ctx.get("slice_id") or unit_id).strip()
        enriched = {
            "slice_id": step_id,
            "slice_type": "step",
            "section_id": item.get("section_id"),
            "unit_id": unit_id,
            "time_range": item.get("time_range"),
            "clip_path": item.get("clip_path"),
            "section": ctx.get("section", {"section_id": item.get("section_id")}),
            "target": ctx.get("target", {"step_id": step_id, "unit_refs": [unit_id]}),
            "source_stage": "teaching-segmentation:atomic-unit-slicing",
        }
        out.append(enriched)
    out.sort(key=lambda s: float((s.get("time_range") or {}).get("start_sec", 0.0)))
    return out


def resolve_clip_path(clip_path: Any) -> Path:
    raw = Path(str(clip_path))
    if raw.is_absolute():
        return raw
    parts = raw.parts
    if parts and parts[0] == "data":
        return (ROOT / raw).resolve()
    return (ROOT / "data" / raw).resolve()


def get_transcript_full_segments(transcript: Dict[str, Any]) -> List[Dict[str, Any]]:
    all_segments = transcript.get("segments")
    if not isinstance(all_segments, list):
        return []
    full: List[Dict[str, Any]] = []
    for seg in all_segments:
        if not isinstance(seg, dict):
            continue
        seg_start = float(seg.get("start_sec", 0.0) or 0.0)
        seg_end = float(seg.get("end_sec", seg_start) or seg_start)
        if seg_end < seg_start:
            seg_end = seg_start
        full.append(
            {
                "segment_id": seg.get("segment_id"),
                "start_sec": seg_start,
                "end_sec": seg_end,
                "speaker": seg.get("speaker", "unknown"),
                "text": seg.get("text", ""),
            }
        )
    return full


def main() -> None:
    parser = argparse.ArgumentParser(description="Build detector plans for all step slices")
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--slice-index", default=None)
    parser.add_argument("--strategy", default=None, help="default data/<case_id>/v2/strategy/teaching_strategy_v2.json")
    parser.add_argument("--small-model-config", default=str(ROOT / "services" / "criteria-trainer" / "configs" / "small_models_v1.json"))
    parser.add_argument("--scene-entities-config", default=None)
    parser.add_argument("--transcript", default=None, help="path to FunASR transcript json; default data/<case_id>/v2/asr/transcript_v1.json")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--model", default=TA_MODEL)
    parser.add_argument("--base-url", default=TA_BASE_URL)
    parser.add_argument("--request-timeout-sec", type=int, default=TA_REQUEST_TIMEOUT_SEC)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-invalid-times", type=int, default=2, help="extra retries per step when generated plan is invalid or step generation fails")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--require-omni", action="store_true")
    args = parser.parse_args()

    api_key = get_api_key()
    if args.require_omni and not api_key:
        raise RuntimeError("require-omni is set but API key is missing")
    if not api_key:
        raise RuntimeError("API key missing. This script requires real Omni calls.")

    case_v2_dir = ROOT / "data" / args.case_id / "v2"
    index_path = (
        Path(args.slice_index)
        if args.slice_index
        else (case_v2_dir / "segmentation" / "atomic-unit-slices" / "index.json")
    )
    strategy_path = Path(args.strategy) if args.strategy else (case_v2_dir / "strategy" / "teaching_strategy_v2.json")
    scene_entities_path = (
        Path(args.scene_entities_config)
        if args.scene_entities_config
        else (case_v2_dir / "strategy" / "scene_entities_v1.json")
    )
    transcript_path = Path(args.transcript) if args.transcript else (case_v2_dir / "asr" / "transcript_v1.json")
    output_dir = Path(args.output_dir) if args.output_dir else (case_v2_dir / "detector-plans")

    index_data = read_json(index_path)
    small_models = filter_small_models_for_detector_plan(read_json(Path(args.small_model_config)))
    scene_entity_catalog = read_json(scene_entities_path)
    transcript = read_json(transcript_path)
    condition_primitives = read_json(CONDITION_PRIMITIVES_PATH)
    primitive_names = extract_primitive_names(condition_primitives)
    system_prompt = read_text(DETECTOR_SYSTEM_PROMPT_PATH)
    user_template = read_text(DETECTOR_USER_PROMPT_TEMPLATE_PATH)

    unit_context = load_strategy_unit_context(args.case_id, strategy_path)
    step_slices = get_atomic_step_slices(index_data, unit_context)
    if not step_slices:
        step_slices = get_step_slices(index_data)
    allowed_object_targets = get_allowed_object_targets(scene_entity_catalog)
    allowed_yolo_targets = load_yolo_entity_names(args.case_id, allowed_object_targets)
    if not step_slices:
        raise RuntimeError("no step slices found in index")
    if not allowed_yolo_targets:
        raise RuntimeError("no YOLO-detectable entity names found; train YOLO from entity presence first")

    successes: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    all_sanitized = True
    all_codes_valid = True
    yolo_detect_targets: Set[str] = set()
    step_attempts = max(1, 1 + int(args.retry_invalid_times))

    for step in step_slices:
        slice_id = str(step.get("slice_id", "")).strip()
        if not slice_id:
            continue
        out_path = output_dir / f"{slice_id}.json"

        if out_path.exists() and not args.force:
            try:
                existing = sanitize_detector_plan(read_json(out_path), allowed_object_targets=allowed_yolo_targets)
                ok, err = validate_plan(existing, primitive_names, allowed_object_targets=allowed_yolo_targets)
                if ok:
                    for t in extract_object_detector_targets(existing):
                        yolo_detect_targets.add(t)
                    successes.append({"slice_id": slice_id, "status": "skipped_existing", "file": str(out_path)})
                    continue
            except Exception:
                pass

        clip_path = resolve_clip_path(step["clip_path"])
        payload = build_detector_user_payload(
            step,
            small_models,
            condition_primitives,
            allowed_yolo_targets,
            scene_entity_catalog,
        )
        payload["transcript_ref"] = str(transcript_path)
        payload["transcript_full"] = get_transcript_full_segments(transcript)
        user_prompt = build_user_prompt(user_template, {"payload_json": json.dumps(payload, ensure_ascii=False, indent=2)})
        video_uri = upload_video_to_dashscope_oss(clip_path, api_key, args.model)

        last_error = ""
        debug_attempts: List[Dict[str, Any]] = []
        generated = False
        for attempt in range(1, step_attempts + 1):
            try:
                raw_plan = call_omni_for_detector_plan(
                    api_key=api_key,
                    model=args.model,
                    base_url=args.base_url,
                    request_timeout_sec=args.request_timeout_sec,
                    max_retries=args.max_retries,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    video_uri=video_uri,
                )
                plan = sanitize_detector_plan(raw_plan, allowed_object_targets=allowed_yolo_targets)
                debug_attempts.append(
                    {
                        "attempt": attempt,
                        "raw_model_selection": raw_plan.get("model_selection") if isinstance(raw_plan, dict) else None,
                        "sanitized_model_selection": plan.get("model_selection"),
                    }
                )
                ok, err = validate_plan(plan, primitive_names, allowed_object_targets=allowed_yolo_targets)
                if not ok:
                    all_codes_valid = False
                    raise RuntimeError(f"invalid plan: {err}")
                for t in extract_object_detector_targets(plan):
                    yolo_detect_targets.add(t)
                write_json(out_path, plan)
                successes.append(
                    {
                        "slice_id": slice_id,
                        "status": "generated",
                        "file": str(out_path),
                        "attempts": attempt,
                    }
                )
                generated = True
                break
            except Exception as exc:
                last_error = str(exc)
                if len(debug_attempts) < attempt:
                    debug_attempts.append({"attempt": attempt, "error": last_error})
                if attempt < step_attempts:
                    continue

        if not generated:
            failures.append(
                {
                    "slice_id": slice_id,
                    "error": last_error,
                    "attempts": step_attempts,
                    "debug_attempts": debug_attempts,
                }
            )

    produced_count = len(successes)
    failed_count = len(failures)
    total_steps = len(step_slices)
    check_count_balance = (total_steps == produced_count + failed_count)

    consistency = {
        "check_total_steps_balance": check_count_balance,
        "check_all_plans_sanitized": all_sanitized,
        "check_all_judgement_codes_valid": all_codes_valid,
    }

    aggregate = {
        "pipeline_stage": "criteria-trainer:detector-plan:all-steps",
        "generated_at": utc_now_iso(),
        "case_id": args.case_id,
        "source": "omni",
        "total_step_slices": total_steps,
        "success_count": produced_count,
        "failed_count": failed_count,
        "successes": successes,
        "failures": failures,
        "grounding_dino_detect_targets": sorted(yolo_detect_targets),
        "yolo_detect_targets": sorted(yolo_detect_targets),
        "slice_index": str(index_path),
        "strategy": str(strategy_path),
        "consistency_checks": consistency,
    }
    write_json(output_dir / "detector_plan_v2.json", aggregate)

    print(f"[OK] total steps: {total_steps}")
    print(f"[OK] success: {produced_count}, failed: {failed_count}")
    print(f"[OK] output: {output_dir / 'detector_plan_v2.json'}")


if __name__ == "__main__":
    main()
