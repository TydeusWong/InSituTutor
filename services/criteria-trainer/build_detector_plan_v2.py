import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

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


def build_detector_user_payload(target_slice: Dict[str, Any], small_models: Dict[str, Any], condition_primitives: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "task": "build_detector_plan_for_single_slice",
        "policy": {
            "minimal_models_only": True,
            "description": "Use the minimum sufficient models to judge completion. Avoid unnecessary entities.",
        },
        "condition_primitives_ref": "services/criteria-trainer/configs/detection_condition_primitives_v1.json",
        "condition_primitives": condition_primitives,
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
                    "code": "string"
                }
            ],
            "code_rule": "code must only call predefined primitive function names in condition_primitives",
        },
    }


def sanitize_detector_plan(plan: Dict[str, Any]) -> Dict[str, Any]:
    allowed = {
        "slice_id",
        "slice_type",
        "models_required",
        "model_selection",
        "execution_plan",
        "judgement_conditions",
    }
    cleaned = {k: v for k, v in plan.items() if k in allowed}
    model_selection = cleaned.get("model_selection")
    if isinstance(model_selection, list):
        agg: Dict[str, List[str]] = {}
        for item in model_selection:
            if not isinstance(item, dict):
                continue
            model_id = str(item.get("model_id", "")).strip()
            if not model_id:
                continue
            targets: List[str] = []
            if isinstance(item.get("detect_targets"), list):
                targets = [str(x).strip() for x in item["detect_targets"] if str(x).strip()]
            elif isinstance(item.get("detect_target"), str):
                raw = item["detect_target"]
                targets = [x.strip() for x in re.split(r",| and ", raw) if x.strip()]
            agg.setdefault(model_id, [])
            for t in targets:
                if t not in agg[model_id]:
                    agg[model_id].append(t)
        cleaned["model_selection"] = [{"model_id": m, "detect_targets": ts} for m, ts in agg.items()]
    return cleaned


def build_mock_detector_plan(target_slice: Dict[str, Any], small_models: Dict[str, Any]) -> Dict[str, Any]:
    _ = small_models
    target_id = target_slice.get("slice_id", "unknown_slice")
    return {
        "slice_id": target_id,
        "slice_type": target_slice.get("slice_type"),
        "models_required": ["grounding-dino"],
        "model_selection": [
            {
                "model_id": "grounding-dino",
                "detect_targets": ["target object position"],
            }
        ],
        "execution_plan": {
            "order": ["grounding-dino"],
            "parallel_groups": [["grounding-dino"]],
        },
        "judgement_conditions": [
            {
                "condition_id": "step_center_alignment",
                "when": "step",
                "code": "abs_pos_distance('blue_box', 'workspace_center') <= 0.05",
            }
        ],
    }


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
            return sanitize_detector_plan(parsed)
        except Exception:
            if i == max_retries:
                raise
            time.sleep(i * 1.1)
    raise RuntimeError("unreachable")


def choose_first_step_slice(index_data: Dict[str, Any]) -> Dict[str, Any]:
    slices = index_data.get("slices", [])
    if not isinstance(slices, list) or not slices:
        raise ValueError("slice index is empty")
    step_slices = [s for s in slices if s.get("slice_type") == "step"]
    if not step_slices:
        raise ValueError("no step slice found")
    step_slices.sort(key=lambda s: float((s.get("time_range") or {}).get("start_sec", 0.0)))
    return step_slices[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build detector plan from first step slice")
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--slice-index", default=None)
    parser.add_argument("--small-model-config", default=str(ROOT / "services" / "criteria-trainer" / "configs" / "small_models_v1.json"))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--model", default=TA_MODEL)
    parser.add_argument("--base-url", default=TA_BASE_URL)
    parser.add_argument("--request-timeout-sec", type=int, default=TA_REQUEST_TIMEOUT_SEC)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--require-omni", action="store_true")
    args = parser.parse_args()

    case_v2_dir = ROOT / "data" / args.case_id / "v2"
    index_path = Path(args.slice_index) if args.slice_index else (case_v2_dir / "slices" / "index.json")
    output_dir = Path(args.output_dir) if args.output_dir else (case_v2_dir / "detector-plans")

    index_data = read_json(index_path)
    small_models = read_json(Path(args.small_model_config))
    condition_primitives = read_json(CONDITION_PRIMITIVES_PATH)
    system_prompt = read_text(DETECTOR_SYSTEM_PROMPT_PATH)
    user_template = read_text(DETECTOR_USER_PROMPT_TEMPLATE_PATH)
    target_slice = choose_first_step_slice(index_data)
    clip_path = ROOT / "data" / Path(target_slice["clip_path"])

    mode = "mock"
    plan: Dict[str, Any]
    api_key = get_api_key()
    can_call_omni = bool(api_key) and not args.mock
    if args.require_omni and not can_call_omni:
        raise RuntimeError("require-omni is set but API key is missing or --mock enabled")

    if can_call_omni:
        user_payload = build_detector_user_payload(
            target_slice=target_slice,
            small_models=small_models,
            condition_primitives=condition_primitives,
        )
        user_prompt = build_user_prompt(
            user_template,
            {"payload_json": json.dumps(user_payload, ensure_ascii=False, indent=2)},
        )
        video_uri = upload_video_to_dashscope_oss(clip_path, api_key, args.model)
        plan = call_omni_for_detector_plan(
            api_key=api_key,
            model=args.model,
            base_url=args.base_url,
            request_timeout_sec=args.request_timeout_sec,
            max_retries=args.max_retries,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            video_uri=video_uri,
        )
        mode = "omni"
    else:
        plan = build_mock_detector_plan(target_slice, small_models)

    plan_bundle = {
        "pipeline_stage": "criteria-trainer:detector-plan",
        "generated_at": utc_now_iso(),
        "case_id": args.case_id,
        "source": mode,
        "selected_slice": {
            "slice_id": target_slice.get("slice_id"),
            "slice_type": target_slice.get("slice_type"),
            "clip_path": target_slice.get("clip_path"),
            "time_range": target_slice.get("time_range"),
        },
        "detector_plan": plan,
    }

    write_json(output_dir / "detector_plan_v2.json", plan_bundle)
    write_json(output_dir / f"{target_slice.get('slice_id')}.json", plan)
    print(f"[OK] output: {output_dir / 'detector_plan_v2.json'}")
    print(f"[OK] mode: {mode}")


if __name__ == "__main__":
    main()
