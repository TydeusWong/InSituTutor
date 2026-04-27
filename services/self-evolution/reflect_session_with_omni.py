import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

import httpx
from openai import OpenAI

from common import extract_json_from_text, read_json, rel_path, session_root, utc_now_iso, write_json, write_json_list


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import TA_BASE_URL, TA_MAX_RETRIES, TA_MODEL, TA_REQUEST_TIMEOUT_SEC, get_api_key  # noqa: E402


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


def sanitize_reflection(raw: Any, case_id: str, session_id: str) -> Dict[str, list[dict]]:
    def normalize_time_range(value: Any) -> Dict[str, float]:
        if isinstance(value, list) and len(value) >= 2:
            return {"start": float(value[0] or 0.0), "end": float(value[1] or value[0] or 0.0)}
        if isinstance(value, dict):
            start = value.get("start", value.get("start_sec", 0.0))
            end = value.get("end", value.get("end_sec", start))
            return {"start": float(start or 0.0), "end": float(end or start or 0.0)}
        return {"start": 0.0, "end": 0.0}

    def ensure_scope(item: dict) -> None:
        scope = item.setdefault("scope", {})
        if not isinstance(scope, dict):
            item["scope"] = scope = {}
        scope.setdefault("applies_to_sections", [])
        scope.setdefault("applies_to_steps", [])
        scope.pop("not_errors_in_sections", None)
        scope.pop("not_errors_in_steps", None)

    def normalize_confusion(item: dict, idx: int) -> None:
        item.setdefault("confusion_id", f"conf_{idx:04d}")
        item.setdefault("session_id", session_id)
        item["time_range_sec"] = normalize_time_range(item.get("time_range_sec"))
        ensure_scope(item)
        if "student_question_original" not in item:
            item["student_question_original"] = str(item.get("evidence", ""))
        if "confusion_reason" not in item:
            item["confusion_reason"] = str(item.get("diagnosis", ""))
        item.setdefault("confidence", 0.5)

    def normalize_error(item: dict, idx: int) -> None:
        item.setdefault("error_id", f"err_{idx:04d}")
        item.setdefault("session_id", session_id)
        item["time_range_sec"] = normalize_time_range(item.get("time_range_sec"))
        ensure_scope(item)
        item.setdefault("error_name", str(item.get("diagnosis", item.get("error_id", "")))[:80])
        item.setdefault("teacher_intervention", {"utterance_original": str(item.get("evidence", "")), "video_observation": ""})

        allowed_cause_types = {
            "object_identity_confusion",
            "wrong_object_selection",
            "wrong_spatial_relation",
            "wrong_orientation",
            "wrong_sequence",
            "wrong_hand_action",
            "insufficient_action",
            "other",
        }
        if item.get("error_cause_type") not in allowed_cause_types:
            item["error_cause_type"] = "other"

        confused_entities = item.get("confused_entities")
        if not isinstance(confused_entities, dict):
            confused_entities = {}
            item["confused_entities"] = confused_entities
        confused_entities.setdefault("student_used_or_pointed_to", "")
        confused_entities.setdefault("should_have_used_or_recognized", "")
        confused_entities.setdefault("evidence_utterance", "")

        item.setdefault("student_wrong_action", str(item.get("diagnosis", "")))
        item.setdefault("why_wrong_in_scope", str(item.get("diagnosis", "")))
        item.setdefault(
            "correction_message",
            {
                "zh": "请指出具体错误对象或动作，并按当前步骤重新操作。",
                "en": "Identify the specific wrong object or action, then redo the current step correctly.",
            },
        )
        item.setdefault("confidence", 0.5)

    data = raw if isinstance(raw, dict) else {}
    learning = data.get("learning_events") if isinstance(data.get("learning_events"), list) else []
    confusion = data.get("confusion_events") if isinstance(data.get("confusion_events"), list) else []
    errors = data.get("error_events") if isinstance(data.get("error_events"), list) else []
    for idx, item in enumerate(confusion, start=1):
        if isinstance(item, dict):
            normalize_confusion(item, idx)
    for idx, item in enumerate(errors, start=1):
        if isinstance(item, dict):
            normalize_error(item, idx)
    if not learning:
        for item in confusion:
            if isinstance(item, dict):
                learning.append({"event_id": item.get("confusion_id"), "event_type": "confusion", **item})
        for item in errors:
            if isinstance(item, dict):
                learning.append({"event_id": item.get("error_id"), "event_type": "error", **item})
    return {
        "learning_events": [x for x in learning if isinstance(x, dict)],
        "confusion_events": [x for x in confusion if isinstance(x, dict)],
        "error_events": [x for x in errors if isinstance(x, dict)],
    }


def build_strategy_outline(step_windows: Dict[str, Any]) -> Dict[str, Any]:
    windows = step_windows.get("step_windows", []) if isinstance(step_windows, dict) else []
    sections: Dict[str, Dict[str, Any]] = {}
    for item in windows if isinstance(windows, list) else []:
        if not isinstance(item, dict):
            continue
        section_id = str(item.get("section_id", "")).strip()
        step_id = str(item.get("step_id", "")).strip()
        if not section_id or not step_id:
            continue
        section = sections.setdefault(section_id, {"section_id": section_id, "steps": []})
        section["steps"].append(
            {
                "step_id": step_id,
                "start_sec": item.get("start_sec"),
                "end_sec": item.get("end_sec"),
                "prompt": item.get("system_prompt", {}),
                "focus_points": item.get("focus_points", []),
                "common_mistakes": item.get("common_mistakes", []),
            }
        )
    return {"sections": list(sections.values())}


def reflect_session(case_id: str, session_id: str, model: str, allow_empty: bool) -> tuple[Path, Path, Path]:
    base = session_root(case_id, session_id)
    transcript = read_json(base / "asr" / "transcript_v1.json")
    step_windows = read_json(base / "alignment" / "step_windows_v1.json")
    entities = read_json(ROOT / "data" / case_id / "v2" / "strategy" / "scene_entities_v1.json")
    manifest = read_json(base / "session_manifest.json") if (base / "session_manifest.json").exists() else {}
    prompt_system = (ROOT / "models" / "prompts" / "self_evolution_reflection_system_prompt_v1.md").read_text(encoding="utf-8")
    prompt_template = (ROOT / "models" / "prompts" / "self_evolution_reflection_user_prompt_template_v1.md").read_text(encoding="utf-8")
    payload = {
        "instructions": prompt_template,
        "case_id": case_id,
        "session_id": session_id,
        "asr_transcript": transcript,
        "step_windows": step_windows,
        "strategy_outline": build_strategy_outline(step_windows),
        "allowed_scene_entities": entities.get("entity_names", []),
    }
    raw: Any = {"learning_events": [], "confusion_events": [], "error_events": []}
    api_key = get_api_key()
    video_uri = ""
    if api_key:
        try:
            review = manifest.get("review_media_path") or str(base / "raw" / "teaching_session_review.mp4")
            video_path = Path(str(review))
            if not video_path.is_absolute():
                video_path = ROOT / video_path
            video_uri = upload_video_to_dashscope_oss(video_path, api_key, model) if video_path.exists() else ""
            extra_headers = {"X-DashScope-OssResourceResolve": "enable"} if video_uri.startswith("oss://") else {}
            content = [{"type": "text", "text": json.dumps(payload, ensure_ascii=False, indent=2)}]
            if video_uri:
                content.append({"type": "video_url", "video_url": {"url": video_uri}})
            client = OpenAI(
                api_key=api_key,
                base_url=TA_BASE_URL,
                timeout=TA_REQUEST_TIMEOUT_SEC,
                http_client=httpx.Client(transport=httpx.HTTPTransport(), timeout=TA_REQUEST_TIMEOUT_SEC),
            )
            for attempt in range(1, TA_MAX_RETRIES + 1):
                try:
                    stream = client.chat.completions.create(
                        model=model,
                        temperature=0,
                        modalities=["text"],
                        stream=True,
                        messages=[{"role": "system", "content": prompt_system}, {"role": "user", "content": content}],
                        extra_headers=extra_headers,
                    )
                    text = "".join(chunk.choices[0].delta.content or "" for chunk in stream if chunk.choices)
                    raw = extract_json_from_text(text)
                    break
                except Exception:
                    if attempt == TA_MAX_RETRIES:
                        raise
                    time.sleep(attempt * 1.2)
        except Exception as exc:
            if not allow_empty:
                raise
            raw = {"learning_events": [], "confusion_events": [], "error_events": [], "error": str(exc)}
    elif not allow_empty:
        raise RuntimeError("No DashScope API key found for Omni reflection.")

    sanitized = sanitize_reflection(raw, case_id, session_id)
    reflection_dir = base / "reflection"
    learning_path = reflection_dir / "learning_events_v1.json"
    confusion_path = reflection_dir / "confusion_events_v1.json"
    error_path = reflection_dir / "error_events_v1.json"
    write_json_list(learning_path, sanitized["learning_events"])
    write_json_list(confusion_path, sanitized["confusion_events"])
    write_json_list(error_path, sanitized["error_events"])
    write_json(
        reflection_dir / "reflection_meta_v1.json",
        {
            "generated_at": utc_now_iso(),
            "model": model,
            "video_uri": video_uri,
            "omni_payload_policy": {
                "sent_session_manifest": False,
                "sent_full_teaching_strategy": False,
                "sent_step_windows": True,
                "sent_strategy_outline": True,
                "sent_allowed_scene_entities": True,
            },
            "raw_response": raw,
        },
    )
    return learning_path, confusion_path, error_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Reflect one self-evolution session with Omni.")
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--model", default=TA_MODEL)
    parser.add_argument("--allow-empty", action="store_true")
    args = parser.parse_args()
    learning_path, confusion_path, error_path = reflect_session(args.case_id, args.session_id, args.model, args.allow_empty)
    print(f"[OK] learning events: {learning_path}")
    print(f"[OK] confusion events: {confusion_path}")
    print(f"[OK] error events: {error_path}")


if __name__ == "__main__":
    main()
