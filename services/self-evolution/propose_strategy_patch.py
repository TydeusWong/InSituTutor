import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any

from openai import OpenAI

from common import extract_json_from_text, read_json, session_root, utc_now_iso, write_json


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import TA_BASE_URL, TA_MAX_RETRIES, TA_MODEL, TA_REQUEST_TIMEOUT_SEC, get_api_key  # noqa: E402


def propose_patch(case_id: str, session_id: str, model: str, allow_empty: bool) -> Path:
    base = session_root(case_id, session_id)
    confusion_path = base / "reflection" / "confusion_events_v1.json"
    error_path = base / "reflection" / "error_events_v1.json"
    confusion = read_json(confusion_path) if confusion_path.exists() else []
    errors = read_json(error_path) if error_path.exists() else []
    strategy = read_json(ROOT / "data" / case_id / "v2" / "strategy" / "teaching_strategy_v2.json")
    system_prompt = (ROOT / "models" / "prompts" / "self_evolution_strategy_patch_system_prompt_v1.md").read_text(encoding="utf-8")
    user_template = (ROOT / "models" / "prompts" / "self_evolution_strategy_patch_user_prompt_template_v1.md").read_text(encoding="utf-8")
    result: dict[str, Any] = {"base_strategy_ref": f"data/{case_id}/v2/strategy/teaching_strategy_v2.json", "patches": []}
    raw_response: dict[str, Any] | None = None
    raw_response_text = ""
    if (confusion or errors) and get_api_key():
        payload = {"instructions": user_template, "confusion_events": confusion, "error_events": errors, "teaching_strategy": strategy}
        try:
            client = OpenAI(api_key=get_api_key(), base_url=TA_BASE_URL, timeout=TA_REQUEST_TIMEOUT_SEC)
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
                    raw_response_text = resp.choices[0].message.content or ""
                    parsed = extract_json_from_text(raw_response_text)
                    if isinstance(parsed, dict):
                        raw_response = copy.deepcopy(parsed)
                        result = parsed
                    break
                except Exception:
                    if attempt == TA_MAX_RETRIES:
                        raise
                    time.sleep(attempt * 1.2)
        except Exception:
            if not allow_empty:
                raise
    elif (confusion or errors) and not allow_empty:
        raise RuntimeError("No API key found for strategy patch proposal.")

    result.setdefault("base_strategy_ref", f"data/{case_id}/v2/strategy/teaching_strategy_v2.json")
    patches = result.get("patches") if isinstance(result.get("patches"), list) else []
    allowed_fields = {"prompt.zh", "prompt.en", "focus_points", "common_mistakes"}
    cleaned = []
    rejected = []
    for idx, patch in enumerate(patches, start=1):
        if not isinstance(patch, dict):
            rejected.append({"patch_index": idx, "reason": "patch_not_object", "patch": patch})
            continue
        target = patch.get("target") if isinstance(patch.get("target"), dict) else {}
        if target.get("field") not in allowed_fields:
            rejected.append(
                {
                    "patch_index": idx,
                    "reason": "target_field_not_allowed",
                    "target_field": target.get("field"),
                    "allowed_fields": sorted(allowed_fields),
                    "patch": patch,
                }
            )
            continue
        patch.setdefault("patch_id", f"patch_{idx:04d}")
        cleaned.append(patch)
    result["patches"] = cleaned
    result.setdefault("no_patch_reasons", [])
    result["rejected_patches"] = rejected
    result["raw_response"] = raw_response
    result["raw_response_text"] = raw_response_text
    result["generated_at"] = utc_now_iso()
    out = base / "reflection" / "strategy_patch_v1.json"
    write_json(out, result)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Propose strategy patches from self-evolution events.")
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--model", default=TA_MODEL)
    parser.add_argument("--allow-empty", action="store_true")
    args = parser.parse_args()
    out = propose_patch(args.case_id, args.session_id, args.model, args.allow_empty)
    print(f"[OK] strategy patch: {out}")


if __name__ == "__main__":
    main()
