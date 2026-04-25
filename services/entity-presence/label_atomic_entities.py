import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import httpx
from openai import OpenAI


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import TA_BASE_URL, TA_MAX_RETRIES, TA_MODEL, TA_REQUEST_TIMEOUT_SEC, get_api_key  # noqa: E402


SYSTEM_PROMPT = """You are a visual entity-presence labeler for short teaching-video clips.

Task:
- Given one short video clip and an allowed entity list, identify which allowed entities are visually present in the clip.

Strict rules:
- Use visual evidence as the primary source. Do not mark an entity present only because speech mentions it.
- Entity names in `present_entities` MUST be copied exactly from `allowed_entities`.
- Do not output aliases, translations, singular/plural variants, capitalization variants, or subpart names.
- If an entity is uncertain or only partially visible, include it only when it is visually recognizable as one of the allowed entities.
- If no allowed entity is visually present, return an empty list.
- Return JSON only. No markdown, no explanation.

Output shape:
{
  "present_entities": ["exact name from allowed_entities"]
}
"""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def to_rel_or_abs(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


def resolve_under_root(path_str: str) -> Path:
    p = Path(path_str)
    if not p.is_absolute():
        p = ROOT / p
    return p.resolve()


def dedupe_str_list(values: Any) -> List[str]:
    if not isinstance(values, list):
        return []
    out: List[str] = []
    seen = set()
    for value in values:
        item = str(value).strip()
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def extract_allowed_entities_from_strategy_catalog(scene_entities_catalog: Dict[str, Any]) -> List[str]:
    return dedupe_str_list(scene_entities_catalog.get("entity_names"))


def build_user_prompt(slice_item: Dict[str, Any], allowed_entities: List[str]) -> str:
    payload = {
        "clip": {
            "video_id": slice_item.get("video_id"),
            "section_id": slice_item.get("section_id"),
            "unit_id": slice_item.get("unit_id"),
            "unit_class": slice_item.get("unit_class"),
            "time_range": slice_item.get("time_range"),
        },
        "allowed_entities": allowed_entities,
    }
    return (
        "Label the visually present entities in this clip.\n"
        "Decide presence from visual understanding first; speech/audio is only a naming aid for objects that are visible.\n"
        "Do not mark an entity present if it is only mentioned in speech but not visible in the clip.\n"
        "Only choose exact strings from `allowed_entities`.\n\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def extract_json_from_text(text: str) -> Dict[str, Any]:
    fenced = re.search(r"```(?:json)?\s*(\{[\s\S]*\})\s*```", text)
    if fenced:
        return json.loads(fenced.group(1))
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return json.loads(text[start : end + 1])
    raise ValueError("no json object found in model output")


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


def call_omni_entity_presence(
    *,
    api_key: str,
    base_url: str,
    model: str,
    user_prompt: str,
    video_uri: str,
    request_timeout_sec: int,
    max_retries: int,
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

    for attempt in range(1, max_retries + 1):
        try:
            stream = client.chat.completions.create(
                model=model,
                temperature=0,
                modalities=["text"],
                stream=True,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
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
            return extract_json_from_text("".join(parts))
        except Exception:
            if attempt == max_retries:
                raise
            time.sleep(attempt * 1.2)
    raise RuntimeError("unreachable")


def sanitize_present_entities(raw: Dict[str, Any], allowed_entities: List[str], slice_id: str) -> List[str]:
    allowed = set(allowed_entities)
    values = raw.get("present_entities")
    if not isinstance(values, list):
        raise ValueError(f"{slice_id}: model output missing list field present_entities")

    out: List[str] = []
    invalid: List[str] = []
    seen = set()
    for value in values:
        name = str(value).strip()
        if name not in allowed:
            invalid.append(name)
            continue
        if name not in seen:
            seen.add(name)
            out.append(name)

    if invalid:
        raise ValueError(f"{slice_id}: present_entities not in allowed entity list: {invalid}")
    return [name for name in allowed_entities if name in seen]


def mock_present_entities(slice_item: Dict[str, Any], allowed_entities: List[str]) -> List[str]:
    text = " ".join(
        str(slice_item.get(key, ""))
        for key in ("section_id", "unit_id", "unit_class", "clip_path")
    ).lower()
    return [name for name in allowed_entities if name.lower() in text]


def main() -> None:
    parser = argparse.ArgumentParser(description="Label scene-entity presence for atomic-unit video clips with Omni")
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--slice-index", default=None, help="default data/<case_id>/v2/segmentation/atomic-unit-slices/index.json")
    parser.add_argument("--sections-units", default=None, help="default data/<case_id>/v2/segmentation/sections_units.json")
    parser.add_argument("--scene-entities", default=None, help="default data/<case_id>/v2/strategy/scene_entities_v1.json")
    parser.add_argument("--output", default=None, help="default data/<case_id>/v2/segmentation/entity-presence/entity_presence.json")
    parser.add_argument("--model", default=TA_MODEL)
    parser.add_argument("--base-url", default=TA_BASE_URL)
    parser.add_argument("--request-timeout-sec", type=int, default=TA_REQUEST_TIMEOUT_SEC)
    parser.add_argument("--max-retries", type=int, default=TA_MAX_RETRIES)
    parser.add_argument("--limit", type=int, default=0, help="debug: process at most N clips; 0 means all")
    parser.add_argument("--mock", action="store_true", help="skip Omni and emit deterministic empty/matched mock labels")
    args = parser.parse_args()

    case_v2_seg_dir = ROOT / "data" / args.case_id / "v2" / "segmentation"
    case_v2_strategy_dir = ROOT / "data" / args.case_id / "v2" / "strategy"
    slice_index_path = Path(args.slice_index) if args.slice_index else (case_v2_seg_dir / "atomic-unit-slices" / "index.json")
    sections_units_path = Path(args.sections_units) if args.sections_units else (case_v2_seg_dir / "sections_units.json")
    scene_entities_path = Path(args.scene_entities) if args.scene_entities else (case_v2_strategy_dir / "scene_entities_v1.json")
    output_path = Path(args.output) if args.output else (case_v2_seg_dir / "entity-presence" / "entity_presence.json")

    slice_index = read_json(slice_index_path)
    if not scene_entities_path.exists():
        raise RuntimeError(
            f"scene entity catalog not found: {scene_entities_path}. "
            f"Run `python services/strategy-builder/build_strategy.py --case-id {args.case_id}` before entity presence labeling."
        )
    scene_entities_catalog = read_json(scene_entities_path)
    allowed_entities = extract_allowed_entities_from_strategy_catalog(scene_entities_catalog)
    if not allowed_entities:
        raise RuntimeError(f"no entity_names found in {scene_entities_path}")

    api_key = ""
    if not args.mock:
        api_key = get_api_key()
        if not api_key:
            raise RuntimeError("API key not found. Use --mock or set DASHSCOPE_API_KEY.")

    slices = slice_index.get("slices")
    if not isinstance(slices, list):
        raise RuntimeError(f"slice index missing list field slices: {slice_index_path}")
    if args.limit and args.limit > 0:
        slices = slices[: args.limit]

    results: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    for idx, slice_item in enumerate(slices, start=1):
        slice_id = f"{slice_item.get('section_id', 'section')}:{slice_item.get('unit_id', 'unit')}"
        try:
            if args.mock:
                present_entities = mock_present_entities(slice_item, allowed_entities)
            else:
                clip_path = resolve_under_root(str(slice_item.get("clip_path", "")))
                if not clip_path.exists():
                    raise FileNotFoundError(f"clip not found: {clip_path}")
                video_uri = upload_video_to_dashscope_oss(clip_path, api_key, args.model)
                raw = call_omni_entity_presence(
                    api_key=api_key,
                    base_url=args.base_url,
                    model=args.model,
                    user_prompt=build_user_prompt(slice_item, allowed_entities),
                    video_uri=video_uri,
                    request_timeout_sec=args.request_timeout_sec,
                    max_retries=args.max_retries,
                )
                present_entities = sanitize_present_entities(raw, allowed_entities, slice_id)

            results.append(
                {
                    "video_id": slice_item.get("video_id"),
                    "section_id": slice_item.get("section_id"),
                    "unit_id": slice_item.get("unit_id"),
                    "unit_class": slice_item.get("unit_class"),
                    "time_range": slice_item.get("time_range"),
                    "clip_path": slice_item.get("clip_path"),
                    "present_entities": present_entities,
                }
            )
            print(f"[OK] {idx}/{len(slices)} {slice_id}: {present_entities}")
        except Exception as exc:
            failures.append(
                {
                    "video_id": slice_item.get("video_id"),
                    "section_id": slice_item.get("section_id"),
                    "unit_id": slice_item.get("unit_id"),
                    "clip_path": slice_item.get("clip_path"),
                    "error": str(exc),
                }
            )
            print(f"[FAIL] {idx}/{len(slices)} {slice_id}: {exc}")

    output = {
        "pipeline_stage": "entity-presence:atomic-unit-omni",
        "generated_at": utc_now_iso(),
        "case_id": args.case_id,
        "model": "mock" if args.mock else args.model,
        "source": {
            "slice_index": to_rel_or_abs(slice_index_path),
            "scene_entities": to_rel_or_abs(scene_entities_path),
            "sections_units": to_rel_or_abs(sections_units_path),
        },
        "allowed_entities": allowed_entities,
        "total_slices": len(slices),
        "success_count": len(results),
        "failure_count": len(failures),
        "results": results,
        "failures": failures,
    }
    write_json(output_path, output)
    print(f"[OK] output: {output_path}")
    print(f"[OK] success: {len(results)}, failed: {len(failures)}")


if __name__ == "__main__":
    main()
