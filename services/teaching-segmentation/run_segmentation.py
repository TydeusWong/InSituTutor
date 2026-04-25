import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import httpx
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import (  # noqa: E402
    TA_BASE_URL,
    TA_MAX_RETRIES,
    TA_MODEL,
    TA_REQUEST_TIMEOUT_SEC,
    V2_SEGMENTATION_OUTPUT_PATH,
    V2_SEGMENTATION_PROMPT_DIR,
    get_api_key,
)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_case_id(raw: str, fallback: str) -> str:
    token = raw.strip().lower().replace("-", "_")
    token = re.sub(r"[^a-z0-9_\-]+", "_", token)
    token = re.sub(r"_{2,}", "_", token).strip("_")
    return token or fallback


def infer_case_id_from_manifest(manifest: Dict[str, Any]) -> str:
    demos = manifest.get("demos", [])
    if isinstance(demos, list) and demos:
        source_video_path = str(demos[0].get("source_video_path", "")).strip()
        if source_video_path:
            stem = Path(source_video_path).stem
            return normalize_case_id(stem, "default_case")
        video_id = str(demos[0].get("video_id", "")).strip()
        if video_id:
            first = video_id.split("__")[0]
            return normalize_case_id(first, "default_case")
    return "default_case"


def resolve_video_ref(item: Dict[str, Any]) -> str:
    if item.get("ingest_video_uri"):
        return str(item["ingest_video_uri"])
    if item.get("ingest_video_path"):
        p = Path(str(item["ingest_video_path"]))
        if not p.is_absolute():
            p = (ROOT / p).resolve()
        return str(p)
    raise ValueError("ingest item must include ingest_video_uri or ingest_video_path")


def resolve_local_ingest_video_path(item: Dict[str, Any]) -> Path:
    video_path = item.get("ingest_video_path")
    if not video_path:
        raise ValueError("ingest_video_path missing; atomic-unit slicing requires a local compressed video")
    p = Path(str(video_path))
    if not p.is_absolute():
        p = (ROOT / p).resolve()
    if not p.exists():
        raise FileNotFoundError(f"compressed ingest video not found: {p}")
    return p


def normalize_time_range(time_range: Dict[str, Any]) -> Tuple[float, float]:
    start = float(time_range.get("start_sec", 0.0) or 0.0)
    end = float(time_range.get("end_sec", start) or start)
    if end < start:
        end = start
    return start, end


def safe_path_token(raw: Any, fallback: str) -> str:
    token = str(raw or "").strip().lower().replace("\\", "_").replace("/", "_")
    token = re.sub(r"[^a-z0-9_\-]+", "_", token)
    token = re.sub(r"_{2,}", "_", token).strip("_")
    return token or fallback


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
        raise RuntimeError(f"ffmpeg failed for {out_file}: {proc.stderr.strip()[:800]}")


def to_rel_or_abs(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


def resolve_workspace_output_dir(path: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(ROOT)
    except ValueError as exc:
        raise ValueError(f"atomic slices output dir must be under project root: {resolved}") from exc
    return resolved


def reset_atomic_slice_dir(output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def slice_atomic_units(
    *,
    case_id: str,
    item: Dict[str, Any],
    parsed: Dict[str, Any],
    output_root: Path,
) -> List[Dict[str, Any]]:
    video_path = resolve_local_ingest_video_path(item)
    video_token = safe_path_token(item.get("video_id"), "video")
    slices: List[Dict[str, Any]] = []

    for section in parsed.get("sections", []):
        if not isinstance(section, dict):
            continue
        section_id = safe_path_token(section.get("section_id"), "section")

        for unit in section.get("atomic_units", []):
            if not isinstance(unit, dict):
                continue
            unit_id = safe_path_token(unit.get("unit_id"), "unit")
            time_range = unit.get("time_range") if isinstance(unit.get("time_range"), dict) else {}
            start, end = normalize_time_range(time_range)
            if end <= start:
                raise ValueError(f"invalid atomic unit time_range for {unit_id}: start={start}, end={end}")

            clip_path = output_root / video_token / section_id / unit_id / "clip.mp4"
            run_ffmpeg_cut(video_path, start, end, clip_path)
            slices.append(
                {
                    "case_id": case_id,
                    "video_id": item.get("video_id", "unknown_video"),
                    "section_id": section.get("section_id", section_id),
                    "unit_id": unit.get("unit_id", unit_id),
                    "unit_class": unit.get("class", "unknown"),
                    "time_range": {"start_sec": start, "end_sec": end},
                    "source_video_path": to_rel_or_abs(video_path),
                    "clip_path": to_rel_or_abs(clip_path),
                }
            )

    return slices


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


def build_user_prompt(template: str, payload: Dict[str, Any]) -> str:
    out = template
    for k, v in payload.items():
        out = out.replace("{{" + k + "}}", str(v))
    return out


def extract_json_from_text(text: str) -> Dict[str, Any]:
    fenced = re.search(r"```(?:json)?\s*(\{[\s\S]*\})\s*```", text)
    if fenced:
        return json.loads(fenced.group(1))
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return json.loads(text[start : end + 1])
    raise ValueError("no json object found in model output")


def call_omni_sections(
    api_key: str,
    base_url: str,
    model: str,
    system_prompt: str,
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
            return extract_json_from_text("".join(parts))
        except Exception:
            if i == max_retries:
                raise
            time.sleep(i * 1.2)
    raise RuntimeError("unreachable")


def mock_sections(item: Dict[str, Any]) -> Dict[str, Any]:
    duration = float(item.get("ingest_duration_sec") or 60.0)
    mid = max(10.0, duration * 0.5)
    end = max(mid + 10.0, duration)
    return {
        "video_overview": {
            "summary": "该视频演示了目标物体摆放与收尾确认流程。",
            "camera_view": "top_down",
            "scene_entities": ["black can", "transparent tape"],
            "total_sections": 2,
            "section_atomic_counts": [
                {"section_ref": "section_01", "atomic_unit_count": 3},
                {"section_ref": "section_02", "atomic_unit_count": 1},
            ],
        },
        "sections": [
            {
                "section_id": "section_01",
                "section_name": "准备与定位",
                "section_summary": "准备关键物体并完成目标对齐。",
                "expected_section_state": "黑色罐子稳定放置在透明胶带上方并保持可见。",
                "time_range": {"start_sec": 0.0, "end_sec": round(mid, 2)},
                "atomic_units": [
                    {
                        "unit_id": "unit_01_001",
                        "time_range": {"start_sec": 3.0, "end_sec": 7.0},
                        "class": "step",
                        "description": "抓取黑色罐子并移动到透明胶带上方",
                        "evidence": {
                            "audio": ["把黑色罐子放到透明胶带上面"],
                            "vision": ["hand_grasp(black_can)", "above(black_can,transparent_tape)"]
                        },
                        "step_fields": {
                            "prompt": {
                                "en": "Place the black can above the transparent tape.",
                                "zh": "请把黑色罐子放到透明胶带上方。"
                            },
                            "focus_points": ["先稳定抓握", "对齐目标中心"],
                            "common_mistakes": ["放在胶带旁边"],
                        },
                        "error_fields": {},
                    },
                    {
                        "unit_id": "unit_01_002",
                        "time_range": {"start_sec": 8.0, "end_sec": 11.0},
                        "class": "error",
                        "description": "黑色罐子摆放偏移，未在透明胶带上方",
                        "evidence": {
                            "audio": [],
                            "vision": ["not_above(black_can,transparent_tape)"]
                        },
                        "step_fields": {},
                        "error_fields": {
                            "trigger_signature": "object_misplacement:black_can_not_above_tape",
                            "correction_prompt": "位置不对，请把黑色罐子放到透明胶带正上方。",
                            "recovery_actions": ["抓取黑色罐子", "移动到透明胶带上方", "松手确认"]
                        },
                    },
                    {
                        "unit_id": "unit_01_003",
                        "time_range": {"start_sec": 12.0, "end_sec": 14.0},
                        "class": "not_related",
                        "description": "与任务无关的短暂停顿",
                        "evidence": {"audio": [], "vision": ["idle"]},
                        "step_fields": {},
                        "error_fields": {},
                    },
                ],
            },
            {
                "section_id": "section_02",
                "section_name": "确认与收尾",
                "section_summary": "确认物体关系保持稳定。",
                "expected_section_state": "黑色罐子在透明胶带上方位置稳定且无遮挡。",
                "time_range": {"start_sec": round(mid, 2), "end_sec": round(end, 2)},
                "atomic_units": [
                    {
                        "unit_id": "unit_02_001",
                        "time_range": {"start_sec": round(mid + 3, 2), "end_sec": round(mid + 7, 2)},
                        "class": "step",
                        "description": "观察关系稳定并完成确认",
                        "evidence": {
                            "audio": ["确认放置完成"],
                            "vision": ["stable(black_can)"]
                        },
                        "step_fields": {
                            "prompt": {
                                "en": "Confirm the black can is stably above the transparent tape.",
                                "zh": "确认黑色罐子稳定位于透明胶带上方。"
                            },
                            "focus_points": ["观察抖动", "检查遮挡"],
                            "common_mistakes": ["确认过早"],
                        },
                        "error_fields": {},
                    }
                ],
            },
        ]
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline A - teaching segmentation")
    parser.add_argument("--manifest", default=str(ROOT / "data" / "processed" / "ingest_manifest.json"))
    parser.add_argument("--output", default=None)
    parser.add_argument("--case-id", help="output case folder under data/, e.g. test_cake")
    parser.add_argument("--model", default=TA_MODEL)
    parser.add_argument("--base-url", default=TA_BASE_URL)
    parser.add_argument("--request-timeout-sec", type=int, default=TA_REQUEST_TIMEOUT_SEC)
    parser.add_argument("--max-retries", type=int, default=TA_MAX_RETRIES)
    parser.add_argument("--mock", action="store_true", help="use local mock parser and skip API")
    parser.add_argument(
        "--atomic-slices-dir",
        default=None,
        help="output directory for atomic-unit clips; default data/<case_id>/v2/segmentation/atomic-unit-slices",
    )
    args = parser.parse_args()

    manifest = read_json(Path(args.manifest))
    system_prompt = read_text(V2_SEGMENTATION_PROMPT_DIR / "segmentation_system_prompt_v2.md")
    user_template = read_text(V2_SEGMENTATION_PROMPT_DIR / "segmentation_user_prompt_template_v2.md")

    api_key = ""
    if not args.mock:
        api_key = get_api_key()
        if not api_key:
            raise RuntimeError("API key not found for Omni parsing. Use --mock or set DASHSCOPE_API_KEY.")

    case_id = args.case_id or infer_case_id_from_manifest(manifest)
    output_path = Path(args.output) if args.output else (ROOT / "data" / case_id / "v2" / "segmentation" / "sections_units.json")
    atomic_slices_dir = (
        Path(args.atomic_slices_dir)
        if args.atomic_slices_dir
        else ROOT / "data" / case_id / "v2" / "segmentation" / "atomic-unit-slices"
    )
    atomic_slices_dir = resolve_workspace_output_dir(atomic_slices_dir)
    reset_atomic_slice_dir(atomic_slices_dir)

    out: Dict[str, Any] = {
        "pipeline_stage": "teaching-segmentation",
        "generated_at": utc_now_iso(),
        "case_id": case_id,
        "demos": [],
    }
    atomic_slice_items: List[Dict[str, Any]] = []

    for item in manifest.get("demos", []):
        if args.mock:
            parsed = mock_sections(item)
        else:
            video_ref = resolve_video_ref(item)
            if video_ref.endswith(".mp4") and not video_ref.startswith("http") and not video_ref.startswith("oss://"):
                video_ref = upload_video_to_dashscope_oss(Path(video_ref), api_key, args.model)

            user_prompt = build_user_prompt(
                user_template,
                {
                    "task_id": item.get("task_id", "unknown_task"),
                    "video_id": item.get("video_id", "unknown_video"),
                    "duration_sec": item.get("ingest_duration_sec", 0),
                    "fps": item.get("fps", "unknown"),
                    "resolution": item.get("resolution", "unknown"),
                    "video_quality": item.get("video_quality", "unknown"),
                    "video_bitrate_kbps": item.get("video_bitrate_kbps", "unknown"),
                    "source_audio_quality": item.get("source_audio_quality", "unknown"),
                    "scene_tags": ",".join(item.get("scene_tags", [])) if isinstance(item.get("scene_tags"), list) else "",
                },
            )
            parsed = call_omni_sections(
                api_key=api_key,
                base_url=args.base_url,
                model=args.model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                video_uri=video_ref,
                request_timeout_sec=args.request_timeout_sec,
                max_retries=args.max_retries,
            )

        atomic_slice_items.extend(
            slice_atomic_units(
                case_id=case_id,
                item=item,
                parsed=parsed,
                output_root=atomic_slices_dir,
            )
        )

        out["demos"].append(
            {
                "task_id": item.get("task_id", "task-demo"),
                "video_id": item.get("video_id", "unknown_video"),
                "scene_tags": item.get("scene_tags", []),
                "source_audio_quality": item.get("source_audio_quality", "unknown"),
                "fps": item.get("fps", "unknown"),
                "resolution": item.get("resolution", "unknown"),
                "video_quality": item.get("video_quality", "unknown"),
                "video_bitrate_kbps": item.get("video_bitrate_kbps", "unknown"),
                "video_overview": parsed.get("video_overview", {}),
                "sections": parsed.get("sections", []),
            }
        )

    atomic_slice_index = {
        "pipeline_stage": "teaching-segmentation:atomic-unit-slicing",
        "generated_at": utc_now_iso(),
        "case_id": case_id,
        "output_root": to_rel_or_abs(atomic_slices_dir),
        "slice_count": len(atomic_slice_items),
        "slices": atomic_slice_items,
    }
    write_json(atomic_slices_dir / "index.json", atomic_slice_index)
    write_json(output_path, out)
    print(f"[OK] output: {output_path}")
    print(f"[OK] atomic unit slices: {atomic_slices_dir / 'index.json'}")
    print(f"[OK] atomic unit slice count: {len(atomic_slice_items)}")


if __name__ == "__main__":
    main()
