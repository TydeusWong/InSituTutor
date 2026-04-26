import argparse
import base64
import json
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


SYSTEM_PROMPT = """You are a strict visual bbox-label verifier.

Task:
- You receive an image with one visible bbox overlay, the target entity name, the full task context, and the full allowed entity list.
- Judge whether the bbox encloses the target entity, not another allowed entity.

Rules:
- Use visual evidence as the primary source.
- The bbox must correspond to the target entity itself.
- If the bbox appears to enclose a different allowed entity, answer no.
- If the bbox is ambiguous, too loose, too partial, or mainly covers another entity, answer no.
- Answer exactly one word: yes or no.
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
    path = Path(path_str)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def image_data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def load_task_context(case_id: str) -> Dict[str, Any]:
    strategy_path = ROOT / "data" / case_id / "v2" / "strategy" / "teaching_strategy_v2.json"
    entities_path = ROOT / "data" / case_id / "v2" / "strategy" / "scene_entities_v1.json"
    strategy = read_json(strategy_path) if strategy_path.exists() else {}
    entities = read_json(entities_path) if entities_path.exists() else {}
    return {
        "strategy_ref": to_rel_or_abs(strategy_path),
        "scene_entities_ref": to_rel_or_abs(entities_path),
        "task": strategy.get("task", {}),
        "allowed_entities": entities.get("entity_names", []),
    }


def call_omni_yes_no(
    *,
    api_key: str,
    base_url: str,
    model: str,
    image_path: Path,
    prompt: str,
    request_timeout_sec: int,
    max_retries: int,
) -> str:
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=request_timeout_sec,
        http_client=httpx.Client(transport=httpx.HTTPTransport(), timeout=request_timeout_sec),
    )
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=0,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": image_data_url(image_path)}},
                        ],
                    },
                ],
            )
            text = str(resp.choices[0].message.content or "").strip().lower()
            if text.startswith("yes"):
                return "yes"
            if text.startswith("no"):
                return "no"
            raise ValueError(f"model did not answer yes/no: {text[:200]}")
        except Exception:
            if attempt == max_retries:
                raise
            time.sleep(attempt * 1.2)
    raise RuntimeError("unreachable")


def iter_accepted_filtered_labels(filtered_root: Path) -> List[Path]:
    if not filtered_root.exists():
        return []
    out: List[Path] = []
    for path in filtered_root.rglob("*.json"):
        try:
            data = read_json(path)
        except Exception:
            continue
        if bool(data.get("accepted")):
            out.append(path)
    out.sort(key=lambda p: str(p).lower())
    return out


def build_user_prompt(task_context: Dict[str, Any], filtered_label: Dict[str, Any]) -> str:
    payload = {
        "instruction": "Answer exactly yes or no. Is the bbox in the image the target entity, not another allowed entity?",
        "target_entity": filtered_label.get("target"),
        "bbox_xyxy_normalized": filtered_label.get("bbox_xyxy"),
        "box_metrics": filtered_label.get("box_metrics"),
        "section_id": filtered_label.get("section_id"),
        "unit_id": filtered_label.get("unit_id"),
        "local_frame_index": filtered_label.get("local_frame_index"),
        "task_context": task_context,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate DINO bbox trace images with Omni yes/no judgement")
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--bootstrap-root", default=None, help="default data/<case_id>/v3/yolo-bootstrap")
    parser.add_argument("--output", default=None, help="default data/<case_id>/v3/yolo-bootstrap/dino_annotation_validation.json")
    parser.add_argument("--model", default=TA_MODEL)
    parser.add_argument("--base-url", default=TA_BASE_URL)
    parser.add_argument("--request-timeout-sec", type=int, default=TA_REQUEST_TIMEOUT_SEC)
    parser.add_argument("--max-retries", type=int, default=TA_MAX_RETRIES)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--mock", action="store_true", help="debug: mark all accepted DINO labels as yes without calling Omni")
    args = parser.parse_args()

    bootstrap_root = Path(args.bootstrap_root) if args.bootstrap_root else ROOT / "data" / args.case_id / "v3" / "yolo-bootstrap"
    filtered_root = bootstrap_root / "labels_dino_filtered"
    annotated_root = bootstrap_root / "annotated_samples"
    validation_root = bootstrap_root / "omni_validation"
    output_path = Path(args.output) if args.output else bootstrap_root / "dino_annotation_validation.json"
    task_context = load_task_context(args.case_id)

    api_key = ""
    if not args.mock:
        api_key = get_api_key()
        if not api_key:
            raise RuntimeError("API key not found. Use --mock or set DASHSCOPE_API_KEY.")

    filtered_paths = iter_accepted_filtered_labels(filtered_root)
    if args.limit and args.limit > 0:
        filtered_paths = filtered_paths[: args.limit]

    results: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    for idx, label_path in enumerate(filtered_paths, start=1):
        label = read_json(label_path)
        target = str(label.get("target", "")).strip()
        frame_key = label_path.stem
        image_path = annotated_root / target / f"{frame_key}.jpg"
        record = {
            "target": target,
            "section_id": label.get("section_id"),
            "unit_id": label.get("unit_id"),
            "local_frame_index": label.get("local_frame_index"),
            "bbox_xyxy": label.get("bbox_xyxy"),
            "score": label.get("score"),
            "filtered_label": to_rel_or_abs(label_path),
            "annotated_image": to_rel_or_abs(image_path),
        }
        try:
            if not image_path.exists():
                raise FileNotFoundError(f"annotated image not found: {image_path}")
            verdict = "yes" if args.mock else call_omni_yes_no(
                api_key=api_key,
                base_url=args.base_url,
                model=args.model,
                image_path=image_path,
                prompt=build_user_prompt(task_context, label),
                request_timeout_sec=args.request_timeout_sec,
                max_retries=args.max_retries,
            )
            record["verdict"] = verdict
            record["accepted_for_yolo"] = verdict == "yes"
            results.append(record)
            write_json(validation_root / target / f"{frame_key}.json", record)
            print(f"[OK] {idx}/{len(filtered_paths)} {target} {frame_key}: {verdict}")
        except Exception as exc:
            failure = dict(record)
            failure["error"] = str(exc)
            failures.append(failure)
            print(f"[FAIL] {idx}/{len(filtered_paths)} {target} {frame_key}: {exc}")

    output = {
        "pipeline_stage": "entity-presence:dino-bbox-omni-validation",
        "generated_at": utc_now_iso(),
        "case_id": args.case_id,
        "model": "mock" if args.mock else args.model,
        "task_context": task_context,
        "source": {
            "bootstrap_root": to_rel_or_abs(bootstrap_root),
            "filtered_root": to_rel_or_abs(filtered_root),
            "annotated_root": to_rel_or_abs(annotated_root),
        },
        "total_candidates": len(filtered_paths),
        "yes_count": sum(1 for item in results if item.get("verdict") == "yes"),
        "no_count": sum(1 for item in results if item.get("verdict") == "no"),
        "failure_count": len(failures),
        "results": results,
        "failures": failures,
    }
    write_json(output_path, output)
    print(f"[OK] output: {output_path}")
    print(f"[OK] yes: {output['yes_count']}, no: {output['no_count']}, failed: {len(failures)}")


if __name__ == "__main__":
    main()
