import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import os

import httpx
from openai import OpenAI

os.environ.setdefault("NO_PROXY", "*")
os.environ.setdefault("no_proxy", "*")

_direct_http = httpx.Client(transport=httpx.HTTPTransport(), timeout=300)

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import (
    TA_API_KEY_ENV_NAMES,
    TA_BASE_URL,
    TA_MANIFEST_PATH,
    TA_MAX_RETRIES,
    TA_MODEL,
    TA_OUTPUT_DIR,
    TA_PROMPT_DIR,
    TA_REQUEST_TIMEOUT_SEC,
    TA_SCHEMA_PATH,
    TA_SEGMENT_MINUTES,
    get_api_key,
)

def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def upload_video_to_dashscope_oss(
    video_path: Path, api_key: str, model: str,
) -> str:
    """上传本地视频到 DashScope 临时 OSS，返回 oss:// URL（48 h 有效）。"""
    policy_resp = _direct_http.get(
        "https://dashscope.aliyuncs.com/api/v1/uploads",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        params={"action": "getPolicy", "model": model},
    )
    if policy_resp.status_code != 200:
        raise RuntimeError(f"获取 OSS 上传凭证失败: {policy_resp.text}")
    data = policy_resp.json()["data"]

    file_name = video_path.name
    upload_key = f"{data['upload_dir']}/{file_name}"
    with open(video_path, "rb") as f:
        upload_resp = _direct_http.post(
            data["upload_host"],
            files={
                "OSSAccessKeyId": (None, data["oss_access_key_id"]),
                "Signature": (None, data["signature"]),
                "policy": (None, data["policy"]),
                "x-oss-object-acl": (None, data["x_oss_object_acl"]),
                "x-oss-forbid-overwrite": (None, data["x_oss_forbid_overwrite"]),
                "key": (None, upload_key),
                "success_action_status": (None, "200"),
                "file": (file_name, f, "video/mp4"),
            },
        )
    if upload_resp.status_code != 200:
        raise RuntimeError(f"上传视频到 OSS 失败: {upload_resp.text}")
    return f"oss://{upload_key}"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_video_id(video_id: str) -> bool:
    # 规范：task__scene__teacher__v01
    return bool(re.match(r"^[a-z0-9\-]+__[a-z0-9\-]+__[a-z0-9\-]+__v\d{2}$", video_id))


def build_user_prompt(template: str, payload: Dict[str, Any]) -> str:
    out = template
    for k, v in payload.items():
        out = out.replace("{{" + k + "}}", str(v))
    return out


def extract_json_from_text(text: str) -> Dict[str, Any]:
    if not text:
        raise ValueError("模型返回为空")
    fenced = re.search(r"```(?:json)?\s*(\{[\s\S]*\})\s*```", text)
    if fenced:
        return json.loads(fenced.group(1))
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return json.loads(text[start : end + 1])
    raise ValueError("未检测到 JSON 内容")


def call_dashscope_chat(
    api_key: str,
    base_url: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    video_uri: str,
    request_timeout_sec: int,
    max_retries: int = 3,
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
                stream_options={"include_usage": True},
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
            text_parts: List[str] = []
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    text_parts.append(chunk.choices[0].delta.content)
            content = "".join(text_parts)
            print(f"  [模型原始输出前 500 字] {content[:500]}")
            return extract_json_from_text(content)
        except Exception as e:
            if i == max_retries:
                raise RuntimeError(f"调用失败，重试 {max_retries} 次后仍失败: {e}") from e
            time.sleep(i * 1.5)
    raise RuntimeError("调用失败")


def ensure_base_shape(d: Dict[str, Any]) -> Dict[str, Any]:
    d.setdefault("task", {})
    # 兼容模型返回 teaching_steps / procedure 等替代键名
    for alt in ("teaching_steps", "procedure", "step_list"):
        if alt in d and "steps" not in d:
            d["steps"] = d.pop(alt)
            break
    d.setdefault("steps", [])
    d.setdefault("error_catalog", [])
    d.setdefault("intervention_policies", [])
    d.setdefault("runtime_rules", [])
    d.setdefault("meta", {})
    return d


def repair_and_validate(d: Dict[str, Any], task_defaults: Dict[str, Any]) -> Dict[str, Any]:
    d = ensure_base_shape(d)
    task = d["task"]
    task.setdefault("task_id", task_defaults["task_id"])
    task.setdefault("task_name", task_defaults["task_name"])
    task.setdefault("environment", task_defaults["environment"])
    task.setdefault("required_objects", [])
    task.setdefault("safety_notes", [])

    fixed_steps = []
    for idx, step in enumerate(d.get("steps", []), start=1):
        if not isinstance(step, dict):
            continue
        step.setdefault("step_id", f"step_{idx:02d}")
        step["step_order"] = idx
        step.setdefault("step_name", f"步骤{idx}")
        step.setdefault("goal", "")
        step.setdefault("actions", [{"action_id": f"action_{idx:02d}_01", "action_name": "执行该步动作"}])
        step.setdefault(
            "attention_cues",
            [{"cue_id": f"cue_{idx:02d}_01", "cue_type": "object_presence", "description": "关键对象可见"}],
        )
        step.setdefault("completion_criteria", ["满足本步主要目标"])
        step.setdefault("next_step_hint", "")
        step.setdefault("common_mistake_warning", "")
        step.setdefault("if_error_then_intervention", "")
        fixed_steps.append(step)
    d["steps"] = fixed_steps

    if not d["intervention_policies"]:
        d["intervention_policies"] = [
            {
                "intervention_id": "intervention_default",
                "title": "默认纠偏提示",
                "message": "请回到当前步骤，按提示重新执行。",
                "channel": "text",
                "cooldown_sec": 5,
                "max_repeat": 3,
            }
        ]

    valid_step_ids = {s["step_id"] for s in d["steps"]}
    valid_interventions = {i["intervention_id"] for i in d["intervention_policies"]}
    valid_errors = set()

    fixed_errors = []
    for idx, e in enumerate(d.get("error_catalog", []), start=1):
        if not isinstance(e, dict):
            continue
        e.setdefault("error_id", f"error_{idx:02d}")
        e.setdefault("step_id", d["steps"][0]["step_id"] if d["steps"] else "step_01")
        if e["step_id"] not in valid_step_ids and d["steps"]:
            e["step_id"] = d["steps"][0]["step_id"]
        e.setdefault("error_pattern", "step_order_violation")
        e.setdefault("trigger_condition", "步骤顺序或动作条件不满足")
        sev = e.get("severity", "medium")
        e["severity"] = sev if sev in ("low", "medium", "high") else "medium"
        e.setdefault("intervention_id", "intervention_default")
        if e["intervention_id"] not in valid_interventions:
            e["intervention_id"] = "intervention_default"
        e.setdefault("fallback_action", "回到当前步骤重新执行")
        fixed_errors.append(e)
        valid_errors.add(e["error_id"])
    d["error_catalog"] = fixed_errors

    fixed_rules = []
    for idx, r in enumerate(d.get("runtime_rules", []), start=1):
        if not isinstance(r, dict):
            continue
        r.setdefault("rule_id", f"rule_{idx:02d}")
        rule_type = r.get("rule_type", "sequence")
        r["rule_type"] = rule_type if rule_type in ("sequence", "presence", "pose", "spatial", "timeout") else "sequence"
        r.setdefault("step_id", d["steps"][0]["step_id"] if d["steps"] else "step_01")
        if r["step_id"] not in valid_step_ids and d["steps"]:
            r["step_id"] = d["steps"][0]["step_id"]
        r.setdefault("simple_rule", {"operator": "must_follow_order", "value": True})
        r.setdefault("on_trigger", d["error_catalog"][0]["error_id"] if d["error_catalog"] else "error_01")
        fixed_rules.append(r)
    d["runtime_rules"] = fixed_rules

    meta = d["meta"]
    meta.setdefault("source_model", "Qwen3.5-Omni")
    meta.setdefault("source_videos", [])
    meta.setdefault("analysis_version", "v0.1.0")
    meta.setdefault("created_at", utc_now_iso())
    meta.setdefault("review_status", "draft")

    if not d["steps"]:
        raise ValueError("修复后 steps 为空，无法继续")
    return d


def make_segment_hints(duration_sec: float, segment_minutes: int) -> List[str]:
    if duration_sec <= 0 or segment_minutes <= 0:
        return []
    seg = segment_minutes * 60
    hints = []
    start = 0
    while start < duration_sec:
        end = min(duration_sec, start + seg)
        hints.append(f"{int(start//60):02d}:{int(start%60):02d}-{int(end//60):02d}:{int(end%60):02d}")
        start += seg
    return hints


def aggregate_knowledges(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not items:
        raise ValueError("无可聚合项")
    if len(items) == 1:
        return items[0]

    base = items[0]
    by_order: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    all_errors = []
    all_interventions = {}
    all_rules = []
    source_videos = []

    for item in items:
        source_videos.extend(item.get("meta", {}).get("source_videos", []))
        for s in item.get("steps", []):
            by_order[int(s.get("step_order", 0))].append(s)
        all_errors.extend(item.get("error_catalog", []))
        for iv in item.get("intervention_policies", []):
            all_interventions[iv["intervention_id"]] = iv
        all_rules.extend(item.get("runtime_rules", []))

    merged_steps = []
    for order in sorted(k for k in by_order.keys() if k > 0):
        group = by_order[order]
        name = Counter([g.get("step_name", f"步骤{order}") for g in group]).most_common(1)[0][0]
        chosen = group[0].copy()
        chosen["step_order"] = order
        chosen["step_name"] = name
        chosen["step_id"] = f"step_{order:02d}"
        merged_steps.append(chosen)

    dedup_errors = {}
    for i, e in enumerate(all_errors, start=1):
        key = (e.get("step_id"), e.get("error_pattern"), e.get("trigger_condition"))
        if key not in dedup_errors:
            new_e = dict(e)
            new_e["error_id"] = f"error_{i:02d}"
            dedup_errors[key] = new_e

    merged = {
        "task": base["task"],
        "steps": merged_steps if merged_steps else base["steps"],
        "error_catalog": list(dedup_errors.values()) if dedup_errors else base["error_catalog"],
        "intervention_policies": list(all_interventions.values()) if all_interventions else base["intervention_policies"],
        "runtime_rules": all_rules if all_rules else base["runtime_rules"],
        "meta": dict(base["meta"]),
    }
    merged["meta"]["source_videos"] = sorted(set(source_videos))
    merged["meta"]["aggregation_notes"] = "按 step_order 聚合，step_name 采用多数投票；错误按语义键去重。"
    return merged


def mock_knowledge(task_id: str, task_name: str, environment: str, video_id: str) -> Dict[str, Any]:
    return {
        "task": {
            "task_id": task_id,
            "task_name": task_name,
            "environment": environment,
            "required_objects": ["目标物体A", "目标物体B"],
            "safety_notes": ["注意保持动作稳定，避免碰撞。"],
            "difficulty_level": "beginner",
            "language": "zh-CN",
        },
        "steps": [
            {
                "step_id": "step_01",
                "step_order": 1,
                "step_name": "准备物体",
                "goal": "确认关键物体已就位",
                "preconditions": [],
                "expected_duration_sec": 20,
                "actions": [{"action_id": "action_01", "action_name": "将A和B放到工作区"}],
                "attention_cues": [{"cue_id": "cue_01", "cue_type": "object_presence", "description": "A、B均可见"}],
                "completion_criteria": ["A、B处于指定区域"],
                "next_step_hint": "开始执行核心动作",
                "common_mistake_warning": "不要遗漏任一关键物体",
                "if_error_then_intervention": "先补齐缺失物体再继续",
                "timeout_hint_sec": 25,
            },
            {
                "step_id": "step_02",
                "step_order": 2,
                "step_name": "执行核心动作",
                "goal": "按顺序完成核心动作",
                "preconditions": ["step_01完成"],
                "expected_duration_sec": 30,
                "actions": [{"action_id": "action_02", "action_name": "执行核心动作"}],
                "attention_cues": [{"cue_id": "cue_02", "cue_type": "pose_keypoint", "description": "关键姿态满足阈值"}],
                "completion_criteria": ["动作完整且顺序正确"],
                "next_step_hint": "进入收尾检查",
                "common_mistake_warning": "注意动作顺序",
                "if_error_then_intervention": "回到当前步起点并慢速重做",
                "timeout_hint_sec": 35,
            },
        ],
        "error_catalog": [
            {
                "error_id": "error_01",
                "step_id": "step_02",
                "error_pattern": "step_order_violation",
                "trigger_condition": "核心动作顺序不一致",
                "severity": "medium",
                "intervention_id": "intervention_01",
                "fallback_action": "回退到 step_02 重新执行",
            }
        ],
        "intervention_policies": [
            {
                "intervention_id": "intervention_01",
                "title": "顺序纠偏",
                "message": "先完成当前动作，再进入下一动作。",
                "channel": "text",
                "cooldown_sec": 5,
                "max_repeat": 3,
            }
        ],
        "runtime_rules": [
            {
                "rule_id": "rule_01",
                "rule_type": "sequence",
                "step_id": "step_02",
                "simple_rule": {"operator": "must_follow_order", "value": True},
                "on_trigger": "error_01",
            }
        ],
        "meta": {
            "source_model": "Qwen3.5-Omni",
            "source_videos": [video_id],
            "analysis_version": "v0.1.0",
            "created_at": utc_now_iso(),
            "review_status": "draft",
        },
    }


def generate_quality_report(knowledge: Dict[str, Any]) -> Dict[str, Any]:
    steps = knowledge.get("steps", [])
    errs = knowledge.get("error_catalog", [])
    rules = knowledge.get("runtime_rules", [])
    report = {
        "step_count": len(steps),
        "error_count": len(errs),
        "rule_count": len(rules),
        "checks": {
            "steps_gte_5": len(steps) >= 5,
            "has_error_catalog": len(errs) > 0,
            "has_runtime_rules": len(rules) > 0,
        },
    }
    report["checks"]["all_pass"] = all(report["checks"].values())
    return report


def analyze_one_video(
    item: Dict[str, Any],
    system_prompt: str,
    user_template: str,
    api_key: str,
    base_url: str,
    model: str,
    use_mock: bool,
    segment_minutes: int,
    request_timeout_sec: int,
    max_retries: int,
) -> Dict[str, Any]:
    task_id = item["task_id"]
    task_name = item["task_name"]
    environment = item.get("environment", "default_space")
    video_id = item["video_id"]
    ingest_video_uri = item.get("ingest_video_uri") or item.get("video_uri", "")
    ingest_video_path = item.get("ingest_video_path", "")
    duration_sec = float(item.get("ingest_duration_sec") or item.get("duration_sec", 0) or 0)

    actual_video_ref = str(ingest_video_uri or "")
    if ingest_video_path:
        local_video = (ROOT / ingest_video_path).resolve() if not Path(ingest_video_path).is_absolute() else Path(ingest_video_path)
        print(f"  [上传] 将本地视频上传到 DashScope OSS: {local_video.name} ...")
        actual_video_ref = upload_video_to_dashscope_oss(local_video, api_key, model)
        print(f"  [上传完成] {actual_video_ref}")
    elif not actual_video_ref:
        raise ValueError(f"{video_id} 缺少 ingest_video_path/ingest_video_uri，请先执行 video-ingest 生成 ingest manifest。")

    if use_mock:
        return mock_knowledge(task_id, task_name, environment, video_id)

    prompt = build_user_prompt(
        user_template,
        {
            "task_name": task_name,
            "environment": environment,
            "video_id": video_id,
            "duration_sec": duration_sec,
            "fps": item.get("fps", "unknown"),
            "analysis_mode": "full_video",
            "segment_hint": "N/A",
        },
    )
    try:
        return call_dashscope_chat(
            api_key,
            base_url,
            model,
            system_prompt,
            prompt,
            actual_video_ref,
            request_timeout_sec=request_timeout_sec,
            max_retries=max_retries,
        )
    except Exception:
        # 降级策略：按时间段二次尝试，然后聚合
        hints = make_segment_hints(duration_sec, segment_minutes)
        if not hints:
            raise
        partials = []
        for h in hints:
            seg_prompt = build_user_prompt(
                user_template,
                {
                    "task_name": task_name,
                    "environment": environment,
                    "video_id": video_id,
                    "duration_sec": duration_sec,
                    "fps": item.get("fps", "unknown"),
                    "analysis_mode": "segmented",
                    "segment_hint": h,
                },
            )
            partial = call_dashscope_chat(
                api_key,
                base_url,
                model,
                system_prompt,
                seg_prompt,
                actual_video_ref,
                request_timeout_sec=request_timeout_sec,
                max_retries=max_retries,
            )
            partials.append(partial)
        return aggregate_knowledges(partials)


def main() -> None:
    parser = argparse.ArgumentParser(description="Step-2: 示教视频分析与结构化知识输出")
    parser.add_argument("--manifest", default=str(TA_MANIFEST_PATH), help="输入 manifest 路径")
    parser.add_argument("--output-dir", default=str(TA_OUTPUT_DIR), help="输出目录")
    parser.add_argument("--model", default=TA_MODEL, help="模型名")
    parser.add_argument("--base-url", default=TA_BASE_URL, help="API Base URL")
    parser.add_argument("--segment-minutes", type=int, default=TA_SEGMENT_MINUTES, help="降级分段分钟数")
    parser.add_argument("--max-retries", type=int, default=TA_MAX_RETRIES, help="API 重试次数")
    parser.add_argument("--request-timeout-sec", type=int, default=TA_REQUEST_TIMEOUT_SEC, help="API 超时秒数")
    parser.add_argument("--mock", action="store_true", help="使用 mock 数据，不调用 API")
    args = parser.parse_args()

    manifest = read_json(Path(args.manifest))
    schema = read_json(TA_SCHEMA_PATH)
    _ = schema  # schema 文件作为契约留档；此脚本执行轻量修复校验

    system_prompt = read_text(TA_PROMPT_DIR / "analysis_system_prompt.md")
    user_template = read_text(TA_PROMPT_DIR / "analysis_user_prompt_template.md")

    api_key = item = None
    if not args.mock:
        api_key = get_api_key()
        if not api_key:
            env_names = ", ".join(TA_API_KEY_ENV_NAMES)
            raise RuntimeError(f"未找到 API Key。请在 .env 或环境变量中设置: {env_names}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    knowledges = []
    for item in manifest.get("demos", []):
        if not validate_video_id(item["video_id"]):
            raise ValueError(f"video_id 不符合命名规范: {item['video_id']}")
        analyzed = analyze_one_video(
            item=item,
            system_prompt=system_prompt,
            user_template=user_template,
            api_key=api_key or "",
            base_url=args.base_url,
            model=args.model,
            use_mock=args.mock,
            segment_minutes=args.segment_minutes,
            request_timeout_sec=args.request_timeout_sec,
            max_retries=args.max_retries,
        )
        repaired = repair_and_validate(
            analyzed,
            task_defaults={
                "task_id": item["task_id"],
                "task_name": item["task_name"],
                "environment": item.get("environment", "default_space"),
            },
        )
        repaired["meta"]["source_videos"] = sorted(set(repaired["meta"].get("source_videos", []) + [item["video_id"]]))
        if item.get("ingest_video_uri") or item.get("video_uri"):
            repaired["meta"]["source_uri"] = item.get("ingest_video_uri") or item.get("video_uri")
        if item.get("ingest_video_path"):
            repaired["meta"]["source_path"] = item["ingest_video_path"]
        if item.get("source_video_path"):
            repaired["meta"]["raw_source_path"] = item["source_video_path"]
        repaired["meta"]["created_at"] = utc_now_iso()
        repaired["meta"]["review_status"] = "draft"
        write_json(output_dir / f"{item['video_id']}.json", repaired)
        knowledges.append(repaired)

    aggregated = aggregate_knowledges(knowledges)
    aggregated["meta"]["source_model"] = "Qwen3.5-Omni"
    aggregated["meta"]["created_at"] = utc_now_iso()
    aggregated["meta"]["review_status"] = "needs_human_review"
    write_json(output_dir / "structured_teaching_knowledge.json", aggregated)
    write_json(output_dir / "quality_report.json", generate_quality_report(aggregated))

    print(f"[OK] 处理示教数: {len(knowledges)}")
    print(f"[OK] 输出: {output_dir / 'structured_teaching_knowledge.json'}")
    print(f"[OK] 质量报告: {output_dir / 'quality_report.json'}")


if __name__ == "__main__":
    main()
