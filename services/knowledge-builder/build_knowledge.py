import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "data" / "processed" / "analysis" / "structured_teaching_knowledge.json"
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "knowledge" / "prompt_knowledge.json"


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def ensure_step_text_fields(step: Dict[str, Any]) -> Dict[str, Any]:
    s = dict(step)
    name = str(s.get("step_name", "当前步骤")).strip()
    s.setdefault("next_step_hint", f"下一步将进入{name}，请提前准备。")
    s.setdefault("common_mistake_warning", "请按提示完成关键动作，避免漏做或顺序错误。")
    s.setdefault("if_error_then_intervention", "请回到当前步骤起点，慢速重做一次关键动作。")
    s.setdefault("timeout_hint_sec", 25)
    return s


def normalize_error_catalog(
    errors_and_interventions: List[Dict[str, Any]],
    step_orders: List[int],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not step_orders:
        step_orders = [1]
    for idx, item in enumerate(errors_and_interventions, start=1):
        step_order = step_orders[(idx - 1) % len(step_orders)]
        error_type = str(item.get("error_type", f"error_{idx:02d}")).strip() or f"error_{idx:02d}"
        out.append(
            {
                "error_id": f"err_{idx:03d}",
                "step_id": f"step_{step_order:02d}",
                "error_pattern": error_type,
                "trigger_condition": str(item.get("trigger_condition", "")).strip(),
                "severity": "medium",
                "intervention_id": f"iv_{error_type}",
                "fallback_action": "回到当前步骤关键动作并重试",
            }
        )
    return out


def normalize_intervention_policies(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if items:
        out = []
        for it in items:
            out.append(
                {
                    "intervention_id": str(it.get("intervention_id", "intervention_default")),
                    "title": str(it.get("title", "默认纠偏提示")),
                    "message": str(it.get("message", "请回到当前步骤，按提示重新执行。")),
                    "channel": str(it.get("channel", "text")),
                    "cooldown_sec": float(it.get("cooldown_sec", 5)),
                    "max_repeat": int(it.get("max_repeat", 3)),
                }
            )
        return out
    return [
        {
            "intervention_id": "intervention_default",
            "title": "默认纠偏提示",
            "message": "请回到当前步骤，按提示重新执行。",
            "channel": "text",
            "cooldown_sec": 5,
            "max_repeat": 3,
        }
    ]


def build_prompt_knowledge(source: Dict[str, Any]) -> Dict[str, Any]:
    steps = source.get("steps", [])
    normalized_steps = [ensure_step_text_fields(s) for s in steps]
    step_orders = [int(s.get("step_order", i + 1)) for i, s in enumerate(normalized_steps)]

    prompt_levels = {
        "pre_notice": {
            "desc": "步骤开始前的预告提示",
            "when": "step_status=not_started",
            "priority": 1,
        },
        "step_hint": {
            "desc": "步骤执行中的常规提示",
            "when": "step_status=in_progress 或 stuck/timeout",
            "priority": 2,
        },
        "instant_correction": {
            "desc": "命中偏差规则后的即时纠偏",
            "when": "step_status=deviation 或 error_type 命中",
            "priority": 3,
        },
        "safety_alert": {
            "desc": "高风险动作的安全提醒，必要时暂停",
            "when": "high_risk=true",
            "priority": 4,
        },
    }

    ban_rules = [
        "禁止模糊表达（如：随便、差不多）",
        "禁止不可执行表达（没有动作指令）",
        "禁止过度责备语气",
    ]

    return {
        "meta": {
            **source.get("meta", {}),
            "knowledge_type": "prompt_knowledge",
        },
        "task": source.get("task", {}),
        "prompt_levels": prompt_levels,
        "style_rules": {
            "tone": "简短、可执行、先肯定后纠偏",
            "banned_phrases": ban_rules,
            "output_mode": "text_only",
        },
        "steps": normalized_steps,
        "errors_and_interventions": source.get("errors_and_interventions", []),
        "runtime_rules": source.get("runtime_rules", []),
        "error_catalog": normalize_error_catalog(source.get("errors_and_interventions", []), step_orders),
        "intervention_policies": normalize_intervention_policies(source.get("intervention_policies", [])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="构建教学提示知识文件（第三部分）")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="输入 structured_teaching_knowledge.json")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="输出 prompt_knowledge.json")
    args = parser.parse_args()

    source = read_json(args.input)
    built = build_prompt_knowledge(source)
    write_json(args.output, built)
    print(f"[OK] 输出: {args.output}")


if __name__ == "__main__":
    main()

