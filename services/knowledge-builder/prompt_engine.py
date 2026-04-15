import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


PRIORITY = {
    "safety_alert": 4,
    "instant_correction": 3,
    "step_hint": 2,
    "pre_notice": 1,
}

BANNED_PHRASES = [
    "你怎么又",
    "太差了",
    "随便做",
    "自己看着办",
    "你不行",
]

ACTION_KEYWORDS = ("请", "保持", "按", "看", "移动", "等待", "重新", "确认", "回到", "开始", "完成")


@dataclass
class PromptPolicy:
    cooldown_sec: float = 5.0
    max_repeat: int = 3
    channel: str = "text"


class PromptEngine:
    """
    MVP 提示引擎：
    1) 提示层级：预告 / 步骤 / 即时纠偏 / 安全提醒
    2) 触发维度：步骤状态 / 时间超时 / 错误命中
    3) 防轰炸：冷却时间 + 最大重复次数
    """

    def __init__(self, knowledge: Dict[str, Any]):
        self.knowledge = knowledge
        self.steps = sorted(knowledge.get("steps", []), key=lambda s: s.get("step_order", 0))
        self.step_map = {int(s.get("step_order", 0)): s for s in self.steps}
        self.error_map = {
            str(e.get("error_type", "")): str(e.get("prompt_text", "")).strip()
            for e in knowledge.get("errors_and_interventions", [])
            if e.get("error_type")
        }
        self.safety_notes = knowledge.get("task", {}).get("safety_notes", []) or []
        self.default_policy = self._build_policy(knowledge.get("intervention_policies", []))
        self.emit_state: Dict[str, Dict[str, float]] = {}

    def _build_policy(self, intervention_policies: List[Dict[str, Any]]) -> PromptPolicy:
        if not intervention_policies:
            return PromptPolicy()
        base = intervention_policies[0]
        cooldown = float(base.get("cooldown_sec", 5))
        max_repeat = int(base.get("max_repeat", 3))
        channel = str(base.get("channel", "text"))
        return PromptPolicy(cooldown_sec=max(0.0, cooldown), max_repeat=max(1, max_repeat), channel=channel)

    def _style_text(self, text: str, level: str) -> str:
        clean = " ".join((text or "").split())
        for banned in BANNED_PHRASES:
            clean = clean.replace(banned, "")
        clean = clean.strip(" ，。")
        if not clean:
            clean = "请按照当前步骤继续操作"
        if level in ("instant_correction", "safety_alert") and not clean.startswith(("做得", "很好", "不错", "请")):
            clean = f"做得不错，{clean}"
        if not any(k in clean for k in ACTION_KEYWORDS):
            clean = f"{clean}，请按当前步骤继续。"
        if clean[-1] not in ("。", "！", "？"):
            clean += "。"
        return clean

    def _allow_emit(self, key: str, now_sec: float, policy: PromptPolicy) -> bool:
        state = self.emit_state.get(key)
        if state is None:
            self.emit_state[key] = {"last_ts": now_sec, "count": 1}
            return True
        delta = now_sec - state["last_ts"]
        if delta < policy.cooldown_sec:
            return False
        if state["count"] >= policy.max_repeat:
            return False
        state["last_ts"] = now_sec
        state["count"] += 1
        return True

    def _build_step_hint_text(self, step: Dict[str, Any], is_stuck: bool) -> str:
        if is_stuck:
            warning = (step.get("common_mistake_warning") or "").strip()
            recover = (step.get("if_error_then_intervention") or "").strip()
            if warning and recover:
                return f"{warning}。建议：{recover}"
            if warning:
                return warning
            if recover:
                return recover
            return "当前步骤有点卡住了，请慢一点按提示继续。"

        action_desc = ""
        if step.get("actions"):
            action_desc = str(step["actions"][0].get("description", "")).strip()
        cue = ""
        if step.get("attention_cues"):
            cue = str(step["attention_cues"][0]).strip()
        if action_desc and cue:
            return f"{action_desc} 观察重点：{cue}"
        if action_desc:
            return action_desc
        if cue:
            return f"请关注：{cue}"
        return f"请执行当前步骤：{step.get('step_name', '未命名步骤')}"

    def _pick_prompt(self, event: Dict[str, Any]) -> Optional[Tuple[str, str, str]]:
        step_order = int(event.get("step_order", 0))
        step_status = str(event.get("step_status", "")).strip()
        elapsed_sec = float(event.get("elapsed_in_step_sec", 0))
        error_type = str(event.get("error_type", "")).strip()
        high_risk = bool(event.get("high_risk", False))
        step = self.step_map.get(step_order, {})

        timeout_hint = float(step.get("timeout_hint_sec", 30) or 30)

        if high_risk:
            text = self.safety_notes[0] if self.safety_notes else "检测到高风险偏差，请立即停止当前动作并等待老师协助。"
            return ("safety_alert", f"safety_{step_order}", text)

        if step_status == "deviation" or error_type:
            text = self.error_map.get(error_type) or step.get("if_error_then_intervention") or step.get("common_mistake_warning")
            if not text:
                text = "动作出现偏差，请回到当前步骤起点重新执行。"
            key = f"correction_{step_order}_{error_type or 'generic'}"
            return ("instant_correction", key, str(text))

        if step_status == "stuck" or (step_status == "in_progress" and elapsed_sec > timeout_hint):
            text = self._build_step_hint_text(step, is_stuck=True)
            return ("step_hint", f"stuck_{step_order}", text)

        if step_status == "in_progress":
            text = self._build_step_hint_text(step, is_stuck=False)
            return ("step_hint", f"step_{step_order}", text)

        if step_status == "not_started":
            text = step.get("next_step_hint") or f"即将开始：{step.get('step_name', '下一步')}，请做好准备。"
            return ("pre_notice", f"pre_{step_order}", str(text))

        return None

    def on_event(self, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        输入事件示例:
        {
          "timestamp_sec": 12.5,
          "step_order": 3,
          "step_status": "in_progress",  # not_started / in_progress / stuck / deviation
          "elapsed_in_step_sec": 4.2,
          "error_type": "step_order_violation",
          "high_risk": false
        }
        """
        ts = float(event.get("timestamp_sec", 0))
        picked = self._pick_prompt(event)
        if not picked:
            return None
        level, key, raw_text = picked
        text = self._style_text(raw_text, level)
        if not self._allow_emit(key=key, now_sec=ts, policy=self.default_policy):
            return None
        step_order = int(event.get("step_order", 0))
        step = self.step_map.get(step_order, {})
        return {
            "level": level,
            "priority": PRIORITY[level],
            "channel": self.default_policy.channel,
            "step_order": step_order,
            "step_name": step.get("step_name", ""),
            "text": text,
            "reason": {
                "step_status": event.get("step_status", ""),
                "error_type": event.get("error_type", ""),
                "high_risk": bool(event.get("high_risk", False)),
            },
        }


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

