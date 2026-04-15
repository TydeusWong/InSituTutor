import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from prompt_engine import PromptEngine, load_json


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "data" / "processed" / "knowledge" / "prompt_knowledge.json"


def simulate_events() -> List[Dict[str, Any]]:
    """
    覆盖 4 类提示场景：
    - 预告提示
    - 步骤提示（含卡住/超时）
    - 即时纠偏
    - 安全提醒
    """
    return [
        {"timestamp_sec": 1.0, "step_order": 1, "step_status": "not_started", "elapsed_in_step_sec": 0},
        {"timestamp_sec": 3.0, "step_order": 1, "step_status": "in_progress", "elapsed_in_step_sec": 2},
        {"timestamp_sec": 5.0, "step_order": 1, "step_status": "in_progress", "elapsed_in_step_sec": 20},
        {"timestamp_sec": 7.0, "step_order": 1, "step_status": "stuck", "elapsed_in_step_sec": 30},
        {
            "timestamp_sec": 9.0,
            "step_order": 1,
            "step_status": "deviation",
            "elapsed_in_step_sec": 10,
            "error_type": "step_order_violation",
        },
        {
            "timestamp_sec": 12.0,
            "step_order": 1,
            "step_status": "deviation",
            "elapsed_in_step_sec": 12,
            "error_type": "step_order_violation",
        },
        {"timestamp_sec": 20.0, "step_order": 2, "step_status": "in_progress", "elapsed_in_step_sec": 3, "high_risk": True},
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="运行提示引擎演示")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="输入 prompt_knowledge.json")
    args = parser.parse_args()

    knowledge = load_json(args.input)
    engine = PromptEngine(knowledge=knowledge)

    outputs = []
    for event in simulate_events():
        emitted = engine.on_event(event)
        if emitted:
            outputs.append({"event": event, "prompt": emitted})

    print(json.dumps(outputs, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

