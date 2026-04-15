# InSituTutor Data Schema (MVP)

## 1. 设计目标

- 为示教分析、知识构建、实时提示、偏差纠错提供统一字段。
- 字段优先可解释、可复盘、可扩展。
- 与 `Qwen3.5-Omni` 输出对齐，便于 JSON schema 校验。

## 2. 顶层结构

```json
{
  "task": {},
  "steps": [],
  "error_catalog": [],
  "intervention_policies": [],
  "runtime_rules": [],
  "meta": {}
}
```

## 3. 任务层（task）

```json
{
  "task_id": "task_bed_making_v1",
  "task_name": "铺床单",
  "environment": "smart_room_a",
  "required_objects": ["床", "床单", "枕头"],
  "safety_notes": ["避免踩踏床沿导致跌倒"],
  "difficulty_level": "beginner",
  "language": "zh-CN"
}
```

字段说明：

- `task_id`: 任务唯一标识
- `task_name`: 任务名称
- `environment`: 场景标识
- `required_objects`: 必需物体清单
- `safety_notes`: 安全提醒
- `difficulty_level`: 提示强度分层依据
- `language`: 提示语言

## 4. 步骤层（steps）

```json
{
  "step_id": "step_01",
  "step_order": 1,
  "step_name": "展开床单",
  "goal": "将床单完全展开在床面上",
  "preconditions": ["床单已拿起"],
  "expected_duration_sec": 20,
  "actions": [],
  "attention_cues": [],
  "completion_criteria": [],
  "next_step_hint": "将床单四角分别对齐床角",
  "common_mistake_warning": "注意不要让床单扭转",
  "if_error_then_intervention": "先固定近侧两个角，再调整远侧",
  "timeout_hint_sec": 25
}
```

字段说明：

- `step_order`: 顺序控制核心字段
- `completion_criteria`: 步骤完成判定条件
- `next_step_hint/common_mistake_warning/if_error_then_intervention`: 学生端提示核心文案
- `timeout_hint_sec`: 超时提醒阈值

## 5. 动作层（actions）

```json
{
  "action_id": "action_01",
  "action_name": "双手拉开床单",
  "body_part": ["left_hand", "right_hand"],
  "object": "床单",
  "pose_hint": "双臂向两侧展开",
  "relative_position": "双手位于床单两边边缘",
  "success_criteria": ["床单宽度方向展开", "边缘可见且不卷曲"]
}
```

## 6. 观测点（attention_cues）

```json
{
  "cue_id": "cue_01",
  "cue_type": "object_state",
  "description": "床单四角中至少两角可见",
  "source": "vision",
  "confidence_threshold": 0.6
}
```

`cue_type` 建议值：

- `pose_keypoint`
- `object_presence`
- `object_state`
- `spatial_relation`

## 7. 错误目录（error_catalog）

```json
{
  "error_id": "error_01",
  "step_id": "step_01",
  "error_pattern": "step_order_violation",
  "trigger_condition": "在 step_01 未完成前进入 step_03 行为",
  "severity": "medium",
  "intervention_id": "intervention_01",
  "fallback_action": "回到 step_01 重新展开床单"
}
```

`severity` 建议值：`low | medium | high`

## 8. 纠偏策略（intervention_policies）

```json
{
  "intervention_id": "intervention_01",
  "title": "顺序纠偏提示",
  "message": "先完成当前步骤：请先把床单铺平，再进行下一步。",
  "channel": "text",
  "cooldown_sec": 5,
  "max_repeat": 3,
  "escalation": {
    "after_repeats": 3,
    "message": "你已连续偏离步骤，建议回看当前步骤要点。"
  }
}
```

## 9. 在线规则（runtime_rules）

```json
{
  "rule_id": "rule_01",
  "rule_type": "sequence",
  "step_id": "step_01",
  "simple_rule": {
    "operator": "must_follow_order",
    "value": true
  },
  "on_trigger": "error_01"
}
```

`rule_type` 建议值：

- `sequence`（顺序）
- `presence`（对象存在性）
- `pose`（姿态粗判）
- `spatial`（相对位置）
- `timeout`（超时）

## 10. 元数据（meta）

```json
{
  "source_model": "Qwen3.5-Omni",
  "source_videos": ["demo_2026-04-13_roomA_teacher01.mp4"],
  "analysis_version": "v0.1.0",
  "created_at": "2026-04-13T10:00:00Z",
  "review_status": "draft"
}
```

## 11. 运行事件（建议日志结构）

```json
{
  "session_id": "session_001",
  "timestamp": "2026-04-13T10:11:22Z",
  "event_type": "deviation_detected",
  "task_id": "task_bed_making_v1",
  "step_id": "step_01",
  "rule_id": "rule_01",
  "error_id": "error_01",
  "intervention_id": "intervention_01",
  "resolved": false
}
```

## 12. Schema 校验建议

- 必填字段：`task.task_id`, `steps[].step_id`, `steps[].step_order`
- `step_order` 必须唯一且从 1 递增
- `error_catalog[].step_id` 必须引用存在的步骤
- `intervention_id` 必须能在 `intervention_policies` 中找到
- `runtime_rules[].on_trigger` 必须能在 `error_catalog` 中找到
