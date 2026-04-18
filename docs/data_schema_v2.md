# InSituTutor Data Schema v2

## 1. Top-level Strategy Object

```json
{
  "task": {},
  "sections": [],
  "detector_registry": [],
  "meta": {}
}
```

## 2. Section

```json
{
  "section_id": "section_01",
  "section_name": "物体摆放",
  "section_goal": "完成核心物体的相对位置摆放",
  "expected_section_state": {
    "object_relations": [
      {"subject": "black_can", "predicate": "above", "object": "transparent_tape"}
    ]
  },
  "time_range": {"start_sec": 12.2, "end_sec": 45.8},
  "atomic_units": [],
  "steps": [],
  "errors": []
}
```

Required fields:

- `section_id`, `section_name`, `section_goal`, `expected_section_state`, `time_range`

## 3. Atomic Unit

```json
{
  "unit_id": "unit_01_003",
  "time_range": {"start_sec": 20.5, "end_sec": 23.9},
  "evidence": {
    "audio": ["把黑色罐子放在透明胶带上面"],
    "vision": ["hand_grasp(black_can)", "iou(black_can,tape)>0.4"]
  },
  "class": "step"
}
```

Constraints:

- `class` must be one of `step | error | not_related`

## 4. Step

```json
{
  "step_id": "step_01_02",
  "unit_refs": ["unit_01_003"],
  "prompt": "请把黑色罐子放到透明胶带上方。",
  "focus_points": ["先抓稳黑色罐子", "保持目标对齐"],
  "common_mistakes": ["放在胶带旁边而非上方"],
  "expected_post_state": {
    "relations": [
      {"subject": "black_can", "predicate": "above", "object": "transparent_tape"}
    ]
  },
  "detector_plan_ref": "dp_step_01_02"
}
```

## 5. Error

```json
{
  "error_id": "error_01_01",
  "unit_refs": ["unit_01_004"],
  "trigger_signature": "object_misplacement:black_can_not_above_tape",
  "correction_prompt": "黑色罐子位置不对，请放到透明胶带正上方。",
  "recovery_actions": ["抓取黑色罐子", "移动到胶带上方", "松手确认"],
  "detector_plan_ref": "dp_error_01_01"
}
```

## 6. Detector Plan (programmatic)

```json
{
  "plan_id": "dp_step_01_02",
  "models_required": [
    "yolo:black_can:v1",
    "yolo:transparent_tape:v1",
    "mediapipe_hand:v1"
  ],
  "features": [
    "bbox(black_can)",
    "bbox(transparent_tape)",
    "hand_state(grasp)"
  ],
  "constraints": [
    "grasp(black_can)==true",
    "above(black_can,transparent_tape)==true",
    "iou(black_can,transparent_tape)>=0.35"
  ],
  "score_fn": "0.4*I(grasp)+0.3*I(above)+0.3*clamp(iou/0.35,0,1)",
  "pass_threshold": 0.85
}
```

## 7. Runtime State

```json
{
  "session_id": "sess_001",
  "current_section": "section_01",
  "current_step": "step_01_02",
  "active_error": null,
  "last_passed_step": "step_01_01",
  "latency_ms": 142
}
```

## 8. Error Memory

```json
{
  "memory_id": "mem_00021",
  "discovered_error": {
    "signature": "unknown_object_relation_mismatch",
    "section_id": "section_01",
    "step_id": "step_01_02"
  },
  "context_clip": {
    "video_uri": "sessions/sess_001/clip_021.mp4",
    "start_sec": 51.2,
    "end_sec": 57.8
  },
  "omni_fix": {
    "prompt": "请先将黑色罐子拿起并放到透明胶带的上方中心位置。",
    "reasoning_summary": "目标关系未满足"
  },
  "review_status": "pending",
  "promoted_to_catalog": false
}
```

Allowed `review_status` values:

- `pending`
- `approved`
- `rejected`

## 9. Schema Files

Machine-readable schemas are provided:

- `models/schemas/teaching_strategy_v2.schema.json`
- `models/schemas/realtime_state_v2.schema.json`
- `models/schemas/error_memory_v2.schema.json`
