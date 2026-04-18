# InSituTutor Realtime Contract v2

## 1. Event Envelope

All realtime events use a transport-agnostic envelope.

```json
{
  "event_id": "evt_123",
  "event_type": "step_evaluated",
  "session_id": "sess_001",
  "task_id": "task_demo",
  "timestamp_ms": 1776508800123,
  "trace": {"source": "realtime-orchestrator", "version": "v2"},
  "payload": {}
}
```

## 2. Event Types

- `section_entered`
- `step_prompted`
- `step_evaluated`
- `step_passed`
- `error_hit`
- `unknown_detected`
- `omni_correction_issued`
- `section_validated`
- `section_completed`

## 3. Core Payload Contracts

### 3.1 step_evaluated

```json
{
  "section_id": "section_01",
  "step_id": "step_01_02",
  "score": 0.79,
  "threshold": 0.85,
  "matched": false,
  "evidence": ["iou=0.28", "grasp=true", "above=false"],
  "latency_ms": 126
}
```

### 3.2 error_hit

```json
{
  "section_id": "section_01",
  "error_id": "error_01_01",
  "trigger_signature": "object_misplacement:black_can_not_above_tape",
  "correction_prompt": "黑色罐子位置不对，请放到透明胶带正上方。",
  "latency_ms": 131
}
```

### 3.3 unknown_detected

```json
{
  "section_id": "section_01",
  "step_id": "step_01_02",
  "reason": "no_step_or_error_match",
  "window": {"start_sec": 51.2, "end_sec": 57.8},
  "latency_ms": 138
}
```

## 4. Online State Machine Freeze

### Section state

`SECTION_ENTERED -> STEP_RUNNING -> SECTION_VALIDATING -> SECTION_DONE`

### Step state

`WAITING -> RUNNING -> PASSED | ERROR_HIT | UNKNOWN`

## 5. Required Runtime Decisions

1. `PASSED`: publish `step_passed` and move to next step
2. `ERROR_HIT`: publish correction and hold in current step
3. `UNKNOWN`: call Omni correction, publish `omni_correction_issued`, write error memory
4. Section exit requires `expected_section_state == true`

## 6. Event Bus Interface

`services/common/event_bus.py` defines the interface:

- `publish(topic, event)`
- `subscribe(topic, handler)`
- `ack(event_id)`
- `nack(event_id, reason)`
- `health()`

Implementation profile:

- `InMemoryEventBus`: default for local development
- `RedisEventBus`: pluggable production adapter
