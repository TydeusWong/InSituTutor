# InSituTutor Architecture v2

## 1. Scope Freeze

This version freezes the new pipeline:

1. Teacher demo video ingest and compression
2. Omni section segmentation and atomic unit extraction
3. Strategy build for step/error and detector plans
4. Criteria training with small models
5. High real-time online orchestration and correction loop

Atomic unit classification is strictly closed-set:

- `step`
- `error`
- `not_related`

`not_related` units are ignored in downstream strategy output.

## 2. Stage Boundaries

### Offline stage (latency tolerant)

- Input: demo videos and audio
- Output: `teaching_strategy_v2.json`, `criteria_bundle_v2`
- Typical latency: minutes to hours

### Online stage (high real-time)

- Input: streaming realtime video
- Output: step progression, correction events, section completion
- Constraint: low-latency event flow and bounded fallback timeouts

## 3. Service Topology

- `services/video-ingest`: keep existing compression and ingest manifest generation
- `services/teaching-segmentation`: Omni-based section and atomic-unit parsing
- `services/strategy-builder`: convert parsed units into executable teaching strategy
- `services/criteria-trainer`: train/select small models and compile detector criteria
- `services/realtime-orchestrator`: online state machine and event scheduling
- `services/error-memory`: persist unknown/new errors and promotion workflow
- `apps/student-web`: event-driven student guidance UI

## 4. Runtime Behavior Freeze

During online execution:

1. If current action matches current-step criteria: advance to next step
2. If action matches any error criteria in current section: enter correction
3. If neither matches: call Omni correction path and write to error memory
4. A section can finish only when `expected_section_state` is satisfied

## 5. Legacy Mapping and Compatibility Layer

- legacy `teaching-analysis` -> v2 `teaching-segmentation` + `strategy-builder`
- legacy `knowledge-builder` -> v2 `strategy-builder` + `error-memory`
- legacy `realtime-coach` + `intervention-engine` -> v2 `realtime-orchestrator`

Compatibility mode:

- Keep legacy outputs readable by introducing adapter transformers:
  - `legacy_structured_to_v2_strategy`
  - `legacy_prompt_knowledge_to_v2_strategy`

Deprecation plan:

- Phase A: dual-write (legacy + v2 artifacts)
- Phase B: v2 read-preferred with legacy fallback
- Phase C: legacy write-off and archive

## 6. Event Bus Contract (transport-agnostic)

Event bus is defined as interface-first:

- Required operations: `publish`, `subscribe`, `ack`, `nack`, `health`
- Default implementation priority:
  1. In-memory local queue for development
  2. Redis streams for multi-process deployment
- Message envelope fields:
  - `event_id`, `event_type`, `session_id`, `task_id`, `timestamp_ms`, `payload`, `trace`

See `docs/realtime_contract_v2.md` for detailed payload schema.
