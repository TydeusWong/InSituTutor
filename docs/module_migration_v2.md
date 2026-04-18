# Module Migration Plan v2

## 1. Mapping

- `services/teaching-analysis` -> `services/teaching-segmentation` + `services/strategy-builder`
- `services/knowledge-builder` -> `services/strategy-builder` + `services/error-memory`
- `services/realtime-coach` + `services/intervention-engine` -> `services/realtime-orchestrator`

## 2. Compatibility Adapters

Planned adapters:

- `legacy_structured_to_v2_strategy`
- `legacy_prompt_knowledge_to_v2_strategy`

## 3. Deprecation Stages

1. Stage A: dual-write artifacts
2. Stage B: v2 read-preferred fallback legacy
3. Stage C: archive legacy writers
