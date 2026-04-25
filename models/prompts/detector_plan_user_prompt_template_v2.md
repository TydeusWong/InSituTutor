Build detector plan for one teaching slice using the payload below.

Input payload (JSON):
{{payload_json}}

Coordinate convention for this task:
- Video perspective: top-down (bird's-eye) view.
- Use a 2D image-plane coordinate system.
- Origin: top-left corner of frame.
- x-axis: horizontal axis in the image plane.
- y-axis: vertical axis in the image plane.
- There is no separate depth/vertical axis (no z-axis) in judgement code.
- x increases left -> right; y increases top -> bottom.
- Coordinates are normalized to [0, 1].

Additional context for this task:
- `transcript_full`: full ASR transcript segments for the entire video, each with `start_sec`, `end_sec`, `speaker`, `text`.
- Use transcript only as auxiliary context to resolve intent/target names; output judgement conditions must remain executable visual rules.

Return JSON only.
`judgement_conditions` must be directly executable code snippets that reference predefined condition primitives.

Output JSON structure (must follow exactly):

```json
{
  "slice_id": "string",
  "slice_type": "step|error",
  "models_required": ["string"],
  "model_selection": [
    {
      "model_id": "string",
      "detect_targets": ["string"]
    }
  ],
  "execution_plan": {
    "order": ["string"],
    "parallel_groups": [["string"]]
  },
  "judgement_conditions": [
    {
      "condition_id": "string",
      "when": "step|error|both",
      "code": "string"
    }
  ]
}
```

Field constraints:
- Do not add extra top-level keys.
- `model_selection` must be aggregated by `model_id` (one entry per model_id).
- `detect_targets` must be array type and deduplicated.
- For `mediapipe-hand-landmarker`, `detect_targets` must be exactly `["hand"]`.
- If step intent (from prompt/focus_points/common_mistakes/transcript) contains hand-action verbs such as remove/open/close/unscrew/screw/twist/press/push/pull/grab/pick/place/hold, you must include `mediapipe-hand-landmarker` when hand-specific evidence is required.
- Minimal-model override for hand:
  - If this step can be decided by final object state/relations only (e.g., place/move/align/stack result), do NOT include `mediapipe-hand-landmarker`; use YOLO-only.
  - Include hand model only when the judgement must verify hand-specific evidence (contact/press/grab/unscrew/twist), not just final object layout.
- `grounding-dino` is NOT available. Never output `grounding-dino` in `models_required`, `model_selection`, or `execution_plan`.
- YOLO has already been trained before this detector-plan step and can detect every entity in payload field `allowed_yolo_detect_targets`.
- For object localization, use `yolo`.
- For `yolo`, `detect_targets` must be chosen only from payload field `allowed_yolo_detect_targets`.
- For `yolo`, `detect_targets` must reuse exact strings from `allowed_yolo_detect_targets` as-is; do not rewrite, translate, abbreviate, normalize, or replace with synonyms.
- Once you decide to use `yolo`, you must select at least one target in `detect_targets`.
- `yolo` target names must be strictly character-exact with `allowed_yolo_detect_targets`; never use a name outside the list.
- Apply "minimal entities only": if the judgement can be decided with fewer entities, do not add extra entities.
- `detect_targets` must contain only real detectable entities; do NOT include spatial anchors/regions such as `workspace center`, `workspace_right`, `workspace_right_half`, etc.
- `judgement_conditions[*].code` must be executable expression-style code.
- `judgement_conditions[*].code` must only call predefined primitive names from `condition_primitives`.
- Some `condition_primitives` groups may use legacy names such as `grounding_dino`; this does not mean Grounding DINO is available. Object bbox primitives are backed by YOLO detections at runtime.
- `judgement_conditions[*].code` must NOT use built-in/helper functions such as `any`, `all`, `len`, `sum`, `min`, `max`, `sorted`, `map`, `filter`.
- `judgement_conditions[*].code` may only use: primitive calls + comparison operators + boolean operators (`and` / `or` / `not`) + parentheses.
- For workspace-relative rules, do not model workspace regions as objects in `rel_x`/`rel_y`; use absolute-position primitives/coordinates instead.
