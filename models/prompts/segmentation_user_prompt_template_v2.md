Demo metadata (context only, not direct evidence):
- task_id: {{task_id}}
- video_id: {{video_id}}
- duration_sec: {{duration_sec}}
- fps: {{fps}}
- resolution: {{resolution}}
- video_quality: {{video_quality}}
- video_bitrate_kbps: {{video_bitrate_kbps}}
- source_audio_quality: {{source_audio_quality}}
- scene_tags: {{scene_tags}}

Instructions:
1) Parse the full video into ordered, non-overlapping sections.
   - Keep section granularity practical: split by objective/state transitions, not only by coarse time blocks.
2) For each section, produce ordered, non-overlapping atomic units fully contained in that section.
   - Each atomic unit must contain exactly one actionable verb/intent.
   - If text/action implies sequence via connectors (`and`, `then`, `after that`, `并`, `然后`, `再`, `再将`), split into separate units.
   - Specifically, split "manipulate object" and "place/move object to location" into two units when they are sequential.
   - Mandatory split pattern: if one sentence implies "remove/unscrew" then "place/move to location", output two separate `step` units.
3) Classify each atomic unit strictly as step, error, or not_related.
4) Add top-level `video_overview` with:
   - summary
   - camera_view (single perspective label for this video)
   - scene_entities (one merged list of all objects + environment markers used in this video; exclude body parts/person descriptors; deduplicate)
   - total_sections
   - section_atomic_counts (each section_ref + atomic_unit_count)
5) Fill fields according to class rules:
   - class=step -> step_fields populated, error_fields={}
   - class=error -> error_fields populated, step_fields={}
   - class=not_related -> step_fields={} and error_fields={}
6) For every step, use bilingual prompt object:
   - prompt.en in English
   - prompt.zh as Chinese translation
7) `focus_points` must be step-relevant execution focus only; use [] if none.
8) `common_mistakes` must be step-relevant likely mistakes only; use [] if none.
9) Keep steps strictly goal-related. If a unit is not directly helping task completion, classify as not_related.
   - Confirmation/showcase/meta narration (e.g., "confirm result", "show finished model", "completion speech") should be `not_related` unless there is a real physical state change.
   - If explanation speech and object manipulation co-occur, classify by the physical manipulation outcome.
   - If audio says an actionable operation (put/place/move/remove/unscrew/press) and vision shows matching state change, classify as `step`, not `not_related`.
10) Use only audio/visual evidence from the video.
    - If there is no direct evidence, do not claim it.
11) Ensure all timestamps are in seconds and realistic for this video.
12) Every atomic unit must be at least 2 seconds long (`end_sec - start_sec >= 2.0`).
13) Return strict JSON only, matching the required shape in system prompt.

Quality checks before returning:
- section timeline global non-overlap
- unit timeline non-overlap within section
- each unit assigned to exactly one section
- no undefined enum values
- no empty required fields
- total_sections equals count of output sections
- every atomic unit duration >= 2.0 seconds
- no atomic unit contains two sequential actionable verbs
- confirmation/showcase/meta-only units are not classified as `step`
- units with observable task-state change are not classified as `not_related`
- no single `step` description/prompt contains both "remove/unscrew" and "place/move to location"
- only pure introduction section may have zero `step`; each later procedural section must contain at least one `step`
- avoid concentrating all steps into one section when multiple procedural phases exist
