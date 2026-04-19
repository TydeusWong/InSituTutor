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
2) For each section, produce ordered, non-overlapping atomic units fully contained in that section.
3) Classify each atomic unit strictly as step, error, or not_related.
4) Add top-level `video_overview` with:
   - summary
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
10) Use only audio/visual evidence from the video.
    - If there is no direct evidence, do not claim it.
11) Ensure all timestamps are in seconds and realistic for this video.
12) Return strict JSON only, matching the required shape in system prompt.

Quality checks before returning:
- section timeline global non-overlap
- unit timeline non-overlap within section
- each unit assigned to exactly one section
- no undefined enum values
- no empty required fields
- total_sections equals count of output sections
