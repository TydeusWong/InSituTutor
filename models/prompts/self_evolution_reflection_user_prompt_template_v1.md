Analyze this self-evolution teaching session.

Inputs:
- ASR transcript with speaker IDs and timestamps
- step_windows with section/step IDs, time ranges, and system prompts
- strategy outline with section/step prompts, focus points, and common mistakes
- allowed scene entities
- optional video evidence attached as video_url

Find only high-value learning events:
- confusion: student question, hesitation, or explicit inability to understand the system instruction.
- error: wrong student action, usually marked by teacher verbal or physical intervention.

For every event:
- include time_range_sec
- include scope.applies_to_sections and scope.applies_to_steps as lists
- assign scope by matching the event time_range_sec to step_windows first; use semantic similarity only as a secondary check
- if the event overlaps multiple step windows, include every affected section/step
- do not include not_errors_in_sections or not_errors_in_steps
- include concrete evidence and diagnosis

For every error:
- include error_cause_type. Allowed values are object_identity_confusion, wrong_object_selection, wrong_spatial_relation, wrong_orientation, wrong_sequence, wrong_hand_action, insufficient_action, other.
- include confused_entities with student_used_or_pointed_to, should_have_used_or_recognized, and evidence_utterance.
- use exact names from allowed_scene_entities when an entity is mentioned; use empty strings if not applicable.
- when the teacher says object-identifying language such as "this is ...", "that is ...", "not X, Y", "这是...", "这个是...", or "不是这个，是那个", explicitly test whether the root error is object identity confusion before considering spatial alignment, distance, or pose errors.
- diagnosis must explain the chosen root cause, not only the visible surface symptom.
- include correction_message.zh and correction_message.en.
- correction_message must be specific: say what wrong action/object to undo first, then say the correct action/object to perform.
- if error_cause_type is object_identity_confusion or wrong_object_selection, correction_message must first tell the student which object is wrong and which object is correct, then tell the next physical action.
- avoid generic correction messages such as "go back to the current step" or "place it correctly".

Return JSON only.
