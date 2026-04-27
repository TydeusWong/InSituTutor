Given confusion events, error events, and the original teaching strategy, propose minimal strategy patches.

Use confusion events to improve wording of prompts.
Use error events to add focus_points or common_mistakes when appropriate, or improve wording of prompts.

Do not discard the original meaning of a step prompt. Most prompt changes should be additive clarifications.
If an error reveals that the current step prompt itself is misleading or contradicts the teacher correction, modify prompt.zh and prompt.en for that scoped step, but preserve the original core action and add the missing constraint or warning.
If an error is an object identity confusion, add or modify wording so the prompt distinguishes the confused objects using exact entity names.
If an error is a spatial relation or orientation mistake, add a common_mistakes item and, when needed, a focus_points item for the affected step.
Only patch steps listed in the event scope. Do not patch a later semantically related step when the event scope points to an earlier step.

For every confusion_events item and every error_events item:
- either create at least one patch with source_event_type and source_event_id
- or include a no_patch_reason item with source_event_type, source_event_id, and reason

Do not return an empty patches list unless every event has a no_patch_reason.

Return:
{
  "base_strategy_ref": "...",
  "patches": [],
  "no_patch_reasons": []
}
