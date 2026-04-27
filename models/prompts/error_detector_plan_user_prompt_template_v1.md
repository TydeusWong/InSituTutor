Create an error detector plan for one reflected error event.

Inputs:
- error event
- error clip metadata
- teaching strategy
- allowed scene entities
- YOLO class map

Use exact entity names from allowed scene entities.
Use yolo for trained object entities and mediapipe-hand-landmarker for hand relations.
Use executable condition helpers such as:
- rel_distance('entity A', 'entity B')
- rel_y('entity A', 'entity B')
- iou('entity A', 'entity B')
- hand_in_bbox('index_fingertip', 'entity')

The output must include judgement_conditions[].code. Do not only describe the condition in natural language.
Return JSON only.
