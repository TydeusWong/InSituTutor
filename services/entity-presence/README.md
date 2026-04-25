# entity-presence

Responsibility:

- Read atomic-unit clip slices from segmentation output.
- Send each clip plus the case-level entity list from strategy-builder to Omni.
- Return which allowed entities are visually present in each clip.
- Keep entity names as exact strings from `data/<case_id>/v2/strategy/scene_entities_v1.json`.

Input:

- `data/<case_id>/v2/strategy/scene_entities_v1.json`
- `data/<case_id>/v2/segmentation/atomic-unit-slices/index.json`
- `data/<case_id>/v2/segmentation/atomic-unit-slices/**/clip.mp4`

Output:

- `data/<case_id>/v2/segmentation/entity-presence/entity_presence.json`

Run:

```powershell
python services/entity-presence/label_atomic_entities.py --case-id test_cake
```

Debug without Omni:

```powershell
python services/entity-presence/label_atomic_entities.py --case-id test_cake --mock
```

Rules enforced by the service:

- `present_entities` must be a list.
- Every returned name must exactly match one string from `scene_entities_v1.json.entity_names`.
- Entity presence should be judged primarily from visual understanding; speech/audio may only help name visible objects and must not by itself make an entity present.
- Invalid names cause that slice to be recorded in `failures`.

## Early YOLO Training

After `entity_presence.json` is available, train YOLO directly from the atomic-unit clips where entities are present:

```powershell
python services/entity-presence/train_yolo_from_entity_presence.py --case-id test_cake --workers 0
```

This replaces the later legacy sequence:

- `services/criteria-trainer/bootstrap_yolo_dataset_from_dino.py`
- `services/criteria-trainer/train_yolo_from_bootstrap.py`

Main outputs:

- `data/<case_id>/v3/yolo-bootstrap/annotated_samples/<entity>/*.jpg`
  - sampled frames with DINO bbox overlays, used as the trace/audit artifact
- `data/<case_id>/v3/yolo-bootstrap/annotation_report.json`
- `data/<case_id>/v3/yolo-dataset/data.yaml`
- `data/<case_id>/v3/yolo-dataset/class_map.json`
- `data/<case_id>/v3/yolo-runs/<run_id>/weights/best.pt`
- `services/criteria-trainer/configs/yolo_registry_v1.json`

By default, only clips whose `unit_class` is `step` are used. Add `--include-non-step` to also use error/not_related clips.

DINO query text:

- YOLO class names remain the canonical entity names from `entity_presence.json`.
- If a canonical entity name uses the `short label . short label . short label` format, that full dot-separated string is sent to DINO as one prompt.
- In the current adapter, each `detect_targets` list item causes one DINO forward pass, so one dot-separated prompt per entity keeps it to one DINO call per entity.
- Optional overrides can still be provided in `services/criteria-trainer/configs/yolo_bootstrap_config_v1.json` field `dino_target_aliases`.
- Example prompt: `black jar . cylindrical jar . black container`.
- DINO detections from this prompt are mapped back to the original canonical entity before writing YOLO labels.
