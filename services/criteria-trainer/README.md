# criteria-trainer

Responsibility:

- Build training dataset from step/error time ranges
- Run DINO-assisted annotation and train/select YOLO models
- Compile detector plans into executable rule criteria

Input:

- `data/<case_id>/v2/strategy/teaching_strategy_v2.json`
- `data/<case_id>/v2/segmentation/sections_units.json`
- `data/<case_id>/ingest_manifest.json`

Output:

- `data/<case_id>/v2/slices/index.json`
- `data/<case_id>/v2/slices/<section_id>/<step_or_error_id>/clip.mp4`
- `data/<case_id>/v2/detector-plans/detector_plan_v2.json`

Utilities:

- `setup_small_models.py`: initialize small-model folder layout and runtime report
- `healthcheck_small_models.py`: run package/adapter/device health checks
- `slice_step_error_clips.py`: cut clips by each step/error time range
- `build_detector_plan_v2.py`: generate detector plan for first step slice (Omni if API key exists, otherwise mock)

Quick start:

```bash
python services/criteria-trainer/setup_small_models.py
python services/criteria-trainer/slice_step_error_clips.py --case-id test_cake
python services/criteria-trainer/build_detector_plan_v2.py --case-id test_cake
```

V3 (DINO -> YOLO acceleration):

1) Bootstrap YOLO dataset from DINO pseudo labels

```bash
python services/criteria-trainer/bootstrap_yolo_dataset_from_dino.py \
  --case-id test_cake \
  --dino-model-id models/small-models/object/grounding-dino \
  --dino-processor-id models/small-models/object/grounding-dino
```

2) Train YOLO from bootstrap dataset

```bash
python services/criteria-trainer/train_yolo_from_bootstrap.py \
  --case-id test_cake \
  --base-model models/small-models/object/yolo/yolo11n.pt \
  --device cuda:0 \
  --workers 0
```

3) Replay validate with YOLO-first and optional DINO fallback

```bash
python services/criteria-trainer/replay_validate_all_steps_yolo.py \
  --case-id test_cake \
  --fallback-dino \
  --fallback-dino-model-id models/small-models/object/grounding-dino \
  --fallback-dino-processor-id models/small-models/object/grounding-dino \
  --fallback-dino-local-files-only \
  --progress-every-seconds 5
```

4) Build speed compare report (YOLO vs DINO)

```bash
python services/criteria-trainer/build_speed_compare_report.py \
  --case-id test_cake \
  --dino-model-id models/small-models/object/grounding-dino \
  --dino-processor-id models/small-models/object/grounding-dino \
  --dino-local-files-only
```

Main V3 outputs:

- `data/<case_id>/v3/yolo-bootstrap/input_snapshot.json`
- `data/<case_id>/v3/yolo-bootstrap/annotation_report.json`
- `data/<case_id>/v3/yolo-dataset/data.yaml`
- `data/<case_id>/v3/yolo-runs/<run_id>/weights/best.pt`
- `services/criteria-trainer/configs/yolo_registry_v1.json`
- `data/<case_id>/v3/replay-validation-yolo/all_steps_result.json`
- `data/<case_id>/v3/replay-validation-yolo/speed_compare_report.json`
