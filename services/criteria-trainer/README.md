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
