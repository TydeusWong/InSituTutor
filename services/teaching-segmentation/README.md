# teaching-segmentation

Responsibility:

- Call Omni on compressed video
- Segment into sections
- Extract atomic units with class `step|error|not_related`
- Output intermediate parse for strategy-builder

Input:

- `data/processed/ingest_manifest.json`

Output:

- `data/processed/v2/segmentation/sections_units.json`

Run:

```powershell
python services/teaching-segmentation/run_segmentation.py --manifest data/test_cake/ingest_manifest.json --mock
```

Use real Omni call (no `--mock`) after setting `DASHSCOPE_API_KEY`.

Case folder behavior:

- default output: `data/<case_id>/v2/segmentation/sections_units.json`
- infer `<case_id>` from first demo `video_id` when not explicitly provided
- override with `--case-id` or `--output`
