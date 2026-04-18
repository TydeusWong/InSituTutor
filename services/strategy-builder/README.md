# strategy-builder

Responsibility:

- Convert sections and atomic units into v2 teaching strategy
- Build step/error artifacts
- Build detector plan references
- Drop `not_related`

Input:

- `data/processed/v2/segmentation/sections_units.json`

Output:

- `data/processed/v2/strategy/teaching_strategy_v2.json`

Run:

```powershell
python services/strategy-builder/build_strategy.py --case-id test_cake
```

Case folder behavior:

- default input: `data/<case_id>/v2/segmentation/sections_units.json`
- default output: `data/<case_id>/v2/strategy/teaching_strategy_v2.json`
- can still override by `--input` and `--output-dir`
