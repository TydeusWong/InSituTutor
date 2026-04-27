import argparse
import copy
from pathlib import Path
from typing import Any

from common import read_json, session_root, utc_now_iso, write_json


ROOT = Path(__file__).resolve().parents[2]


def find_step(strategy: dict, section_id: str, step_id: str) -> dict | None:
    for section in strategy.get("sections", []):
        if str(section.get("section_id")) != section_id:
            continue
        for step in section.get("steps", []):
            if str(step.get("step_id")) == step_id:
                return step
    return None


def apply_one_patch(strategy: dict, patch: dict) -> bool:
    target = patch.get("target") if isinstance(patch.get("target"), dict) else {}
    step = find_step(strategy, str(target.get("section_id", "")), str(target.get("step_id", "")))
    if step is None:
        return False
    field = str(target.get("field", ""))
    op = str(patch.get("operation", "replace"))
    new_value: Any = patch.get("new_value")
    if field == "prompt.zh":
        step.setdefault("prompt", {})["zh"] = str(new_value)
    elif field == "prompt.en":
        step.setdefault("prompt", {})["en"] = str(new_value)
    elif field in {"focus_points", "common_mistakes"}:
        if not isinstance(new_value, list):
            new_value = [str(new_value)]
        if op == "append":
            existing = step.get(field) if isinstance(step.get(field), list) else []
            step[field] = [*existing, *new_value]
        else:
            step[field] = new_value
    else:
        return False
    return True


def apply_strategy_patch(case_id: str, session_id: str, review_approved: bool) -> tuple[Path, Path, Path]:
    base = session_root(case_id, session_id)
    review_path = base / "review" / "human_review_v1.json"
    if not review_path.exists():
        write_json(
            review_path,
            {
                "session_id": session_id,
                "review_status": "approved" if review_approved else "pending",
                "approved_patch_ids": "all" if review_approved else [],
                "reviewed_at": utc_now_iso() if review_approved else "",
            },
        )
    review = read_json(review_path)
    if review_approved:
        review["review_status"] = "approved"
        review["approved_patch_ids"] = "all"
        review["reviewed_at"] = utc_now_iso()
        write_json(review_path, review)
    approved = review_approved or str(review.get("review_status", "")).lower() == "approved"
    patch_path = base / "reflection" / "strategy_patch_v1.json"
    patch_data = read_json(patch_path) if patch_path.exists() else {"patches": []}
    base_strategy_path = ROOT / "data" / case_id / "v2" / "strategy" / "teaching_strategy_v2.json"
    strategy = read_json(base_strategy_path)
    evolved = copy.deepcopy(strategy)
    applied = []
    if approved:
        approved_ids = "all" if review_approved else review.get("approved_patch_ids", "all")
        for patch in patch_data.get("patches", []):
            if not isinstance(patch, dict):
                continue
            patch_id = str(patch.get("patch_id", ""))
            if approved_ids != "all" and patch_id not in set(approved_ids if isinstance(approved_ids, list) else []):
                continue
            if apply_one_patch(evolved, patch):
                applied.append(patch_id)
    out_strategy = ROOT / "data" / case_id / "v5" / "strategy" / "teaching_strategy_evolved_v1.json"
    manifest_path = ROOT / "data" / case_id / "v5" / "strategy" / "strategy_evolution_manifest.json"
    write_json(out_strategy, evolved)
    write_json(
        manifest_path,
        {
            "generated_at": utc_now_iso(),
            "case_id": case_id,
            "session_id": session_id,
            "base_strategy_ref": str(base_strategy_path),
            "evolved_strategy_ref": str(out_strategy),
            "patch_ref": str(patch_path),
            "review_ref": str(review_path),
            "review_approved": approved,
            "applied_patch_ids": applied,
            "rollback_ref": str(base_strategy_path),
        },
    )
    return review_path, out_strategy, manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply reviewed self-evolution strategy patches.")
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--review-approved", action="store_true")
    args = parser.parse_args()
    review_path, strategy_path, manifest_path = apply_strategy_patch(args.case_id, args.session_id, args.review_approved)
    print(f"[OK] human review: {review_path}")
    print(f"[OK] evolved strategy: {strategy_path}")
    print(f"[OK] evolution manifest: {manifest_path}")


if __name__ == "__main__":
    main()
