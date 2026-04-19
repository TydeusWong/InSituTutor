import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def resolve_video_path(case_id: str) -> Path:
    manifest = read_json(ROOT / "data" / case_id / "ingest_manifest.json")
    demos = manifest.get("demos", [])
    if not demos:
        raise ValueError("ingest manifest has no demos")
    video_path = Path(str(demos[0].get("ingest_video_path", "")))
    if not video_path:
        raise ValueError("ingest_video_path missing")
    if not video_path.is_absolute():
        video_path = (ROOT / video_path).resolve()
    return video_path


def choose_step_plan(case_id: str) -> Tuple[str, Dict[str, Any]]:
    bundle_path = ROOT / "data" / case_id / "v2" / "detector-plans" / "detector_plan_v2.json"
    bundle = read_json(bundle_path)
    selected_slice = bundle.get("selected_slice", {})
    slice_id = str(selected_slice.get("slice_id", "")).strip()
    if not slice_id:
        raise ValueError("selected_slice.slice_id missing in detector_plan_v2.json")
    step_path = ROOT / "data" / case_id / "v2" / "detector-plans" / f"{slice_id}.json"
    return slice_id, read_json(step_path)


def sample_offsets_per_second(fps_int: int) -> List[int]:
    if fps_int < 2:
        return [0]
    n = fps_int // 2
    # 1-based: frame 1 and frame n+1  -> 0-based offsets: 0 and n
    offsets = [0, n]
    return sorted(set([min(max(x, 0), fps_int - 1) for x in offsets]))


def detect_blue_box_center(frame_bgr: np.ndarray) -> Tuple[bool, float, float, float]:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([90, 50, 40], dtype=np.uint8)
    upper_blue = np.array([135, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False, 0.0, 0.0, 0.0

    best = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(best))
    if area < 250.0:
        return False, 0.0, 0.0, 0.0

    x, y, w, h = cv2.boundingRect(best)
    cx = x + w / 2.0
    cy = y + h / 2.0
    return True, cx, cy, area


def make_eval_context(frame_bgr: np.ndarray) -> Dict[str, Any]:
    h, w = frame_bgr.shape[:2]
    found, cx, cy, _ = detect_blue_box_center(frame_bgr)
    objects: Dict[str, Dict[str, float]] = {}
    if found:
        objects["blue box"] = {"x": cx / w, "y": cy / h}
    anchors = {"workspace_center": {"x": 0.5, "y": 0.5}}
    return {"objects": objects, "anchors": anchors}


def eval_condition_code(code: str, context: Dict[str, Any]) -> bool:
    def abs_pos_distance(object_id: str, anchor_id: str) -> float:
        obj = context["objects"].get(object_id)
        anchor = context["anchors"].get(anchor_id)
        if not obj or not anchor:
            return 1.0
        dx = float(obj["x"] - anchor["x"])
        dy = float(obj["y"] - anchor["y"])
        return math.sqrt(dx * dx + dy * dy) / math.sqrt(2.0)

    safe_globals: Dict[str, Any] = {"__builtins__": {}}
    safe_locals: Dict[str, Any] = {"abs_pos_distance": abs_pos_distance}
    return bool(eval(code, safe_globals, safe_locals))


def first_step_replay_validate(case_id: str) -> Dict[str, Any]:
    slice_id, step_plan = choose_step_plan(case_id)
    video_path = resolve_video_path(case_id)

    conditions = step_plan.get("judgement_conditions", [])
    if not conditions:
        raise ValueError("step plan has no judgement_conditions")
    code = str(conditions[0].get("code", "")).strip()
    if not code:
        raise ValueError("judgement condition code is empty")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        raise RuntimeError("invalid fps from video")
    fps_int = max(1, int(round(fps)))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    offsets = sample_offsets_per_second(fps_int)

    matched = False
    matched_frame_idx = -1
    matched_sec = -1.0
    sampled_frames = 0

    sec = 0
    while sec * fps_int < total_frames:
        sec_base = sec * fps_int
        for off in offsets:
            frame_idx = sec_base + off
            if frame_idx >= total_frames:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            sampled_frames += 1
            ctx = make_eval_context(frame)
            if eval_condition_code(code, ctx):
                matched = True
                matched_frame_idx = frame_idx
                matched_sec = frame_idx / fps
                break
        if matched:
            break
        sec += 1

    cap.release()
    return {
        "pipeline_stage": "criteria-trainer:replay-validation:first-step",
        "generated_at": utc_now_iso(),
        "case_id": case_id,
        "slice_id": slice_id,
        "video_path": str(video_path),
        "fps": fps,
        "sampling_rule": {
            "per_second_frames": 2,
            "fps_formula": "if fps = 2n, sample frame #1 and frame #n+1 each second",
            "effective_offsets_zero_based": offsets,
        },
        "judgement_code": code,
        "sampled_frames": sampled_frames,
        "matched": matched,
        "first_match_frame_index": matched_frame_idx,
        "first_match_second": matched_sec if matched else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay validation for first step detector plan")
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    result = first_step_replay_validate(args.case_id)
    out_path = (
        Path(args.output)
        if args.output
        else ROOT / "data" / args.case_id / "v2" / "replay-validation" / "first_step_result.json"
    )
    write_json(out_path, result)
    print(f"[OK] output: {out_path}")
    if result["matched"]:
        print(f"[OK] first match second: {result['first_match_second']:.3f}")
    else:
        print("[OK] no match found")


if __name__ == "__main__":
    main()
