import argparse
import json
import math
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from adapters.base import InferenceInput  # noqa: E402
from adapters.grounding_dino import GroundingDINOAdapter  # noqa: E402
from adapters.mediapipe_hand import MediaPipeHandAdapter  # noqa: E402


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def normalize_name(name: str) -> str:
    s = str(name or "").strip().lower().replace("_", " ")
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def tokenize_name(name: str) -> List[str]:
    n = normalize_name(name)
    return [x for x in n.split(" ") if x]


def split_detect_target_string(raw: str) -> List[str]:
    if not isinstance(raw, str):
        return []
    parts = re.split(r",| and |，|、|及|和", raw)
    out: List[str] = []
    for p in parts:
        t = str(p).strip()
        if t and t not in out:
            out.append(t)
    return out


ANCHOR_COORDS: Dict[str, Dict[str, float]] = {
    "workspace_center": {"x": 0.5, "y": 0.5},
    "workspace_left": {"x": 0.2, "y": 0.5},
    "workspace_right": {"x": 0.8, "y": 0.5},
    "workspace_top": {"x": 0.5, "y": 0.2},
    "workspace_bottom": {"x": 0.5, "y": 0.8},
    "workspace_top_left": {"x": 0.2, "y": 0.2},
    "workspace_top_right": {"x": 0.8, "y": 0.2},
    "workspace_bottom_left": {"x": 0.2, "y": 0.8},
    "workspace_bottom_right": {"x": 0.8, "y": 0.8},
}


ANCHOR_ALIASES: Dict[str, str] = {
    "workspace center": "workspace_center",
    "workspace_center": "workspace_center",
    "center of workspace": "workspace_center",
    "workspace right": "workspace_right",
    "workspace_right": "workspace_right",
    "workspace left": "workspace_left",
    "workspace_left": "workspace_left",
    "workspace top": "workspace_top",
    "workspace_top": "workspace_top",
    "workspace bottom": "workspace_bottom",
    "workspace_bottom": "workspace_bottom",
    "workspace top left": "workspace_top_left",
    "workspace_top_left": "workspace_top_left",
    "workspace top right": "workspace_top_right",
    "workspace_top_right": "workspace_top_right",
    "workspace bottom left": "workspace_bottom_left",
    "workspace_bottom_left": "workspace_bottom_left",
    "workspace bottom right": "workspace_bottom_right",
    "workspace_bottom_right": "workspace_bottom_right",
}


def canonical_anchor_id(name: str) -> Optional[str]:
    n = normalize_name(name)
    if not n:
        return None
    return ANCHOR_ALIASES.get(n)


def build_anchors_from_targets(global_targets: List[str]) -> Dict[str, Dict[str, float]]:
    anchors: Dict[str, Dict[str, float]] = {}
    for item in global_targets:
        aid = canonical_anchor_id(str(item))
        if not aid:
            continue
        coords = ANCHOR_COORDS.get(aid)
        if coords and aid not in anchors:
            anchors[aid] = {"x": float(coords["x"]), "y": float(coords["y"])}
    # Backward-compatible fallback: if no anchor target is declared, keep default center.
    if not anchors:
        anchors["workspace_center"] = {"x": 0.5, "y": 0.5}
    return anchors


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


def sample_offsets_per_second(fps_int: int) -> List[int]:
    if fps_int < 2:
        return [0]
    n = fps_int // 2
    # 1-based frame#1 and frame#(n+1) => 0-based offsets 0 and n
    return [0, n]


def load_step_sequence(case_id: str) -> List[Dict[str, Any]]:
    index_path = ROOT / "data" / case_id / "v2" / "slices" / "index.json"
    index_data = read_json(index_path)
    slices = index_data.get("slices", [])
    steps = [s for s in slices if s.get("slice_type") == "step"]
    steps.sort(key=lambda s: float((s.get("time_range") or {}).get("start_sec", 0.0)))
    return steps


def load_step_plan(case_id: str, step_id: str) -> Dict[str, Any]:
    return read_json(ROOT / "data" / case_id / "v2" / "detector-plans" / f"{step_id}.json")


def load_detector_plan_bundle(case_id: str) -> Dict[str, Any]:
    bundle_path = ROOT / "data" / case_id / "v2" / "detector-plans" / "detector_plan_v2.json"
    if not bundle_path.exists():
        return {}
    return read_json(bundle_path)


def extract_grounding_targets_from_plan(step_plan: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    model_selection = step_plan.get("model_selection", [])
    if not isinstance(model_selection, list):
        return out
    for item in model_selection:
        if not isinstance(item, dict):
            continue
        if str(item.get("model_id", "")).strip() != "grounding-dino":
            continue
        if isinstance(item.get("detect_targets"), list):
            for x in item["detect_targets"]:
                t = str(x).strip()
                if t and t not in out:
                    out.append(t)
        elif isinstance(item.get("detect_target"), str):
            for t in split_detect_target_string(item["detect_target"]):
                if t not in out:
                    out.append(t)
    return out


def step_requires_hand_model(step_plan: Dict[str, Any], code: str) -> bool:
    models_required = step_plan.get("models_required", [])
    if isinstance(models_required, list) and any(str(x).strip() == "mediapipe-hand-landmarker" for x in models_required):
        return True
    model_selection = step_plan.get("model_selection", [])
    if isinstance(model_selection, list):
        for item in model_selection:
            if isinstance(item, dict) and str(item.get("model_id", "")).strip() == "mediapipe-hand-landmarker":
                return True
    return ("hand_" in (code or ""))


def register_object(objects: Dict[str, Dict[str, float]], name: str, x: float, y: float) -> None:
    descriptors = {"clear", "black", "blue", "colored", "colour", "color", "small", "large"}
    candidates: List[str] = []

    raw = str(name).strip()
    if raw:
        candidates.append(raw)
        candidates.append(raw.replace(" ", "_"))

    norm = normalize_name(name)
    if norm:
        candidates.append(norm)
        candidates.append(norm.replace(" ", "_"))
        tokens = tokenize_name(norm)
        if tokens:
            reduced = " ".join([t for t in tokens if t not in descriptors]).strip()
            if reduced and reduced != norm:
                candidates.append(reduced)
                candidates.append(reduced.replace(" ", "_"))

    for k in candidates:
        kk = str(k).strip()
        if kk and kk not in objects:
            objects[kk] = {"x": x, "y": y}


def bind_missing_targets_as_aliases(objects: Dict[str, Dict[str, float]], targets: List[str]) -> None:
    if not targets or not objects:
        return
    existing_keys = list(objects.keys())
    existing_token_map = {k: set(tokenize_name(k)) for k in existing_keys}
    for target in targets:
        target_name = str(target).strip()
        if not target_name:
            continue
        if target_name in objects or normalize_name(target_name) in existing_token_map:
            continue
        target_tokens = set(tokenize_name(target_name))
        if not target_tokens:
            continue
        best_key = None
        best_score = 0
        for k in existing_keys:
            overlap = len(target_tokens.intersection(existing_token_map.get(k, set())))
            if overlap > best_score:
                best_score = overlap
                best_key = k
        # Require at least 2 token overlap, or exact single-token noun overlap.
        if best_key and (best_score >= 2 or (best_score == 1 and len(target_tokens) == 1)):
            obj = objects[best_key]
            register_object(objects, target_name, float(obj["x"]), float(obj["y"]))


def run_grounding_dino_objects(
    adapter: Optional[GroundingDINOAdapter],
    frame_bgr: np.ndarray,
    frame_index: int,
    sample_seq: int,
    sec: float,
    targets: List[str],
    object_memory: Dict[str, Dict[str, Any]],
    edge_margin: float,
    persist_max_miss_samples: int,
) -> Dict[str, Dict[str, float]]:
    objects: Dict[str, Dict[str, float]] = {}
    if adapter is None:
        return objects
    object_targets = [t for t in targets if t and not canonical_anchor_id(t)]
    if not object_targets:
        return objects

    payload = InferenceInput(
        frame_bgr=frame_bgr,
        timestamp_sec=sec,
        frame_index=frame_index,
        context={"detect_targets": object_targets},
    )
    out = adapter.infer(payload)
    by_target = out.features.get("by_target", {})
    if not isinstance(by_target, dict):
        by_target = {}

    for target in object_targets:
        t = str(target).strip()
        if not t:
            continue
        det = by_target.get(t)
        if isinstance(det, dict):
            center = det.get("center", {})
            if isinstance(center, dict):
                try:
                    cx = float(center["x"])
                    cy = float(center["y"])
                    register_object(objects, t, cx, cy)
                    bbox = det.get("bbox_xyxy", {})
                    if isinstance(bbox, dict):
                        x1 = float(bbox.get("x1", cx))
                        y1 = float(bbox.get("y1", cy))
                        x2 = float(bbox.get("x2", cx))
                        y2 = float(bbox.get("y2", cy))
                    else:
                        x1, y1, x2, y2 = cx, cy, cx, cy
                    near_edge = (
                        x1 <= edge_margin or y1 <= edge_margin or x2 >= (1.0 - edge_margin) or y2 >= (1.0 - edge_margin)
                    )
                    object_memory[t] = {
                        "x": cx,
                        "y": cy,
                        "last_seen_frame": frame_index,
                        "last_seen_sample": sample_seq,
                        "near_edge": bool(near_edge),
                    }
                    continue
                except Exception:
                    pass

        # Missed detection fallback: carry previous box center for limited sampled frames,
        # only when previous box was not near frame boundary.
        mem = object_memory.get(t)
        if isinstance(mem, dict):
            last_sample = int(mem.get("last_seen_sample", -10))
            near_edge = bool(mem.get("near_edge", True))
            if (sample_seq - last_sample) <= max(1, int(persist_max_miss_samples)) and (not near_edge):
                try:
                    cx = float(mem["x"])
                    cy = float(mem["y"])
                    register_object(objects, t, cx, cy)
                except Exception:
                    pass
    return objects


def run_mediapipe_hand_points(
    adapter: Optional[MediaPipeHandAdapter],
    frame_bgr: np.ndarray,
    frame_index: int,
    sec: float,
    objects: Dict[str, Dict[str, float]],
) -> Dict[str, Dict[str, float]]:
    points: Dict[str, Dict[str, float]] = {}
    if adapter is None:
        return points

    payload = InferenceInput(
        frame_bgr=frame_bgr,
        timestamp_sec=sec,
        frame_index=frame_index,
        context={},
    )
    out = adapter.infer(payload)
    hands = out.features.get("hand_landmarks", [])
    if not isinstance(hands, list) or not hands:
        return points

    candidate_tips: List[Tuple[float, float]] = []
    for hand in hands:
        if not isinstance(hand, dict):
            continue
        tip = hand.get("index_fingertip")
        if not isinstance(tip, dict):
            continue
        try:
            candidate_tips.append((float(tip["x"]), float(tip["y"])))
        except Exception:
            continue
    if not candidate_tips:
        return points

    if objects:
        xs = [float(v["x"]) for v in objects.values()]
        ys = [float(v["y"]) for v in objects.values()]
        cx = float(sum(xs) / len(xs))
        cy = float(sum(ys) / len(ys))
        best = min(candidate_tips, key=lambda p: (p[0] - cx) ** 2 + (p[1] - cy) ** 2)
    else:
        best = candidate_tips[0]
    points["index_fingertip"] = {"x": best[0], "y": best[1]}
    return points


def build_context(
    step_id: str,
    frame_bgr: np.ndarray,
    frame_index: int,
    sample_seq: int,
    sec: float,
    step_range: Dict[str, Any],
    dino_targets: List[str],
    anchors: Dict[str, Dict[str, float]],
    dino_adapter: Optional[GroundingDINOAdapter],
    hand_adapter: Optional[MediaPipeHandAdapter],
    use_hand_model: bool,
    object_memory: Dict[str, Dict[str, Any]],
    edge_margin: float,
    persist_max_miss_samples: int,
) -> Dict[str, Any]:
    objects = run_grounding_dino_objects(
        adapter=dino_adapter,
        frame_bgr=frame_bgr,
        frame_index=frame_index,
        sample_seq=sample_seq,
        sec=sec,
        targets=dino_targets,
        object_memory=object_memory,
        edge_margin=edge_margin,
        persist_max_miss_samples=persist_max_miss_samples,
    )
    hand_points: Dict[str, Dict[str, float]] = {}
    if use_hand_model:
        hand_points = run_mediapipe_hand_points(
            adapter=hand_adapter,
            frame_bgr=frame_bgr,
            frame_index=frame_index,
            sec=sec,
            objects=objects,
        )

    bind_missing_targets_as_aliases(objects, dino_targets)
    object_index: Dict[str, str] = {}
    for k in objects.keys():
        nk = normalize_name(k)
        if nk and nk not in object_index:
            object_index[nk] = k

    return {"objects": objects, "object_index": object_index, "anchors": anchors, "hand_points": hand_points}


def eval_condition_code(code: str, context: Dict[str, Any]) -> bool:
    object_index = context.get("object_index", {})
    anchor_index = {normalize_name(k): k for k in context.get("anchors", {}).keys()}

    def _obj(name: str) -> Optional[Dict[str, float]]:
        obj = context["objects"].get(name)
        if obj:
            return obj
        key = object_index.get(normalize_name(name))
        if key:
            return context["objects"].get(key)
        return None

    def abs_pos_distance(object_id: str, anchor_id: str) -> float:
        obj = _obj(object_id)
        anchor = context["anchors"].get(anchor_id)
        if not anchor:
            key = anchor_index.get(normalize_name(anchor_id))
            if key:
                anchor = context["anchors"].get(key)
        if not obj or not anchor:
            return 1.0
        dx = float(obj["x"] - anchor["x"])
        dy = float(obj["y"] - anchor["y"])
        return math.sqrt(dx * dx + dy * dy)

    def abs_center_xy(object_id: str) -> Tuple[float, float]:
        obj = _obj(object_id)
        if not obj:
            return (0.0, 0.0)
        return (float(obj["x"]), float(obj["y"]))

    def rel_x(left_obj: str, right_obj: str) -> float:
        a = _obj(left_obj)
        b = _obj(right_obj)
        if not a or not b:
            return 1.0
        return float(a["x"] - b["x"])

    def rel_y(top_obj: str, bottom_obj: str) -> float:
        a = _obj(top_obj)
        b = _obj(bottom_obj)
        if not a or not b:
            return 1.0
        return float(a["y"] - b["y"])

    def rel_distance(obj_a: str, obj_b: str) -> float:
        a = _obj(obj_a)
        b = _obj(obj_b)
        if not a or not b:
            return 1.0
        dx = float(a["x"] - b["x"])
        dy = float(a["y"] - b["y"])
        return math.sqrt(dx * dx + dy * dy)

    def hand_to_bbox_distance(hand_point: str, object_id: str) -> float:
        hp = context["hand_points"].get(hand_point)
        obj = _obj(object_id)
        if not hp or not obj:
            return 1.0
        dx = float(hp["x"] - obj["x"])
        dy = float(hp["y"] - obj["y"])
        return math.sqrt(dx * dx + dy * dy)

    def hand_in_bbox(hand_point: str, object_id: str) -> bool:
        # v1 proxy: approximate by near-center threshold
        return hand_to_bbox_distance(hand_point, object_id) <= 0.06

    safe_globals: Dict[str, Any] = {"__builtins__": {}}
    safe_locals: Dict[str, Any] = {
        "abs_pos_distance": abs_pos_distance,
        "abs_center_xy": abs_center_xy,
        "rel_x": rel_x,
        "rel_y": rel_y,
        "rel_distance": rel_distance,
        "hand_to_bbox_distance": hand_to_bbox_distance,
        "hand_in_bbox": hand_in_bbox,
    }
    return bool(eval(code, safe_globals, safe_locals))


def find_first_match_for_step(
    cap: cv2.VideoCapture,
    fps: float,
    total_frames: int,
    offsets: List[int],
    start_frame: int,
    step_id: str,
    step_range: Dict[str, Any],
    dino_targets: List[str],
    anchors: Dict[str, Dict[str, float]],
    dino_adapter: Optional[GroundingDINOAdapter],
    hand_adapter: Optional[MediaPipeHandAdapter],
    use_hand_model: bool,
    code: str,
    progress_every_seconds: int,
    persist_edge_margin: float,
    persist_max_miss_samples: int,
    progress_cb: Optional[Any] = None,
) -> Tuple[bool, int, float, int]:
    fps_int = max(1, int(round(fps)))
    sampled = 0

    sec_start = start_frame // fps_int
    sec = sec_start
    next_progress_sec = sec_start
    object_memory: Dict[str, Dict[str, Any]] = {}
    while sec * fps_int < total_frames:
        if progress_cb and progress_every_seconds > 0 and sec >= next_progress_sec:
            progress_cb(sec, sampled)
            next_progress_sec += progress_every_seconds
        base = sec * fps_int
        for off in offsets:
            frame_idx = base + off
            if frame_idx < start_frame or frame_idx >= total_frames:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            sampled += 1
            sec_float = frame_idx / fps
            ctx = build_context(
                step_id=step_id,
                frame_bgr=frame,
                frame_index=frame_idx,
                sample_seq=sampled,
                sec=sec_float,
                step_range=step_range,
                dino_targets=dino_targets,
                anchors=anchors,
                dino_adapter=dino_adapter,
                hand_adapter=hand_adapter,
                use_hand_model=use_hand_model,
                object_memory=object_memory,
                edge_margin=persist_edge_margin,
                persist_max_miss_samples=persist_max_miss_samples,
            )
            if eval_condition_code(code, ctx):
                return True, frame_idx, sec_float, sampled
        sec += 1
    return False, -1, -1.0, sampled


def run_all_steps_replay(
    case_id: str,
    dino_model_id: str,
    dino_processor_id: Optional[str],
    dino_box_threshold: float,
    dino_text_threshold: float,
    dino_local_files_only: bool,
    progress_every_seconds: int,
    persist_edge_margin: float,
    persist_max_miss_samples: int,
) -> Dict[str, Any]:
    video_path = resolve_video_path(case_id)
    step_slices = load_step_sequence(case_id)
    bundle = load_detector_plan_bundle(case_id)
    global_dino_targets = bundle.get("grounding_dino_detect_targets", [])
    if not isinstance(global_dino_targets, list):
        global_dino_targets = []
    anchors = build_anchors_from_targets([str(x).strip() for x in global_dino_targets if str(x).strip()])
    if not step_slices:
        raise ValueError("no step slices found")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        raise RuntimeError("invalid fps")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_int = max(1, int(round(fps)))
    offsets = sample_offsets_per_second(fps_int)
    dino_adapter = GroundingDINOAdapter(
        model_id=dino_model_id,
        processor_id=dino_processor_id,
        box_threshold=dino_box_threshold,
        text_threshold=dino_text_threshold,
        local_files_only=dino_local_files_only,
    )
    try:
        dino_adapter.load()
    except Exception as exc:
        cap.release()
        raise RuntimeError(f"failed to load Grounding DINO adapter: {exc}") from exc
    hand_adapter: Optional[MediaPipeHandAdapter] = None

    cursor_frame = 0
    sampled_total = 0
    results: List[Dict[str, Any]] = []
    stop_reason = "all_steps_processed"
    t_start = time.perf_counter()

    print(f"[INFO] replay start case={case_id} steps={len(step_slices)} fps={fps:.2f} total_frames={total_frames}")
    print(
        "[INFO] dino runtime: "
        f"model={dino_model_id} processor={dino_processor_id or dino_model_id} "
        f"box_th={dino_box_threshold} text_th={dino_text_threshold} local_only={dino_local_files_only}"
    )

    for step_idx, step in enumerate(step_slices, start=1):
        step_id = str(step.get("slice_id", "")).strip()
        if not step_id:
            continue
        plan = load_step_plan(case_id, step_id)
        step_dino_targets = extract_grounding_targets_from_plan(plan)
        merged_dino_targets = list(dict.fromkeys([*step_dino_targets, *[str(x).strip() for x in global_dino_targets if str(x).strip()]]))
        conditions = plan.get("judgement_conditions", [])
        code = ""
        for cond in conditions:
            if isinstance(cond, dict) and str(cond.get("when", "")).strip() in {"step", "both"}:
                code = str(cond.get("code", "")).strip()
                if code:
                    break
        if not code:
            code = str((conditions[0] if conditions else {}).get("code", "")).strip()
        if not code:
            results.append({"step_id": step_id, "matched": False, "error": "missing judgement code"})
            stop_reason = "step_has_no_code"
            break
        use_hand_model = step_requires_hand_model(plan, code)
        if use_hand_model and hand_adapter is None:
            hand_adapter = MediaPipeHandAdapter()
            try:
                hand_adapter.load()
            except Exception as exc:
                cap.release()
                raise RuntimeError(f"failed to load MediaPipe hand adapter: {exc}") from exc

        print(f"[STEP] {step_idx}/{len(step_slices)} {step_id} start_from={cursor_frame/fps:.2f}s")

        def _progress_cb(sec_cursor: int, sampled_in_step: int) -> None:
            elapsed = time.perf_counter() - t_start
            print(
                f"[PROGRESS] step={step_id} scan_sec={sec_cursor}s "
                f"sampled_in_step={sampled_in_step} sampled_total={sampled_total + sampled_in_step} "
                f"elapsed={elapsed:.1f}s"
            )

        matched, frame_idx, sec_float, sampled = find_first_match_for_step(
            cap=cap,
            fps=fps,
            total_frames=total_frames,
            offsets=offsets,
            start_frame=cursor_frame,
            step_id=step_id,
            step_range=step.get("time_range", {}),
            dino_targets=merged_dino_targets,
            anchors=anchors,
            dino_adapter=dino_adapter,
            hand_adapter=hand_adapter,
            use_hand_model=use_hand_model,
            code=code,
            progress_every_seconds=progress_every_seconds,
            persist_edge_margin=persist_edge_margin,
            persist_max_miss_samples=persist_max_miss_samples,
            progress_cb=_progress_cb,
        )
        sampled_total += sampled
        if not matched:
            print(f"[STEP] {step_id} not matched after sampled={sampled}")
            results.append({"step_id": step_id, "matched": False, "first_match_second": None})
            stop_reason = "video_exhausted_or_step_not_matched"
            break
        print(f"[STEP] {step_id} matched at {sec_float:.3f}s sampled={sampled}")

        results.append(
            {
                "step_id": step_id,
                "matched": True,
                "first_match_frame_index": frame_idx,
                "first_match_second": round(sec_float, 3),
                "judgement_code": code,
                "grounding_dino_detect_targets": step_dino_targets,
            }
        )
        cursor_frame = frame_idx + 1
        if cursor_frame >= total_frames:
            stop_reason = "video_exhausted"
            break

    cap.release()
    return {
        "pipeline_stage": "criteria-trainer:replay-validation:all-steps",
        "generated_at": utc_now_iso(),
        "case_id": case_id,
        "video_path": str(video_path),
        "fps": fps,
        "sampling_rule": {
            "per_second_frames": 2,
            "fps_formula": "if fps = 2n, sample frame #1 and frame #n+1 each second",
            "effective_offsets_zero_based": offsets,
        },
        "anchors_source": "detector_plan_v2.grounding_dino_detect_targets",
        "anchors": anchors,
        "grounding_dino_runtime": {
            "model_id": dino_model_id,
            "processor_id": dino_processor_id or dino_model_id,
            "box_threshold": dino_box_threshold,
            "text_threshold": dino_text_threshold,
            "local_files_only": dino_local_files_only,
            "persist_on_miss_when_not_edge": True,
            "persist_edge_margin": persist_edge_margin,
            "persist_max_miss_samples": persist_max_miss_samples,
        },
        "total_steps": len(step_slices),
        "matched_steps": sum(1 for x in results if x.get("matched")),
        "sampled_frames_total": sampled_total,
        "stop_reason": stop_reason,
        "step_results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay validation for all step plans")
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--output", default=None)
    local_dino_dir = ROOT / "models" / "small-models" / "object" / "grounding-dino"
    default_dino_model = str(local_dino_dir) if local_dino_dir.exists() else "IDEA-Research/grounding-dino-base"
    parser.add_argument("--dino-model-id", default=default_dino_model)
    parser.add_argument("--dino-processor-id", default=None)
    parser.add_argument("--dino-box-threshold", type=float, default=0.30)
    parser.add_argument("--dino-text-threshold", type=float, default=0.25)
    parser.add_argument("--dino-local-files-only", action="store_true")
    parser.add_argument("--progress-every-seconds", type=int, default=5)
    parser.add_argument("--persist-edge-margin", type=float, default=0.05)
    parser.add_argument("--persist-max-miss-samples", type=int, default=6)
    args = parser.parse_args()

    result = run_all_steps_replay(
        case_id=args.case_id,
        dino_model_id=args.dino_model_id,
        dino_processor_id=args.dino_processor_id,
        dino_box_threshold=args.dino_box_threshold,
        dino_text_threshold=args.dino_text_threshold,
        dino_local_files_only=args.dino_local_files_only,
        progress_every_seconds=max(1, int(args.progress_every_seconds)),
        persist_edge_margin=max(0.0, min(0.3, float(args.persist_edge_margin))),
        persist_max_miss_samples=max(1, int(args.persist_max_miss_samples)),
    )
    out_path = (
        Path(args.output)
        if args.output
        else ROOT / "data" / args.case_id / "v2" / "replay-validation" / "all_steps_result.json"
    )
    write_json(out_path, result)
    print(f"[OK] output: {out_path}")
    print(f"[OK] matched steps: {result['matched_steps']}/{result['total_steps']}")


if __name__ == "__main__":
    main()
