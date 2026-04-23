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

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from adapters.base import InferenceInput  # noqa: E402
from adapters.grounding_dino import GroundingDINOAdapter  # noqa: E402
from adapters.mediapipe_hand import MediaPipeHandAdapter  # noqa: E402
from adapters.yolo import YOLOAdapter  # noqa: E402

DEFAULT_REGISTRY_PATH = ROOT / "services" / "criteria-trainer" / "configs" / "yolo_registry_v1.json"


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


def split_detect_target_string(raw: str) -> List[str]:
    if not isinstance(raw, str):
        return []
    parts = re.split(r",| and |，|、", raw)
    out: List[str] = []
    for p in parts:
        t = str(p).strip()
        if t and t not in out:
            out.append(t)
    return out


ANCHOR_COORDS: Dict[str, Dict[str, float]] = {
    "workspace_center": {"x": 0.5, "y": 0.5},
}
ANCHOR_ALIASES: Dict[str, str] = {
    "workspace center": "workspace_center",
    "workspace_center": "workspace_center",
    "center of workspace": "workspace_center",
}


def canonical_anchor_id(name: str) -> Optional[str]:
    return ANCHOR_ALIASES.get(normalize_name(name))


def build_anchors_from_targets(global_targets: List[str]) -> Dict[str, Dict[str, float]]:
    anchors: Dict[str, Dict[str, float]] = {}
    for item in global_targets:
        aid = canonical_anchor_id(str(item))
        if not aid:
            continue
        coords = ANCHOR_COORDS.get(aid)
        if coords and aid not in anchors:
            anchors[aid] = {"x": float(coords["x"]), "y": float(coords["y"])}
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


def extract_object_targets_from_plan(step_plan: Dict[str, Any]) -> List[str]:
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
    raw = str(name).strip()
    if raw and raw not in objects:
        objects[raw] = {"x": x, "y": y}
    norm = normalize_name(raw)
    if norm and norm not in objects:
        objects[norm] = {"x": x, "y": y}
    alias = norm.replace(" ", "_")
    if alias and alias not in objects:
        objects[alias] = {"x": x, "y": y}


def bbox_iou(a: Dict[str, float], b: Dict[str, float]) -> float:
    ax1, ay1, ax2, ay2 = float(a["x1"]), float(a["y1"]), float(a["x2"]), float(a["y2"])
    bx1, by1, bx2, by2 = float(b["x1"]), float(b["y1"]), float(b["x2"]), float(b["y2"])
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    a_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = a_area + b_area - inter
    if union <= 1e-12:
        return 0.0
    return float(inter / union)


def pick_candidate_least_like_others(
    target: str,
    candidates: List[Dict[str, Any]],
    detections_by_norm: Dict[str, List[Dict[str, Any]]],
    disambiguation_targets: List[str],
) -> Optional[Dict[str, Any]]:
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    other_targets = [t for t in disambiguation_targets if normalize_name(t) != normalize_name(target)]
    if not other_targets:
        return max(candidates, key=lambda d: float(d.get("score", 0.0)))

    best_det: Optional[Dict[str, Any]] = None
    best_key: Optional[Tuple[float, float]] = None
    for cand in candidates:
        cand_bbox = cand.get("bbox_xyxy")
        if not isinstance(cand_bbox, dict):
            continue
        like_other_sum = 0.0
        for ot in other_targets:
            for od in detections_by_norm.get(normalize_name(ot), []):
                od_bbox = od.get("bbox_xyxy")
                if not isinstance(od_bbox, dict):
                    continue
                iou = bbox_iou(cand_bbox, od_bbox)
                if iou <= 0.0:
                    continue
                like_other_sum += float(od.get("score", 0.0)) * iou
        key = (like_other_sum, -float(cand.get("score", 0.0)))
        if best_key is None or key < best_key:
            best_key = key
            best_det = cand
    if isinstance(best_det, dict):
        return best_det
    return max(candidates, key=lambda d: float(d.get("score", 0.0)))


def run_yolo_objects(
    adapter: YOLOAdapter,
    frame_bgr: Any,
    frame_index: int,
    sample_seq: int,
    sec: float,
    targets: List[str],
    disambiguation_targets: List[str],
    object_memory: Dict[str, Dict[str, Any]],
    edge_margin: float,
    persist_max_miss_samples: int,
    precomputed_detections: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Dict[str, float]]:
    objects: Dict[str, Dict[str, float]] = {}
    infer_targets = disambiguation_targets if disambiguation_targets else targets
    if isinstance(precomputed_detections, list):
        detections = [d for d in precomputed_detections if isinstance(d, dict)]
    else:
        payload = InferenceInput(
            frame_bgr=frame_bgr,
            timestamp_sec=sec,
            frame_index=frame_index,
            context={"detect_targets": infer_targets},
        )
        out = adapter.infer(payload)
        detections = out.features.get("detections", [])
        if not isinstance(detections, list):
            detections = []
    detections_by_norm: Dict[str, List[Dict[str, Any]]] = {}
    for det in detections:
        if not isinstance(det, dict):
            continue
        label = str(det.get("label", det.get("target", ""))).strip()
        if not label:
            continue
        key = normalize_name(label)
        if key not in detections_by_norm:
            detections_by_norm[key] = []
        detections_by_norm[key].append(det)
    # Resolve hit/miss per requested target first, then decide whether persistence should trigger.
    target_hits: Dict[str, Optional[Dict[str, Any]]] = {}
    for target in targets:
        cands = detections_by_norm.get(normalize_name(target), [])
        det = pick_candidate_least_like_others(
            target=target,
            candidates=cands,
            detections_by_norm=detections_by_norm,
            disambiguation_targets=infer_targets,
        )
        target_hits[str(target)] = det if isinstance(det, dict) else None

    detected_count = sum(1 for v in target_hits.values() if isinstance(v, dict))
    meta = object_memory.get("__meta__")
    if not isinstance(meta, dict):
        meta = {}
    prev_detected_count = int(meta.get("prev_detected_count", detected_count))
    # New policy: occlusion persistence is triggered when detected target count drops suddenly.
    drop_trigger = detected_count < prev_detected_count

    for target in targets:
        det = target_hits.get(str(target))
        if isinstance(det, dict):
            center = det.get("center", {})
            if isinstance(center, dict):
                try:
                    cx = float(center["x"])
                    cy = float(center["y"])
                    register_object(objects, target, cx, cy)
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
                    object_memory[str(target)] = {
                        "x": cx,
                        "y": cy,
                        "last_seen_frame": frame_index,
                        "last_seen_sample": sample_seq,
                        "near_edge": bool(near_edge),
                        "persist_active": False,
                    }
                    continue
                except Exception:
                    pass

        # Miss fallback: only trigger when target count drops (occlusion sign), then keep active for limited samples.
        mem = object_memory.get(str(target))
        if isinstance(mem, dict):
            last_sample = int(mem.get("last_seen_sample", -10))
            near_edge = bool(mem.get("near_edge", True))
            persist_active = bool(mem.get("persist_active", False))
            can_persist = drop_trigger or persist_active
            if can_persist and (sample_seq - last_sample) <= max(1, int(persist_max_miss_samples)) and (not near_edge):
                try:
                    cx = float(mem["x"])
                    cy = float(mem["y"])
                    register_object(objects, target, cx, cy)
                    mem["persist_active"] = True
                    object_memory[str(target)] = mem
                except Exception:
                    pass
            else:
                mem["persist_active"] = False
                object_memory[str(target)] = mem

    meta["prev_detected_count"] = detected_count
    object_memory["__meta__"] = meta
    return objects


def run_grounding_dino_objects(
    adapter: Optional[GroundingDINOAdapter],
    frame_bgr: Any,
    frame_index: int,
    sec: float,
    targets: List[str],
) -> Dict[str, Dict[str, float]]:
    objects: Dict[str, Dict[str, float]] = {}
    if adapter is None:
        return objects
    payload = InferenceInput(
        frame_bgr=frame_bgr,
        timestamp_sec=sec,
        frame_index=frame_index,
        context={"detect_targets": targets},
    )
    out = adapter.infer(payload)
    by_target = out.features.get("by_target", {})
    if not isinstance(by_target, dict):
        by_target = {}
    for target in targets:
        det = by_target.get(target)
        if not isinstance(det, dict):
            for k, v in by_target.items():
                if normalize_name(str(k)) == normalize_name(target):
                    det = v
                    break
        if not isinstance(det, dict):
            continue
        center = det.get("center", {})
        if not isinstance(center, dict):
            continue
        try:
            cx = float(center["x"])
            cy = float(center["y"])
        except Exception:
            continue
        register_object(objects, target, cx, cy)
    return objects


def run_mediapipe_hand_points(
    adapter: Optional[MediaPipeHandAdapter],
    frame_bgr: Any,
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
    if not isinstance(hands, list):
        return points
    candidates: List[Tuple[float, float]] = []
    for hand in hands:
        if not isinstance(hand, dict):
            continue
        tip = hand.get("index_fingertip")
        if not isinstance(tip, dict):
            continue
        try:
            candidates.append((float(tip["x"]), float(tip["y"])))
        except Exception:
            continue
    if not candidates:
        return points
    if objects:
        ox = sum(v["x"] for v in objects.values()) / len(objects)
        oy = sum(v["y"] for v in objects.values()) / len(objects)
        best = min(candidates, key=lambda p: (p[0] - ox) ** 2 + (p[1] - oy) ** 2)
    else:
        best = candidates[0]
    points["index_fingertip"] = {"x": best[0], "y": best[1]}
    return points


def build_context(
    frame_bgr: Any,
    frame_index: int,
    sample_seq: int,
    sec: float,
    object_targets: List[str],
    disambiguation_targets: List[str],
    anchors: Dict[str, Dict[str, float]],
    yolo_adapter: YOLOAdapter,
    object_memory: Dict[str, Dict[str, Any]],
    persist_edge_margin: float,
    persist_max_miss_samples: int,
    dino_adapter: Optional[GroundingDINOAdapter],
    hand_adapter: Optional[MediaPipeHandAdapter],
    use_hand_model: bool,
    enable_dino_fallback: bool,
    yolo_detections: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    objects = run_yolo_objects(
        adapter=yolo_adapter,
        frame_bgr=frame_bgr,
        frame_index=frame_index,
        sample_seq=sample_seq,
        sec=sec,
        targets=object_targets,
        disambiguation_targets=disambiguation_targets,
        object_memory=object_memory,
        edge_margin=persist_edge_margin,
        persist_max_miss_samples=persist_max_miss_samples,
        precomputed_detections=yolo_detections,
    )
    detector_source = "yolo"
    if enable_dino_fallback and object_targets and len(objects) == 0:
        dino_objects = run_grounding_dino_objects(dino_adapter, frame_bgr, frame_index, sec, object_targets)
        if dino_objects:
            objects = dino_objects
            detector_source = "dino_fallback"
    hand_points: Dict[str, Dict[str, float]] = {}
    if use_hand_model:
        hand_points = run_mediapipe_hand_points(hand_adapter, frame_bgr, frame_index, sec, objects)
    object_index = {normalize_name(k): k for k in objects.keys()}
    return {
        "objects": objects,
        "object_index": object_index,
        "anchors": anchors,
        "hand_points": hand_points,
        "detector_source": detector_source,
    }


def eval_condition_code(code: str, context: Dict[str, Any], hand_in_bbox_threshold: float) -> bool:
    class MissingDetectionError(Exception):
        pass

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
        if not obj:
            raise MissingDetectionError(f"missing object: {object_id}")
        if not anchor:
            raise MissingDetectionError(f"missing anchor: {anchor_id}")
        return math.hypot(float(obj["x"] - anchor["x"]), float(obj["y"] - anchor["y"]))

    def abs_center_xy(object_id: str) -> Tuple[float, float]:
        obj = _obj(object_id)
        if not obj:
            raise MissingDetectionError(f"missing object: {object_id}")
        return (float(obj["x"]), float(obj["y"]))

    def rel_x(left_obj: str, right_obj: str) -> float:
        a = _obj(left_obj)
        b = _obj(right_obj)
        if not a:
            raise MissingDetectionError(f"missing object: {left_obj}")
        if not b:
            raise MissingDetectionError(f"missing object: {right_obj}")
        return float(a["x"] - b["x"])

    def rel_y(top_obj: str, bottom_obj: str) -> float:
        a = _obj(top_obj)
        b = _obj(bottom_obj)
        if not a:
            raise MissingDetectionError(f"missing object: {top_obj}")
        if not b:
            raise MissingDetectionError(f"missing object: {bottom_obj}")
        return float(a["y"] - b["y"])

    def rel_distance(obj_a: str, obj_b: str) -> float:
        a = _obj(obj_a)
        b = _obj(obj_b)
        if not a:
            raise MissingDetectionError(f"missing object: {obj_a}")
        if not b:
            raise MissingDetectionError(f"missing object: {obj_b}")
        return math.hypot(float(a["x"] - b["x"]), float(a["y"] - b["y"]))

    def hand_to_bbox_distance(hand_point: str, object_id: str) -> float:
        hp = context["hand_points"].get(hand_point)
        obj = _obj(object_id)
        # Hand-point missing should not hard-fail the frame.
        # Return +inf so distance-based "close-to-hand" checks evaluate to False.
        if not hp:
            return float("inf")
        if not obj:
            raise MissingDetectionError(f"missing object: {object_id}")
        return math.hypot(float(hp["x"] - obj["x"]), float(hp["y"] - obj["y"]))

    def hand_in_bbox(hand_point: str, object_id: str) -> bool:
        # Hand-point missing => not in bbox (so `not hand_in_bbox(...)` can still pass).
        if not context["hand_points"].get(hand_point):
            return False
        return hand_to_bbox_distance(hand_point, object_id) <= hand_in_bbox_threshold

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
    try:
        return bool(eval(code, safe_globals, safe_locals))
    except MissingDetectionError:
        # Strict policy: if referenced detections are missing, this frame cannot pass.
        return False


def resolve_yolo_weights(case_id: str, explicit_weights: Optional[str], registry_path: Path) -> Path:
    if explicit_weights:
        return Path(explicit_weights)
    if not registry_path.exists():
        raise RuntimeError(f"YOLO registry not found: {registry_path}")
    registry = read_json(registry_path)
    latest = registry.get("latest_by_case", {})
    if not isinstance(latest, dict) or case_id not in latest:
        raise RuntimeError(f"no latest YOLO run for case_id={case_id} in {registry_path}")
    weights = Path(str(latest[case_id].get("weights_path", "")))
    if not weights:
        raise RuntimeError(f"bad weights path in registry for case_id={case_id}")
    return weights


def find_first_match_for_step(
    cap: cv2.VideoCapture,
    fps: float,
    total_frames: int,
    offsets: List[int],
    start_frame: int,
    step_id: str,
    object_targets: List[str],
    disambiguation_targets: List[str],
    anchors: Dict[str, Dict[str, float]],
    yolo_adapter: YOLOAdapter,
    dino_adapter: Optional[GroundingDINOAdapter],
    hand_adapter: Optional[MediaPipeHandAdapter],
    use_hand_model: bool,
    enable_dino_fallback: bool,
    code: str,
    progress_every_seconds: int,
    hand_in_bbox_threshold: float,
    persist_edge_margin: float,
    persist_max_miss_samples: int,
    progress_cb: Optional[Any] = None,
) -> Tuple[bool, int, float, int, List[Dict[str, Any]]]:
    fps_int = max(1, int(round(fps)))
    sampled = 0
    sec_start = start_frame // fps_int
    sec = sec_start
    next_progress_sec = sec_start
    trace: List[Dict[str, Any]] = []
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
                frame_bgr=frame,
                frame_index=frame_idx,
                sample_seq=sampled,
                sec=sec_float,
                object_targets=object_targets,
                disambiguation_targets=disambiguation_targets,
                anchors=anchors,
                yolo_adapter=yolo_adapter,
                object_memory=object_memory,
                persist_edge_margin=persist_edge_margin,
                persist_max_miss_samples=persist_max_miss_samples,
                dino_adapter=dino_adapter,
                hand_adapter=hand_adapter,
                use_hand_model=use_hand_model,
                enable_dino_fallback=enable_dino_fallback,
            )
            matched = eval_condition_code(code, ctx, hand_in_bbox_threshold=hand_in_bbox_threshold)
            trace.append(
                {
                    "frame_index": frame_idx,
                    "second": round(sec_float, 3),
                    "matched": bool(matched),
                    "objects": ctx.get("objects", {}),
                    "hand_points": ctx.get("hand_points", {}),
                    "detector_source": ctx.get("detector_source", "yolo"),
                }
            )
            if matched:
                return True, frame_idx, sec_float, sampled, trace
        sec += 1
    return False, -1, -1.0, sampled, trace


def run_all_steps_replay_yolo(
    case_id: str,
    yolo_weights: Path,
    yolo_conf: float,
    yolo_iou: float,
    yolo_device: str,
    yolo_class_map: Optional[Path],
    enable_dino_fallback: bool,
    dino_model_id: Optional[str],
    dino_processor_id: Optional[str],
    dino_box_threshold: float,
    dino_text_threshold: float,
    dino_local_files_only: bool,
    progress_every_seconds: int,
    hand_in_bbox_threshold: float,
    persist_edge_margin: float,
    persist_max_miss_samples: int,
) -> Dict[str, Any]:
    video_path = resolve_video_path(case_id)
    step_slices = load_step_sequence(case_id)
    bundle = load_detector_plan_bundle(case_id)
    global_targets = bundle.get("grounding_dino_detect_targets", [])
    if not isinstance(global_targets, list):
        global_targets = []
    anchors = build_anchors_from_targets([str(x).strip() for x in global_targets if str(x).strip()])
    global_disambiguation_targets = [
        str(x).strip()
        for x in global_targets
        if str(x).strip() and not canonical_anchor_id(str(x))
    ]

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        raise RuntimeError("invalid fps")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    offsets = sample_offsets_per_second(max(1, int(round(fps))))

    class_names: Optional[List[str]] = None
    if yolo_class_map and yolo_class_map.exists():
        cm = read_json(yolo_class_map)
        classes = cm.get("classes", [])
        if isinstance(classes, list):
            names: List[str] = []
            for item in classes:
                if isinstance(item, dict):
                    n = str(item.get("name", "")).strip()
                    if n:
                        names.append(n)
            if names:
                class_names = names

    yolo_adapter = YOLOAdapter(
        weights_path=str(yolo_weights),
        conf=yolo_conf,
        iou=yolo_iou,
        device=yolo_device,
        class_names=class_names,
    )
    yolo_adapter.load()
    dino_adapter: Optional[GroundingDINOAdapter] = None
    if enable_dino_fallback:
        if not dino_model_id:
            local_dino_dir = ROOT / "models" / "small-models" / "object" / "grounding-dino"
            dino_model_id = str(local_dino_dir) if local_dino_dir.exists() else "IDEA-Research/grounding-dino-base"
        dino_adapter = GroundingDINOAdapter(
            model_id=str(dino_model_id),
            processor_id=(str(dino_processor_id) if dino_processor_id else None),
            box_threshold=float(dino_box_threshold),
            text_threshold=float(dino_text_threshold),
            local_files_only=bool(dino_local_files_only),
        )
        dino_adapter.load()
    hand_adapter: Optional[MediaPipeHandAdapter] = None

    cursor_frame = 0
    sampled_total = 0
    results: List[Dict[str, Any]] = []
    stop_reason = "all_steps_processed"
    t_start = time.perf_counter()

    trace_root = ROOT / "data" / case_id / "v3" / "replay-validation-yolo" / "trace"
    trace_root.mkdir(parents=True, exist_ok=True)

    for idx, step in enumerate(step_slices, start=1):
        step_id = str(step.get("slice_id", ""))
        plan = load_step_plan(case_id, step_id)
        codes = plan.get("judgement_conditions", [])
        code = str(codes[0].get("code", "")).strip() if isinstance(codes, list) and codes else ""
        object_targets = extract_object_targets_from_plan(plan)
        disambiguation_targets = list(dict.fromkeys([*object_targets, *global_disambiguation_targets]))
        use_hand_model = step_requires_hand_model(plan, code)
        if use_hand_model and hand_adapter is None:
            hand_adapter = MediaPipeHandAdapter()
            hand_adapter.load()
        print(f"[STEP] {idx}/{len(step_slices)} {step_id} start_from={cursor_frame/fps:.2f}s")

        def _progress_cb(sec_cursor: int, sampled_in_step: int) -> None:
            elapsed = time.perf_counter() - t_start
            print(
                f"[PROGRESS] step={step_id} scan_sec={sec_cursor}s "
                f"sampled_in_step={sampled_in_step} sampled_total={sampled_total + sampled_in_step} elapsed={elapsed:.1f}s"
            )

        matched, frame_idx, sec_float, sampled, trace = find_first_match_for_step(
            cap=cap,
            fps=fps,
            total_frames=total_frames,
            offsets=offsets,
            start_frame=cursor_frame,
            step_id=step_id,
            object_targets=object_targets,
            disambiguation_targets=disambiguation_targets,
            anchors=anchors,
            yolo_adapter=yolo_adapter,
            dino_adapter=dino_adapter,
            hand_adapter=hand_adapter if use_hand_model else None,
            use_hand_model=use_hand_model,
            enable_dino_fallback=enable_dino_fallback,
            code=code,
            progress_every_seconds=progress_every_seconds,
            hand_in_bbox_threshold=hand_in_bbox_threshold,
            persist_edge_margin=persist_edge_margin,
            persist_max_miss_samples=persist_max_miss_samples,
            progress_cb=_progress_cb,
        )
        sampled_total += sampled
        write_json(trace_root / f"{step_id}.json", {"step_id": step_id, "sampled": sampled, "trace": trace})

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
                "object_targets": object_targets,
            }
        )
        cursor_frame = frame_idx + 1

    cap.release()
    return {
        "pipeline_stage": "criteria-trainer:replay-validation:all-steps:yolo",
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
        "yolo_runtime": {
            "weights_path": str(yolo_weights),
            "conf": yolo_conf,
            "iou": yolo_iou,
            "device": yolo_device,
            "class_map": str(yolo_class_map) if yolo_class_map else None,
            "hand_in_bbox_threshold": hand_in_bbox_threshold,
            "persist_on_miss_when_not_edge": True,
            "persist_edge_margin": persist_edge_margin,
            "persist_max_miss_samples": persist_max_miss_samples,
        },
        "dino_fallback_runtime": {
            "enabled": bool(enable_dino_fallback),
            "model_id": str(dino_model_id) if dino_model_id else None,
            "processor_id": str(dino_processor_id) if dino_processor_id else None,
            "box_threshold": float(dino_box_threshold),
            "text_threshold": float(dino_text_threshold),
            "local_files_only": bool(dino_local_files_only),
        },
        "total_steps": len(step_slices),
        "matched_steps": sum(1 for x in results if x.get("matched")),
        "sampled_frames_total": sampled_total,
        "stop_reason": stop_reason,
        "step_results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay validate all steps using YOLO object detection backend")
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--yolo-weights", default=None, help="default from yolo_registry_v1.json latest_by_case")
    parser.add_argument("--yolo-registry", default=str(DEFAULT_REGISTRY_PATH))
    parser.add_argument("--yolo-conf", type=float, default=0.2)
    parser.add_argument("--yolo-iou", type=float, default=0.45)
    parser.add_argument("--yolo-device", default="cuda:0")
    parser.add_argument("--yolo-class-map", default=None, help="default data/<case_id>/v3/yolo-dataset/class_map.json")
    parser.add_argument("--fallback-dino", action="store_true", help="enable DINO fallback when YOLO misses all targets")
    local_dino_dir = ROOT / "models" / "small-models" / "object" / "grounding-dino"
    default_dino_model = str(local_dino_dir) if local_dino_dir.exists() else "IDEA-Research/grounding-dino-base"
    parser.add_argument("--fallback-dino-model-id", default=default_dino_model)
    parser.add_argument("--fallback-dino-processor-id", default=None)
    parser.add_argument("--fallback-dino-box-threshold", type=float, default=0.30)
    parser.add_argument("--fallback-dino-text-threshold", type=float, default=0.25)
    parser.add_argument("--fallback-dino-local-files-only", action="store_true")
    parser.add_argument("--progress-every-seconds", type=int, default=5)
    parser.add_argument("--hand-in-bbox-threshold", type=float, default=0.12)
    parser.add_argument("--persist-edge-margin", type=float, default=0.05)
    parser.add_argument("--persist-max-miss-samples", type=int, default=6)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    yolo_weights = resolve_yolo_weights(
        case_id=args.case_id,
        explicit_weights=args.yolo_weights,
        registry_path=Path(args.yolo_registry),
    )
    yolo_class_map = Path(args.yolo_class_map) if args.yolo_class_map else (ROOT / "data" / args.case_id / "v3" / "yolo-dataset" / "class_map.json")

    result = run_all_steps_replay_yolo(
        case_id=args.case_id,
        yolo_weights=yolo_weights,
        yolo_conf=max(0.01, min(0.95, float(args.yolo_conf))),
        yolo_iou=max(0.01, min(0.95, float(args.yolo_iou))),
        yolo_device=str(args.yolo_device),
        yolo_class_map=yolo_class_map if yolo_class_map.exists() else None,
        enable_dino_fallback=bool(args.fallback_dino),
        dino_model_id=(str(args.fallback_dino_model_id) if args.fallback_dino else None),
        dino_processor_id=(str(args.fallback_dino_processor_id) if (args.fallback_dino and args.fallback_dino_processor_id) else None),
        dino_box_threshold=max(0.01, min(0.95, float(args.fallback_dino_box_threshold))),
        dino_text_threshold=max(0.01, min(0.95, float(args.fallback_dino_text_threshold))),
        dino_local_files_only=bool(args.fallback_dino_local_files_only),
        progress_every_seconds=max(1, int(args.progress_every_seconds)),
        hand_in_bbox_threshold=max(0.01, min(0.5, float(args.hand_in_bbox_threshold))),
        persist_edge_margin=max(0.0, min(0.3, float(args.persist_edge_margin))),
        persist_max_miss_samples=max(1, int(args.persist_max_miss_samples)),
    )

    out_path = Path(args.output) if args.output else (ROOT / "data" / args.case_id / "v3" / "replay-validation-yolo" / "all_steps_result.json")
    write_json(out_path, result)
    print(f"[OK] output: {out_path}")
    print(f"[OK] matched steps: {result.get('matched_steps')}/{result.get('total_steps')}")


if __name__ == "__main__":
    main()
