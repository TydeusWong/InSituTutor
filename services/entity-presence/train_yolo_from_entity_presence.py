import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2


ROOT = Path(__file__).resolve().parents[2]
CRITERIA_DIR = ROOT / "services" / "criteria-trainer"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(CRITERIA_DIR) not in sys.path:
    sys.path.insert(0, str(CRITERIA_DIR))

from adapters.base import InferenceInput  # noqa: E402
from adapters.grounding_dino import GroundingDINOAdapter  # noqa: E402


DEFAULT_BOOTSTRAP_CONFIG = CRITERIA_DIR / "configs" / "yolo_bootstrap_config_v1.json"
DEFAULT_REGISTRY_PATH = CRITERIA_DIR / "configs" / "yolo_registry_v1.json"
DEFAULT_BASE_MODEL = ROOT / "models" / "small-models" / "object" / "yolo" / "yolo11n.pt"
STOPWORDS_FOR_DINO_QUERY = {
    "large",
    "small",
    "medium",
    "body",
    "surface",
    "object",
    "item",
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def to_rel_or_abs(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


def resolve_under_root(path_str: str) -> Path:
    p = Path(path_str)
    if not p.is_absolute():
        p = ROOT / p
    return p.resolve()


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def normalize_name(value: Any) -> str:
    return str(value).strip().lower()


def dedupe_str_list(values: Any) -> List[str]:
    if not isinstance(values, list):
        return []
    out: List[str] = []
    seen = set()
    for value in values:
        item = str(value).strip()
        if not item:
            continue
        key = normalize_name(item)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def split_entity_name_words(name: str) -> List[str]:
    return [x for x in re.split(r"[^a-zA-Z0-9]+", str(name).lower()) if x]


def canonical_to_dino_prompt(entity_name: str) -> str:
    name = str(entity_name).strip().lower()
    if not name:
        return ""

    dot_parts = [p.strip() for p in name.split(".") if p.strip()]
    if len(dot_parts) > 1:
        return " . ".join(dedupe_str_list(dot_parts))

    tokens = split_entity_name_words(name)
    if not tokens:
        return name

    queries: List[str] = []
    if len(tokens) <= 4:
        queries.append(" ".join(tokens))

    compact = [t for t in tokens if t not in STOPWORDS_FOR_DINO_QUERY]
    core_tokens = compact if compact else tokens
    core = core_tokens[-1]
    if len(core_tokens) >= 2:
        queries.append(f"{core_tokens[-2]} {core}")
        queries.append(core)
    if len(core_tokens) >= 3:
        queries.append(f"{core_tokens[-3]} {core}")

    if compact and compact != tokens:
        compact_phrase = " ".join(compact[:4])
        if compact_phrase:
            queries.append(compact_phrase)
        if len(compact) >= 2:
            queries.append(f"{compact[-2]} {compact[-1]}")

    queries = dedupe_str_list([q for q in queries if 1 <= len(q.split()) <= 4])
    return " . ".join(queries) if queries else name


def build_dino_query_map(class_names: List[str], cfg: Dict[str, Any]) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    aliases_cfg = cfg.get("dino_target_aliases", {})
    if not isinstance(aliases_cfg, dict):
        aliases_cfg = {}

    queries_by_entity: Dict[str, List[str]] = {}
    entity_by_query: Dict[str, str] = {}
    for entity in class_names:
        raw_aliases = aliases_cfg.get(entity)
        if raw_aliases is None:
            raw_aliases = aliases_cfg.get(normalize_name(entity))
        if isinstance(raw_aliases, str):
            prompt = " . ".join(dedupe_str_list([p.strip() for p in raw_aliases.split(".") if p.strip()]))
        elif isinstance(raw_aliases, list):
            prompt = " . ".join(dedupe_str_list([str(x).strip() for x in raw_aliases if str(x).strip()]))
        else:
            prompt = canonical_to_dino_prompt(entity)
        if not prompt:
            prompt = entity
        queries_by_entity[entity] = [prompt]
        # The current GroundingDINOAdapter runs one forward per detect_targets item.
        # Keep one dot-separated prompt per entity so DINO handles phrase alternatives in a single forward.
        entity_by_query.setdefault(prompt, entity)
    return queries_by_entity, entity_by_query


def split_train_val(key: str, val_ratio: float, seed: int) -> str:
    h = hashlib.sha1(f"{seed}:{key}".encode("utf-8")).hexdigest()
    bucket = int(h[:8], 16) % 10000
    return "val" if bucket < int(val_ratio * 10000) else "train"


def uniform_sample_indices(total_frames: int, n: int) -> List[int]:
    if total_frames <= 0:
        return []
    if n >= total_frames:
        return list(range(total_frames))
    if n <= 1:
        return [0]
    span = total_frames - 1
    out = set()
    for i in range(n):
        out.add(min(total_frames - 1, max(0, int(round((i * span) / (n - 1))))))
    return sorted(out)


def norm_box_metrics(norm: Dict[str, float]) -> Tuple[float, float, float, float]:
    x1 = float(norm["x1"])
    y1 = float(norm["y1"])
    x2 = float(norm["x2"])
    y2 = float(norm["y2"])
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    area = w * h
    ratio = (max(w, h) / min(w, h)) if min(w, h) > 1e-6 else 999.0
    return w, h, area, ratio


def to_yolo_line(class_id: int, norm: Dict[str, float]) -> str:
    x1 = float(norm["x1"])
    y1 = float(norm["y1"])
    x2 = float(norm["x2"])
    y2 = float(norm["y2"])
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = x2 - x1
    h = y2 - y1
    return f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


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
    return float(inter / union) if union > 1e-12 else 0.0


def pick_candidate_least_like_others(
    target: str,
    candidates: List[Dict[str, Any]],
    detections_by_target: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    if len(candidates) <= 1:
        return candidates[0]
    best_det = None
    best_key = None
    for cand in candidates:
        cand_bbox = cand.get("bbox_xyxy")
        if not isinstance(cand_bbox, dict):
            continue
        like_other_sum = 0.0
        for other_target, other_dets in detections_by_target.items():
            if other_target == target:
                continue
            for other_det in other_dets:
                other_bbox = other_det.get("bbox_xyxy")
                if isinstance(other_bbox, dict):
                    like_other_sum += float(other_det.get("score", 0.0)) * bbox_iou(cand_bbox, other_bbox)
        key = (like_other_sum, -float(cand.get("score", 0.0)))
        if best_key is None or key < best_key:
            best_key = key
            best_det = cand
    return best_det if isinstance(best_det, dict) else max(candidates, key=lambda d: float(d.get("score", 0.0)))


def draw_detections(frame: Any, detections: List[Dict[str, Any]]) -> Any:
    annotated = frame.copy()
    h, w = annotated.shape[:2]
    for det in detections:
        bbox = det.get("bbox_xyxy")
        if not isinstance(bbox, dict):
            continue
        x1 = int(max(0, min(w - 1, float(bbox["x1"]) * w)))
        y1 = int(max(0, min(h - 1, float(bbox["y1"]) * h)))
        x2 = int(max(0, min(w - 1, float(bbox["x2"]) * w)))
        y2 = int(max(0, min(h - 1, float(bbox["y2"]) * h)))
        label = f"{det.get('target', '')} {float(det.get('score', 0.0)):.2f}"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (38, 211, 111), 2)
        cv2.rectangle(annotated, (x1, max(0, y1 - 22)), (min(w - 1, x1 + max(90, len(label) * 8)), y1), (38, 211, 111), -1)
        cv2.putText(annotated, label, (x1 + 4, max(14, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (8, 17, 28), 1, cv2.LINE_AA)
    return annotated


def load_presence_items(path: Path, include_non_step: bool) -> Tuple[List[str], List[Dict[str, Any]]]:
    presence = read_json(path)
    allowed_entities = [str(x).strip() for x in presence.get("allowed_entities", []) if str(x).strip()]
    items: List[Dict[str, Any]] = []
    for item in presence.get("results", []):
        if not isinstance(item, dict):
            continue
        if not include_non_step and str(item.get("unit_class", "")).strip() != "step":
            continue
        present = [str(x).strip() for x in item.get("present_entities", []) if str(x).strip()]
        present = [x for x in allowed_entities if x in set(present)]
        if not present:
            continue
        merged = dict(item)
        merged["present_entities"] = present
        items.append(merged)
    return allowed_entities, items


def build_dataset_from_presence(
    *,
    case_id: str,
    presence_path: Path,
    config_path: Path,
    cfg: Dict[str, Any],
    dino_model_id: str,
    dino_processor_id: str,
    out_root: Path,
    include_non_step: bool,
    progress_every_frames: int,
) -> Dict[str, Any]:
    sampling_cfg = cfg.get("sampling", {})
    split_cfg = cfg.get("split", {})
    dino_cfg = cfg.get("dino", {})
    filter_cfg = cfg.get("filters", {})

    max_frames_per_step = int(sampling_cfg.get("max_frames_per_step", 30))
    min_frames_per_step = int(sampling_cfg.get("min_frames_per_step", 10))
    seed = int(sampling_cfg.get("seed", 42))
    val_ratio = float(split_cfg.get("val_ratio", 0.2))

    min_conf = float(filter_cfg.get("min_confidence", 0.25))
    min_area = float(filter_cfg.get("min_box_area_norm", 0.0005))
    max_area = float(filter_cfg.get("max_box_area_norm", 0.8))
    max_ar = float(filter_cfg.get("max_aspect_ratio", 8.0))

    dino_box_th = float(dino_cfg.get("box_threshold", 0.3))
    dino_text_th = float(dino_cfg.get("text_threshold", 0.25))
    dino_local_only = bool(dino_cfg.get("local_files_only", True))

    allowed_entities, presence_items = load_presence_items(presence_path, include_non_step=include_non_step)
    entity_has_step = {entity: False for entity in allowed_entities}
    for item in presence_items:
        for entity in item["present_entities"]:
            entity_has_step[entity] = True
    class_names = [entity for entity in allowed_entities if entity_has_step.get(entity)]
    if not class_names:
        raise RuntimeError("no entities are present in eligible step slices")
    class_to_id = {name: i for i, name in enumerate(class_names)}
    dino_queries_by_entity, entity_by_dino_query = build_dino_query_map(class_names, cfg)

    bootstrap_root = out_root / "yolo-bootstrap"
    dataset_root = out_root / "yolo-dataset"
    annotated_root = bootstrap_root / "annotated_samples"
    raw_root = bootstrap_root / "labels_dino_raw"
    filtered_root = bootstrap_root / "labels_dino_filtered"
    images_train = dataset_root / "images" / "train"
    images_val = dataset_root / "images" / "val"
    labels_train = dataset_root / "labels" / "train"
    labels_val = dataset_root / "labels" / "val"
    reset_dir(bootstrap_root)
    reset_dir(dataset_root)
    for p in [images_train, images_val, labels_train, labels_val, annotated_root, raw_root, filtered_root]:
        p.mkdir(parents=True, exist_ok=True)

    dino = GroundingDINOAdapter(
        model_id=dino_model_id,
        processor_id=dino_processor_id,
        box_threshold=dino_box_th,
        text_threshold=dino_text_th,
        local_files_only=dino_local_only,
    )
    dino.load()

    per_target_stats = defaultdict(lambda: {"raw": 0, "accepted": 0, "rejected": defaultdict(int)})
    sampled_total = 0
    labeled_images = 0
    annotated_images = 0
    planned_sampled_total = 0
    plans: List[Dict[str, Any]] = []

    for item in presence_items:
        clip_path = resolve_under_root(str(item.get("clip_path", "")))
        cap = cv2.VideoCapture(str(clip_path))
        if not cap.isOpened():
            raise RuntimeError(f"cannot open clip: {clip_path}")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        cap.release()
        if total_frames <= 0:
            continue
        n = min(max_frames_per_step, total_frames)
        n = max(1, min(total_frames, max(min_frames_per_step, n)))
        frame_indices = uniform_sample_indices(total_frames, n)
        planned_sampled_total += len(frame_indices)
        plans.append({"item": item, "clip_path": clip_path, "frame_indices": frame_indices, "fps": fps})

    t0 = time.perf_counter()
    print(
        f"[INFO] entity-presence YOLO bootstrap case={case_id} "
        f"eligible_slices={len(plans)} classes={len(class_names)} planned_sampled_frames={planned_sampled_total}"
    )
    print("[INFO] DINO query aliases:")
    for entity in class_names:
        print(f"  - {entity}: {' . '.join(dino_queries_by_entity.get(entity, [entity]))}")

    for plan in plans:
        item = plan["item"]
        clip_path = plan["clip_path"]
        targets = [x for x in item["present_entities"] if x in class_to_id]
        if not targets:
            continue
        dino_queries = dedupe_str_list(
            [query for target in targets for query in dino_queries_by_entity.get(target, [target])]
        )
        cap = cv2.VideoCapture(str(clip_path))
        if not cap.isOpened():
            continue
        unit_id = str(item.get("unit_id", "unit"))
        section_id = str(item.get("section_id", "section"))
        print(f"[SLICE] {section_id}/{unit_id} targets={len(targets)} frames={len(plan['frame_indices'])}")

        for local_frame_idx in plan["frame_indices"]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, local_frame_idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            sampled_total += 1
            frame_key = f"{section_id}__{unit_id}__f{local_frame_idx:06d}"
            img_name = f"{frame_key}.jpg"
            split = split_train_val(frame_key, val_ratio=val_ratio, seed=seed)

            payload = InferenceInput(
                frame_bgr=frame,
                timestamp_sec=(local_frame_idx / plan["fps"] if plan["fps"] else 0.0),
                frame_index=local_frame_idx,
                context={"detect_targets": dino_queries},
            )
            out = dino.infer(payload)
            detections = out.features.get("detections", [])
            if not isinstance(detections, list):
                detections = []

            detections_by_target: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
            raw_detections: List[Dict[str, Any]] = []
            for det in detections:
                if not isinstance(det, dict):
                    continue
                query = str(det.get("target", "")).strip()
                entity = entity_by_dino_query.get(query)
                raw_detections.append(det)
                if entity not in targets:
                    continue
                mapped_det = dict(det)
                mapped_det["dino_query"] = query
                mapped_det["target"] = entity
                detections_by_target[entity].append(mapped_det)

            raw_dump = {
                "section_id": section_id,
                "unit_id": unit_id,
                "clip_path": to_rel_or_abs(clip_path),
                "local_frame_index": local_frame_idx,
                "targets": targets,
                "dino_queries": dino_queries,
                "dino_queries_by_entity": {t: dino_queries_by_entity.get(t, [t]) for t in targets},
                "detections_raw": raw_detections,
                "detections": [det for values in detections_by_target.values() for det in values],
            }
            write_json(raw_root / section_id / unit_id / f"{frame_key}.json", raw_dump)

            accepted_lines: List[str] = []
            accepted_detections: List[Dict[str, Any]] = []
            for target in targets:
                per_target_stats[target]["raw"] += 1
                cands = detections_by_target.get(target, [])
                filtered_out: Dict[str, Any] = {
                    "section_id": section_id,
                    "unit_id": unit_id,
                    "local_frame_index": local_frame_idx,
                    "target": target,
                    "accepted": False,
                }
                if not cands:
                    per_target_stats[target]["rejected"]["missing_detection"] += 1
                    write_json(filtered_root / target / f"{frame_key}.json", filtered_out)
                    continue
                det = pick_candidate_least_like_others(target, cands, detections_by_target)
                score = float(det.get("score", 0.0))
                bbox = det.get("bbox_xyxy")
                if not isinstance(bbox, dict):
                    per_target_stats[target]["rejected"]["missing_bbox"] += 1
                    write_json(filtered_root / target / f"{frame_key}.json", filtered_out)
                    continue
                w, h, area, ar = norm_box_metrics(bbox)
                reject_reason = ""
                if score < min_conf:
                    reject_reason = "low_confidence"
                elif area < min_area:
                    reject_reason = "small_box"
                elif area > max_area:
                    reject_reason = "large_box"
                elif ar > max_ar:
                    reject_reason = "bad_aspect_ratio"
                if reject_reason:
                    per_target_stats[target]["rejected"][reject_reason] += 1
                    filtered_out["reject_reason"] = reject_reason
                    filtered_out["score"] = score
                    filtered_out["bbox_xyxy"] = bbox
                    write_json(filtered_root / target / f"{frame_key}.json", filtered_out)
                    continue

                filtered_out.update(
                    {
                        "accepted": True,
                        "score": score,
                        "bbox_xyxy": bbox,
                        "box_metrics": {"w": w, "h": h, "area": area, "aspect_ratio": ar},
                    }
                )
                write_json(filtered_root / target / f"{frame_key}.json", filtered_out)
                accepted_lines.append(to_yolo_line(class_to_id[target], bbox))
                accepted_detections.append(det)
                per_target_stats[target]["accepted"] += 1

                annotated = draw_detections(frame, [det])
                annotated_path = annotated_root / target / f"{frame_key}.jpg"
                annotated_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(annotated_path), annotated)
                annotated_images += 1

            img_out = (images_val if split == "val" else images_train) / img_name
            lbl_out = (labels_val if split == "val" else labels_train) / f"{frame_key}.txt"
            cv2.imwrite(str(img_out), frame)
            write_text(lbl_out, "\n".join(accepted_lines) + ("\n" if accepted_lines else ""))
            if accepted_lines:
                labeled_images += 1

            if sampled_total == 1 or sampled_total % max(1, progress_every_frames) == 0 or sampled_total == planned_sampled_total:
                elapsed = max(1e-6, time.perf_counter() - t0)
                speed = sampled_total / elapsed
                eta = max(0, planned_sampled_total - sampled_total) / max(1e-6, speed)
                print(
                    f"[PROGRESS] sampled={sampled_total}/{planned_sampled_total} "
                    f"labeled={labeled_images} annotated={annotated_images} eta={eta:.1f}s speed={speed:.2f}"
                )
        cap.release()

    data_yaml = dataset_root / "data.yaml"
    yaml_text = "\n".join(
        [
            f"path: {dataset_root.as_posix()}",
            "train: images/train",
            "val: images/val",
            f"nc: {len(class_names)}",
            "names:",
        ]
        + [f"  - {n}" for n in class_names]
    )
    write_text(data_yaml, yaml_text + "\n")
    write_json(
        dataset_root / "class_map.json",
        {
            "version": "v1",
            "case_id": case_id,
            "generated_at": utc_now_iso(),
            "source": "entity-presence",
            "classes": [{"id": i, "name": n} for i, n in enumerate(class_names)],
        },
    )

    input_snapshot = {
        "pipeline_stage": "entity-presence:yolo-bootstrap:input-snapshot",
        "generated_at": utc_now_iso(),
        "case_id": case_id,
        "refs": {
            "entity_presence": to_rel_or_abs(presence_path),
            "bootstrap_config": to_rel_or_abs(config_path),
        },
        "include_non_step": include_non_step,
        "eligible_slice_count": len(plans),
        "classes": class_names,
        "dino_queries_by_entity": dino_queries_by_entity,
    }
    write_json(bootstrap_root / "input_snapshot.json", input_snapshot)

    annotation_report = {
        "pipeline_stage": "entity-presence:yolo-bootstrap:annotation-report",
        "generated_at": utc_now_iso(),
        "case_id": case_id,
        "sampled_frames_total": sampled_total,
        "labeled_images": labeled_images,
        "annotated_bbox_images": annotated_images,
        "class_count": len(class_names),
        "classes": class_names,
        "dino_queries_by_entity": dino_queries_by_entity,
        "per_target_stats": {
            k: {
                "raw": int(v["raw"]),
                "accepted": int(v["accepted"]),
                "rejected": {rk: int(rv) for rk, rv in v["rejected"].items()},
            }
            for k, v in per_target_stats.items()
        },
        "outputs": {
            "annotated_samples_root": to_rel_or_abs(annotated_root),
            "labels_dino_raw_root": to_rel_or_abs(raw_root),
            "labels_dino_filtered_root": to_rel_or_abs(filtered_root),
            "dataset_root": to_rel_or_abs(dataset_root),
            "data_yaml": to_rel_or_abs(data_yaml),
        },
    }
    write_json(bootstrap_root / "annotation_report.json", annotation_report)
    return {
        "dataset_root": dataset_root,
        "data_yaml": data_yaml,
        "class_names": class_names,
        "bootstrap_root": bootstrap_root,
        "annotation_report": annotation_report,
    }

def train_yolo(
    *,
    case_id: str,
    cfg: Dict[str, Any],
    dataset_root: Path,
    registry_path: Path,
    base_model: str | None,
    epochs: int | None,
    imgsz: int | None,
    batch: int | None,
    device: str | None,
    workers: int,
    amp: bool,
    plots: bool,
) -> Dict[str, Any]:
    train_cfg = cfg.get("yolo_train", {})
    base_model = base_model or str(train_cfg.get("base_model", str(DEFAULT_BASE_MODEL)))
    epochs = int(epochs if epochs is not None else train_cfg.get("epochs", 80))
    imgsz = int(imgsz if imgsz is not None else train_cfg.get("imgsz", 640))
    batch = int(batch if batch is not None else train_cfg.get("batch", 16))
    device = str(device if device is not None else train_cfg.get("device", "cuda:0"))

    data_yaml = dataset_root / "data.yaml"
    class_map_path = dataset_root / "class_map.json"
    if not data_yaml.exists():
        raise RuntimeError(f"missing dataset yaml: {data_yaml}")
    class_map = read_json(class_map_path) if class_map_path.exists() else {"classes": []}

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    project_dir = ROOT / "data" / case_id / "v3" / "yolo-runs"
    project_dir.mkdir(parents=True, exist_ok=True)

    yolo_cfg_dir = ROOT / ".ultralytics"
    yolo_cfg_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(yolo_cfg_dir))
    arial_src = Path(r"C:\Windows\Fonts\arial.ttf")
    arial_dst = yolo_cfg_dir / "Arial.ttf"
    if arial_src.exists() and not arial_dst.exists():
        try:
            arial_dst.write_bytes(arial_src.read_bytes())
        except Exception:
            pass

    from ultralytics import YOLO

    model = YOLO(base_model)
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(project_dir),
        name=run_id,
        exist_ok=True,
        workers=workers,
        amp=amp,
        plots=plots,
        verbose=True,
    )

    run_dir = project_dir / run_id
    best_pt = run_dir / "weights" / "best.pt"
    last_pt = run_dir / "weights" / "last.pt"
    results_csv = run_dir / "results.csv"
    metrics_summary: Dict[str, Any] = {}
    try:
        metrics_summary["results_dict"] = dict(getattr(results, "results_dict", {}) or {})
    except Exception:
        metrics_summary["results_dict"] = {}
    metrics_summary["best_exists"] = best_pt.exists()
    metrics_summary["last_exists"] = last_pt.exists()
    metrics_summary["results_csv_exists"] = results_csv.exists()

    train_args = {
        "case_id": case_id,
        "run_id": run_id,
        "generated_at": utc_now_iso(),
        "base_model": base_model,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "device": device,
        "workers": workers,
        "amp": bool(amp),
        "plots": bool(plots),
        "dataset_root": str(dataset_root),
        "data_yaml": str(data_yaml),
        "source": "entity-presence",
    }
    write_json(run_dir / "train_args.json", train_args)
    write_json(run_dir / "metrics_summary.json", metrics_summary)

    if registry_path.exists():
        registry = read_json(registry_path)
    else:
        registry = {"version": "v1", "generated_at": utc_now_iso(), "runs": []}
    runs: List[Dict[str, Any]] = registry.get("runs", [])
    if not isinstance(runs, list):
        runs = []
    entry = {
        "case_id": case_id,
        "run_id": run_id,
        "created_at": utc_now_iso(),
        "weights_path": str(best_pt if best_pt.exists() else last_pt),
        "run_dir": str(run_dir),
        "classes": class_map.get("classes", []),
        "train_args": train_args,
        "metrics_summary": metrics_summary,
    }
    runs.append(entry)
    registry["runs"] = runs
    registry["generated_at"] = utc_now_iso()
    registry["latest_by_case"] = registry.get("latest_by_case", {})
    if not isinstance(registry["latest_by_case"], dict):
        registry["latest_by_case"] = {}
    registry["latest_by_case"][case_id] = {
        "run_id": run_id,
        "weights_path": entry["weights_path"],
        "run_dir": str(run_dir),
    }
    write_json(registry_path, registry)
    return entry


def main() -> None:
    parser = argparse.ArgumentParser(description="Early DINO bootstrap and YOLO training from entity-presence step clips")
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--entity-presence", default=None, help="default data/<case_id>/v2/segmentation/entity-presence/entity_presence.json")
    parser.add_argument("--config", default=str(DEFAULT_BOOTSTRAP_CONFIG))
    parser.add_argument("--dino-model-id", default=str(ROOT / "models" / "small-models" / "object" / "grounding-dino"))
    parser.add_argument("--dino-processor-id", default=str(ROOT / "models" / "small-models" / "object" / "grounding-dino"))
    parser.add_argument("--output-root", default=None, help="default data/<case_id>/v3")
    parser.add_argument("--include-non-step", action="store_true", help="also use not_related/error clips when entities are present")
    parser.add_argument("--progress-every-frames", type=int, default=10)
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY_PATH))
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--amp", action="store_true", default=False)
    parser.add_argument("--plots", action="store_true", default=False)
    parser.add_argument("--skip-train", action="store_true", help="only build DINO-labeled YOLO dataset and bbox trace images")
    args = parser.parse_args()

    cfg = read_json(Path(args.config))
    out_root = Path(args.output_root) if args.output_root else (ROOT / "data" / args.case_id / "v3")
    presence_path = (
        Path(args.entity_presence)
        if args.entity_presence
        else ROOT / "data" / args.case_id / "v2" / "segmentation" / "entity-presence" / "entity_presence.json"
    )

    dataset_info = build_dataset_from_presence(
        case_id=args.case_id,
        presence_path=presence_path,
        config_path=Path(args.config),
        cfg=cfg,
        dino_model_id=args.dino_model_id,
        dino_processor_id=args.dino_processor_id,
        out_root=out_root,
        include_non_step=bool(args.include_non_step),
        progress_every_frames=max(1, int(args.progress_every_frames)),
    )

    train_entry = None
    if not args.skip_train:
        train_entry = train_yolo(
            case_id=args.case_id,
            cfg=cfg,
            dataset_root=dataset_info["dataset_root"],
            registry_path=Path(args.registry),
            base_model=args.base_model,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            amp=bool(args.amp),
            plots=bool(args.plots),
        )

    summary = {
        "pipeline_stage": "entity-presence:dino-bootstrap-yolo-train",
        "generated_at": utc_now_iso(),
        "case_id": args.case_id,
        "entity_presence": to_rel_or_abs(presence_path),
        "dataset_root": to_rel_or_abs(dataset_info["dataset_root"]),
        "annotation_report": to_rel_or_abs(dataset_info["bootstrap_root"] / "annotation_report.json"),
        "annotated_samples_root": dataset_info["annotation_report"]["outputs"]["annotated_samples_root"],
        "train": train_entry,
    }
    write_json(out_root / "entity_presence_yolo_pipeline_summary.json", summary)
    print(f"[OK] dataset: {dataset_info['dataset_root']}")
    print(f"[OK] annotated bbox images: {dataset_info['annotation_report']['outputs']['annotated_samples_root']}")
    print(f"[OK] annotation report: {dataset_info['bootstrap_root'] / 'annotation_report.json'}")
    if train_entry:
        print(f"[OK] yolo weights: {train_entry['weights_path']}")
        print(f"[OK] registry: {Path(args.registry)}")
    print(f"[OK] summary: {out_root / 'entity_presence_yolo_pipeline_summary.json'}")


if __name__ == "__main__":
    main()
