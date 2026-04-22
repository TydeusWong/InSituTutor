import argparse
import hashlib
import json
import math
import random
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from adapters.base import InferenceInput  # noqa: E402
from adapters.grounding_dino import GroundingDINOAdapter  # noqa: E402

DEFAULT_BOOTSTRAP_CONFIG = ROOT / "services" / "criteria-trainer" / "configs" / "yolo_bootstrap_config_v1.json"


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


def normalize_name(name: Any) -> str:
    return str(name).strip()


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


def load_step_slices(case_id: str) -> List[Dict[str, Any]]:
    index_path = ROOT / "data" / case_id / "v2" / "slices" / "index.json"
    index_data = read_json(index_path)
    slices = index_data.get("slices", [])
    out = [x for x in slices if x.get("slice_type") == "step"]
    out.sort(key=lambda x: float((x.get("time_range") or {}).get("start_sec", 0.0)))
    return out


def load_step_plan(case_id: str, step_id: str) -> Dict[str, Any]:
    return read_json(ROOT / "data" / case_id / "v2" / "detector-plans" / f"{step_id}.json")


def extract_dino_targets(step_plan: Dict[str, Any]) -> List[str]:
    targets: List[str] = []
    for item in step_plan.get("model_selection", []):
        if not isinstance(item, dict):
            continue
        if str(item.get("model_id", "")).strip() != "grounding-dino":
            continue
        if isinstance(item.get("detect_targets"), list):
            for t in item["detect_targets"]:
                ts = normalize_name(t)
                if ts and ts not in targets:
                    targets.append(ts)
    return targets


def has_dino(step_plan: Dict[str, Any]) -> bool:
    models_required = step_plan.get("models_required", [])
    if isinstance(models_required, list) and any(str(x).strip() == "grounding-dino" for x in models_required):
        return True
    return bool(extract_dino_targets(step_plan))


def uniform_sample_indices(start_frame: int, end_frame_exclusive: int, n: int) -> List[int]:
    if end_frame_exclusive <= start_frame:
        return [start_frame]
    frame_len = end_frame_exclusive - start_frame
    if n >= frame_len:
        return list(range(start_frame, end_frame_exclusive))
    if n <= 1:
        return [start_frame]
    span = frame_len - 1
    out = set()
    for i in range(n):
        pos = start_frame + int(round((i * span) / (n - 1)))
        out.add(min(end_frame_exclusive - 1, max(start_frame, pos)))
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
    detections_by_target: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    if len(candidates) <= 1:
        return candidates[0]
    other_targets = [t for t in detections_by_target.keys() if t != target]
    if not other_targets:
        return max(candidates, key=lambda d: float(d.get("score", 0.0)))

    # "like other targets" score: sum(other_score * IoU) over all other-target detections.
    # Pick minimal score; tie-break by higher self score.
    best_det = None
    best_key = None
    for cand in candidates:
        cand_bbox = cand.get("bbox_xyxy")
        if not isinstance(cand_bbox, dict):
            continue
        like_other_sum = 0.0
        for ot in other_targets:
            for od in detections_by_target.get(ot, []):
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


def split_train_val(key: str, val_ratio: float, seed: int) -> str:
    h = hashlib.sha1(f"{seed}:{key}".encode("utf-8")).hexdigest()
    bucket = int(h[:8], 16) % 10000
    return "val" if bucket < int(val_ratio * 10000) else "train"


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap YOLO dataset from DINO pseudo labels")
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--config", default=str(DEFAULT_BOOTSTRAP_CONFIG))
    parser.add_argument("--dino-model-id", default=str(ROOT / "models" / "small-models" / "object" / "grounding-dino"))
    parser.add_argument("--dino-processor-id", default=str(ROOT / "models" / "small-models" / "object" / "grounding-dino"))
    parser.add_argument("--output-root", default=None, help="default: data/<case_id>/v3")
    parser.add_argument("--progress-every-frames", type=int, default=10, help="print progress every N sampled frames")
    args = parser.parse_args()

    case_id = args.case_id
    cfg = read_json(Path(args.config))
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

    out_root = Path(args.output_root) if args.output_root else (ROOT / "data" / case_id / "v3")
    bootstrap_root = out_root / "yolo-bootstrap"
    dataset_root = out_root / "yolo-dataset"
    frames_root = bootstrap_root / "frames"
    raw_root = bootstrap_root / "labels_dino_raw"
    filtered_root = bootstrap_root / "labels_dino_filtered"
    images_train = dataset_root / "images" / "train"
    images_val = dataset_root / "images" / "val"
    labels_train = dataset_root / "labels" / "train"
    labels_val = dataset_root / "labels" / "val"
    for p in [images_train, images_val, labels_train, labels_val]:
        p.mkdir(parents=True, exist_ok=True)

    step_slices = load_step_slices(case_id)
    scene_entities_path = ROOT / "data" / case_id / "v2" / "strategy" / "scene_entities_v1.json"
    scene_entities = read_json(scene_entities_path).get("entity_names", [])

    dino_steps: List[Tuple[Dict[str, Any], Dict[str, Any], List[str]]] = []
    all_targets: List[str] = []
    for step in step_slices:
        step_id = str(step.get("slice_id", "")).strip()
        if not step_id:
            continue
        plan = load_step_plan(case_id, step_id)
        if not has_dino(plan):
            continue
        targets = extract_dino_targets(plan)
        if not targets:
            continue
        dino_steps.append((step, plan, targets))
        for t in targets:
            if t not in all_targets:
                all_targets.append(t)

    if not dino_steps:
        raise RuntimeError("no dino-based steps found")

    # Keep class list constrained by declared targets; prefer scene_entities order.
    class_names: List[str] = []
    for name in scene_entities:
        ns = normalize_name(name)
        if ns in all_targets and ns not in class_names:
            class_names.append(ns)
    for name in all_targets:
        nn = normalize_name(name)
        if nn not in class_names:
            class_names.append(nn)
    class_to_id = {name: i for i, name in enumerate(class_names)}

    video_path = resolve_video_path(case_id)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        raise RuntimeError("invalid fps")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    dino = GroundingDINOAdapter(
        model_id=args.dino_model_id,
        processor_id=args.dino_processor_id,
        box_threshold=dino_box_th,
        text_threshold=dino_text_th,
        local_files_only=dino_local_only,
    )
    dino.load()

    random.seed(seed)
    per_target_stats = defaultdict(lambda: {"raw": 0, "accepted": 0, "rejected": defaultdict(int)})
    sampled_total = 0
    labeled_images = 0
    progress_every = max(1, int(args.progress_every_frames))

    # Pre-calc planned sampled frames for progress/ETA.
    planned_sampled_total = 0
    for step, _plan, _targets in dino_steps:
        tr = step.get("time_range", {})
        start_sec = float(tr.get("start_sec", 0.0) or 0.0)
        end_sec = float(tr.get("end_sec", start_sec) or start_sec)
        if end_sec < start_sec:
            end_sec = start_sec
        start_frame = max(0, int(math.floor(start_sec * fps)))
        end_frame = min(total_frames, int(math.ceil(end_sec * fps)))
        frame_len = max(1, end_frame - start_frame)
        n = min(max_frames_per_step, frame_len)
        n = max(1, min(frame_len, max(min_frames_per_step, n)))
        planned_sampled_total += len(uniform_sample_indices(start_frame, end_frame, n))

    t_bootstrap_start = time.perf_counter()
    print(f"[INFO] bootstrap start case={case_id} dino_steps={len(dino_steps)} planned_sampled_frames={planned_sampled_total}")

    # Use all DINO targets seen across all steps as disambiguation pool.
    # This makes "least-like-other-entities" compare against global entities, not only current step targets.
    global_disambiguation_targets = [t for t in all_targets if t]

    for step, _plan, targets in dino_steps:
        step_id = str(step.get("slice_id", "")).strip()
        tr = step.get("time_range", {})
        start_sec = float(tr.get("start_sec", 0.0) or 0.0)
        end_sec = float(tr.get("end_sec", start_sec) or start_sec)
        if end_sec < start_sec:
            end_sec = start_sec
        start_frame = max(0, int(math.floor(start_sec * fps)))
        end_frame = min(total_frames, int(math.ceil(end_sec * fps)))
        frame_len = max(1, end_frame - start_frame)
        n = min(max_frames_per_step, frame_len)
        n = max(1, min(frame_len, max(min_frames_per_step, n)))
        frame_indices = uniform_sample_indices(start_frame, end_frame, n)
        print(
            f"[STEP] {step_id} range={start_sec:.2f}-{end_sec:.2f}s "
            f"targets={len(targets)} sampled_frames={len(frame_indices)}"
        )

        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            sampled_total += 1
            sec = frame_idx / fps
            frame_key = f"{step_id}__f{frame_idx:06d}"
            img_name = f"{frame_key}.jpg"

            # Keep requested trace layout: one frame copy per target folder
            for t in targets:
                out_img = frames_root / step_id / t / img_name
                out_img.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out_img), frame)

            payload = InferenceInput(
                frame_bgr=frame,
                timestamp_sec=sec,
                frame_index=frame_idx,
                context={"detect_targets": global_disambiguation_targets},
            )
            out = dino.infer(payload)
            detections = out.features.get("detections", [])
            if not isinstance(detections, list):
                detections = []

            detections_by_target: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
            for det in detections:
                if not isinstance(det, dict):
                    continue
                tname = normalize_name(det.get("target", ""))
                if not tname:
                    continue
                detections_by_target[tname].append(det)

            accepted_lines: List[str] = []
            selected_by_target: Dict[str, Dict[str, Any]] = {}
            for t in targets:
                cands = detections_by_target.get(t, [])
                if not cands:
                    continue
                selected_by_target[t] = pick_candidate_least_like_others(
                    target=t,
                    candidates=cands,
                    detections_by_target=detections_by_target,
                )

            raw_frame_dump = {
                "frame_index": frame_idx,
                "timestamp_sec": sec,
                "step_targets": targets,
                "disambiguation_targets": global_disambiguation_targets,
                "detections": detections,
                "detections_by_target_count": {k: len(v) for k, v in detections_by_target.items()},
                "selected_by_target": selected_by_target,
                "selection_policy": "min_sum(other_score * iou) with self-score tiebreak",
            }

            for t in targets:
                det = selected_by_target.get(t)
                per_target_stats[t]["raw"] += 1
                raw_file = raw_root / step_id / t / f"{frame_key}.json"
                write_json(raw_file, raw_frame_dump)

                filtered_out: Dict[str, Any] = {"frame_index": frame_idx, "timestamp_sec": sec, "target": t, "accepted": False}
                if not isinstance(det, dict):
                    per_target_stats[t]["rejected"]["missing_detection"] += 1
                    write_json(filtered_root / step_id / t / f"{frame_key}.json", filtered_out)
                    continue

                score = float(det.get("score", 0.0))
                bbox = det.get("bbox_xyxy")
                if not isinstance(bbox, dict):
                    per_target_stats[t]["rejected"]["missing_bbox"] += 1
                    write_json(filtered_root / step_id / t / f"{frame_key}.json", filtered_out)
                    continue
                w, h, area, ar = norm_box_metrics(bbox)
                if score < min_conf:
                    per_target_stats[t]["rejected"]["low_confidence"] += 1
                    write_json(filtered_root / step_id / t / f"{frame_key}.json", filtered_out)
                    continue
                if area < min_area:
                    per_target_stats[t]["rejected"]["small_box"] += 1
                    write_json(filtered_root / step_id / t / f"{frame_key}.json", filtered_out)
                    continue
                if area > max_area:
                    per_target_stats[t]["rejected"]["large_box"] += 1
                    write_json(filtered_root / step_id / t / f"{frame_key}.json", filtered_out)
                    continue
                if ar > max_ar:
                    per_target_stats[t]["rejected"]["bad_aspect_ratio"] += 1
                    write_json(filtered_root / step_id / t / f"{frame_key}.json", filtered_out)
                    continue

                class_id = class_to_id.get(t)
                if class_id is None:
                    per_target_stats[t]["rejected"]["missing_class_map"] += 1
                    write_json(filtered_root / step_id / t / f"{frame_key}.json", filtered_out)
                    continue

                filtered_out = {
                    "frame_index": frame_idx,
                    "timestamp_sec": sec,
                    "target": t,
                    "accepted": True,
                    "score": score,
                    "bbox_xyxy": bbox,
                    "box_metrics": {"w": w, "h": h, "area": area, "aspect_ratio": ar},
                }
                write_json(filtered_root / step_id / t / f"{frame_key}.json", filtered_out)
                accepted_lines.append(to_yolo_line(class_id, bbox))
                per_target_stats[t]["accepted"] += 1

            split = split_train_val(frame_key, val_ratio=val_ratio, seed=seed)
            img_out = (images_val if split == "val" else images_train) / img_name
            lbl_out = (labels_val if split == "val" else labels_train) / f"{frame_key}.txt"
            cv2.imwrite(str(img_out), frame)
            write_text(lbl_out, "\n".join(accepted_lines) + ("\n" if accepted_lines else ""))
            if accepted_lines:
                labeled_images += 1

            if sampled_total == 1 or (sampled_total % progress_every == 0) or (sampled_total == planned_sampled_total):
                elapsed = max(1e-6, time.perf_counter() - t_bootstrap_start)
                speed = sampled_total / elapsed
                remain = max(0, planned_sampled_total - sampled_total)
                eta_sec = remain / max(1e-6, speed)
                pct = (sampled_total / max(1, planned_sampled_total)) * 100.0
                print(
                    f"[PROGRESS] sampled={sampled_total}/{planned_sampled_total} "
                    f"({pct:.1f}%) elapsed={elapsed:.1f}s eta={eta_sec:.1f}s "
                    f"speed={speed:.2f}fps-sampled"
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

    class_map = {
        "version": "v1",
        "case_id": case_id,
        "generated_at": utc_now_iso(),
        "classes": [{"id": i, "name": n} for i, n in enumerate(class_names)],
    }
    write_json(dataset_root / "class_map.json", class_map)

    input_snapshot = {
        "pipeline_stage": "criteria-trainer:yolo-bootstrap:input-snapshot",
        "generated_at": utc_now_iso(),
        "case_id": case_id,
        "refs": {
            "slice_index": f"data/{case_id}/v2/slices/index.json",
            "detector_plan_bundle": f"data/{case_id}/v2/detector-plans/detector_plan_v2.json",
            "scene_entities": f"data/{case_id}/v2/strategy/scene_entities_v1.json",
            "video": str(video_path),
            "bootstrap_config": str(Path(args.config)),
        },
    }
    write_json(bootstrap_root / "input_snapshot.json", input_snapshot)

    annotation_report = {
        "pipeline_stage": "criteria-trainer:yolo-bootstrap:annotation-report",
        "generated_at": utc_now_iso(),
        "case_id": case_id,
        "sampled_frames_total": sampled_total,
        "labeled_images": labeled_images,
        "class_count": len(class_names),
        "classes": class_names,
        "per_target_stats": {
            k: {
                "raw": int(v["raw"]),
                "accepted": int(v["accepted"]),
                "rejected": {rk: int(rv) for rk, rv in v["rejected"].items()},
            }
            for k, v in per_target_stats.items()
        },
        "outputs": {
            "frames_root": str(frames_root),
            "labels_dino_raw_root": str(raw_root),
            "labels_dino_filtered_root": str(filtered_root),
            "dataset_root": str(dataset_root),
            "data_yaml": str(data_yaml),
        },
    }
    write_json(bootstrap_root / "annotation_report.json", annotation_report)

    print(f"[OK] sampled frames: {sampled_total}")
    print(f"[OK] labeled images: {labeled_images}")
    print(f"[OK] dataset: {dataset_root}")
    print(f"[OK] report: {bootstrap_root / 'annotation_report.json'}")


if __name__ == "__main__":
    main()
