import argparse
import importlib.util
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import cv2


ROOT = Path(__file__).resolve().parents[2]
THIS_DIR = Path(__file__).resolve().parent
TRAIN_SCRIPT = THIS_DIR / "train_yolo_from_entity_presence.py"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

spec = importlib.util.spec_from_file_location("entity_presence_train", TRAIN_SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load {TRAIN_SCRIPT}")
entity_train = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entity_train)


DEFAULT_BOOTSTRAP_CONFIG = ROOT / "services" / "criteria-trainer" / "configs" / "yolo_bootstrap_config_v1.json"
DEFAULT_REGISTRY_PATH = ROOT / "services" / "criteria-trainer" / "configs" / "yolo_registry_v1.json"


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
    path = Path(path_str)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def reset_dir(path: Path) -> None:
    if path.exists():
        import shutil

        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def load_class_names(validation: Dict[str, Any]) -> List[str]:
    task_context = validation.get("task_context", {})
    allowed = task_context.get("allowed_entities", []) if isinstance(task_context, dict) else []
    names = [str(x).strip() for x in allowed if str(x).strip()]
    if names:
        return names
    seen = []
    for item in validation.get("results", []):
        if not isinstance(item, dict) or not item.get("accepted_for_yolo"):
            continue
        target = str(item.get("target", "")).strip()
        if target and target not in seen:
            seen.append(target)
    return seen


def source_frame_for_validation_item(bootstrap_root: Path, item: Dict[str, Any]) -> Any:
    filtered_label = resolve_under_root(str(item.get("filtered_label", "")))
    label = read_json(filtered_label)
    section_id = str(label.get("section_id", item.get("section_id", "section")))
    unit_id = str(label.get("unit_id", item.get("unit_id", "unit")))
    frame_key = filtered_label.stem
    raw_path = bootstrap_root / "labels_dino_raw" / section_id / unit_id / f"{frame_key}.json"
    raw = read_json(raw_path)
    clip_path = resolve_under_root(str(raw.get("clip_path", "")))
    frame_index = int(raw.get("local_frame_index", label.get("local_frame_index", 0)) or 0)
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open clip: {clip_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"cannot read frame {frame_index} from {clip_path}")
    return frame


def build_validated_dataset(
    *,
    case_id: str,
    validation_path: Path,
    cfg: Dict[str, Any],
    out_root: Path,
) -> Dict[str, Any]:
    validation = read_json(validation_path)
    bootstrap_root = out_root / "yolo-bootstrap"
    dataset_root = out_root / "yolo-dataset"
    images_train = dataset_root / "images" / "train"
    images_val = dataset_root / "images" / "val"
    labels_train = dataset_root / "labels" / "train"
    labels_val = dataset_root / "labels" / "val"
    reset_dir(dataset_root)
    for path in [images_train, images_val, labels_train, labels_val]:
        path.mkdir(parents=True, exist_ok=True)

    split_cfg = cfg.get("split", {})
    sampling_cfg = cfg.get("sampling", {})
    val_ratio = float(split_cfg.get("val_ratio", 0.2))
    seed = int(sampling_cfg.get("seed", 42))

    class_names = load_class_names(validation)
    if not class_names:
        raise RuntimeError("no class names found in validation file")
    class_to_id = {name: idx for idx, name in enumerate(class_names)}

    labels_by_frame: Dict[str, List[str]] = defaultdict(list)
    records_by_frame: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in validation.get("results", []):
        if not isinstance(item, dict) or not item.get("accepted_for_yolo"):
            continue
        target = str(item.get("target", "")).strip()
        if target not in class_to_id:
            continue
        filtered_label = resolve_under_root(str(item.get("filtered_label", "")))
        frame_key = filtered_label.stem
        bbox = item.get("bbox_xyxy")
        if not isinstance(bbox, dict):
            continue
        labels_by_frame[frame_key].append(entity_train.to_yolo_line(class_to_id[target], bbox))
        records_by_frame[frame_key].append(item)

    if not labels_by_frame:
        raise RuntimeError("no Omni-validated yes labels found")

    for frame_key, labels in labels_by_frame.items():
        split = entity_train.split_train_val(frame_key, val_ratio=val_ratio, seed=seed)
        image_out = (images_val if split == "val" else images_train) / f"{frame_key}.jpg"
        label_out = (labels_val if split == "val" else labels_train) / f"{frame_key}.txt"
        frame = source_frame_for_validation_item(bootstrap_root, records_by_frame[frame_key][0])
        cv2.imwrite(str(image_out), frame)
        write_text(label_out, "\n".join(labels) + "\n")

    data_yaml = dataset_root / "data.yaml"
    yaml_text = "\n".join(
        [
            f"path: {dataset_root.as_posix()}",
            "train: images/train",
            "val: images/val",
            f"nc: {len(class_names)}",
            "names:",
        ]
        + [f"  - {name}" for name in class_names]
    )
    write_text(data_yaml, yaml_text + "\n")
    write_json(
        dataset_root / "class_map.json",
        {
            "version": "v1",
            "case_id": case_id,
            "generated_at": utc_now_iso(),
            "source": "omni-validated-dino",
            "classes": [{"id": i, "name": name} for i, name in enumerate(class_names)],
        },
    )

    summary = {
        "pipeline_stage": "entity-presence:omni-validated-dino-yolo-dataset",
        "generated_at": utc_now_iso(),
        "case_id": case_id,
        "validation": to_rel_or_abs(validation_path),
        "dataset_root": to_rel_or_abs(dataset_root),
        "data_yaml": to_rel_or_abs(data_yaml),
        "class_count": len(class_names),
        "classes": class_names,
        "validated_label_count": sum(len(v) for v in labels_by_frame.values()),
        "image_count": len(labels_by_frame),
    }
    write_json(out_root / "validated_dino_yolo_dataset_summary.json", summary)
    return {
        "dataset_root": dataset_root,
        "data_yaml": data_yaml,
        "class_names": class_names,
        "summary": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLO using only Omni-validated DINO bbox labels")
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--validation", default=None, help="default data/<case_id>/v3/yolo-bootstrap/dino_annotation_validation.json")
    parser.add_argument("--config", default=str(DEFAULT_BOOTSTRAP_CONFIG))
    parser.add_argument("--output-root", default=None, help="default data/<case_id>/v3")
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY_PATH))
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--amp", action="store_true", default=False)
    parser.add_argument("--plots", action="store_true", default=False)
    parser.add_argument("--skip-train", action="store_true", help="only build the validated YOLO dataset")
    args = parser.parse_args()

    cfg = read_json(Path(args.config))
    out_root = Path(args.output_root) if args.output_root else ROOT / "data" / args.case_id / "v3"
    validation_path = (
        Path(args.validation)
        if args.validation
        else out_root / "yolo-bootstrap" / "dino_annotation_validation.json"
    )
    dataset_info = build_validated_dataset(
        case_id=args.case_id,
        validation_path=validation_path,
        cfg=cfg,
        out_root=out_root,
    )

    train_entry = None
    if not args.skip_train:
        train_entry = entity_train.train_yolo(
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
        "pipeline_stage": "entity-presence:train-yolo-from-omni-validated-dino",
        "generated_at": utc_now_iso(),
        "case_id": args.case_id,
        "validation": to_rel_or_abs(validation_path),
        "dataset_root": to_rel_or_abs(dataset_info["dataset_root"]),
        "dataset_summary": dataset_info["summary"],
        "train": train_entry,
    }
    write_json(out_root / "validated_dino_yolo_pipeline_summary.json", summary)
    print(f"[OK] dataset: {dataset_info['dataset_root']}")
    print(f"[OK] validated labels: {dataset_info['summary']['validated_label_count']}")
    if train_entry:
        print(f"[OK] yolo weights: {train_entry['weights_path']}")
        print(f"[OK] registry: {Path(args.registry)}")
    print(f"[OK] summary: {out_root / 'validated_dino_yolo_pipeline_summary.json'}")


if __name__ == "__main__":
    main()
