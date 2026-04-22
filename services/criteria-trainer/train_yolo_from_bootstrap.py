import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

DEFAULT_BOOTSTRAP_CONFIG = ROOT / "services" / "criteria-trainer" / "configs" / "yolo_bootstrap_config_v1.json"
DEFAULT_REGISTRY_PATH = ROOT / "services" / "criteria-trainer" / "configs" / "yolo_registry_v1.json"
DEFAULT_BASE_MODEL = ROOT / "models" / "small-models" / "object" / "yolo" / "yolo11n.pt"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLO from DINO-bootstrap dataset")
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--dataset-root", default=None, help="default: data/<case_id>/v3/yolo-dataset")
    parser.add_argument("--config", default=str(DEFAULT_BOOTSTRAP_CONFIG))
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY_PATH))
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--amp", action="store_true", default=False)
    parser.add_argument("--plots", action="store_true", default=False)
    args = parser.parse_args()

    cfg = read_json(Path(args.config))
    train_cfg = cfg.get("yolo_train", {})

    base_model = args.base_model or str(train_cfg.get("base_model", str(DEFAULT_BASE_MODEL)))
    epochs = int(args.epochs if args.epochs is not None else train_cfg.get("epochs", 80))
    imgsz = int(args.imgsz if args.imgsz is not None else train_cfg.get("imgsz", 640))
    batch = int(args.batch if args.batch is not None else train_cfg.get("batch", 16))
    device = str(args.device if args.device is not None else train_cfg.get("device", "cuda:0"))

    dataset_root = Path(args.dataset_root) if args.dataset_root else (ROOT / "data" / args.case_id / "v3" / "yolo-dataset")
    data_yaml = dataset_root / "data.yaml"
    class_map_path = dataset_root / "class_map.json"
    if not data_yaml.exists():
        raise RuntimeError(f"missing dataset yaml: {data_yaml}")
    class_map = read_json(class_map_path) if class_map_path.exists() else {"classes": []}

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    project_dir = ROOT / "data" / args.case_id / "v3" / "yolo-runs"
    project_dir.mkdir(parents=True, exist_ok=True)

    # Keep Ultralytics config/cache inside workspace to avoid host-profile permission issues.
    yolo_cfg_dir = ROOT / ".ultralytics"
    yolo_cfg_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(yolo_cfg_dir))
    # Optional offline font seed for Ultralytics plotting path.
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
        workers=args.workers,
        amp=args.amp,
        plots=args.plots,
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
        "case_id": args.case_id,
        "run_id": run_id,
        "generated_at": utc_now_iso(),
        "base_model": base_model,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "device": device,
        "workers": args.workers,
        "amp": bool(args.amp),
        "plots": bool(args.plots),
        "dataset_root": str(dataset_root),
        "data_yaml": str(data_yaml),
    }
    write_json(run_dir / "train_args.json", train_args)
    write_json(run_dir / "metrics_summary.json", metrics_summary)

    registry_path = Path(args.registry)
    if registry_path.exists():
        registry = read_json(registry_path)
    else:
        registry = {"version": "v1", "generated_at": utc_now_iso(), "runs": []}

    runs: List[Dict[str, Any]] = registry.get("runs", [])
    if not isinstance(runs, list):
        runs = []
    entry = {
        "case_id": args.case_id,
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
    registry["latest_by_case"][args.case_id] = {
        "run_id": run_id,
        "weights_path": entry["weights_path"],
        "run_dir": str(run_dir),
    }
    write_json(registry_path, registry)

    print(f"[OK] run dir: {run_dir}")
    print(f"[OK] weights: {entry['weights_path']}")
    print(f"[OK] registry: {registry_path}")


if __name__ == "__main__":
    main()
