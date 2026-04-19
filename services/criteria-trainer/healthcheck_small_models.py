import argparse
import importlib
import json
import platform
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
CRITERIA_TRAINER_DIR = Path(__file__).resolve().parent

import sys

if str(CRITERIA_TRAINER_DIR) not in sys.path:
    sys.path.insert(0, str(CRITERIA_TRAINER_DIR))


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def check_package(import_name: str) -> Dict[str, Any]:
    try:
        mod = importlib.import_module(import_name)
        return {"ok": True, "version": getattr(mod, "__version__", "unknown")}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def check_adapters(catalog: Dict[str, Any]) -> List[Dict[str, Any]]:
    checks: List[Dict[str, Any]] = []
    for item in catalog.get("models", []):
        adapter_path = str(item.get("adapter", ""))
        parts = adapter_path.rsplit(".", 1)
        if len(parts) != 2:
            checks.append({"model_id": item.get("id"), "ok": False, "error": f"bad adapter path: {adapter_path}"})
            continue
        module_name, class_name = parts
        try:
            module = importlib.import_module(module_name)
            cls = getattr(module, class_name)
            adapter = cls()
            status = adapter.healthcheck()
            checks.append({"model_id": item.get("id"), **status})
        except Exception as exc:
            checks.append({"model_id": item.get("id"), "ok": False, "error": str(exc)})
    return checks


def detect_runtime() -> Dict[str, Any]:
    runtime: Dict[str, Any] = {
        "os": platform.platform(),
        "python": platform.python_version(),
        "ffmpeg_available": shutil.which("ffmpeg") is not None,
    }
    try:
        import torch

        runtime["torch_cuda"] = bool(torch.cuda.is_available())
        runtime["torch_cuda_device_count"] = int(torch.cuda.device_count())
    except Exception as exc:
        runtime["torch_cuda"] = False
        runtime["torch_error"] = str(exc)

    try:
        import onnxruntime as ort

        runtime["onnxruntime_providers"] = ort.get_available_providers()
    except Exception as exc:
        runtime["onnxruntime_error"] = str(exc)
    return runtime


def main() -> None:
    parser = argparse.ArgumentParser(description="Health check for small model dependencies and adapters")
    parser.add_argument("--config", default=str(ROOT / "services" / "criteria-trainer" / "configs" / "small_models_v1.json"))
    parser.add_argument("--output", default=str(ROOT / "services" / "criteria-trainer" / "configs" / "healthcheck_report.json"))
    args = parser.parse_args()

    catalog = read_json(Path(args.config))
    package_checks = {
        "mediapipe": check_package("mediapipe"),
        "opencv": check_package("cv2"),
        "ultralytics": check_package("ultralytics"),
        "torch": check_package("torch"),
    }
    adapter_checks = check_adapters(catalog)
    runtime = detect_runtime()

    report = {
        "pipeline_stage": "criteria-trainer:small-model-healthcheck",
        "generated_at": utc_now_iso(),
        "catalog_version": catalog.get("version", "unknown"),
        "package_checks": package_checks,
        "adapter_checks": adapter_checks,
        "runtime": runtime,
    }
    out_path = Path(args.output)
    write_json(out_path, report)
    failed = [x for x in adapter_checks if not x.get("ok")]
    print(f"[OK] healthcheck report: {out_path}")
    print(f"[OK] adapters checked: {len(adapter_checks)}, failed: {len(failed)}")


if __name__ == "__main__":
    main()
