import argparse
import json
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]

MODEL_DIRS = [
    ROOT / "models" / "small-models" / "pose" / "mediapipe-pose-landmarker",
    ROOT / "models" / "small-models" / "hand" / "mediapipe-hand-landmarker",
    ROOT / "models" / "small-models" / "object" / "grounding-dino",
    ROOT / "models" / "small-models" / "object" / "yolo",
    ROOT / "models" / "small-models" / "color" / "opencv",
]

README_CONTENT = """# Small Model Slot

This folder keeps runtime assets or weights for this model family.
"""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def ensure_dirs() -> List[str]:
    created: List[str] = []
    for model_dir in MODEL_DIRS:
        model_dir.mkdir(parents=True, exist_ok=True)
        readme = model_dir / "README.md"
        if not readme.exists():
            readme.write_text(README_CONTENT, encoding="utf-8")
        created.append(str(model_dir.relative_to(ROOT)))
    (ROOT / "services" / "criteria-trainer" / "adapters").mkdir(parents=True, exist_ok=True)
    (ROOT / "services" / "criteria-trainer" / "configs").mkdir(parents=True, exist_ok=True)
    return created


def detect_runtime() -> Dict[str, Any]:
    gpu = {"torch_cuda": False, "torch_device_count": 0, "onnxruntime_cuda": False}
    try:
        import torch

        gpu["torch_cuda"] = bool(torch.cuda.is_available())
        gpu["torch_device_count"] = int(torch.cuda.device_count())
    except Exception:
        pass

    try:
        import onnxruntime as ort

        providers = ort.get_available_providers()
        gpu["onnxruntime_cuda"] = "CUDAExecutionProvider" in providers
    except Exception:
        pass

    return {
        "os": platform.platform(),
        "python": platform.python_version(),
        "ffmpeg_available": shutil.which("ffmpeg") is not None,
        "gpu": gpu,
    }


def run_healthcheck(healthcheck_script: Path) -> Dict[str, Any]:
    proc = subprocess.run(
        ["python", str(healthcheck_script), "--output", str(ROOT / "services" / "criteria-trainer" / "configs" / "healthcheck_report.json")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "exit_code": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize small-model environment and verify dependencies.")
    parser.add_argument("--skip-healthcheck", action="store_true")
    args = parser.parse_args()

    created_dirs = ensure_dirs()
    runtime = detect_runtime()
    health = {"skipped": True}
    if not args.skip_healthcheck:
        health = run_healthcheck(ROOT / "services" / "criteria-trainer" / "healthcheck_small_models.py")

    setup_report = {
        "pipeline_stage": "criteria-trainer:small-model-setup",
        "generated_at": utc_now_iso(),
        "created_dirs": created_dirs,
        "runtime": runtime,
        "healthcheck": health,
    }
    out_path = ROOT / "services" / "criteria-trainer" / "configs" / "small_model_setup_report.json"
    write_json(out_path, setup_report)
    print(f"[OK] setup report: {out_path}")


if __name__ == "__main__":
    main()
