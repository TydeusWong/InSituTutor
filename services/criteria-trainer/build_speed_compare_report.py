import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from adapters.base import InferenceInput  # noqa: E402
from adapters.grounding_dino import GroundingDINOAdapter  # noqa: E402
from adapters.yolo import YOLOAdapter  # noqa: E402

DEFAULT_YOLO_REGISTRY = ROOT / "services" / "criteria-trainer" / "configs" / "yolo_registry_v1.json"


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


def resolve_yolo_weights(case_id: str, explicit_weights: Optional[str], registry_path: Path) -> Path:
    if explicit_weights:
        return Path(explicit_weights)
    registry = read_json(registry_path)
    latest = registry.get("latest_by_case", {})
    if not isinstance(latest, dict) or case_id not in latest:
        raise RuntimeError(f"no latest YOLO run for case_id={case_id} in {registry_path}")
    return Path(str(latest[case_id]["weights_path"]))


def pick_targets_for_benchmark(case_id: str) -> List[str]:
    bundle = read_json(ROOT / "data" / case_id / "v2" / "detector-plans" / "detector_plan_v2.json")
    targets = bundle.get("grounding_dino_detect_targets", [])
    out: List[str] = []
    if isinstance(targets, list):
        for x in targets:
            t = str(x).strip()
            if t and t not in out:
                out.append(t)
    if not out:
        out = ["blue box"]
    return out[:6]


def benchmark_single_frame(
    case_id: str,
    yolo_weights: Path,
    dino_model_id: str,
    dino_processor_id: Optional[str],
    dino_local_files_only: bool,
    runs: int,
) -> Dict[str, Any]:
    video = resolve_video_path(case_id)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video}")
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError("cannot read first frame for benchmark")
    targets = pick_targets_for_benchmark(case_id)

    yolo = YOLOAdapter(weights_path=str(yolo_weights), conf=0.25, iou=0.45, device="cuda:0")
    yolo.load()
    dino = GroundingDINOAdapter(
        model_id=dino_model_id,
        processor_id=dino_processor_id,
        box_threshold=0.30,
        text_threshold=0.25,
        local_files_only=dino_local_files_only,
    )
    dino.load()

    payload = InferenceInput(frame_bgr=frame, timestamp_sec=0.0, frame_index=0, context={"detect_targets": targets})

    # warmup
    _ = yolo.infer(payload)
    _ = dino.infer(payload)

    yolo_times: List[float] = []
    dino_times: List[float] = []
    for _ in range(max(1, runs)):
        t0 = time.perf_counter()
        _ = yolo.infer(payload)
        yolo_times.append((time.perf_counter() - t0) * 1000.0)
        t1 = time.perf_counter()
        _ = dino.infer(payload)
        dino_times.append((time.perf_counter() - t1) * 1000.0)

    return {
        "targets_used": targets,
        "runs": max(1, runs),
        "yolo_ms_avg": round(sum(yolo_times) / len(yolo_times), 3),
        "dino_ms_avg": round(sum(dino_times) / len(dino_times), 3),
        "speedup_x": round((sum(dino_times) / len(dino_times)) / max(1e-6, (sum(yolo_times) / len(yolo_times))), 3),
    }


def index_step_results(items: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for x in items:
        sid = str(x.get("step_id", "")).strip()
        if sid:
            out[sid] = x
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build YOLO vs DINO replay speed/quality compare report")
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--dino-result", default=None, help="default data/<case_id>/v2/replay-validation/all_steps_result.json")
    parser.add_argument("--yolo-result", default=None, help="default data/<case_id>/v3/replay-validation-yolo/all_steps_result.json")
    parser.add_argument("--yolo-weights", default=None)
    parser.add_argument("--yolo-registry", default=str(DEFAULT_YOLO_REGISTRY))
    parser.add_argument("--dino-model-id", default=None)
    parser.add_argument("--dino-processor-id", default=None)
    parser.add_argument("--dino-local-files-only", action="store_true")
    parser.add_argument("--benchmark-runs", type=int, default=5)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    dino_result_path = Path(args.dino_result) if args.dino_result else (ROOT / "data" / args.case_id / "v2" / "replay-validation" / "all_steps_result.json")
    yolo_result_path = Path(args.yolo_result) if args.yolo_result else (ROOT / "data" / args.case_id / "v3" / "replay-validation-yolo" / "all_steps_result.json")
    if not dino_result_path.exists():
        raise RuntimeError(f"missing dino result: {dino_result_path}")
    if not yolo_result_path.exists():
        raise RuntimeError(f"missing yolo result: {yolo_result_path}")

    dino_result = read_json(dino_result_path)
    yolo_result = read_json(yolo_result_path)

    dino_steps = index_step_results(dino_result.get("step_results", []))
    yolo_steps = index_step_results(yolo_result.get("step_results", []))
    all_steps = sorted(set(dino_steps.keys()) | set(yolo_steps.keys()))

    yolo_only_matched: List[str] = []
    yolo_missed_but_dino_matched: List[str] = []
    match_time_delta: List[Dict[str, Any]] = []
    for sid in all_steps:
        d = dino_steps.get(sid, {})
        y = yolo_steps.get(sid, {})
        dm = bool(d.get("matched"))
        ym = bool(y.get("matched"))
        if ym and not dm:
            yolo_only_matched.append(sid)
        if dm and not ym:
            yolo_missed_but_dino_matched.append(sid)
        if dm and ym:
            ds = d.get("first_match_second")
            ys = y.get("first_match_second")
            if isinstance(ds, (int, float)) and isinstance(ys, (int, float)):
                match_time_delta.append({"step_id": sid, "dino_sec": float(ds), "yolo_sec": float(ys), "delta_sec": round(float(ys) - float(ds), 3)})

    yolo_weights = resolve_yolo_weights(args.case_id, args.yolo_weights, Path(args.yolo_registry))
    local_dino_dir = ROOT / "models" / "small-models" / "object" / "grounding-dino"
    dino_model_id = args.dino_model_id or (str(local_dino_dir) if local_dino_dir.exists() else "IDEA-Research/grounding-dino-base")
    single_frame = benchmark_single_frame(
        case_id=args.case_id,
        yolo_weights=yolo_weights,
        dino_model_id=dino_model_id,
        dino_processor_id=args.dino_processor_id,
        dino_local_files_only=bool(args.dino_local_files_only),
        runs=max(1, int(args.benchmark_runs)),
    )

    report = {
        "pipeline_stage": "criteria-trainer:speed-compare:yolo-vs-dino",
        "generated_at": utc_now_iso(),
        "case_id": args.case_id,
        "inputs": {
            "dino_result": str(dino_result_path),
            "yolo_result": str(yolo_result_path),
            "yolo_weights": str(yolo_weights),
            "dino_model_id": dino_model_id,
        },
        "single_frame_latency_ms": single_frame,
        "full_video_replay_summary": {
            "dino": {
                "matched_steps": int(dino_result.get("matched_steps", 0)),
                "total_steps": int(dino_result.get("total_steps", 0)),
                "sampled_frames_total": int(dino_result.get("sampled_frames_total", 0)),
            },
            "yolo": {
                "matched_steps": int(yolo_result.get("matched_steps", 0)),
                "total_steps": int(yolo_result.get("total_steps", 0)),
                "sampled_frames_total": int(yolo_result.get("sampled_frames_total", 0)),
            },
        },
        "misdetect_cases": {
            "yolo_false_positive_suspects": yolo_only_matched,
            "yolo_miss_cases": yolo_missed_but_dino_matched,
            "first_match_time_delta": match_time_delta,
        },
    }

    output = Path(args.output) if args.output else (ROOT / "data" / args.case_id / "v3" / "replay-validation-yolo" / "speed_compare_report.json")
    write_json(output, report)
    print(f"[OK] output: {output}")
    print(f"[OK] single-frame speedup x: {single_frame['speedup_x']}")


if __name__ == "__main__":
    main()

