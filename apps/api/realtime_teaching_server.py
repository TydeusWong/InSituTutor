import argparse
import base64
import copy
import json
import math
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

import cv2
import httpx
import numpy as np
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[2]
APPS_ROOT = ROOT / "apps"
WEB_ROOT = APPS_ROOT / "student-web"

import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "services" / "criteria-trainer") not in sys.path:
    sys.path.insert(0, str(ROOT / "services" / "criteria-trainer"))

from config import (  # noqa: E402
    RT_DEFAULT_BOX_MAX_STALE_MS,
    RT_DEFAULT_CAPTURE_FPS,
    RT_DEFAULT_HAND_IN_BBOX_THRESHOLD,
    RT_DEFAULT_IP_WEBCAM_URL,
    RT_DEFAULT_OMNI_MODEL,
    RT_DEFAULT_OMNI_SECTION_PASS_THRESHOLD,
    RT_DEFAULT_PERSIST_EDGE_MARGIN,
    RT_DEFAULT_PERSIST_MAX_MISS_SECONDS,
    RT_DEFAULT_RENDER_FPS,
    RT_DEFAULT_SERVER_SIDE_OVERLAY,
    RT_DEFAULT_SKIP_OMNI_VALIDATION,
    RT_DEFAULT_STREAM_FPS,
    RT_DEFAULT_STREAM_MAX_WIDTH,
    RT_DEFAULT_YOLO_CONF,
    RT_DEFAULT_YOLO_DEVICE,
    RT_DEFAULT_YOLO_FPS,
    RT_DEFAULT_YOLO_IOU,
    TA_BASE_URL,
    TA_REQUEST_TIMEOUT_SEC,
    get_api_key,
    load_env_file,
)
from adapters.mediapipe_hand import MediaPipeHandAdapter  # noqa: E402
from adapters.yolo import YOLOAdapter  # noqa: E402
from adapters.base import InferenceInput  # noqa: E402
from replay_validate_all_steps_yolo import (  # noqa: E402
    build_anchors_from_targets,
    build_context,
    eval_condition_code,
    extract_object_targets_from_plan,
    normalize_name,
    step_requires_hand_model,
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, item: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def env_str(name: str, default: str) -> str:
    load_env_file()
    val = str(os.environ.get(name, default)).strip()
    return val if val else default


def env_float(name: str, default: float) -> float:
    s = env_str(name, str(default))
    try:
        return float(s)
    except Exception:
        return float(default)


def env_bool(name: str, default: bool) -> bool:
    raw = env_str(name, "1" if default else "0").strip().lower()
    return raw in {"1", "true", "yes", "y", "on"}


class FrameSource:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.snapshot_urls = self._build_snapshot_urls(self.base_url)
        self.stream_url = self._build_stream_url(self.base_url)
        self.last_source_kind = "unknown"
        self.last_frame_jpg: bytes = b""
        self.last_frame_ts: float = 0.0
        self._stream_client: Optional[httpx.Client] = None
        self._stream_ctx: Any = None
        self._stream_resp: Optional[httpx.Response] = None
        self._stream_iter: Any = None
        self._stream_buffer = bytearray()

    @staticmethod
    def _build_snapshot_urls(base_url: str) -> List[str]:
        out: List[str] = []
        if base_url.endswith(".jpg") or base_url.endswith(".jpeg"):
            out.append(base_url)
        else:
            out.extend([f"{base_url}/shot.jpg", f"{base_url}/photo.jpg"])
        # raw URL fallback in case user already points to snapshot endpoint.
        out.append(base_url)
        dedup: List[str] = []
        for u in out:
            if u not in dedup:
                dedup.append(u)
        return dedup

    @staticmethod
    def _build_stream_url(base_url: str) -> str:
        if base_url.endswith("/video") or base_url.endswith(".mjpg"):
            return base_url
        return f"{base_url}/video"

    def stream_display_url(self) -> str:
        return self.stream_url

    def _read_snapshot(self) -> Optional[Any]:
        with httpx.Client(timeout=2.0) as client:
            for url in self.snapshot_urls:
                try:
                    resp = client.get(url)
                    if resp.status_code != 200:
                        continue
                    content_type = (resp.headers.get("content-type") or "").lower()
                    if "image" not in content_type and not url.lower().endswith((".jpg", ".jpeg")):
                        continue
                    jpg = bytes(resp.content)
                    arr = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if arr is not None:
                        self.last_source_kind = "snapshot"
                        self.last_frame_jpg = jpg
                        self.last_frame_ts = time.time()
                        return arr
                except Exception:
                    continue
        return None

    def _close_stream(self) -> None:
        try:
            if self._stream_resp is not None:
                self._stream_resp.close()
        except Exception:
            pass
        try:
            if self._stream_ctx is not None:
                self._stream_ctx.__exit__(None, None, None)
        except Exception:
            pass
        try:
            if self._stream_client is not None:
                self._stream_client.close()
        except Exception:
            pass
        self._stream_client = None
        self._stream_ctx = None
        self._stream_resp = None
        self._stream_iter = None
        self._stream_buffer = bytearray()

    def _ensure_stream_open(self) -> bool:
        if self._stream_iter is not None:
            return True
        self._close_stream()
        try:
            timeout = httpx.Timeout(connect=2.0, read=2.0, write=2.0, pool=2.0)
            self._stream_client = httpx.Client(timeout=timeout)
            self._stream_ctx = self._stream_client.stream("GET", self.stream_url, headers={"Connection": "keep-alive"})
            self._stream_resp = self._stream_ctx.__enter__()
            if self._stream_resp.status_code != 200:
                self._close_stream()
                return False
            self._stream_iter = self._stream_resp.iter_bytes()
            return True
        except Exception:
            self._close_stream()
            return False

    def _read_stream(self) -> Optional[Any]:
        if not self._ensure_stream_open():
            return None
        try:
            for _ in range(32):
                chunk = next(self._stream_iter)
                if not chunk:
                    continue
                self._stream_buffer.extend(chunk)
                if len(self._stream_buffer) > 8_000_000:
                    # Drop old backlog aggressively to avoid stale-frame queueing.
                    self._stream_buffer = self._stream_buffer[-2_000_000:]
                end = self._stream_buffer.rfind(b"\xff\xd9")
                if end < 0:
                    continue
                start = self._stream_buffer.rfind(b"\xff\xd8", 0, end)
                if start < 0:
                    del self._stream_buffer[: end + 2]
                    continue
                jpg = bytes(self._stream_buffer[start : end + 2])
                del self._stream_buffer[: end + 2]
                arr = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                if arr is not None:
                    self.last_source_kind = "stream"
                    self.last_frame_jpg = jpg
                    self.last_frame_ts = time.time()
                    return arr
            return None
        except StopIteration:
            self._close_stream()
            return None
        except Exception:
            self._close_stream()
            return None

    def read_frame(self) -> Optional[Any]:
        # Prefer continuous stream endpoint (/video) for lower request overhead.
        frame = self._read_stream()
        if frame is not None:
            return frame
        # Fallback to snapshot endpoint when stream is temporarily unavailable.
        return self._read_snapshot()


def frame_to_jpeg_bytes(frame_bgr: Any, quality: int = 80) -> bytes:
    ok, encoded = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok or encoded is None:
        return b""
    return bytes(encoded.tobytes())


def blank_jpeg_bytes(width: int = 640, height: int = 360, text: str = "waiting for camera...") -> bytes:
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(canvas, text, (20, int(height / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2, cv2.LINE_AA)
    return frame_to_jpeg_bytes(canvas, quality=70)


def load_ingest_video_meta(case_id: str) -> Dict[str, Any]:
    manifest_path = ROOT / "data" / case_id / "ingest_manifest.json"
    if not manifest_path.exists():
        return {"target_width": 1280, "target_height": 720, "target_fps": 10.0, "ingest_video_path": ""}
    data = read_json(manifest_path)
    demos = data.get("demos", [])
    demo = demos[0] if isinstance(demos, list) and demos else {}
    resolution = str(demo.get("resolution", "1280x720")).lower().strip()
    width, height = 1280, 720
    if "x" in resolution:
        left, right = resolution.split("x", 1)
        try:
            width = int(float(left.strip()))
            height = int(float(right.strip()))
        except Exception:
            width, height = 1280, 720
    try:
        fps = float(demo.get("fps", 10.0))
    except Exception:
        fps = 10.0
    return {
        "target_width": width,
        "target_height": height,
        "target_fps": fps,
        "ingest_video_path": str(demo.get("ingest_video_path", "")),
    }


def jpeg_bytes_to_data_uri(blob: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(blob).decode("ascii")


def extract_json_from_text(text: str) -> Dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return json.loads(text[start : end + 1])
    raise ValueError("no json found in model output")


@dataclass
class RuntimeConfig:
    case_id: str
    mode: str
    strategy_version: str
    error_library: str
    ip_webcam_url: str
    realtime_yolo_fps: float
    omni_section_pass_threshold: float
    skip_omni_validation: bool
    omni_model: str
    omni_base_url: str
    omni_timeout_sec: int
    yolo_conf: float
    yolo_iou: float
    yolo_device: str
    capture_fps: float
    render_fps: float
    stream_fps: float
    stream_max_width: int
    server_side_overlay_enabled: bool
    box_max_stale_ms: int
    persist_edge_margin: float
    persist_max_miss_seconds: float
    persist_max_miss_samples: int
    hand_in_bbox_threshold: float
    target_width: int
    target_height: int
    target_fps: float
    ingest_video_path: str


@dataclass
class SectionRuntime:
    section_id: str
    section_name: str
    section_goal: str
    expected_section_state: str
    steps: List[Dict[str, Any]]


@dataclass
class SessionState:
    session_id: str
    case_id: str
    created_at: str
    state: str = "idle"
    section_idx: int = 0
    step_idx: int = 0
    state_until: float = 0.0
    started: bool = False
    step_sample_seq: int = 0
    step_object_memory: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    recent_frames: Deque[Tuple[float, bytes]] = field(default_factory=lambda: deque(maxlen=20))
    next_sample_at: float = 0.0
    last_eval: Optional[Dict[str, Any]] = None
    last_omni: Optional[Dict[str, Any]] = None
    last_error: Optional[Dict[str, Any]] = None
    completed_steps: List[str] = field(default_factory=list)
    last_overlay_detections: List[Dict[str, Any]] = field(default_factory=list)
    last_overlay_object_targets: List[str] = field(default_factory=list)
    last_overlay_step_id: str = ""
    last_overlay_matched: bool = False
    last_overlay_ts: float = 0.0
    last_overlay_frame_width: int = 0
    last_overlay_frame_height: int = 0
    latest_annotated_jpg: bytes = b""
    last_meta_write_ts: float = 0.0
    self_evolution_recording_enabled: bool = False
    capture_meta: Dict[str, Any] = field(default_factory=dict)
    started_at_monotonic: float = 0.0
    ended_at: str = ""
    raw_media_path: str = ""
    raw_audio_path: str = ""
    raw_video_path: str = ""
    review_media_path: str = ""
    review_media_error: str = ""
    record_stop_event: Optional[threading.Event] = None
    record_thread: Optional[threading.Thread] = None
    recorded_frame_count: int = 0


class RealtimeTutorEngine:
    def __init__(self, cfg: RuntimeConfig) -> None:
        self.cfg = cfg
        self.api_key = get_api_key()
        self.frame_source = FrameSource(cfg.ip_webcam_url)
        self.strategy_path = self._resolve_strategy_path()
        self.strategy = read_json(self.strategy_path)
        self.bundle = read_json(ROOT / "data" / cfg.case_id / "v2" / "detector-plans" / "detector_plan_v2.json")
        self.sections = self._build_sections(self.strategy)
        self.global_targets = [
            str(x).strip()
            for x in (self.bundle.get("yolo_detect_targets") or self.bundle.get("grounding_dino_detect_targets", []))
            if str(x).strip()
        ]
        self.anchors = build_anchors_from_targets(self.global_targets)
        self.global_disambiguation_targets = [
            t for t in self.global_targets if normalize_name(t) not in {"workspace center", "workspace_center"}
        ]
        self.error_plans_by_step = self._load_error_plans_by_step()
        self.yolo_adapter = self._load_yolo_adapter(cfg.case_id)
        self.hand_adapter: Optional[MediaPipeHandAdapter] = None
        self.sessions: Dict[str, SessionState] = {}
        self._lock = threading.Lock()
        self._latest_frame_bgr: Optional[Any] = None
        self._latest_frame_jpg: bytes = b""
        self._latest_frame_ts: float = 0.0
        self._latest_frame_seq: int = 0
        self._latest_annotated_jpg: bytes = b""
        self._blank_jpg: bytes = blank_jpeg_bytes()
        self._stop_event = threading.Event()
        self._capture_thread = threading.Thread(target=self._capture_loop, name="realtime-capture", daemon=True)
        self._infer_thread = threading.Thread(target=self._infer_loop, name="realtime-infer", daemon=True)
        self._render_thread: Optional[threading.Thread] = None
        self._capture_thread.start()
        self._infer_thread.start()
        if self.cfg.server_side_overlay_enabled:
            self._render_thread = threading.Thread(target=self._render_loop, name="realtime-render", daemon=True)
            self._render_thread.start()

    def _resolve_strategy_path(self) -> Path:
        if self.cfg.strategy_version == "evolved":
            path = ROOT / "data" / self.cfg.case_id / "v5" / "strategy" / "teaching_strategy_evolved_v1.json"
            if path.exists():
                return path
            raise FileNotFoundError(f"missing evolved strategy: {path}")
        return ROOT / "data" / self.cfg.case_id / "v2" / "strategy" / "teaching_strategy_v2.json"

    def _load_error_plans_by_step(self) -> Dict[str, List[Dict[str, Any]]]:
        if self.cfg.error_library == "none":
            return {}
        library_path = ROOT / "data" / self.cfg.case_id / "v5" / "errors" / "error_library_v1.json"
        if not library_path.exists():
            return {}
        library = read_json(library_path)
        by_step: Dict[str, List[Dict[str, Any]]] = {}
        for item in library.get("errors", []) if isinstance(library.get("errors"), list) else []:
            if not isinstance(item, dict):
                continue
            plan_value = str(item.get("plan_path", "")).strip()
            if not plan_value:
                continue
            plan_path = Path(plan_value)
            if not plan_path.is_absolute():
                plan_path = ROOT / plan_path
            if not plan_path.exists():
                continue
            plan = read_json(plan_path)
            scope = plan.get("scope") if isinstance(plan.get("scope"), dict) else item.get("scope", {})
            if not isinstance(scope, dict):
                scope = {}
            steps = [str(x).strip() for x in scope.get("applies_to_steps", []) if str(x).strip()]
            for step_id in steps:
                by_step.setdefault(step_id, []).append(plan)
        return by_step

    def _capture_loop(self) -> None:
        interval = 1.0 / max(1.0, float(self.cfg.capture_fps))
        while not self._stop_event.is_set():
            frame = self.frame_source.read_frame()
            if frame is None:
                time.sleep(0.03)
                continue
            jpg = self.frame_source.last_frame_jpg or frame_to_jpeg_bytes(frame, quality=70)
            self._latest_frame_bgr = frame.copy()
            self._latest_frame_jpg = jpg
            self._latest_frame_ts = float(self.frame_source.last_frame_ts or time.time())
            self._latest_frame_seq += 1
            # Continuous MJPEG sources must be drained as fast as possible to avoid
            # old frames backing up in socket buffers. Only throttle snapshot fallback.
            if self.frame_source.last_source_kind != "stream":
                time.sleep(interval)

    def _infer_loop(self) -> None:
        while not self._stop_event.is_set():
            with self._lock:
                session_ids = [sid for sid, s in self.sessions.items() if s.started and s.state != "course_done"]
            if not session_ids:
                time.sleep(0.05)
                continue
            for sid in session_ids:
                try:
                    self.evaluate_step(sid)
                except Exception:
                    # Keep background loop alive even if one session fails.
                    pass
            time.sleep(0.01)

    def _render_loop(self) -> None:
        interval = 1.0 / max(1.0, float(self.cfg.render_fps))
        stale_sec = max(0.05, float(self.cfg.box_max_stale_ms) / 1000.0)
        while not self._stop_event.is_set():
            t0 = time.time()
            raw_frame = self._get_latest_frame_copy()
            if raw_frame is None:
                time.sleep(0.02)
                continue
            frame = self._prepare_stream_frame(raw_frame)
            can_reuse_raw_jpg = frame.shape[:2] == raw_frame.shape[:2]

            with self._lock:
                sessions = [
                    (
                        sid,
                        bool(session.started),
                        str(session.state),
                        list(session.last_overlay_detections) if isinstance(session.last_overlay_detections, list) else [],
                        list(session.last_overlay_object_targets) if isinstance(session.last_overlay_object_targets, list) else [],
                        str(session.last_overlay_step_id or "-"),
                        bool(session.last_overlay_matched),
                        float(session.last_overlay_ts or 0.0),
                    )
                    for sid, session in self.sessions.items()
                ]

            now = time.time()
            for sid, started, state, detections, targets, step_id, matched, overlay_ts in sessions:
                if not started or state == "course_done":
                    continue
                fresh = (now - overlay_ts) <= stale_sec

                if fresh and detections and targets:
                    annotated = self._render_overlay_frame(
                        frame_bgr=frame,
                        detections=detections,
                        object_targets=targets,
                        matched=matched,
                        step_id=step_id,
                    )
                else:
                    annotated = self._get_latest_frame_jpg() if can_reuse_raw_jpg else frame_to_jpeg_bytes(frame, quality=70)
                if not annotated:
                    continue

                with self._lock:
                    s = self.sessions.get(sid)
                    if s is not None:
                        s.latest_annotated_jpg = annotated
                self._latest_annotated_jpg = annotated

            remain = interval - (time.time() - t0)
            if remain > 0:
                time.sleep(remain)

    def _get_latest_frame_copy(self) -> Optional[Any]:
        frame = self._latest_frame_bgr
        if frame is None:
            return None
        return frame.copy()

    def _get_latest_frame_jpg(self) -> bytes:
        return self._latest_frame_jpg

    def _get_latest_frame_seq(self) -> int:
        return int(self._latest_frame_seq)

    def _prepare_stream_frame(self, frame_bgr: Any) -> Any:
        if frame_bgr is None:
            return None
        max_width = max(0, int(self.cfg.stream_max_width))
        if max_width <= 0:
            return frame_bgr
        h, w = frame_bgr.shape[:2]
        if w <= max_width:
            return frame_bgr
        scale = float(max_width) / float(max(1, w))
        target_h = max(1, int(round(h * scale)))
        return cv2.resize(frame_bgr, (max_width, target_h), interpolation=cv2.INTER_AREA)

    @staticmethod
    def _safe_float(v: Any, default: float = 0.0) -> float:
        try:
            return float(v)
        except Exception:
            return default

    def _infer_yolo_detections(self, frame_bgr: Any, targets: List[str], frame_index: int) -> List[Dict[str, Any]]:
        payload = InferenceInput(
            frame_bgr=frame_bgr,
            timestamp_sec=time.time(),
            frame_index=frame_index,
            context={"detect_targets": targets},
        )
        out = self.yolo_adapter.infer(payload)
        detections = out.features.get("detections", [])
        if not isinstance(detections, list):
            return []
        return [d for d in detections if isinstance(d, dict)]

    def _render_overlay_frame(
        self,
        frame_bgr: Any,
        detections: List[Dict[str, Any]],
        object_targets: List[str],
        matched: bool,
        step_id: str,
    ) -> bytes:
        vis = frame_bgr.copy()
        target_norm = {normalize_name(x) for x in object_targets}
        for det in detections:
            label = str(det.get("label", det.get("target", ""))).strip()
            if not label:
                continue
            if target_norm and normalize_name(label) not in target_norm:
                continue
            bbox = det.get("bbox_xyxy", {})
            if not isinstance(bbox, dict):
                continue
            x1 = int(self._safe_float(bbox.get("x1", 0.0)) * vis.shape[1])
            y1 = int(self._safe_float(bbox.get("y1", 0.0)) * vis.shape[0])
            x2 = int(self._safe_float(bbox.get("x2", 0.0)) * vis.shape[1])
            y2 = int(self._safe_float(bbox.get("y2", 0.0)) * vis.shape[0])
            score = self._safe_float(det.get("score", 0.0))
            color = (30, 200, 30)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            text = f"{label} {score:.2f}"
            cv2.putText(vis, text, (max(4, x1), max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        status = "MATCHED" if matched else "RUNNING"
        status_color = (20, 180, 20) if matched else (0, 180, 255)
        cv2.putText(vis, f"{step_id} | {status}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, status_color, 2, cv2.LINE_AA)
        return frame_to_jpeg_bytes(vis, quality=80)

    def _build_sections(self, strategy: Dict[str, Any]) -> List[SectionRuntime]:
        out: List[SectionRuntime] = []
        for s in strategy.get("sections", []):
            steps = s.get("steps", [])
            if not isinstance(steps, list):
                steps = []
            out.append(
                SectionRuntime(
                    section_id=str(s.get("section_id", "")),
                    section_name=str(s.get("section_name", "")),
                    section_goal=str(s.get("section_goal", "")),
                    expected_section_state=str(s.get("expected_section_state", "")),
                    steps=steps,
                )
            )
        return out

    def _load_yolo_adapter(self, case_id: str) -> YOLOAdapter:
        registry = read_json(ROOT / "services" / "criteria-trainer" / "configs" / "yolo_registry_v1.json")
        latest = (registry.get("latest_by_case", {}) or {}).get(case_id, {})
        weights = Path(str(latest.get("weights_path", "")))
        if not weights.exists():
            raise RuntimeError(f"YOLO weights not found for case '{case_id}': {weights}")
        class_map = read_json(ROOT / "data" / case_id / "v3" / "yolo-dataset" / "class_map.json")
        class_names = [str(c.get("name")) for c in class_map.get("classes", []) if isinstance(c, dict) and c.get("name")]
        adapter = YOLOAdapter(
            weights_path=str(weights),
            conf=float(self.cfg.yolo_conf),
            iou=float(self.cfg.yolo_iou),
            device=str(self.cfg.yolo_device),
            class_names=class_names,
        )
        adapter.load()
        return adapter

    def _session_dir(self, session: SessionState) -> Path:
        return ROOT / "data" / session.case_id / "v4" / "realtime-logs" / session.session_id

    def _self_evolution_session_dir(self, session: SessionState) -> Path:
        return ROOT / "data" / session.case_id / "v5" / "sessions" / session.session_id

    def _self_evolution_logs_dir(self, session: SessionState) -> Path:
        return self._self_evolution_session_dir(session) / "logs"

    def _self_evolution_raw_dir(self, session: SessionState) -> Path:
        return self._self_evolution_session_dir(session) / "raw"

    def _session_meta_path(self, session: SessionState) -> Path:
        return self._session_dir(session) / "session_meta.json"

    def _self_evolution_session_meta_path(self, session: SessionState) -> Path:
        return self._self_evolution_logs_dir(session) / "session_meta.json"

    def _events_path(self, session: SessionState) -> Path:
        return self._session_dir(session) / "events.jsonl"

    def _omni_path(self, session: SessionState) -> Path:
        return self._session_dir(session) / "omni_calls.jsonl"

    def _trace_path(self, session: SessionState) -> Path:
        return self._session_dir(session) / "step_trace.jsonl"

    def _self_evolution_path(self, session: SessionState, name: str) -> Path:
        return self._self_evolution_logs_dir(session) / name

    def _session_elapsed_sec(self, session: SessionState) -> float:
        if session.started_at_monotonic <= 0:
            return 0.0
        return max(0.0, time.time() - session.started_at_monotonic)

    def _prepare_self_evolution_session(self, session: SessionState) -> None:
        base = self._self_evolution_session_dir(session)
        for rel in [
            "raw",
            "logs",
            "asr",
            "alignment",
            "reflection",
            "error-slices",
            "detector-plans/errors",
            "review",
        ]:
            (base / rel).mkdir(parents=True, exist_ok=True)
        for name in ["events.jsonl", "step_trace.jsonl", "omni_calls.jsonl", "system_prompts.jsonl", "teacher_interventions.jsonl"]:
            p = self._self_evolution_path(session, name)
            if not p.exists():
                p.write_text("", encoding="utf-8")

    def _start_ipcam_recording(self, session: SessionState) -> None:
        if session.record_thread is not None and session.record_thread.is_alive():
            return
        raw_dir = self._self_evolution_raw_dir(session)
        raw_dir.mkdir(parents=True, exist_ok=True)
        video_path = raw_dir / "teaching_session_ipcam.mp4"
        stop_event = threading.Event()
        session.record_stop_event = stop_event
        session.raw_video_path = str(video_path.relative_to(ROOT))
        session.recorded_frame_count = 0
        thread = threading.Thread(
            target=self._record_ipcam_loop,
            args=(session.session_id, video_path, stop_event),
            name=f"self-evolution-record-{session.session_id}",
            daemon=True,
        )
        session.record_thread = thread
        thread.start()

    def _record_ipcam_loop(self, session_id: str, video_path: Path, stop_event: threading.Event) -> None:
        fps = max(1.0, float(self.cfg.target_fps or self.cfg.capture_fps or 10.0))
        width = max(1, int(self.cfg.target_width or 1280))
        height = max(1, int(self.cfg.target_height or 720))
        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        interval = 1.0 / fps
        next_at = time.time()
        frame_count = 0
        try:
            while not stop_event.is_set():
                now = time.time()
                if now < next_at:
                    time.sleep(min(0.02, next_at - now))
                    continue
                next_at += interval
                frame = self._get_latest_frame_copy()
                if frame is None:
                    continue
                if frame.shape[1] != width or frame.shape[0] != height:
                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                writer.write(frame)
                frame_count += 1
                with self._lock:
                    session = self.sessions.get(session_id)
                    if session is not None:
                        session.recorded_frame_count = frame_count
        finally:
            writer.release()

    def _stop_ipcam_recording(self, session: SessionState) -> None:
        stop_event = session.record_stop_event
        thread = session.record_thread
        if stop_event is not None:
            stop_event.set()
        if thread is not None and thread.is_alive():
            thread.join(timeout=3.0)
        session.record_stop_event = None
        session.record_thread = None

    def _build_review_media(self, session: SessionState) -> None:
        session.review_media_path = ""
        session.review_media_error = ""
        video_path = ROOT / session.raw_video_path if session.raw_video_path else None
        audio_path = ROOT / session.raw_audio_path if session.raw_audio_path else None
        if video_path is None or not video_path.exists():
            session.review_media_error = "missing_ipcam_video"
            return
        if audio_path is None or not audio_path.exists():
            session.review_media_error = "missing_microphone_audio"
            return
        out_path = self._self_evolution_raw_dir(session) / "teaching_session_review.mp4"
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-i",
            str(audio_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            str(out_path),
        ]
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            session.review_media_error = proc.stderr.strip()[:800] or "ffmpeg_mux_failed"
            return
        session.review_media_path = str(out_path.relative_to(ROOT))

    def _write_session_meta(self, session: SessionState) -> None:
        actual_width = int(session.capture_meta.get("actual_width") or session.capture_meta.get("width") or 0)
        actual_height = int(session.capture_meta.get("actual_height") or session.capture_meta.get("height") or 0)
        actual_fps = float(session.capture_meta.get("actual_fps") or session.capture_meta.get("fps") or 0.0)
        capture_degraded = bool(
            session.self_evolution_recording_enabled
            and (
                (actual_width > 0 and actual_width != int(self.cfg.target_width))
                or (actual_height > 0 and actual_height != int(self.cfg.target_height))
                or (actual_fps > 0 and abs(actual_fps - float(self.cfg.target_fps)) > 0.5)
            )
        )
        meta = {
            "session_id": session.session_id,
            "case_id": session.case_id,
            "created_at": session.created_at,
            "updated_at": utc_now_iso(),
            "state": session.state,
            "section_idx": session.section_idx,
            "step_idx": session.step_idx,
            "started": session.started,
            "ended_at": session.ended_at,
            "mode": self.cfg.mode,
            "self_evolution_recording_enabled": session.self_evolution_recording_enabled,
            "target_width": int(self.cfg.target_width),
            "target_height": int(self.cfg.target_height),
            "target_fps": float(self.cfg.target_fps),
            "actual_width": actual_width,
            "actual_height": actual_height,
            "actual_fps": actual_fps,
            "capture_degraded": capture_degraded,
            "degrade_reason": str(session.capture_meta.get("degrade_reason") or ("actual_capture_differs_from_target" if capture_degraded else "")),
            "capture_meta": session.capture_meta,
            "raw_media_path": session.raw_media_path,
            "raw_audio_path": session.raw_audio_path,
            "raw_video_path": session.raw_video_path,
            "review_media_path": session.review_media_path,
            "review_media_error": session.review_media_error,
            "recorded_frame_count": session.recorded_frame_count,
            "runtime_config": {
                "ip_webcam_url": self.cfg.ip_webcam_url,
                "strategy_version": self.cfg.strategy_version,
                "strategy_path": str(self.strategy_path),
                "error_library": self.cfg.error_library,
                "loaded_error_count": sum(len(v) for v in self.error_plans_by_step.values()),
                "realtime_yolo_fps": self.cfg.realtime_yolo_fps,
                "realtime_capture_fps": self.cfg.capture_fps,
                "realtime_render_fps": self.cfg.render_fps,
                "realtime_stream_fps": self.cfg.stream_fps,
                "box_max_stale_ms": self.cfg.box_max_stale_ms,
                "omni_section_pass_threshold": self.cfg.omni_section_pass_threshold,
                "skip_omni_validation": self.cfg.skip_omni_validation,
                "omni_model": self.cfg.omni_model,
                "yolo_conf": self.cfg.yolo_conf,
                "yolo_iou": self.cfg.yolo_iou,
                "yolo_device": self.cfg.yolo_device,
                "persist_max_miss_seconds": self.cfg.persist_max_miss_seconds,
                "persist_max_miss_samples": self.cfg.persist_max_miss_samples,
                "ingest_video_path": self.cfg.ingest_video_path,
            },
        }
        write_json(self._session_meta_path(session), meta)
        if session.self_evolution_recording_enabled:
            self._prepare_self_evolution_session(session)
            write_json(self._self_evolution_session_meta_path(session), meta)

    def _write_session_meta_if_due(self, session: SessionState, force: bool = False) -> None:
        now = time.time()
        if not force and session.last_meta_write_ts > 0 and (now - session.last_meta_write_ts) < 1.0:
            return
        self._write_session_meta(session)
        session.last_meta_write_ts = now

    def _log_event(self, session: SessionState, event: str, payload: Dict[str, Any]) -> None:
        item = {
            "ts": utc_now_iso(),
            "elapsed_sec": self._session_elapsed_sec(session),
            "event": event,
            "session_id": session.session_id,
            "section_idx": session.section_idx,
            "step_idx": session.step_idx,
            "state": session.state,
            "payload": payload,
        }
        append_jsonl(self._events_path(session), item)
        if session.self_evolution_recording_enabled:
            append_jsonl(self._self_evolution_path(session, "events.jsonl"), item)

    def _log_trace(self, session: SessionState, payload: Dict[str, Any]) -> None:
        item = {
            "ts": utc_now_iso(),
            "elapsed_sec": self._session_elapsed_sec(session),
            "session_id": session.session_id,
            "section_idx": session.section_idx,
            "step_idx": session.step_idx,
            "payload": payload,
        }
        append_jsonl(self._trace_path(session), item)
        if session.self_evolution_recording_enabled:
            append_jsonl(self._self_evolution_path(session, "step_trace.jsonl"), item)

    def _log_omni(self, session: SessionState, payload: Dict[str, Any]) -> None:
        item = {
            "ts": utc_now_iso(),
            "elapsed_sec": self._session_elapsed_sec(session),
            "session_id": session.session_id,
            "section_idx": session.section_idx,
            "step_idx": session.step_idx,
            "payload": payload,
        }
        append_jsonl(self._omni_path(session), item)
        if session.self_evolution_recording_enabled:
            append_jsonl(self._self_evolution_path(session, "omni_calls.jsonl"), item)

    def _current_section(self, session: SessionState) -> Optional[SectionRuntime]:
        if session.section_idx < 0 or session.section_idx >= len(self.sections):
            return None
        return self.sections[session.section_idx]

    def _current_step(self, session: SessionState) -> Optional[Dict[str, Any]]:
        sec = self._current_section(session)
        if sec is None:
            return None
        if session.step_idx < 0 or session.step_idx >= len(sec.steps):
            return None
        return sec.steps[session.step_idx]

    def _load_step_plan(self, step_id: str) -> Dict[str, Any]:
        return read_json(ROOT / "data" / self.cfg.case_id / "v2" / "detector-plans" / f"{step_id}.json")

    def _plan_object_targets(self, plan: Dict[str, Any]) -> List[str]:
        targets = extract_object_targets_from_plan(plan)
        global_by_norm = {normalize_name(t): t for t in self.global_targets}
        for cond in plan.get("judgement_conditions", []) if isinstance(plan.get("judgement_conditions"), list) else []:
            if not isinstance(cond, dict):
                continue
            code = str(cond.get("code", ""))
            for quoted in re.findall(r"'([^']+)'", code):
                key = normalize_name(quoted)
                if key in global_by_norm and global_by_norm[key] not in targets:
                    targets.append(global_by_norm[key])
        for item in plan.get("object_targets", []) if isinstance(plan.get("object_targets"), list) else []:
            target = str(item).strip()
            if target and target not in targets:
                targets.append(target)
        return targets

    def _error_plans_for_step(self, section_id: str, step_id: str) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for plan in self.error_plans_by_step.get(step_id, []):
            scope = plan.get("scope") if isinstance(plan.get("scope"), dict) else {}
            applies_sections = {str(x).strip() for x in scope.get("applies_to_sections", []) if str(x).strip()}
            applies_steps = {str(x).strip() for x in scope.get("applies_to_steps", []) if str(x).strip()}
            not_sections = {str(x).strip() for x in scope.get("not_errors_in_sections", []) if str(x).strip()}
            not_steps = {str(x).strip() for x in scope.get("not_errors_in_steps", []) if str(x).strip()}
            if section_id in not_sections or step_id in not_steps:
                continue
            if applies_steps and step_id not in applies_steps:
                continue
            if applies_sections and section_id not in applies_sections:
                continue
            out.append(plan)
        return out

    def _section_intro_payload(self, session: SessionState) -> Dict[str, Any]:
        sec = self._current_section(session)
        if sec is None:
            return {"title": "完成", "body": "全部章节已完成"}
        return {
            "title": f"{sec.section_name}",
            "body": sec.section_goal,
            "step_count": len(sec.steps),
        }

    def _reset_step_runtime(self, session: SessionState) -> None:
        session.step_sample_seq = 0
        session.step_object_memory = {}

    def _goto_next_section(self, session: SessionState, reason: str) -> None:
        session.section_idx += 1
        session.step_idx = 0
        self._reset_step_runtime(session)
        if session.section_idx >= len(self.sections):
            session.state = "course_done"
            session.state_until = 0.0
            self._log_event(session, "course_done", {"reason": reason})
            return
        session.state = "chapter_intro"
        session.state_until = time.time() + 3.0
        self._log_event(session, "section_enter", {"reason": reason, "section": self._section_intro_payload(session)})

    def _start_step_running(self, session: SessionState) -> None:
        sec = self._current_section(session)
        if sec is not None and len(sec.steps) == 0:
            # Empty section should still go through Omni validation once.
            session.state = "section_validating"
            session.state_until = 0.0
            self._log_event(session, "section_ready_for_omni_empty", {"section_id": sec.section_id})
            return
        session.state = "step_running"
        session.state_until = 0.0
        session.next_sample_at = 0.0
        self._reset_step_runtime(session)
        step = self._current_step(session)
        if step:
            prompt = (step.get("prompt") or {}) if isinstance(step, dict) else {}
            payload = {
                "step_id": step.get("step_id"),
                "step_order": step.get("step_order"),
                "section_id": sec.section_id if sec else None,
                "start_sec": self._session_elapsed_sec(session),
                "prompt": prompt,
                "focus_points": step.get("focus_points", []),
                "common_mistakes": step.get("common_mistakes", []),
            }
            self._log_event(session, "step_start", payload)
            if session.self_evolution_recording_enabled:
                append_jsonl(self._self_evolution_path(session, "system_prompts.jsonl"), payload)
                append_jsonl(self._self_evolution_path(session, "step_trace.jsonl"), {**payload, "event": "step_start"})

    def _advance_state_machine(self, session: SessionState) -> None:
        now = time.time()
        if session.state == "chapter_intro" and now >= session.state_until:
            self._start_step_running(session)
            return
        if session.state == "step_done_hold" and now >= session.state_until:
            sec = self._current_section(session)
            if sec is None:
                session.state = "course_done"
                return
            session.step_idx += 1
            if session.step_idx >= len(sec.steps):
                session.state = "section_validating"
                session.state_until = 0.0
                self._log_event(session, "section_ready_for_omni", {"section_id": sec.section_id})
            else:
                self._start_step_running(session)
            return

    def _tick_state_machine(self, session: SessionState) -> None:
        self._advance_state_machine(session)
        if session.state == "section_validating":
            self._run_omni_section_validation(session)
            return

    def _omni_validate_payload(self, session: SessionState) -> Tuple[str, List[Dict[str, Any]]]:
        sec = self._current_section(session)
        if sec is None:
            return "", []
        frames = list(session.recent_frames)[-5:]
        images: List[Dict[str, Any]] = []
        for ts, blob in frames:
            if not blob:
                continue
            images.append(
                {
                    "type": "image_url",
                    "image_url": {"url": jpeg_bytes_to_data_uri(blob)},
                }
            )
        steps_summary = [
            {
                "step_id": st.get("step_id"),
                "prompt": st.get("prompt"),
                "focus_points": st.get("focus_points", []),
            }
            for st in sec.steps
        ]
        payload_text = {
            "task": "validate_section_completion",
            "constraints": {
                "no_cross_section_retry": True,
                "must_return_json_only": True,
            },
            "current_section_index_0_based": session.section_idx,
            "current_section_id": sec.section_id,
            "current_section_name": sec.section_name,
            "expected_section_state": sec.expected_section_state,
            "steps": steps_summary,
            "completed_steps_in_session": session.completed_steps,
            "threshold": self.cfg.omni_section_pass_threshold,
            "output_schema": {
                "confidence": "float in [0,1]",
                "pass": "bool",
                "retry_step_id": "string|null, must belong to current section steps when pass=false",
                "reason": "string",
            },
            "decision_rule": "if confidence >= threshold then pass=true else pass=false and set retry_step_id",
        }
        return json.dumps(payload_text, ensure_ascii=False, indent=2), images

    def _run_omni_section_validation(self, session: SessionState) -> None:
        sec = self._current_section(session)
        if sec is None:
            session.state = "course_done"
            return
        text, images = self._omni_validate_payload(session)
        step_ids = [str(s.get("step_id", "")) for s in sec.steps if str(s.get("step_id", ""))]
        result: Dict[str, Any]
        if self.cfg.skip_omni_validation:
            result = {
                "confidence": 1.0,
                "pass": True,
                "retry_step_id": None,
                "reason": "omni_validation_skipped_by_config",
            }
            self._log_omni(
                session,
                {
                    "mode": "skipped_by_config",
                    "request_text": text,
                    "images_count": len(images),
                    "response": result,
                },
            )
        elif not self.api_key:
            result = {
                "confidence": 0.99,
                "pass": True,
                "retry_step_id": None,
                "reason": "omni_skipped_no_api_key",
            }
            self._log_omni(session, {"mode": "mock_no_api_key", "request_text": text, "response": result})
        else:
            try:
                client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.cfg.omni_base_url,
                    timeout=self.cfg.omni_timeout_sec,
                )
                content = [{"type": "text", "text": text}, *images]
                resp = client.chat.completions.create(
                    model=self.cfg.omni_model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a strict teaching-progress validator. Return JSON only.",
                        },
                        {"role": "user", "content": content},
                    ],
                    stream=False,
                )
                text_out = (resp.choices[0].message.content or "").strip()
                parsed = extract_json_from_text(text_out)
                result = {
                    "confidence": float(parsed.get("confidence", 0.0)),
                    "pass": bool(parsed.get("pass", False)),
                    "retry_step_id": parsed.get("retry_step_id"),
                    "reason": str(parsed.get("reason", "")),
                }
                self._log_omni(
                    session,
                    {
                        "mode": "omni",
                        "request_text": text,
                        "images_count": len(images),
                        "raw_response_text": text_out,
                        "response": result,
                    },
                )
            except Exception as exc:
                result = {
                    "confidence": 0.99,
                    "pass": True,
                    "retry_step_id": None,
                    "reason": f"omni_error_fallback_pass:{exc}",
                }
                self._log_omni(
                    session,
                    {
                        "mode": "omni_error_fallback",
                        "request_text": text,
                        "images_count": len(images),
                        "error": str(exc),
                        "response": result,
                    },
                )

        # Normalize fallback within current section only
        retry_step_id = str(result.get("retry_step_id") or "").strip()
        if retry_step_id and retry_step_id not in step_ids:
            retry_step_id = ""
        confidence = max(0.0, min(1.0, float(result.get("confidence", 0.0))))
        passed = bool(result.get("pass", False)) or (confidence >= self.cfg.omni_section_pass_threshold)
        if not step_ids:
            # Empty section has no retry target; still run Omni but always continue.
            passed = True
        final = {
            "confidence": confidence,
            "pass": passed,
            "retry_step_id": (retry_step_id or None),
            "reason": str(result.get("reason", "")) if step_ids else f"{str(result.get('reason', ''))}|empty_section_forced_pass",
            "threshold": self.cfg.omni_section_pass_threshold,
        }
        session.last_omni = final
        self._log_event(session, "section_omni_result", final)

        if passed:
            self._goto_next_section(session, reason="omni_pass")
            return

        # Fail: retry within current section
        if retry_step_id:
            idx = step_ids.index(retry_step_id)
        else:
            idx = 0
        session.step_idx = idx
        self._start_step_running(session)
        self._log_event(
            session,
            "section_retry",
            {"retry_step_id": step_ids[idx], "retry_step_index": idx, "reason": final.get("reason", "")},
        )

    def init_session(self, session_id: Optional[str]) -> Dict[str, Any]:
        with self._lock:
            sid = (session_id or "").strip() or str(uuid.uuid4())
            session = self.sessions.get(sid)
            if session is None:
                session = SessionState(
                    session_id=sid,
                    case_id=self.cfg.case_id,
                    created_at=utc_now_iso(),
                )
                self.sessions[sid] = session
                self._write_session_meta_if_due(session, force=True)
                self._log_event(session, "session_init", {})
            return self._snapshot(session)

    def start_session(
        self,
        session_id: str,
        self_evolution_recording_enabled: bool = False,
        capture_meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        with self._lock:
            session = self.sessions[session_id]
            session.self_evolution_recording_enabled = bool(self_evolution_recording_enabled)
            session.capture_meta = capture_meta if isinstance(capture_meta, dict) else {}
            session.started_at_monotonic = time.time()
            session.ended_at = ""
            session.review_media_path = ""
            session.review_media_error = ""
            if session.self_evolution_recording_enabled:
                self._prepare_self_evolution_session(session)
                self._start_ipcam_recording(session)
            session.started = True
            session.state = "chapter_intro"
            session.state_until = time.time() + 3.0
            session.section_idx = 0
            session.step_idx = 0
            session.completed_steps = []
            session.recent_frames.clear()
            session.last_eval = None
            session.last_omni = None
            session.last_error = None
            session.last_overlay_detections = []
            session.last_overlay_object_targets = []
            session.last_overlay_step_id = ""
            session.last_overlay_matched = False
            session.last_overlay_ts = 0.0
            session.last_overlay_frame_width = 0
            session.last_overlay_frame_height = 0
            session.latest_annotated_jpg = b""
            self._reset_step_runtime(session)
            self._write_session_meta_if_due(session, force=True)
            self._log_event(
                session,
                "session_start",
                {
                    "section": self._section_intro_payload(session),
                    "self_evolution_recording_enabled": session.self_evolution_recording_enabled,
                    "capture_meta": session.capture_meta,
                },
            )
            return self._snapshot(session)

    def reset_session(self, session_id: str) -> Dict[str, Any]:
        return self.start_session(session_id)

    def evaluate_step(self, session_id: str) -> Dict[str, Any]:
        needs_omni = False
        step_id = ""
        code = ""
        object_targets: List[str] = []
        disambiguation_targets: List[str] = []
        error_plans: List[Dict[str, Any]] = []
        use_hand_model = False
        sample_seq = 0
        object_memory: Dict[str, Dict[str, Any]] = {}
        session_step_idx = -1

        with self._lock:
            session = self.sessions[session_id]
            if not session.started:
                return self._snapshot(session)

            self._advance_state_machine(session)
            needs_omni = session.state == "section_validating"
            if not needs_omni and session.state != "step_running":
                self._write_session_meta_if_due(session, force=needs_omni)
                return self._snapshot(session)
            if needs_omni:
                pass
            else:
                now = time.time()
                interval = 1.0 / max(0.1, float(self.cfg.realtime_yolo_fps))
                if now < session.next_sample_at:
                    return self._snapshot(session)
                session.next_sample_at = now + interval

                step = self._current_step(session)
                if not step:
                    sec = self._current_section(session)
                    if sec is not None and len(sec.steps) == 0:
                        session.state = "section_validating"
                        session.state_until = 0.0
                        self._log_event(session, "section_ready_for_omni_empty", {"section_id": sec.section_id})
                    else:
                        session.state = "course_done"
                    return self._snapshot(session)
                step_id = str(step.get("step_id", ""))
                sec = self._current_section(session)
                section_id = sec.section_id if sec else ""
                plan = self._load_step_plan(step_id)
                code_items = plan.get("judgement_conditions", [])
                code = str(code_items[0].get("code", "")).strip() if isinstance(code_items, list) and code_items else ""
                if not code:
                    self._log_event(session, "step_error", {"step_id": step_id, "error": "empty_judgement_code"})
                    return self._snapshot(session)

                object_targets = self._plan_object_targets(plan)
                error_plans = self._error_plans_for_step(section_id, step_id)
                for error_plan in error_plans:
                    for target in self._plan_object_targets(error_plan):
                        if target not in object_targets:
                            object_targets.append(target)
                disambiguation_targets = list(dict.fromkeys([*object_targets, *self.global_disambiguation_targets]))
                use_hand_model = step_requires_hand_model(plan, code) or any(
                    step_requires_hand_model(
                        ep,
                        str(((ep.get("judgement_conditions") or [{}])[0] if isinstance(ep.get("judgement_conditions"), list) and ep.get("judgement_conditions") else {}).get("code", "")),
                    )
                    for ep in error_plans
                )
                sample_seq = session.step_sample_seq + 1
                session.step_sample_seq = sample_seq
                object_memory = copy.deepcopy(session.step_object_memory)
                session_step_idx = int(session.step_idx)

        if needs_omni:
            session = self.sessions.get(session_id)
            if session is not None:
                self._run_omni_section_validation(session)
                with self._lock:
                    session = self.sessions[session_id]
                    self._write_session_meta_if_due(session, force=True)
                    return self._snapshot(session)

        if use_hand_model and self.hand_adapter is None:
            hand_adapter = MediaPipeHandAdapter()
            hand_adapter.load()
            self.hand_adapter = hand_adapter

        frame = self._get_latest_frame_copy()
        if frame is None:
            with self._lock:
                session = self.sessions[session_id]
                self._log_event(session, "frame_error", {"reason": "failed_to_read_frame"})
                return self._snapshot(session)
        ts = time.time()
        latest_jpg = self._get_latest_frame_jpg()

        overlay_detections = self._infer_yolo_detections(
            frame_bgr=frame,
            targets=disambiguation_targets,
            frame_index=sample_seq,
        )

        ctx = build_context(
            frame_bgr=frame,
            frame_index=sample_seq,
            sample_seq=sample_seq,
            sec=ts,
            object_targets=object_targets,
            disambiguation_targets=disambiguation_targets,
            anchors=self.anchors,
            yolo_adapter=self.yolo_adapter,
            object_memory=object_memory,
            persist_edge_margin=self.cfg.persist_edge_margin,
            persist_max_miss_samples=self.cfg.persist_max_miss_samples,
            dino_adapter=None,
            hand_adapter=self.hand_adapter if use_hand_model else None,
            use_hand_model=use_hand_model,
            enable_dino_fallback=False,
            yolo_detections=overlay_detections,
        )
        matched = eval_condition_code(code, ctx, hand_in_bbox_threshold=self.cfg.hand_in_bbox_threshold)
        matched_error: Optional[Dict[str, Any]] = None
        for error_plan in error_plans:
            conds = error_plan.get("judgement_conditions", [])
            err_code = str(conds[0].get("code", "")).strip() if isinstance(conds, list) and conds else ""
            if not err_code:
                continue
            error_matched = eval_condition_code(err_code, ctx, hand_in_bbox_threshold=self.cfg.hand_in_bbox_threshold)
            if not error_matched:
                continue
            correction = error_plan.get("correction_message") if isinstance(error_plan.get("correction_message"), dict) else {}
            matched_error = {
                "error_id": str(error_plan.get("error_id") or error_plan.get("slice_id") or ""),
                "message": str(correction.get("zh") or correction.get("en") or ""),
                "scope": error_plan.get("scope", {}),
                "evidence": {
                    "matched_condition": err_code,
                    "confidence": 1.0,
                    "detector_source": ctx.get("detector_source", "yolo"),
                },
            }
            break
        filtered_detections = [d for d in overlay_detections if isinstance(d, dict)]
        trace_payload = {
            "step_id": step_id,
            "sample_seq": sample_seq,
            "matched": bool(matched),
            "error_matched": bool(matched_error),
            "last_error": matched_error,
            "objects": ctx.get("objects", {}),
            "hand_points": ctx.get("hand_points", {}),
            "detector_source": ctx.get("detector_source", "yolo"),
        }

        with self._lock:
            session = self.sessions[session_id]
            if not session.started or session.state != "step_running":
                return self._snapshot(session)
            current_step = self._current_step(session)
            current_step_id = str((current_step or {}).get("step_id", ""))
            if current_step_id != step_id or int(session.step_idx) != session_step_idx:
                return self._snapshot(session)

            session.step_object_memory = object_memory
            if latest_jpg:
                session.recent_frames.append((ts, latest_jpg))
            session.last_overlay_detections = filtered_detections
            session.last_overlay_object_targets = [str(x) for x in object_targets]
            session.last_overlay_step_id = step_id
            session.last_overlay_matched = bool(matched)
            session.last_overlay_ts = ts
            session.last_overlay_frame_width = int(frame.shape[1]) if getattr(frame, "shape", None) is not None else 0
            session.last_overlay_frame_height = int(frame.shape[0]) if getattr(frame, "shape", None) is not None else 0
            self._log_trace(session, trace_payload)

            session.last_eval = {
                "step_id": step_id,
                "sample_seq": sample_seq,
                "matched": bool(matched),
                "detector_source": ctx.get("detector_source", "yolo"),
            }

            if matched_error:
                session.last_error = matched_error
                self._log_event(session, "known_error_detected", matched_error)
            else:
                session.last_error = None

            if matched and not matched_error:
                session.completed_steps.append(step_id)
                session.state = "step_done_hold"
                session.state_until = time.time() + 3.0
                self._log_event(
                    session,
                    "step_done",
                    {
                        "step_id": step_id,
                        "sample_seq": sample_seq,
                        "matched_at": utc_now_iso(),
                    },
                )

            self._write_session_meta_if_due(session, force=bool(matched))
            return self._snapshot(session)

    def poll_session(self, session_id: str) -> Dict[str, Any]:
        with self._lock:
            session = self.sessions[session_id]
            return self._snapshot(session)

    def get_visual_frame(self, session_id: str) -> bytes:
        session = self.sessions.get(session_id)
        if session is not None and session.latest_annotated_jpg:
            return session.latest_annotated_jpg
        if self._latest_annotated_jpg:
            return self._latest_annotated_jpg
        raw = self._get_latest_frame_jpg()
        return raw if raw else self._blank_jpg

    def get_overlay_payload(self, session_id: str) -> Dict[str, Any]:
        with self._lock:
            session = self.sessions.get(session_id)
            if session is None:
                return {"ok": False, "error": "unknown_session", "session_id": session_id}

            now = time.time()
            stale_sec = max(0.05, float(self.cfg.box_max_stale_ms) / 1000.0)
            fresh = (now - float(session.last_overlay_ts or 0.0)) <= stale_sec
            targets = [str(x) for x in session.last_overlay_object_targets if str(x)]
            target_norm = {normalize_name(x) for x in targets}
            detections: List[Dict[str, Any]] = []
            for det in session.last_overlay_detections if isinstance(session.last_overlay_detections, list) else []:
                if not isinstance(det, dict):
                    continue
                label = str(det.get("label", det.get("target", ""))).strip()
                if not label:
                    continue
                if target_norm and normalize_name(label) not in target_norm:
                    continue
                bbox = det.get("bbox_xyxy", {})
                if not isinstance(bbox, dict):
                    continue
                detections.append(
                    {
                        "label": label,
                        "score": self._safe_float(det.get("score", 0.0)),
                        "bbox_xyxy": {
                            "x1": self._safe_float(bbox.get("x1", 0.0)),
                            "y1": self._safe_float(bbox.get("y1", 0.0)),
                            "x2": self._safe_float(bbox.get("x2", 0.0)),
                            "y2": self._safe_float(bbox.get("y2", 0.0)),
                        },
                    }
                )

            overlay_age_ms = max(0, int((now - float(session.last_overlay_ts or 0.0)) * 1000.0))
            capture_age_ms = max(0, int((now - float(self._latest_frame_ts or 0.0)) * 1000.0))
            return {
                "ok": True,
                "session_id": session.session_id,
                "started": session.started,
                "state": session.state,
                "step_id": session.last_overlay_step_id or None,
                "matched": bool(session.last_overlay_matched),
                "fresh": bool(fresh),
                "overlay_age_ms": overlay_age_ms,
                "capture_age_ms": capture_age_ms,
                "frame_width": int(session.last_overlay_frame_width or 0),
                "frame_height": int(session.last_overlay_frame_height or 0),
                "object_targets": targets,
                "detections": detections if fresh else [],
            }

    def force_next_section(self, session_id: str) -> Dict[str, Any]:
        with self._lock:
            session = self.sessions[session_id]
            self._goto_next_section(session, reason="manual_next_section")
            self._write_session_meta_if_due(session, force=True)
            self._log_event(session, "manual_next_section", {})
            return self._snapshot(session)

    def force_next_step(self, session_id: str) -> Dict[str, Any]:
        with self._lock:
            session = self.sessions[session_id]
            sec = self._current_section(session)
            if sec is None:
                return self._snapshot(session)
            current_step = self._current_step(session)
            current_step_id = str((current_step or {}).get("step_id", ""))
            session.last_error = None
            session.step_idx += 1
            if session.step_idx >= len(sec.steps):
                self._goto_next_section(session, reason="manual_next_step_at_section_end")
            else:
                self._start_step_running(session)
            self._write_session_meta_if_due(session, force=True)
            self._log_event(
                session,
                "manual_next_step",
                {
                    "from_step_id": current_step_id,
                    "to_step_id": str((self._current_step(session) or {}).get("step_id", "")),
                    "section_id": sec.section_id,
                },
            )
            return self._snapshot(session)

    def retry_section(self, session_id: str) -> Dict[str, Any]:
        with self._lock:
            session = self.sessions[session_id]
            sec = self._current_section(session)
            if sec is None:
                return self._snapshot(session)
            session.step_idx = 0
            self._start_step_running(session)
            self._write_session_meta_if_due(session, force=True)
            self._log_event(session, "manual_retry_section", {"section_id": sec.section_id})
            return self._snapshot(session)

    def validate_section_now(self, session_id: str) -> Dict[str, Any]:
        with self._lock:
            session = self.sessions[session_id]
            session.state = "section_validating"
        self._run_omni_section_validation(session)
        with self._lock:
            session = self.sessions[session_id]
            self._write_session_meta_if_due(session, force=True)
            return self._snapshot(session)

    def get_logs(self, session_id: str) -> Dict[str, Any]:
        with self._lock:
            session = self.sessions[session_id]
            base = self._session_dir(session)
            out: Dict[str, Any] = {"session_id": session_id, "base_dir": str(base)}
            for name, path in {
                "session_meta": self._session_meta_path(session),
                "events": self._events_path(session),
                "omni_calls": self._omni_path(session),
                "step_trace": self._trace_path(session),
            }.items():
                if not path.exists():
                    out[name] = []
                    continue
                if path.suffix == ".json":
                    out[name] = read_json(path)
                else:
                    lines = path.read_text(encoding="utf-8").splitlines()
                    out[name] = [json.loads(x) for x in lines if x.strip()]
            return out

    def store_self_evolution_media(self, session_id: str, media_bytes: bytes, content_type: str) -> Dict[str, Any]:
        with self._lock:
            session = self.sessions[session_id]
            if not session.self_evolution_recording_enabled:
                return {"ok": False, "error": "self_evolution_recording_not_enabled", "session_id": session_id}
            raw_dir = self._self_evolution_raw_dir(session)
            raw_dir.mkdir(parents=True, exist_ok=True)
            lower_type = (content_type or "").lower()
            suffix = ".webm"
            if "wav" in lower_type:
                suffix = ".wav"
            elif "mpeg" in lower_type or "mp3" in lower_type:
                suffix = ".mp3"
            path = raw_dir / f"teaching_audio{suffix}"
            path.write_bytes(media_bytes)
            session.raw_audio_path = str(path.relative_to(ROOT))
            session.raw_media_path = session.raw_audio_path
            session.ended_at = utc_now_iso()
            self._log_event(
                session,
                "self_evolution_audio_uploaded",
                {"path": session.raw_audio_path, "content_type": content_type, "bytes": len(media_bytes)},
            )
            self._write_session_meta_if_due(session, force=True)
            return {
                "ok": True,
                "session_id": session_id,
                "raw_audio_path": session.raw_audio_path,
                "bytes": len(media_bytes),
            }

    def finish_self_evolution_session(self, session_id: str) -> Dict[str, Any]:
        with self._lock:
            session = self.sessions[session_id]
            self._stop_ipcam_recording(session)
            self._build_review_media(session)
            session.ended_at = session.ended_at or utc_now_iso()
            self._log_event(
                session,
                "self_evolution_session_finished",
                {
                    "auto_reflection_pending": True,
                    "review_media_path": session.review_media_path,
                    "review_media_error": session.review_media_error,
                },
            )
            self._write_session_meta_if_due(session, force=True)
            return {
                "ok": True,
                "session_id": session_id,
                "self_evolution_recording_enabled": session.self_evolution_recording_enabled,
                "raw_media_path": session.raw_media_path,
                "raw_audio_path": session.raw_audio_path,
                "raw_video_path": session.raw_video_path,
                "review_media_path": session.review_media_path,
                "review_media_error": session.review_media_error,
                "recorded_frame_count": session.recorded_frame_count,
            }

    def discard_self_evolution_session(self, session_id: str) -> Dict[str, Any]:
        with self._lock:
            session = self.sessions.get(session_id)
            if session is None:
                return {"ok": True, "session_id": session_id, "discarded": False, "reason": "unknown_session"}
            self._stop_ipcam_recording(session)
            v5_dir = self._self_evolution_session_dir(session)
            session.self_evolution_recording_enabled = False
            session.capture_meta = {}
            session.raw_media_path = ""
            session.raw_audio_path = ""
            session.raw_video_path = ""
            session.review_media_path = ""
            session.review_media_error = ""
            session.ended_at = utc_now_iso()
        if v5_dir.exists():
            shutil.rmtree(v5_dir)
        return {"ok": True, "session_id": session_id, "discarded": True, "deleted_path": str(v5_dir)}

    def _snapshot(self, session: SessionState) -> Dict[str, Any]:
        sec = self._current_section(session)
        step = self._current_step(session)
        section_count = len(self.sections)
        step_count = len(sec.steps) if sec else 0
        is_done = session.state == "course_done"
        message = ""
        if session.state == "chapter_intro":
            intro = self._section_intro_payload(session)
            message = f"开始章节：{intro.get('title', '')}（步骤 {intro.get('step_count', 0)}）"
        elif session.state == "step_done_hold":
            message = "步骤完成，3秒后进入下一步"
        elif session.state == "section_validating":
            message = "章节完成，正在进行 Omni 校验"
        elif session.state == "course_done":
            message = "课程完成，恭喜！"
        elif step:
            message = str((step.get("prompt") or {}).get("zh") or (step.get("prompt") or {}).get("en") or "")
        raw_stream_url = f"/api/realtime/stream.mjpg?session_id={session.session_id}"
        return {
            "ok": True,
            "session_id": session.session_id,
            "started": session.started,
            "state": session.state,
            "message": message,
            "video_feed_url": self.frame_source.stream_display_url(),
            "video_feed_overlay_url": f"/api/realtime/frame?session_id={session.session_id}",
            "video_feed_stream_url": raw_stream_url,
            "video_primary_url": raw_stream_url,
            "overlay_meta_url": f"/api/realtime/overlay?session_id={session.session_id}",
            "section": {
                "index": session.section_idx,
                "count": section_count,
                "section_id": (sec.section_id if sec else None),
                "section_name": (sec.section_name if sec else None),
                "section_goal": (sec.section_goal if sec else None),
                "expected_section_state": (sec.expected_section_state if sec else None),
            },
            "step": {
                "index": session.step_idx,
                "count": step_count,
                "step_id": (step.get("step_id") if step else None),
                "prompt": ((step.get("prompt") if step else {}) or {}),
                "focus_points": ((step.get("focus_points") if step else []) or []),
                "common_mistakes": ((step.get("common_mistakes") if step else []) or []),
            },
            "last_eval": session.last_eval,
            "last_omni": session.last_omni,
            "last_error": session.last_error,
            "is_done": is_done,
            "self_evolution": {
                "mode": self.cfg.mode,
                "recording_enabled": session.self_evolution_recording_enabled,
                "session_dir": str(self._self_evolution_session_dir(session).relative_to(ROOT)),
                "raw_media_path": session.raw_media_path,
                "raw_audio_path": session.raw_audio_path,
                "raw_video_path": session.raw_video_path,
                "review_media_path": session.review_media_path,
                "review_media_error": session.review_media_error,
                "video_source": "ipcam",
                "audio_source": "browser_microphone",
                "recorded_frame_count": session.recorded_frame_count,
                "capture": {
                    "target_width": int(self.cfg.target_width),
                    "target_height": int(self.cfg.target_height),
                    "target_fps": float(self.cfg.target_fps),
                    "actual_width": int(session.capture_meta.get("actual_width") or session.capture_meta.get("width") or 0),
                    "actual_height": int(session.capture_meta.get("actual_height") or session.capture_meta.get("height") or 0),
                    "actual_fps": float(session.capture_meta.get("actual_fps") or session.capture_meta.get("fps") or 0.0),
                },
            },
            "config": {
                "realtime_yolo_fps": self.cfg.realtime_yolo_fps,
                "realtime_capture_fps": self.cfg.capture_fps,
                "realtime_render_fps": self.cfg.render_fps,
                "realtime_stream_fps": self.cfg.stream_fps,
                "realtime_stream_max_width": self.cfg.stream_max_width,
                "server_side_overlay_enabled": self.cfg.server_side_overlay_enabled,
                "strategy_version": self.cfg.strategy_version,
                "strategy_path": str(self.strategy_path),
                "error_library": self.cfg.error_library,
                "loaded_error_count": sum(len(v) for v in self.error_plans_by_step.values()),
                "box_max_stale_ms": self.cfg.box_max_stale_ms,
                "omni_section_pass_threshold": self.cfg.omni_section_pass_threshold,
                "omni_model": self.cfg.omni_model,
                "yolo_conf": self.cfg.yolo_conf,
                "yolo_iou": self.cfg.yolo_iou,
                "persist_edge_margin": self.cfg.persist_edge_margin,
                "persist_max_miss_seconds": self.cfg.persist_max_miss_seconds,
                "persist_max_miss_samples": self.cfg.persist_max_miss_samples,
            },
        }


class RealtimeHandler(BaseHTTPRequestHandler):
    engine: RealtimeTutorEngine = None  # type: ignore

    def _json(self, status: int, payload: Dict[str, Any]) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(raw)

    def _read_json_body(self) -> Dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except Exception:
            length = 0
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        if not raw:
            return {}
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception:
            return {}

    def _jpeg(self, status: int, blob: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(blob)))
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.end_headers()
        if blob:
            self.wfile.write(blob)

    def _stream_mjpeg(self, session_id: str) -> None:
        boundary = "frame"
        self.send_response(200)
        self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={boundary}")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.end_headers()
        min_interval = 1.0 / max(1.0, float(self.engine.cfg.stream_fps))
        last_seq = -1
        last_sent_at = 0.0
        while True:
            try:
                seq = self.engine._get_latest_frame_seq()
                if seq <= 0 or seq == last_seq:
                    time.sleep(0.005)
                    continue
                now = time.time()
                wait = min_interval - (now - last_sent_at)
                if wait > 0:
                    time.sleep(wait)
                blob = self.engine._get_latest_frame_jpg()
                if not blob:
                    blob = blank_jpeg_bytes()
                head = (
                    f"--{boundary}\r\n"
                    f"Content-Type: image/jpeg\r\n"
                    f"Content-Length: {len(blob)}\r\n\r\n"
                ).encode("ascii")
                self.wfile.write(head)
                self.wfile.write(blob)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
                last_seq = seq
                last_sent_at = time.time()
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                return
            except Exception:
                return

    def _static_file(self, rel_path: str) -> None:
        p = (WEB_ROOT / rel_path).resolve()
        if not str(p).startswith(str(WEB_ROOT.resolve())) or not p.exists() or not p.is_file():
            self.send_error(404, "Not Found")
            return
        if p.suffix == ".html":
            ctype = "text/html; charset=utf-8"
        elif p.suffix == ".js":
            ctype = "application/javascript; charset=utf-8"
        elif p.suffix == ".css":
            ctype = "text/css; charset=utf-8"
        else:
            ctype = "application/octet-stream"
        data = p.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)
        if path == "/" or path == "/index.html":
            return self._static_file("index.html")
        if path == "/app.js":
            return self._static_file("app.js")
        if path == "/styles.css":
            return self._static_file("styles.css")
        if path == "/api/realtime/session/init":
            sid = (qs.get("session_id", [""])[0] or "").strip() or None
            try:
                payload = self.engine.init_session(sid)
                return self._json(200, payload)
            except Exception as exc:
                return self._json(500, {"ok": False, "error": str(exc)})
        if path == "/api/realtime/stream.mjpg":
            sid = (qs.get("session_id", [""])[0] or "").strip()
            if not sid:
                sid = str(uuid.uuid4())
            if sid not in self.engine.sessions:
                self.engine.init_session(sid)
            return self._stream_mjpeg(sid)
        if path == "/api/realtime/overlay":
            sid = (qs.get("session_id", [""])[0] or "").strip()
            if not sid:
                return self._json(400, {"ok": False, "error": "missing session_id"})
            try:
                payload = self.engine.get_overlay_payload(sid)
                status = 200 if payload.get("ok") else 404
                return self._json(status, payload)
            except Exception as exc:
                return self._json(500, {"ok": False, "error": str(exc)})
        if path == "/api/realtime/frame":
            sid = (qs.get("session_id", [""])[0] or "").strip()
            if not sid:
                blob = blank_jpeg_bytes(text="missing session_id")
                return self._jpeg(200, blob)
            try:
                blob = self.engine.get_visual_frame(sid)
                if not blob:
                    blob = blank_jpeg_bytes()
                return self._jpeg(200, blob)
            except Exception as exc:
                blob = blank_jpeg_bytes(text=f"frame error: {str(exc)[:40]}")
                return self._jpeg(200, blob)
        if path.startswith("/api/realtime/logs/"):
            sid = path.rsplit("/", 1)[-1].strip()
            try:
                payload = self.engine.get_logs(sid)
                return self._json(200, {"ok": True, "data": payload})
            except Exception as exc:
                return self._json(500, {"ok": False, "error": str(exc)})
        if path == "/healthz":
            return self._json(200, {"ok": True, "ts": utc_now_iso()})
        self.send_error(404, "Not Found")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/api/self-evolution/session/media":
            qs = parse_qs(parsed.query)
            sid = (qs.get("session_id", [""])[0] or self.headers.get("X-Session-Id", "")).strip()
            if not sid:
                return self._json(400, {"ok": False, "error": "missing session_id"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except Exception:
                length = 0
            if length <= 0:
                return self._json(400, {"ok": False, "error": "empty_media_body"})
            media = self.rfile.read(length)
            try:
                payload = self.engine.store_self_evolution_media(
                    sid,
                    media,
                    self.headers.get("Content-Type", "application/octet-stream"),
                )
                return self._json(200 if payload.get("ok") else 400, payload)
            except Exception as exc:
                return self._json(500, {"ok": False, "error": str(exc)})

        body = self._read_json_body()
        sid = str(body.get("session_id", "")).strip()

        try:
            if path == "/api/realtime/session/start":
                payload = self.engine.start_session(
                    sid,
                    self_evolution_recording_enabled=bool(body.get("self_evolution_recording_enabled", False)),
                    capture_meta=body.get("capture_meta") if isinstance(body.get("capture_meta"), dict) else {},
                )
                return self._json(200, payload)
            if path == "/api/self-evolution/session/finish":
                payload = self.engine.finish_self_evolution_session(sid)
                return self._json(200, payload)
            if path == "/api/self-evolution/session/discard":
                payload = self.engine.discard_self_evolution_session(sid)
                return self._json(200, payload)
            if path == "/api/realtime/session/reset":
                payload = self.engine.reset_session(sid)
                return self._json(200, payload)
            if path == "/api/realtime/step/evaluate":
                payload = self.engine.poll_session(sid)
                return self._json(200, payload)
            if path == "/api/realtime/section/validate":
                payload = self.engine.validate_section_now(sid)
                return self._json(200, payload)
            if path == "/api/realtime/section/next":
                payload = self.engine.force_next_section(sid)
                return self._json(200, payload)
            if path == "/api/realtime/step/next":
                payload = self.engine.force_next_step(sid)
                return self._json(200, payload)
            if path == "/api/realtime/section/retry":
                payload = self.engine.retry_section(sid)
                return self._json(200, payload)
        except Exception as exc:
            return self._json(500, {"ok": False, "error": str(exc)})

        self.send_error(404, "Not Found")


def build_runtime_config(case_id: str, mode: str = "realtime", strategy_version: str = "base", error_library: str = "none") -> RuntimeConfig:
    realtime_yolo_fps = max(0.1, env_float("REALTIME_YOLO_FPS", RT_DEFAULT_YOLO_FPS))
    ingest_meta = load_ingest_video_meta(case_id)
    persist_max_miss_seconds = max(
        0.1,
        env_float("REALTIME_PERSIST_MAX_MISS_SECONDS", RT_DEFAULT_PERSIST_MAX_MISS_SECONDS),
    )
    return RuntimeConfig(
        case_id=case_id,
        mode=mode,
        strategy_version=strategy_version,
        error_library=error_library,
        ip_webcam_url=env_str("IP_WEBCAM_URL", RT_DEFAULT_IP_WEBCAM_URL),
        realtime_yolo_fps=realtime_yolo_fps,
        omni_section_pass_threshold=max(
            0.0,
            min(1.0, env_float("OMNI_SECTION_PASS_THRESHOLD", RT_DEFAULT_OMNI_SECTION_PASS_THRESHOLD)),
        ),
        skip_omni_validation=env_bool("REALTIME_SKIP_OMNI_VALIDATION", RT_DEFAULT_SKIP_OMNI_VALIDATION),
        omni_model=env_str("OMNI_MODEL", RT_DEFAULT_OMNI_MODEL),
        omni_base_url=TA_BASE_URL,
        omni_timeout_sec=int(TA_REQUEST_TIMEOUT_SEC),
        yolo_conf=max(0.01, min(0.95, env_float("REALTIME_YOLO_CONF", RT_DEFAULT_YOLO_CONF))),
        yolo_iou=max(0.01, min(0.95, env_float("REALTIME_YOLO_IOU", RT_DEFAULT_YOLO_IOU))),
        yolo_device=env_str("REALTIME_YOLO_DEVICE", RT_DEFAULT_YOLO_DEVICE),
        capture_fps=max(1.0, min(60.0, env_float("REALTIME_CAPTURE_FPS", RT_DEFAULT_CAPTURE_FPS))),
        render_fps=max(1.0, min(60.0, env_float("REALTIME_RENDER_FPS", RT_DEFAULT_RENDER_FPS))),
        stream_fps=max(1.0, min(60.0, env_float("REALTIME_STREAM_FPS", RT_DEFAULT_STREAM_FPS))),
        stream_max_width=max(0, int(env_float("REALTIME_STREAM_MAX_WIDTH", float(RT_DEFAULT_STREAM_MAX_WIDTH)))),
        server_side_overlay_enabled=env_bool("REALTIME_SERVER_SIDE_OVERLAY", RT_DEFAULT_SERVER_SIDE_OVERLAY),
        box_max_stale_ms=max(50, min(1000, int(env_float("BOX_MAX_STALE_MS", float(RT_DEFAULT_BOX_MAX_STALE_MS))))),
        persist_edge_margin=max(
            0.0,
            min(0.3, env_float("REALTIME_PERSIST_EDGE_MARGIN", RT_DEFAULT_PERSIST_EDGE_MARGIN)),
        ),
        persist_max_miss_seconds=persist_max_miss_seconds,
        persist_max_miss_samples=max(1, int(math.ceil(persist_max_miss_seconds * realtime_yolo_fps))),
        hand_in_bbox_threshold=max(
            0.01,
            min(0.5, env_float("REALTIME_HAND_IN_BBOX_THRESHOLD", RT_DEFAULT_HAND_IN_BBOX_THRESHOLD)),
        ),
        target_width=int(ingest_meta["target_width"]),
        target_height=int(ingest_meta["target_height"]),
        target_fps=float(ingest_meta["target_fps"]),
        ingest_video_path=str(ingest_meta["ingest_video_path"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Realtime tutor web server")
    parser.add_argument("--case-id", default="test_cake")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--mode", choices=["realtime", "self-evolution"], default="realtime")
    parser.add_argument("--strategy-version", choices=["base", "evolved"], default="base")
    parser.add_argument("--error-library", choices=["none", "v5"], default="none")
    args = parser.parse_args()

    cfg = build_runtime_config(
        args.case_id,
        mode=args.mode,
        strategy_version=args.strategy_version,
        error_library=args.error_library,
    )
    engine = RealtimeTutorEngine(cfg)
    RealtimeHandler.engine = engine
    server = ThreadingHTTPServer((args.host, int(args.port)), RealtimeHandler)
    print(f"[INFO] realtime tutor server: http://{args.host}:{args.port}")
    print(
        f"[INFO] case={args.case_id} mode={cfg.mode} strategy_version={cfg.strategy_version} "
        f"error_library={cfg.error_library} webcam={cfg.ip_webcam_url} "
        f"capture_fps={cfg.capture_fps} yolo_fps={cfg.realtime_yolo_fps} "
        f"render_fps={cfg.render_fps} stream_fps={cfg.stream_fps} "
        f"stream_max_width={cfg.stream_max_width} "
        f"server_side_overlay_enabled={cfg.server_side_overlay_enabled}"
    )
    print(
        f"[INFO] omni_model={cfg.omni_model} threshold={cfg.omni_section_pass_threshold} "
        f"skip_omni_validation={cfg.skip_omni_validation}"
    )
    print(f"[INFO] self_evolution_target={cfg.target_width}x{cfg.target_height}@{cfg.target_fps}fps")
    print(f"[INFO] strategy_path={engine.strategy_path}")
    print(f"[INFO] loaded_error_count={sum(len(v) for v in engine.error_plans_by_step.values())}")
    server.serve_forever()


if __name__ == "__main__":
    main()
