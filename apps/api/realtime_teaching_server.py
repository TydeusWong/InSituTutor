import argparse
import base64
import copy
import json
import math
import os
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
    ip_webcam_url: str
    realtime_yolo_fps: float
    omni_section_pass_threshold: float
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


class RealtimeTutorEngine:
    def __init__(self, cfg: RuntimeConfig) -> None:
        self.cfg = cfg
        self.api_key = get_api_key()
        self.frame_source = FrameSource(cfg.ip_webcam_url)
        self.strategy = read_json(ROOT / "data" / cfg.case_id / "v2" / "strategy" / "teaching_strategy_v2.json")
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

    def _session_meta_path(self, session: SessionState) -> Path:
        return self._session_dir(session) / "session_meta.json"

    def _events_path(self, session: SessionState) -> Path:
        return self._session_dir(session) / "events.jsonl"

    def _omni_path(self, session: SessionState) -> Path:
        return self._session_dir(session) / "omni_calls.jsonl"

    def _trace_path(self, session: SessionState) -> Path:
        return self._session_dir(session) / "step_trace.jsonl"

    def _write_session_meta(self, session: SessionState) -> None:
        write_json(
            self._session_meta_path(session),
            {
                "session_id": session.session_id,
                "case_id": session.case_id,
                "created_at": session.created_at,
                "updated_at": utc_now_iso(),
                "state": session.state,
                "section_idx": session.section_idx,
                "step_idx": session.step_idx,
                "started": session.started,
                "runtime_config": {
                    "ip_webcam_url": self.cfg.ip_webcam_url,
                    "realtime_yolo_fps": self.cfg.realtime_yolo_fps,
                    "realtime_capture_fps": self.cfg.capture_fps,
                    "realtime_render_fps": self.cfg.render_fps,
                    "realtime_stream_fps": self.cfg.stream_fps,
                    "box_max_stale_ms": self.cfg.box_max_stale_ms,
                    "omni_section_pass_threshold": self.cfg.omni_section_pass_threshold,
                    "omni_model": self.cfg.omni_model,
                    "yolo_conf": self.cfg.yolo_conf,
                    "yolo_iou": self.cfg.yolo_iou,
                    "yolo_device": self.cfg.yolo_device,
                    "persist_max_miss_seconds": self.cfg.persist_max_miss_seconds,
                    "persist_max_miss_samples": self.cfg.persist_max_miss_samples,
                },
            },
        )

    def _write_session_meta_if_due(self, session: SessionState, force: bool = False) -> None:
        now = time.time()
        if not force and session.last_meta_write_ts > 0 and (now - session.last_meta_write_ts) < 1.0:
            return
        self._write_session_meta(session)
        session.last_meta_write_ts = now

    def _log_event(self, session: SessionState, event: str, payload: Dict[str, Any]) -> None:
        append_jsonl(
            self._events_path(session),
            {
                "ts": utc_now_iso(),
                "event": event,
                "session_id": session.session_id,
                "section_idx": session.section_idx,
                "step_idx": session.step_idx,
                "state": session.state,
                "payload": payload,
            },
        )

    def _log_trace(self, session: SessionState, payload: Dict[str, Any]) -> None:
        append_jsonl(
            self._trace_path(session),
            {
                "ts": utc_now_iso(),
                "session_id": session.session_id,
                "section_idx": session.section_idx,
                "step_idx": session.step_idx,
                "payload": payload,
            },
        )

    def _log_omni(self, session: SessionState, payload: Dict[str, Any]) -> None:
        append_jsonl(
            self._omni_path(session),
            {
                "ts": utc_now_iso(),
                "session_id": session.session_id,
                "section_idx": session.section_idx,
                "step_idx": session.step_idx,
                "payload": payload,
            },
        )

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
            self._log_event(session, "step_start", {"step_id": step.get("step_id"), "step_order": step.get("step_order")})

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
        if not self.api_key:
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

    def start_session(self, session_id: str) -> Dict[str, Any]:
        with self._lock:
            session = self.sessions[session_id]
            session.started = True
            session.state = "chapter_intro"
            session.state_until = time.time() + 3.0
            session.section_idx = 0
            session.step_idx = 0
            session.completed_steps = []
            session.recent_frames.clear()
            session.last_eval = None
            session.last_omni = None
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
            self._log_event(session, "session_start", {"section": self._section_intro_payload(session)})
            return self._snapshot(session)

    def reset_session(self, session_id: str) -> Dict[str, Any]:
        return self.start_session(session_id)

    def evaluate_step(self, session_id: str) -> Dict[str, Any]:
        needs_omni = False
        step_id = ""
        code = ""
        object_targets: List[str] = []
        disambiguation_targets: List[str] = []
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
                plan = self._load_step_plan(step_id)
                code_items = plan.get("judgement_conditions", [])
                code = str(code_items[0].get("code", "")).strip() if isinstance(code_items, list) and code_items else ""
                if not code:
                    self._log_event(session, "step_error", {"step_id": step_id, "error": "empty_judgement_code"})
                    return self._snapshot(session)

                object_targets = extract_object_targets_from_plan(plan)
                disambiguation_targets = list(dict.fromkeys([*object_targets, *self.global_disambiguation_targets]))
                use_hand_model = step_requires_hand_model(plan, code)
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
        filtered_detections = [d for d in overlay_detections if isinstance(d, dict)]
        trace_payload = {
            "step_id": step_id,
            "sample_seq": sample_seq,
            "matched": bool(matched),
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

            if matched:
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
            "is_done": is_done,
            "config": {
                "realtime_yolo_fps": self.cfg.realtime_yolo_fps,
                "realtime_capture_fps": self.cfg.capture_fps,
                "realtime_render_fps": self.cfg.render_fps,
                "realtime_stream_fps": self.cfg.stream_fps,
                "realtime_stream_max_width": self.cfg.stream_max_width,
                "server_side_overlay_enabled": self.cfg.server_side_overlay_enabled,
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
        body = self._read_json_body()
        sid = str(body.get("session_id", "")).strip()

        try:
            if path == "/api/realtime/session/start":
                payload = self.engine.start_session(sid)
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
            if path == "/api/realtime/section/retry":
                payload = self.engine.retry_section(sid)
                return self._json(200, payload)
        except Exception as exc:
            return self._json(500, {"ok": False, "error": str(exc)})

        self.send_error(404, "Not Found")


def build_runtime_config(case_id: str) -> RuntimeConfig:
    realtime_yolo_fps = max(0.1, env_float("REALTIME_YOLO_FPS", RT_DEFAULT_YOLO_FPS))
    persist_max_miss_seconds = max(
        0.1,
        env_float("REALTIME_PERSIST_MAX_MISS_SECONDS", RT_DEFAULT_PERSIST_MAX_MISS_SECONDS),
    )
    return RuntimeConfig(
        case_id=case_id,
        ip_webcam_url=env_str("IP_WEBCAM_URL", RT_DEFAULT_IP_WEBCAM_URL),
        realtime_yolo_fps=realtime_yolo_fps,
        omni_section_pass_threshold=max(
            0.0,
            min(1.0, env_float("OMNI_SECTION_PASS_THRESHOLD", RT_DEFAULT_OMNI_SECTION_PASS_THRESHOLD)),
        ),
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
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Realtime tutor web server")
    parser.add_argument("--case-id", default="test_cake")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8787)
    args = parser.parse_args()

    cfg = build_runtime_config(args.case_id)
    engine = RealtimeTutorEngine(cfg)
    RealtimeHandler.engine = engine
    server = ThreadingHTTPServer((args.host, int(args.port)), RealtimeHandler)
    print(f"[INFO] realtime tutor server: http://{args.host}:{args.port}")
    print(
        f"[INFO] case={args.case_id} webcam={cfg.ip_webcam_url} "
        f"capture_fps={cfg.capture_fps} yolo_fps={cfg.realtime_yolo_fps} "
        f"render_fps={cfg.render_fps} stream_fps={cfg.stream_fps} "
        f"stream_max_width={cfg.stream_max_width} "
        f"server_side_overlay_enabled={cfg.server_side_overlay_enabled}"
    )
    print(f"[INFO] omni_model={cfg.omni_model} threshold={cfg.omni_section_pass_threshold}")
    server.serve_forever()


if __name__ == "__main__":
    main()
