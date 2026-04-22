import argparse
import base64
import json
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

from config import TA_BASE_URL, TA_MODEL, TA_REQUEST_TIMEOUT_SEC, get_api_key, load_env_file  # noqa: E402
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


class FrameSource:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.snapshot_urls = self._build_snapshot_urls(self.base_url)
        self.stream_url = self._build_stream_url(self.base_url)
        self._capture: Optional[cv2.VideoCapture] = None

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
                    arr = cv2.imdecode(np.frombuffer(resp.content, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if arr is not None:
                        return arr
                except Exception:
                    continue
        return None

    def _read_stream(self) -> Optional[Any]:
        if self._capture is None or not self._capture.isOpened():
            self._capture = cv2.VideoCapture(self.stream_url)
        if self._capture is None or not self._capture.isOpened():
            return None
        ok, frame = self._capture.read()
        if not ok or frame is None:
            return None
        return frame

    def read_frame(self) -> Optional[Any]:
        frame = self._read_snapshot()
        if frame is not None:
            return frame
        return self._read_stream()


def frame_to_jpeg_bytes(frame_bgr: Any, quality: int = 80) -> bytes:
    ok, encoded = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok or encoded is None:
        return b""
    return bytes(encoded.tobytes())


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
    persist_edge_margin: float
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
            for x in self.bundle.get("grounding_dino_detect_targets", [])
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
                    "omni_section_pass_threshold": self.cfg.omni_section_pass_threshold,
                    "omni_model": self.cfg.omni_model,
                    "yolo_conf": self.cfg.yolo_conf,
                    "yolo_iou": self.cfg.yolo_iou,
                    "yolo_device": self.cfg.yolo_device,
                },
            },
        )

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
        session.state = "step_running"
        session.state_until = 0.0
        session.next_sample_at = 0.0
        self._reset_step_runtime(session)
        step = self._current_step(session)
        if step:
            self._log_event(session, "step_start", {"step_id": step.get("step_id"), "step_order": step.get("step_order")})

    def _tick_state_machine(self, session: SessionState) -> None:
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
        final = {
            "confidence": confidence,
            "pass": passed,
            "retry_step_id": (retry_step_id or None),
            "reason": str(result.get("reason", "")),
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
                self._write_session_meta(session)
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
            self._reset_step_runtime(session)
            self._write_session_meta(session)
            self._log_event(session, "session_start", {"section": self._section_intro_payload(session)})
            return self._snapshot(session)

    def reset_session(self, session_id: str) -> Dict[str, Any]:
        return self.start_session(session_id)

    def evaluate_step(self, session_id: str) -> Dict[str, Any]:
        with self._lock:
            session = self.sessions[session_id]
            if not session.started:
                return self._snapshot(session)

            self._tick_state_machine(session)
            if session.state != "step_running":
                self._write_session_meta(session)
                return self._snapshot(session)

            now = time.time()
            interval = 1.0 / max(0.1, float(self.cfg.realtime_yolo_fps))
            if now < session.next_sample_at:
                return self._snapshot(session)
            session.next_sample_at = now + interval

            step = self._current_step(session)
            if not step:
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
            if use_hand_model and self.hand_adapter is None:
                self.hand_adapter = MediaPipeHandAdapter()
                self.hand_adapter.load()

            frame = self.frame_source.read_frame()
            if frame is None:
                self._log_event(session, "frame_error", {"reason": "failed_to_read_frame"})
                return self._snapshot(session)
            session.step_sample_seq += 1
            ts = time.time()
            jpg = frame_to_jpeg_bytes(frame, quality=70)
            if jpg:
                session.recent_frames.append((ts, jpg))

            ctx = build_context(
                frame_bgr=frame,
                frame_index=session.step_sample_seq,
                sample_seq=session.step_sample_seq,
                sec=ts,
                object_targets=object_targets,
                disambiguation_targets=disambiguation_targets,
                anchors=self.anchors,
                yolo_adapter=self.yolo_adapter,
                object_memory=session.step_object_memory,
                persist_edge_margin=self.cfg.persist_edge_margin,
                persist_max_miss_samples=self.cfg.persist_max_miss_samples,
                dino_adapter=None,
                hand_adapter=self.hand_adapter if use_hand_model else None,
                use_hand_model=use_hand_model,
                enable_dino_fallback=False,
            )
            matched = eval_condition_code(code, ctx, hand_in_bbox_threshold=self.cfg.hand_in_bbox_threshold)
            trace_payload = {
                "step_id": step_id,
                "sample_seq": session.step_sample_seq,
                "matched": bool(matched),
                "objects": ctx.get("objects", {}),
                "hand_points": ctx.get("hand_points", {}),
                "detector_source": ctx.get("detector_source", "yolo"),
            }
            self._log_trace(session, trace_payload)

            session.last_eval = {
                "step_id": step_id,
                "sample_seq": session.step_sample_seq,
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
                        "sample_seq": session.step_sample_seq,
                        "matched_at": utc_now_iso(),
                    },
                )

            self._write_session_meta(session)
            return self._snapshot(session)

    def force_next_section(self, session_id: str) -> Dict[str, Any]:
        with self._lock:
            session = self.sessions[session_id]
            self._goto_next_section(session, reason="manual_next_section")
            self._write_session_meta(session)
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
            self._write_session_meta(session)
            self._log_event(session, "manual_retry_section", {"section_id": sec.section_id})
            return self._snapshot(session)

    def validate_section_now(self, session_id: str) -> Dict[str, Any]:
        with self._lock:
            session = self.sessions[session_id]
            session.state = "section_validating"
            self._tick_state_machine(session)
            self._write_session_meta(session)
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
        return {
            "ok": True,
            "session_id": session.session_id,
            "started": session.started,
            "state": session.state,
            "message": message,
            "video_feed_url": self.frame_source.stream_display_url(),
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
                "omni_section_pass_threshold": self.cfg.omni_section_pass_threshold,
                "omni_model": self.cfg.omni_model,
                "yolo_conf": self.cfg.yolo_conf,
                "yolo_iou": self.cfg.yolo_iou,
                "persist_edge_margin": self.cfg.persist_edge_margin,
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
                payload = self.engine.evaluate_step(sid)
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
    return RuntimeConfig(
        case_id=case_id,
        ip_webcam_url=env_str("IP_WEBCAM_URL", "http://127.0.0.1:8080"),
        realtime_yolo_fps=max(0.1, env_float("REALTIME_YOLO_FPS", 2.0)),
        omni_section_pass_threshold=max(0.0, min(1.0, env_float("OMNI_SECTION_PASS_THRESHOLD", 0.80))),
        omni_model=env_str("OMNI_MODEL", TA_MODEL),
        omni_base_url=TA_BASE_URL,
        omni_timeout_sec=int(TA_REQUEST_TIMEOUT_SEC),
        yolo_conf=max(0.01, min(0.95, env_float("REALTIME_YOLO_CONF", 0.2))),
        yolo_iou=max(0.01, min(0.95, env_float("REALTIME_YOLO_IOU", 0.45))),
        yolo_device=env_str("REALTIME_YOLO_DEVICE", "cuda:0"),
        persist_edge_margin=max(0.0, min(0.3, env_float("REALTIME_PERSIST_EDGE_MARGIN", 0.05))),
        persist_max_miss_samples=max(1, int(env_float("REALTIME_PERSIST_MAX_MISS_SAMPLES", 6))),
        hand_in_bbox_threshold=max(0.01, min(0.5, env_float("REALTIME_HAND_IN_BBOX_THRESHOLD", 0.12))),
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
    print(f"[INFO] case={args.case_id} webcam={cfg.ip_webcam_url} fps={cfg.realtime_yolo_fps}")
    print(f"[INFO] omni_model={cfg.omni_model} threshold={cfg.omni_section_pass_threshold}")
    server.serve_forever()


if __name__ == "__main__":
    main()
