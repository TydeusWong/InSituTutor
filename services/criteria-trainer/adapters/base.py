from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Protocol


@dataclass
class InferenceInput:
    frame_bgr: Any
    timestamp_sec: float
    frame_index: int
    context: Dict[str, Any]


@dataclass
class InferenceOutput:
    model_name: str
    features: Dict[str, Any]
    confidence: float
    latency_ms: float
    raw: Dict[str, Any]


class ModelAdapter(Protocol):
    model_name: str

    def load(self) -> None:
        ...

    def infer(self, payload: InferenceInput) -> InferenceOutput:
        ...

    def healthcheck(self) -> Dict[str, Any]:
        ...


def normalize_bbox_xyxy(box: List[float], width: int, height: int) -> Dict[str, float]:
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be > 0")
    x1, y1, x2, y2 = box
    return {
        "x1": max(0.0, min(1.0, x1 / width)),
        "y1": max(0.0, min(1.0, y1 / height)),
        "x2": max(0.0, min(1.0, x2 / width)),
        "y2": max(0.0, min(1.0, y2 / height)),
    }
