from __future__ import annotations

from typing import Any, Dict

from .base import InferenceInput, InferenceOutput


class MediaPipePoseAdapter:
    model_name = "mediapipe-pose-landmarker"

    def __init__(self) -> None:
        self._loaded = False

    def load(self) -> None:
        import mediapipe as mp  # noqa: F401

        self._loaded = True

    def infer(self, payload: InferenceInput) -> InferenceOutput:
        if not self._loaded:
            raise RuntimeError("adapter not loaded")
        return InferenceOutput(
            model_name=self.model_name,
            features={"pose_landmarks": [], "note": "stub inference"},
            confidence=0.0,
            latency_ms=0.0,
            raw={},
        )

    def healthcheck(self) -> Dict[str, Any]:
        try:
            self.load()
            return {"ok": True, "model_name": self.model_name}
        except Exception as exc:  # pragma: no cover
            return {"ok": False, "model_name": self.model_name, "error": str(exc)}
