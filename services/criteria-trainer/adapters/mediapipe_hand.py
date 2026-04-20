from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict

import cv2

from .base import InferenceInput, InferenceOutput


class MediaPipeHandAdapter:
    model_name = "mediapipe-hand-landmarker"

    def __init__(self, max_num_hands: int = 2, min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5) -> None:
        self._loaded = False
        self._mp = None
        self._hands = None
        self._task_hand_landmarker = None
        self._backend = None
        self._max_num_hands = int(max_num_hands)
        self._min_detection_confidence = float(min_detection_confidence)
        self._min_tracking_confidence = float(min_tracking_confidence)

    def load(self) -> None:
        import mediapipe as mp

        self._mp = mp
        if hasattr(mp, "solutions"):
            self._hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=self._max_num_hands,
                min_detection_confidence=self._min_detection_confidence,
                min_tracking_confidence=self._min_tracking_confidence,
            )
            self._backend = "solutions"
        else:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision

            root = Path(__file__).resolve().parents[3]
            default_task_path = root / "models" / "small-models" / "hand" / "mediapipe-hand-landmarker" / "hand_landmarker.task"
            task_path = Path(str(default_task_path))
            if not task_path.exists():
                raise RuntimeError(
                    "MediaPipe tasks backend requires hand model asset. Missing file: "
                    f"{task_path}. Please place hand_landmarker.task there."
                )
            base_options = mp_python.BaseOptions(model_asset_path=str(task_path))
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                num_hands=self._max_num_hands,
                min_hand_detection_confidence=self._min_detection_confidence,
                min_tracking_confidence=self._min_tracking_confidence,
            )
            self._task_hand_landmarker = vision.HandLandmarker.create_from_options(options)
            self._backend = "tasks"

        self._loaded = True

    def infer(self, payload: InferenceInput) -> InferenceOutput:
        if not self._loaded:
            raise RuntimeError("adapter not loaded")
        if self._backend is None:
            raise RuntimeError("hand model backend is not initialized")

        frame_bgr = payload.frame_bgr
        if frame_bgr is None:
            raise ValueError("payload.frame_bgr is empty")

        t0 = time.perf_counter()
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        hand_landmarks_out = []
        scores = []
        if self._backend == "solutions":
            if self._hands is None:
                raise RuntimeError("solutions hands model is not initialized")
            results = self._hands.process(frame_rgb)
            if results and results.multi_hand_landmarks:
                for idx, hand_lm in enumerate(results.multi_hand_landmarks):
                    handedness = None
                    score = 0.0
                    if results.multi_handedness and idx < len(results.multi_handedness):
                        cls = results.multi_handedness[idx].classification[0]
                        handedness = cls.label
                        score = float(cls.score)
                        scores.append(score)
                    lms = [{"x": float(lm.x), "y": float(lm.y), "z": float(lm.z)} for lm in hand_lm.landmark]
                    index_tip = lms[8] if len(lms) > 8 else None
                    hand_landmarks_out.append(
                        {
                            "handedness": handedness,
                            "score": score,
                            "landmarks": lms,
                            "index_fingertip": index_tip,
                        }
                    )
        elif self._backend == "tasks":
            if self._task_hand_landmarker is None:
                raise RuntimeError("tasks hand landmarker is not initialized")
            mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=frame_rgb)
            result = self._task_hand_landmarker.detect(mp_image)
            hand_lms = getattr(result, "hand_landmarks", []) or []
            handedness_list = getattr(result, "handedness", []) or []
            for idx, hand_lm_list in enumerate(hand_lms):
                handedness = None
                score = 0.0
                if idx < len(handedness_list) and handedness_list[idx]:
                    cat = handedness_list[idx][0]
                    handedness = str(getattr(cat, "category_name", None) or "")
                    score = float(getattr(cat, "score", 0.0))
                    scores.append(score)
                lms = [{"x": float(lm.x), "y": float(lm.y), "z": float(lm.z)} for lm in hand_lm_list]
                index_tip = lms[8] if len(lms) > 8 else None
                hand_landmarks_out.append(
                    {
                        "handedness": handedness,
                        "score": score,
                        "landmarks": lms,
                        "index_fingertip": index_tip,
                    }
                )
        else:
            raise RuntimeError(f"unknown backend: {self._backend}")

        latency_ms = (time.perf_counter() - t0) * 1000.0
        confidence = float(sum(scores) / len(scores)) if scores else 0.0

        return InferenceOutput(
            model_name=self.model_name,
            features={"hand_landmarks": hand_landmarks_out},
            confidence=confidence,
            latency_ms=latency_ms,
            raw={"num_hands": len(hand_landmarks_out)},
        )

    def healthcheck(self) -> Dict[str, Any]:
        try:
            self.load()
            return {"ok": True, "model_name": self.model_name, "backend": self._backend}
        except Exception as exc:  # pragma: no cover
            return {"ok": False, "model_name": self.model_name, "error": str(exc)}
