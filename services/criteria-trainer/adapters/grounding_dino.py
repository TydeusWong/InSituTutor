from __future__ import annotations

import os
import time
from typing import Any, Dict

import cv2
import numpy as np

from .base import InferenceInput, InferenceOutput
from .base import normalize_bbox_xyxy


class GroundingDINOAdapter:
    model_name = "grounding-dino"

    def __init__(
        self,
        model_id: str | None = None,
        processor_id: str | None = None,
        box_threshold: float = 0.30,
        text_threshold: float = 0.25,
        local_files_only: bool = False,
    ) -> None:
        self._loaded = False
        self._model_id = model_id or os.getenv("GROUNDING_DINO_MODEL_ID", "IDEA-Research/grounding-dino-base")
        self._processor_id = processor_id or os.getenv("GROUNDING_DINO_PROCESSOR_ID", self._model_id)
        self._box_threshold = float(box_threshold)
        self._text_threshold = float(text_threshold)
        self._local_files_only = bool(local_files_only)
        self._torch = None
        self._processor = None
        self._model = None
        self._device = "cpu"

    def load(self) -> None:
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self._torch = torch
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self._processor = AutoProcessor.from_pretrained(
                self._processor_id,
                local_files_only=self._local_files_only,
            )
        except Exception as exc:
            if self._local_files_only and os.path.isdir(self._processor_id):
                missing = self._check_local_processor_files(self._processor_id)
                if missing:
                    raise RuntimeError(
                        f"GroundingDINO processor files missing in '{self._processor_id}': {', '.join(missing)}"
                    ) from exc
            raise
        self._model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self._model_id,
            local_files_only=self._local_files_only,
        ).to(self._device).eval()
        self._loaded = True

    @staticmethod
    def _check_local_processor_files(path: str) -> list[str]:
        required_any = [
            "preprocessor_config.json",
        ]
        tokenizer_any_group = [
            "tokenizer.json",
            "vocab.txt",
        ]
        required_optional = [
            "tokenizer_config.json",
            "special_tokens_map.json",
        ]
        missing: list[str] = []
        for name in required_any:
            if not os.path.exists(os.path.join(path, name)):
                missing.append(name)
        if not any(os.path.exists(os.path.join(path, n)) for n in tokenizer_any_group):
            missing.extend(tokenizer_any_group)
        for name in required_optional:
            if not os.path.exists(os.path.join(path, name)):
                missing.append(name)
        return missing

    def infer(self, payload: InferenceInput) -> InferenceOutput:
        if not self._loaded:
            raise RuntimeError("adapter not loaded")

        t0 = time.perf_counter()
        targets = [str(x).strip() for x in (payload.context.get("detect_targets") or []) if str(x).strip()]
        if not targets:
            return InferenceOutput(
                model_name=self.model_name,
                features={"detections": [], "by_target": {}},
                confidence=0.0,
                latency_ms=0.0,
                raw={"model_id": self._model_id, "device": self._device},
            )

        frame_bgr = payload.frame_bgr
        if frame_bgr is None or not isinstance(frame_bgr, np.ndarray):
            raise ValueError("payload.frame_bgr must be a valid numpy ndarray")
        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        detections = []
        by_target: Dict[str, Dict[str, Any]] = {}
        scores: list[float] = []

        for target in targets:
            inputs = self._processor(images=frame_rgb, text=target, return_tensors="pt")
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            with self._torch.no_grad():
                outputs = self._model(**inputs)
            try:
                results = self._processor.post_process_grounded_object_detection(
                    outputs=outputs,
                    input_ids=inputs["input_ids"],
                    target_sizes=[(h, w)],
                    box_threshold=self._box_threshold,
                    text_threshold=self._text_threshold,
                )
            except TypeError:
                # transformers versions differ: some use `threshold` instead of `box_threshold`
                results = self._processor.post_process_grounded_object_detection(
                    outputs=outputs,
                    input_ids=inputs["input_ids"],
                    target_sizes=[(h, w)],
                    threshold=self._box_threshold,
                    text_threshold=self._text_threshold,
                )
            if not results:
                continue
            result0 = results[0]
            boxes = result0.get("boxes", [])
            labels = result0.get("labels", [])
            confs = result0.get("scores", [])
            if len(boxes) == 0:
                continue
            for box, lbl, score in zip(boxes, labels, confs):
                box_xyxy = [float(x) for x in box.tolist()]
                norm = normalize_bbox_xyxy(box_xyxy, w, h)
                center_x = float((norm["x1"] + norm["x2"]) / 2.0)
                center_y = float((norm["y1"] + norm["y2"]) / 2.0)
                det = {
                    "target": target,
                    "label": str(lbl),
                    "score": float(score),
                    "bbox_xyxy": norm,
                    "center": {"x": center_x, "y": center_y},
                }
                detections.append(det)
                scores.append(float(score))
                best = by_target.get(target)
                if best is None or float(score) > float(best.get("score", -1.0)):
                    by_target[target] = det

        latency_ms = (time.perf_counter() - t0) * 1000.0
        confidence = float(sum(scores) / len(scores)) if scores else 0.0
        return InferenceOutput(
            model_name=self.model_name,
            features={"detections": detections, "by_target": by_target},
            confidence=confidence,
            latency_ms=latency_ms,
            raw={"model_id": self._model_id, "device": self._device, "targets": targets},
        )

    def healthcheck(self) -> Dict[str, Any]:
        try:
            import torch
            import transformers  # noqa: F401

            return {
                "ok": True,
                "model_name": self.model_name,
                "torch_cuda_available": bool(torch.cuda.is_available()),
                "model_id": self._model_id,
                "processor_id": self._processor_id,
                "local_files_only": self._local_files_only,
            }
        except Exception as exc:  # pragma: no cover
            return {"ok": False, "model_name": self.model_name, "error": str(exc)}
