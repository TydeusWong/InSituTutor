from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict

from .base import InferenceInput, InferenceOutput
from .base import normalize_bbox_xyxy


class YOLOAdapter:
    model_name = "yolo"

    def __init__(
        self,
        weights_path: str | None = None,
        conf: float = 0.25,
        iou: float = 0.45,
        device: str = "cuda:0",
        class_names: list[str] | None = None,
    ) -> None:
        self._loaded = False
        self._weights_path = weights_path
        self._conf = float(conf)
        self._iou = float(iou)
        self._device = str(device)
        self._class_names = list(class_names) if isinstance(class_names, list) else None
        self._model = None

    def load(self) -> None:
        from ultralytics import YOLO

        if not self._weights_path:
            raise RuntimeError("YOLOAdapter requires weights_path")
        path = Path(self._weights_path)
        if not path.exists():
            raise RuntimeError(f"YOLO weights not found: {path}")
        self._model = YOLO(str(path))
        self._loaded = True

    def infer(self, payload: InferenceInput) -> InferenceOutput:
        if not self._loaded:
            raise RuntimeError("adapter not loaded")
        if self._model is None:
            raise RuntimeError("YOLO model is not initialized")
        frame_bgr = payload.frame_bgr
        if frame_bgr is None:
            raise ValueError("payload.frame_bgr is empty")
        h, w = frame_bgr.shape[:2]
        t0 = time.perf_counter()

        preds = self._model.predict(
            source=frame_bgr,
            conf=self._conf,
            iou=self._iou,
            device=self._device,
            verbose=False,
        )

        detections: list[Dict[str, Any]] = []
        by_target: Dict[str, Dict[str, Any]] = {}
        target_set = {str(x).strip() for x in (payload.context.get("detect_targets") or []) if str(x).strip()}
        scores: list[float] = []

        if preds:
            result = preds[0]
            boxes = getattr(result, "boxes", None)
            names = self._class_names if self._class_names is not None else getattr(result, "names", {})
            if boxes is not None:
                for box in boxes:
                    try:
                        cls_id = int(box.cls.item())
                        score = float(box.conf.item())
                        xyxy = [float(x) for x in box.xyxy[0].tolist()]
                    except Exception:
                        continue
                    if isinstance(names, dict):
                        label = str(names.get(cls_id, cls_id))
                    elif isinstance(names, list) and cls_id < len(names):
                        label = str(names[cls_id])
                    else:
                        label = str(cls_id)
                    norm = normalize_bbox_xyxy(xyxy, w, h)
                    center_x = float((norm["x1"] + norm["x2"]) / 2.0)
                    center_y = float((norm["y1"] + norm["y2"]) / 2.0)
                    det = {
                        "target": label,
                        "label": label,
                        "score": score,
                        "bbox_xyxy": norm,
                        "center": {"x": center_x, "y": center_y},
                    }
                    detections.append(det)
                    scores.append(score)
                    if not target_set or label in target_set:
                        best = by_target.get(label)
                        if best is None or float(score) > float(best.get("score", -1.0)):
                            by_target[label] = det

        # Try relaxed underscore/space alias when caller gives target names with different separators.
        if target_set:
            for t in target_set:
                if t in by_target:
                    continue
                alias = t.replace("_", " ")
                for det in detections:
                    if str(det.get("label", "")).replace("_", " ") != alias:
                        continue
                    best = by_target.get(t)
                    if best is None or float(det.get("score", 0.0)) > float(best.get("score", -1.0)):
                        by_target[t] = det

        return InferenceOutput(
            model_name=self.model_name,
            features={"detections": detections, "by_target": by_target},
            confidence=float(sum(scores) / len(scores)) if scores else 0.0,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            raw={
                "weights_path": self._weights_path,
                "device": self._device,
                "conf": self._conf,
                "iou": self._iou,
            },
        )

    def healthcheck(self) -> Dict[str, Any]:
        try:
            import ultralytics  # noqa: F401

            if self._weights_path and not Path(self._weights_path).exists():
                return {"ok": False, "model_name": self.model_name, "error": f"weights not found: {self._weights_path}"}
            return {"ok": True, "model_name": self.model_name, "weights_path": self._weights_path}
        except Exception as exc:  # pragma: no cover
            return {"ok": False, "model_name": self.model_name, "error": str(exc)}

