"""Detección y conteo de ganado con YOLO.

Notas de campo (probado con las fotos nocturnas del 30-sep-2026):
- En infrarrojo el modelo general a veces confunde vaca con borrego o caballo,
  por eso cualquiera de esas clases se cuenta como "ganado".
- Los paneles de arriba (izq/der) ven casi la misma zona: una misma vaca aparece
  en ambos. Por eso se configuran "grupos solapados" y de cada grupo se toma
  el máximo, no la suma.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .panes import Pane, split_panes

# Clases COCO que tratamos como ganado (el IR nocturno las confunde entre sí)
LIVESTOCK_CLASSES = {17: "caballo", 18: "borrego", 19: "vaca"}

DEFAULT_OVERLAP_GROUPS = [["arriba_izq", "arriba_der"], ["abajo"]]


@dataclass
class Detection:
    pane: str
    box: tuple[int, int, int, int]  # x1, y1, x2, y2 dentro del panel
    confidence: float
    track_id: int | None = None


@dataclass
class FrameCount:
    per_pane: dict[str, int]
    total: int
    detections: list[Detection] = field(default_factory=list)


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def merge_detections(dets: list[Detection], iou_threshold: float = 0.4) -> list[Detection]:
    """Une detecciones de varias escalas: si dos cajas se enciman, es el mismo animal."""
    kept: list[Detection] = []
    for d in sorted(dets, key=lambda d: d.confidence, reverse=True):
        if all(_iou(d.box, k.box) < iou_threshold for k in kept):
            kept.append(d)
    return kept


class LivestockCounter:
    def __init__(
        self,
        model_path: str = "yolo11s.pt",
        confidence: float = 0.25,
        image_sizes: tuple[int, ...] = (640, 1280),
        layout: str = "2+1",
        overlap_groups: list[list[str]] | None = None,
    ):
        from ultralytics import YOLO  # import tardío: es pesado

        self.model = YOLO(model_path)
        self.confidence = confidence
        # Varias escalas: 640 ve bien las vacas cercanas, 1280 las lejanas y becerros
        self.image_sizes = tuple(image_sizes)
        self.layout = layout
        self.overlap_groups = overlap_groups or DEFAULT_OVERLAP_GROUPS
        # Si el modelo es propio (entrenado con fotos del rancho) usamos todas sus clases
        names = self.model.names
        self.classes = list(LIVESTOCK_CLASSES) if names.get(19) == "cow" else None

    def _predict(self, pane: Pane, track: bool, image_size: int) -> list[Detection]:
        kwargs = dict(
            conf=self.confidence,
            imgsz=image_size,
            classes=self.classes,
            agnostic_nms=True,  # una vaca no puede ser vaca y borrego a la vez
            verbose=False,
        )
        if track:
            result = self.model.track(pane.image, persist=True, tracker="bytetrack.yaml", **kwargs)[0]
        else:
            result = self.model.predict(pane.image, **kwargs)[0]

        ids = result.boxes.id.tolist() if result.boxes.id is not None else [None] * len(result.boxes)
        return [
            Detection(pane.name, tuple(int(v) for v in box), round(float(conf), 3), None if tid is None else int(tid))
            for box, conf, tid in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist(), ids)
        ]

    def combine(self, per_pane: dict[str, int]) -> int:
        """Suma entre grupos independientes y toma el máximo dentro de cada grupo solapado."""
        grouped = {name for group in self.overlap_groups for name in group}
        total = sum(max((per_pane.get(n, 0) for n in group), default=0) for group in self.overlap_groups)
        total += sum(c for n, c in per_pane.items() if n not in grouped)
        return total

    def count_frame(self, image: np.ndarray) -> FrameCount:
        detections: list[Detection] = []
        per_pane: dict[str, int] = {}
        for pane in split_panes(image, self.layout):
            dets = merge_detections(
                [d for size in self.image_sizes for d in self._predict(pane, track=False, image_size=size)]
            )
            per_pane[pane.name] = len(dets)
            detections += dets
        return FrameCount(per_pane, self.combine(per_pane), detections)

    def count_video(self, frames, every_n: int = 5) -> dict:
        """Cuenta ganado en un video.

        Devuelve dos cifras:
        - maximo_simultaneo: el mayor número de animales visto en un mismo cuadro
          (conservador, el más confiable).
        - individuos_rastreados: animales distintos que el rastreador siguió
          (puede inflarse si una vaca sale y vuelve a entrar).
        """
        # Un rastreador por panel: se usa una instancia de modelo por panel
        from ultralytics import YOLO

        trackers: dict[str, object] = {}
        seen_ids: dict[str, set[int]] = {}
        best = FrameCount({}, 0)
        processed = 0

        for i, frame in enumerate(frames):
            if i % every_n:
                continue
            processed += 1
            per_pane: dict[str, int] = {}
            dets_all: list[Detection] = []
            for pane in split_panes(frame, self.layout):
                if pane.name not in trackers:
                    trackers[pane.name] = YOLO(self.model.ckpt_path)
                saved, self.model = self.model, trackers[pane.name]
                try:
                    dets = self._predict(pane, track=True, image_size=self.image_sizes[-1])
                finally:
                    self.model = saved
                per_pane[pane.name] = len(dets)
                dets_all += dets
                seen_ids.setdefault(pane.name, set()).update(d.track_id for d in dets if d.track_id is not None)
            total = self.combine(per_pane)
            if total > best.total:
                best = FrameCount(per_pane, total, dets_all)

        tracked = self.combine({k: len(v) for k, v in seen_ids.items()})
        return {
            "cuadros_analizados": processed,
            "maximo_simultaneo": best.total,
            "maximo_por_panel": best.per_pane,
            "individuos_rastreados": tracked,
        }
