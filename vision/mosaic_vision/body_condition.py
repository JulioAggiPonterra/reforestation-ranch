"""Condición corporal (CC) del ganado.

La condición corporal NO se puede estimar con el modelo genérico: hay que
enseñarle con fotos del propio rancho calificadas por alguien que sabe (Augusto).
Este módulo cubre el ciclo completo:

1. export_crops(): recorta cada vaca detectada y la guarda para calificarla.
2. Augusto acomoda cada recorte en una carpeta por calificación (escala 1 a 9):
       dataset_cc/train/cc_3/...jpg   dataset_cc/train/cc_5/...jpg   ...
       dataset_cc/val/cc_3/...jpg     ...
3. train(): entrena un clasificador con esas carpetas.
4. BodyConditionScorer: califica cada vaca en las fotos nuevas.

Recomendación: las fotos de día y de costado dan mucho mejor resultado que el
infrarrojo nocturno. Se necesitan ~50 fotos por calificación para empezar.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .livestock import FrameCount
from .panes import split_panes


def export_crops(image: np.ndarray, count: FrameCount, out_dir: str | Path, prefix: str, layout: str = "2+1",
                 min_size: int = 60) -> list[Path]:
    """Guarda un recorte por vaca detectada. Ignora animales muy pequeños/lejanos."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    panes = {p.name: p for p in split_panes(image, layout)}
    saved = []
    for i, det in enumerate(count.detections):
        x1, y1, x2, y2 = det.box
        if min(x2 - x1, y2 - y1) < min_size:
            continue
        crop = panes[det.pane].image[max(0, y1):y2, max(0, x1):x2]
        path = out_dir / f"{prefix}_{det.pane}_{i}.jpg"
        cv2.imwrite(str(path), crop)
        saved.append(path)
    return saved


def train(dataset_dir: str, epochs: int = 50, base_model: str = "yolo11s-cls.pt") -> str:
    """Entrena el clasificador de condición corporal. Devuelve la ruta del modelo."""
    from ultralytics import YOLO

    result = YOLO(base_model).train(data=dataset_dir, epochs=epochs, imgsz=224, project="modelos", name="condicion_corporal")
    return str(Path(result.save_dir) / "weights" / "best.pt")


class BodyConditionScorer:
    def __init__(self, model_path: str):
        from ultralytics import YOLO

        self.model = YOLO(model_path)

    def score(self, crop: np.ndarray) -> dict:
        probs = self.model.predict(crop, imgsz=224, verbose=False)[0].probs
        label = self.model.names[int(probs.top1)]  # p. ej. "cc_5"
        return {"condicion_corporal": int(label.split("_")[-1]), "confianza": round(float(probs.top1conf), 3)}
