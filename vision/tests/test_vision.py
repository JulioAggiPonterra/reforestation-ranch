"""Pruebas que no necesitan el modelo YOLO (rápidas)."""

import sys
from datetime import date
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mosaic_vision.events import last_day_with_cattle
from mosaic_vision.livestock import Detection, merge_detections
from mosaic_vision.panes import is_night_infrared, split_panes
from mosaic_vision.pasture import green_cover, pasture_status

SAMPLES = Path(__file__).resolve().parent.parent / "samples"


def test_split_panes_finds_separators():
    img = cv2.imread(str(SAMPLES / "noche_2026-09-30_2133.jpg"))
    panes = {p.name: p for p in split_panes(img)}
    assert set(panes) == {"arriba_izq", "arriba_der", "abajo"}
    assert 640 < panes["abajo"].y0 < 660
    assert 570 < panes["arriba_der"].x0 < 585


def test_night_photo_detected_as_infrared():
    assert is_night_infrared(cv2.imread(str(SAMPLES / "noche_2026-09-30_2130.jpg")))
    day = np.zeros((10, 10, 3), np.uint8)
    day[..., 1] = 200  # verde
    assert not is_night_infrared(day)


def test_merge_detections_removes_duplicates():
    dets = [Detection("abajo", (0, 0, 100, 100), 0.9), Detection("abajo", (5, 5, 100, 100), 0.5),
            Detection("abajo", (200, 200, 250, 250), 0.4)]
    assert len(merge_detections(dets)) == 2


def test_green_cover():
    img = np.zeros((100, 100, 3), np.uint8)
    img[50:, :50] = (30, 180, 40)  # mitad del pasto verde
    img[50:, 50:] = (60, 90, 120)  # mitad tierra
    assert abs(green_cover(img) - 0.5) < 0.05
    assert green_cover(cv2.imread(str(SAMPLES / "noche_2026-09-30_2130.jpg"))) is None


def test_pasture_status_rules():
    hoy = date(2026, 11, 5)
    assert pasture_status(date(2026, 11, 5), hoy)["estado"] == "ocupado"
    assert pasture_status(date(2026, 10, 20), hoy)["estado"] == "en_descanso"
    assert pasture_status(date(2026, 9, 30), hoy, 0.6, 0.65)["estado"] == "listo"
    assert pasture_status(date(2026, 9, 30), hoy, 0.3, 0.65)["estado"] == "en_descanso"
    assert pasture_status(date(2026, 8, 1), hoy)["estado"] == "pasado"


def test_last_day_with_cattle():
    evts = [
        {"tipo": "conteo_ganado", "potrero": "norte", "conteo": 5, "fecha_hora": "2026-09-30T21:33:57"},
        {"tipo": "conteo_ganado", "potrero": "norte", "conteo": 0, "fecha_hora": "2026-10-02T10:00:00"},
        {"tipo": "conteo_ganado", "potrero": "sur", "conteo": 3, "fecha_hora": "2026-10-03T10:00:00"},
    ]
    assert last_day_with_cattle(evts, "norte") == date(2026, 9, 30)
