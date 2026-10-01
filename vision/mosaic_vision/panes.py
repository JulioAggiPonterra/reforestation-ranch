"""Separa la imagen compuesta de la cámara (mosaico de 3 lentes) en paneles.

Las cámaras del rancho entregan una sola imagen con este acomodo:

    +-----------+-----------+
    |  arriba_  |  arriba_  |
    |  izq      |  der      |
    +-----------+-----------+
    |        abajo          |
    +-----------------------+

Las líneas negras que dividen los paneles se detectan automáticamente;
si no se encuentran se usan las proporciones por defecto (mitad y mitad).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SEPARATOR_MAX_BRIGHTNESS = 8.0  # una fila/columna más oscura que esto es línea divisoria


@dataclass
class Pane:
    name: str
    image: np.ndarray
    x0: int  # posición dentro de la imagen original
    y0: int


def _find_separator(profile: np.ndarray, lo: int, hi: int) -> tuple[int, int] | None:
    """Busca la franja oscura (línea divisoria) dentro de profile[lo:hi]."""
    dark = [i for i in range(lo, hi) if profile[i] < SEPARATOR_MAX_BRIGHTNESS]
    if not dark:
        return None
    # Tomar el bloque contiguo más cercano al centro de la búsqueda
    mid = (lo + hi) / 2
    blocks, start = [], dark[0]
    for a, b in zip(dark, dark[1:] + [None]):
        if b != a + 1:
            blocks.append((start, a))
            start = b
    return min(blocks, key=lambda bl: abs((bl[0] + bl[1]) / 2 - mid))


def split_panes(image: np.ndarray, layout: str = "2+1") -> list[Pane]:
    """Divide la imagen en paneles según el acomodo de la cámara.

    layout: "2+1" (dos arriba, uno abajo, el de las fotos del rancho) o "1" (sin dividir).
    """
    if layout == "1":
        return [Pane("unico", image, 0, 0)]
    if layout != "2+1":
        raise ValueError(f"Acomodo de paneles no soportado: {layout}")

    gray = image.mean(axis=2) if image.ndim == 3 else image.astype(float)
    h, w = gray.shape

    row_sep = _find_separator(gray.mean(axis=1), int(h * 0.4), int(h * 0.6))
    top_end, bottom_start = row_sep if row_sep else (h // 2, h // 2)
    bottom_start += 1

    col_sep = _find_separator(gray[:top_end].mean(axis=0), int(w * 0.4), int(w * 0.6))
    left_end, right_start = col_sep if col_sep else (w // 2, w // 2)
    right_start += 1

    return [
        Pane("arriba_izq", image[:top_end, :left_end], 0, 0),
        Pane("arriba_der", image[:top_end, right_start:], right_start, 0),
        Pane("abajo", image[bottom_start:, :], 0, bottom_start),
    ]


def is_night_infrared(image: np.ndarray, tolerance: float = 4.0) -> bool:
    """True si la imagen es infrarroja nocturna (blanco y negro).

    Importa porque el análisis de pasto necesita color: de noche solo se cuentan animales.
    """
    if image.ndim < 3:
        return True
    px = image.reshape(-1, 3).astype(float)
    b, g, r = px[:, 0], px[:, 1], px[:, 2]
    return float(np.mean(np.abs(r - g)) + np.mean(np.abs(g - b))) / 2 < tolerance
