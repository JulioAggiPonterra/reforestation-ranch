"""Estado del potrero: ¿ya está en punto óptimo de retorno?

Combina dos señales:
1. Días de descanso: se calculan solos a partir de cuándo la cámara dejó de ver
   ganado en el potrero (eventos de conteo).
2. Recuperación del verde: índice de verdor (ExG) de fotos DE DÍA comparado
   contra la referencia del potrero cuando estaba listo.

De noche (infrarrojo) no hay color, así que solo se usa la regla de días.
Los umbrales son punto de partida; se ajustan con el criterio de Augusto.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np

from .panes import is_night_infrared


@dataclass
class PastureRules:
    descanso_minimo_dias: int = 30   # antes de esto no se regresa aunque se vea verde
    descanso_maximo_dias: int = 60   # después de esto el pasto se pasa (lignifica)
    recuperacion_objetivo: float = 0.9  # fracción del verdor de referencia


def green_cover(image_bgr: np.ndarray, roi: tuple[float, float, float, float] | None = None) -> float | None:
    """Fracción (0-1) del área con vegetación verde, usando el índice ExG = 2G - R - B.

    roi: zona a analizar en fracciones (x0, y0, x1, y1); por defecto la mitad
    inferior, para no medir cerros ni cielo. Devuelve None si la foto es nocturna.
    """
    if is_night_infrared(image_bgr):
        return None
    h, w = image_bgr.shape[:2]
    x0, y0, x1, y1 = roi or (0.0, 0.5, 1.0, 1.0)
    crop = image_bgr[int(h * y0):int(h * y1), int(w * x0):int(w * x1)].astype(float)
    total = crop.sum(axis=2) + 1e-6
    b, g, r = (crop[..., i] / total for i in range(3))
    exg = 2 * g - r - b
    return float(np.mean(exg > 0.05))


def pasture_status(
    ultimo_dia_con_ganado: date | None,
    hoy: date,
    verdor_actual: float | None = None,
    verdor_referencia: float | None = None,
    reglas: PastureRules = PastureRules(),
) -> dict:
    """Dictamen del potrero para MosaicOS."""
    if ultimo_dia_con_ganado is None:
        return {"estado": "sin_datos", "motivo": "La cámara aún no registra ocupación en este potrero."}

    dias = (hoy - ultimo_dia_con_ganado).days
    recuperacion = None
    if verdor_actual is not None and verdor_referencia:
        recuperacion = round(verdor_actual / verdor_referencia, 2)

    out = {"dias_descanso": dias, "recuperacion_verdor": recuperacion}
    if dias == 0:
        out.update(estado="ocupado", motivo="Hay ganado en el potrero hoy.")
    elif dias < reglas.descanso_minimo_dias:
        out.update(estado="en_descanso", motivo=f"Faltan {reglas.descanso_minimo_dias - dias} días de descanso mínimo.")
    elif dias > reglas.descanso_maximo_dias:
        out.update(estado="pasado", motivo="Superó el descanso máximo: el pasto pierde calidad, conviene pastorear ya.")
    elif recuperacion is None:
        out.update(estado="probablemente_listo", motivo="Cumple días de descanso; falta foto de día para confirmar el verdor.")
    elif recuperacion >= reglas.recuperacion_objetivo:
        out.update(estado="listo", motivo="Cumple descanso y el verdor se recuperó.")
    else:
        out.update(estado="en_descanso", motivo=f"Verdor al {int(recuperacion * 100)}% de la referencia.")
    return out
