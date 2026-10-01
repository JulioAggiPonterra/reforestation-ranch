"""Eventos que se envían a MosaicOS.

Cada análisis produce un evento JSON. Se guarda en un archivo de bitácora
(una línea por evento) y, si está configurado, se manda por HTTP a MosaicOS.

Ejemplo de evento de conteo:
{
  "tipo": "conteo_ganado",
  "camara": "potrero-norte",
  "potrero": "norte",
  "fecha_hora": "2026-09-30T21:33:57",
  "conteo": 5,
  "por_panel": {"arriba_izq": 0, "arriba_der": 0, "abajo": 5},
  "nocturna": true,
  "fuente": "samples/noche_2026-09-30_2133.jpg"
}
"""

from __future__ import annotations

import json
import urllib.request
from datetime import date, datetime
from pathlib import Path


def emit(event: dict, log_path: str | Path, webhook_url: str | None = None, token: str | None = None) -> None:
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")

    if webhook_url:
        headers = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        req = urllib.request.Request(webhook_url, json.dumps(event).encode(), headers, method="POST")
        try:
            urllib.request.urlopen(req, timeout=10)
        except OSError as e:  # sin internet en el rancho: queda en la bitácora
            print(f"Aviso: no se pudo enviar a MosaicOS ({e}); quedó guardado en {log_path}")


def read_events(log_path: str | Path) -> list[dict]:
    path = Path(log_path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def last_day_with_cattle(events: list[dict], potrero: str, min_animals: int = 1) -> date | None:
    """Último día en que la cámara vio ganado en el potrero (para contar días de descanso)."""
    days = [
        datetime.fromisoformat(e["fecha_hora"]).date()
        for e in events
        if e.get("tipo") == "conteo_ganado" and e.get("potrero") == potrero and e.get("conteo", 0) >= min_animals
    ]
    return max(days) if days else None
