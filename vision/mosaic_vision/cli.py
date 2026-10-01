"""Comandos para usar la visión del rancho desde la terminal.

    python -m mosaic_vision contar samples/noche_2026-09-30_2133.jpg --camara potrero-norte
    python -m mosaic_vision contar video.mp4 --camara potrero-norte
    python -m mosaic_vision vigilar --cada 600
    python -m mosaic_vision potrero norte
    python -m mosaic_vision referencia foto_de_dia.jpg --camara potrero-norte
    python -m mosaic_vision recortes foto.jpg --salida dataset_cc/sin_calificar
"""

from __future__ import annotations

import argparse
import time
from datetime import date, datetime
from pathlib import Path

import cv2
import yaml

from . import events
from .livestock import DEFAULT_OVERLAP_GROUPS, FrameCount, LivestockCounter
from .panes import is_night_infrared, split_panes
from .pasture import PastureRules, green_cover, pasture_status

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".h264", ".264"}


def load_config(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        p = Path(__file__).resolve().parent.parent / "config.example.yaml"
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def camera_config(cfg: dict, name: str | None) -> dict:
    for cam in cfg.get("camaras", []):
        if cam["nombre"] == name:
            return cam
    return {"nombre": name or "sin_nombre", "potrero": None, "acomodo": "2+1", "grupos_solapados": DEFAULT_OVERLAP_GROUPS}


def make_counter(cfg: dict, cam: dict) -> LivestockCounter:
    return LivestockCounter(
        model_path=cfg.get("modelo", "yolo11s.pt"),
        confidence=cfg.get("confianza", 0.25),
        layout=cam.get("acomodo", "2+1"),
        overlap_groups=cam.get("grupos_solapados"),
    )


def read_frames(source: str):
    cap = cv2.VideoCapture(source)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
    finally:
        cap.release()


def grab_frame(source: str):
    cap = cv2.VideoCapture(source)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"No se pudo leer imagen de {source}")
    return frame


def draw(image, count: FrameCount, layout: str):
    out = image.copy()
    offsets = {p.name: (p.x0, p.y0) for p in split_panes(image, layout)}
    for d in count.detections:
        ox, oy = offsets[d.pane]
        x1, y1, x2, y2 = d.box
        cv2.rectangle(out, (x1 + ox, y1 + oy), (x2 + ox, y2 + oy), (0, 255, 0), 2)
        cv2.putText(out, f"{d.confidence:.2f}", (x1 + ox, y1 + oy - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(out, f"Ganado: {count.total}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    return out


def analyze_image(cfg: dict, cam: dict, frame, source: str, when: datetime, annotated: str | None = None) -> list[dict]:
    counter = make_counter(cfg, cam)
    count = counter.count_frame(frame)
    night = is_night_infrared(frame)
    out = [{
        "tipo": "conteo_ganado",
        "camara": cam["nombre"],
        "potrero": cam.get("potrero"),
        "fecha_hora": when.isoformat(timespec="seconds"),
        "conteo": count.total,
        "por_panel": count.per_pane,
        "nocturna": night,
        "fuente": source,
    }]
    if not night:
        pane = split_panes(frame, cam.get("acomodo", "2+1"))[-1].image
        out.append({
            "tipo": "verdor_potrero",
            "camara": cam["nombre"],
            "potrero": cam.get("potrero"),
            "fecha_hora": when.isoformat(timespec="seconds"),
            "verdor": round(green_cover(pane, cam.get("zona_pasto")), 3),
            "fuente": source,
        })
    if annotated:
        cv2.imwrite(annotated, draw(frame, count, cam.get("acomodo", "2+1")))
    return out


def cmd_contar(args, cfg):
    cam = camera_config(cfg, args.camara)
    when = datetime.fromisoformat(args.fecha) if args.fecha else datetime.now()
    if Path(args.fuente).suffix.lower() in VIDEO_EXT:
        result = make_counter(cfg, cam).count_video(read_frames(args.fuente), every_n=args.cada_n)
        evts = [{
            "tipo": "conteo_ganado",
            "camara": cam["nombre"],
            "potrero": cam.get("potrero"),
            "fecha_hora": when.isoformat(timespec="seconds"),
            "conteo": result["maximo_simultaneo"],
            "por_panel": result["maximo_por_panel"],
            "individuos_rastreados": result["individuos_rastreados"],
            "cuadros_analizados": result["cuadros_analizados"],
            "fuente": args.fuente,
        }]
    else:
        frame = cv2.imread(args.fuente) if Path(args.fuente).exists() else grab_frame(args.fuente)
        evts = analyze_image(cfg, cam, frame, args.fuente, when, args.anotada)
    for e in evts:
        print(e)
        if not args.no_guardar:
            events.emit(e, cfg["bitacora"], cfg.get("mosaicos", {}).get("webhook_url"), cfg.get("mosaicos", {}).get("token"))


def cmd_vigilar(args, cfg):
    print(f"Vigilando {len(cfg['camaras'])} cámara(s) cada {args.cada} segundos. Ctrl+C para detener.")
    while True:
        for cam in cfg["camaras"]:
            try:
                frame = grab_frame(cam["url"])
                for e in analyze_image(cfg, cam, frame, cam["url"].split("@")[-1], datetime.now()):
                    print(e)
                    events.emit(e, cfg["bitacora"], cfg.get("mosaicos", {}).get("webhook_url"), cfg.get("mosaicos", {}).get("token"))
            except Exception as e:  # una cámara caída no detiene a las demás
                print(f"[{cam['nombre']}] error: {e}")
        time.sleep(args.cada)


def cmd_potrero(args, cfg):
    evts = events.read_events(cfg["bitacora"])
    cam = next((c for c in cfg.get("camaras", []) if c.get("potrero") == args.nombre), {})
    verdor = [e for e in evts if e.get("tipo") == "verdor_potrero" and e.get("potrero") == args.nombre]
    status = pasture_status(
        events.last_day_with_cattle(evts, args.nombre),
        date.fromisoformat(args.hoy) if args.hoy else date.today(),
        verdor[-1]["verdor"] if verdor else None,
        cam.get("verdor_referencia"),
        PastureRules(**cfg.get("reglas_potrero", {})),
    )
    print({"potrero": args.nombre, **status})


def cmd_referencia(args, cfg):
    cam = camera_config(cfg, args.camara)
    frame = cv2.imread(args.imagen)
    pane = split_panes(frame, cam.get("acomodo", "2+1"))[-1].image
    value = green_cover(pane, cam.get("zona_pasto"))
    if value is None:
        print("La foto es nocturna (infrarrojo): usa una foto de día con buena luz.")
    else:
        print(f"Verdor de referencia: {value:.3f}  -> ponlo en config.yaml como verdor_referencia")


def cmd_recortes(args, cfg):
    from .body_condition import export_crops

    cam = camera_config(cfg, args.camara)
    frame = cv2.imread(args.imagen)
    count = make_counter(cfg, cam).count_frame(frame)
    saved = export_crops(frame, count, args.salida, Path(args.imagen).stem, cam.get("acomodo", "2+1"))
    print(f"{len(saved)} recorte(s) guardados en {args.salida}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="mosaic_vision", description="Visión por cámaras para MosaicOS")
    ap.add_argument("--config", default="config.yaml")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("contar", help="Contar ganado en una foto, video o cámara")
    p.add_argument("fuente", help="foto, video o URL rtsp://")
    p.add_argument("--camara")
    p.add_argument("--fecha", help="fecha y hora de la toma, p. ej. 2026-09-30T21:33:57")
    p.add_argument("--anotada", help="guardar copia de la foto con las vacas marcadas")
    p.add_argument("--cada-n", type=int, default=5, help="en video, analizar 1 de cada N cuadros")
    p.add_argument("--no-guardar", action="store_true", help="no escribir en la bitácora")
    p.set_defaults(func=cmd_contar)

    p = sub.add_parser("vigilar", help="Tomar foto de todas las cámaras cada cierto tiempo")
    p.add_argument("--cada", type=int, default=600, help="segundos entre rondas")
    p.set_defaults(func=cmd_vigilar)

    p = sub.add_parser("potrero", help="¿El potrero está listo para regresar el ganado?")
    p.add_argument("nombre")
    p.add_argument("--hoy", help="fecha a evaluar (por defecto hoy)")
    p.set_defaults(func=cmd_potrero)

    p = sub.add_parser("referencia", help="Medir verdor de referencia con una foto de día")
    p.add_argument("imagen")
    p.add_argument("--camara")
    p.set_defaults(func=cmd_referencia)

    p = sub.add_parser("recortes", help="Recortar vacas para calificar condición corporal")
    p.add_argument("imagen")
    p.add_argument("--camara")
    p.add_argument("--salida", default="dataset_cc/sin_calificar")
    p.set_defaults(func=cmd_recortes)

    args = ap.parse_args(argv)
    args.func(args, load_config(args.config))


if __name__ == "__main__":
    main()
