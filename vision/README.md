# Visión por cámaras para MosaicOS

Usa las cámaras del rancho para que MosaicOS sepa, sin que nadie vaya a revisar:

| Pregunta | Estado | Cómo |
|---|---|---|
| ¿Cuántas vacas hay en el potrero? | **Funciona hoy** | Detección con IA (YOLO) en foto, video o cámara en vivo |
| ¿Ya está listo el potrero para regresar? | **Funciona hoy (reglas)** | Días de descanso (los calcula la cámara) + verdor en fotos de día |
| ¿Qué condición corporal tiene cada vaca? | **Listo para entrenar** | Hay que enseñarle con fotos calificadas por Augusto |

## Prueba con las fotos del 30 de septiembre

| Foto | Lo que hay | Lo que contó |
|---|---|---|
| 21:30:42 | 1 vaca arriba (se ve en los 2 paneles) + 2 abajo | **3** ✅ |
| 21:33:57 | 4 vacas + 1 becerro abajo | **5** ✅ (incluido el becerro lejano) |

Las fotos están en `samples/`.

### Lo que aprendimos de las fotos

- **La imagen trae 3 paneles** (2 arriba, 1 abajo). El sistema los separa solo.
- **Los dos paneles de arriba ven la misma zona**: la misma vaca sale dos veces. Por eso
  se toma el máximo de esos dos, no la suma (`grupos_solapados` en la configuración).
- **De noche (infrarrojo) la IA a veces cree que una vaca es borrego.** Se cuentan
  vaca, borrego y caballo como "ganado". Cuando haya un modelo entrenado con fotos del rancho esto desaparece.
- **Se analiza a dos tamaños**: uno ve bien las vacas cercanas y otro las lejanas y becerros.
- **De noche no se puede medir el pasto** (no hay color). Para eso se usan fotos de día.

## Cómo se usa

Instalación (una vez, en la computadora del rancho o un servidor):

```bash
cd vision
pip install -r requirements.txt
cp config.example.yaml config.yaml   # y llenar con los datos de las cámaras
```

Comandos:

```bash
# Contar vacas en una foto (y guardar copia con las vacas marcadas)
python -m mosaic_vision contar samples/noche_2026-09-30_2133.jpg --camara potrero-norte --anotada marcada.jpg

# Contar vacas en un video
python -m mosaic_vision contar video.mp4 --camara potrero-norte

# Vigilar todas las cámaras: una foto cada 10 minutos, cada resultado va a MosaicOS
python -m mosaic_vision vigilar --cada 600

# ¿Está listo el potrero norte?
python -m mosaic_vision potrero norte

# Guardar el verdor de un potrero cuando está en su punto (foto de día)
python -m mosaic_vision referencia foto_dia.jpg --camara potrero-norte
```

## Cómo llega a MosaicOS

Cada análisis genera un **evento** (un registro en formato JSON) que:
1. Se guarda siempre en `datos/eventos.jsonl` (funciona aunque no haya internet).
2. Se envía a MosaicOS si se pone su dirección en `mosaicos.webhook_url`.

```json
{"tipo": "conteo_ganado", "camara": "potrero-norte", "potrero": "norte",
 "fecha_hora": "2026-09-30T21:33:57", "conteo": 5,
 "por_panel": {"arriba_izq": 0, "arriba_der": 0, "abajo": 5}, "nocturna": true}

{"tipo": "verdor_potrero", "potrero": "norte", "fecha_hora": "...", "verdor": 0.62}
```

Respuesta del comando `potrero`:

```json
{"potrero": "norte", "dias_descanso": 36, "recuperacion_verdor": 0.93,
 "estado": "listo", "motivo": "Cumple descanso y el verdor se recuperó."}
```

Estados posibles: `ocupado`, `en_descanso`, `probablemente_listo`, `listo`, `pasado`, `sin_datos`.

## Conexión con las cámaras

El enlace `res.av380.net` es la nube del fabricante (app AV380/V380). Esa nube **no** permite
que otro programa tome el video. Para que MosaicOS vea las cámaras hay dos caminos:

1. **RTSP en la red local (recomendado).** La mayoría de las cámaras V380 lo tienen. Se necesita
   la IP de cada cámara y su usuario/contraseña. Rutas comunes: `rtsp://IP:554/live/ch00_0`.
   Se prueba abriendo esa dirección en VLC (Medio → Abrir ubicación de red).
2. **Clips guardados.** Si RTSP no está disponible, se pueden subir los videos/fotos que guarda la
   cámara y analizarlos con `contar`.

Una computadora pequeña en el rancho (mini PC o Jetson) puede correr `vigilar` todo el día;
solo manda a MosaicOS los resultados, no el video, así que usa muy poco internet.

## Condición corporal: cómo entrenarla

La IA general no sabe calificar condición corporal; hay que enseñarle:

1. Juntar fotos (mejor **de día y de costado**):
   `python -m mosaic_vision recortes foto.jpg --salida dataset_cc/sin_calificar`
2. Augusto mueve cada recorte a la carpeta de su calificación (escala 1–9):
   `dataset_cc/train/cc_4/`, `dataset_cc/train/cc_5/`, … y una parte a `dataset_cc/val/cc_X/`.
   Meta inicial: ~50 fotos por calificación.
3. Entrenar (en Python): `from mosaic_vision.body_condition import train; train("dataset_cc")`
4. Con el modelo entrenado, `BodyConditionScorer` califica cada vaca nueva.

## Siguientes pasos sugeridos

1. Conseguir la IP y acceso RTSP de cada cámara y llenar `config.yaml`.
2. Definir con Augusto los días de descanso mínimo/máximo por tipo de pasto (hoy: 30–60).
3. Tomar una foto de día de cada potrero "en su punto" para fijar el verdor de referencia.
4. Juntar ~300 fotos del rancho para entrenar un modelo propio (mejora el conteo nocturno
   y permite identificar becerros vs. adultos).
5. Agregar la dirección (webhook) de MosaicOS para que los eventos lleguen solos.

Pruebas: `python -m pytest tests`
