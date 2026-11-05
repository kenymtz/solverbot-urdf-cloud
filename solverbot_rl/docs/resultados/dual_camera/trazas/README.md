# Trazas del experimento de dos cámaras

Copia compacta de `runs/dual_camera/` (los `runs/` no se versionan). Por corrida: `summary.json` (agregados + IC de
Wilson), `index.jsonl` (resumen por episodio). `audit_ep/*/cams_<semilla>.json`: registro POR PASO de ambas cámaras
(estado de visibilidad, píxeles, estimación, error contra la pose real — solo diagnóstico —, antigüedad, profundidad
−inf, ángulos del cabezal, acción, q y eje del TCP). `cf/*/cf_<semilla>.json`: lo que vio la política en cada paso con
cada contrafáctico. `audit_static/audit_static.json`: CameraInfo, frecuencias, barrido pan/tilt, mapa de la mesa y
control positivo. Las trazas completas (`episodes/*.npz`, cuadros de ambas cámaras `frames_*.npz` y PNG de cada pose)
se regeneran con los trabajos de `solverbot_rl/experiments/dual_camera_jobs/`.
