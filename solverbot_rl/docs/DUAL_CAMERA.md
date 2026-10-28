# Percepción cooperativa con dos cámaras — experimento `exp/solverbot-dual-camera-perception`

Rama experimental desde `a38c2e2` (`feat/solverbot-reach-bc-sac`). Solo simulación (Gazebo Harmonic + DART, WSL,
2026-10-28). Sin cambios en física, URDF, pinza dual, recompensas, criterios de éxito, expertos, observación (31) ni
acciones (7). Tablas completas y figuras: `docs/resultados/dual_camera/` (generadas por `scripts/report_dual_camera.py`).

## Conclusión

**La cámara de cabeza no puede ver el cubo en ninguna posición de la mesa con el montaje actual: la fusión de dos
cámaras NO es físicamente viable y no se implementó** (fases C, D y E detenidas según el criterio del encargo). No se
movió la cámara ni se cambió la geometría del robot.

Separando visión y control (fase F) con la misma política congelada y las mismas semillas:

1. **El fallo dominante de bc_v2w es de control (cambio de rama de joint4), no de visión.** Persiste con la posición
   EXACTA del cubo en cada paso (oráculo, diagnóstico privilegiado) y en el backend cinemático, donde la detección es
   continua y no hay física. En 102 de 107 cambios de rama de las evaluaciones previas, la política empieza a dejar su
   rama mientras la detección de la muñeca todavía es válida; la pérdida de detección viene **después**.
2. **La información visual agrava parte de los fallos.** Con una estimación mejor (oráculo, o la primera detección
   congelada SIN verdad del simulador) se recuperan algunos episodios (`reach`: 16/20 → 20/20) pero no todos
   (`var_m`: 18/20 → 17–18/20; `var_s`: 22/30 → 25/30). En la prueba pareada más grande, 60 semillas nuevas de
   `var_s` (905000+), la primera detección congelada da 52/60 frente a 49/60 (7 recuperadas, 4 perdidas, p = 0,55):
   **ninguna mejora demostrada**. Para bc_v1 en extrapolación (`var_l`), la posición exacta **empeora** el resultado
   (20/30 → 12/30; 1 recuperada, 9 perdidas, p = 0,021): la política se entrenó con estimaciones ruidosas e
   intermitentes y una entrada "mejor" la saca de su distribución. Mejorar la percepción de una política congelada
   no es una mejora gratuita.
3. **La semilla `902003` no es un fallo reproducible.** Con la misma pose inicial (d0 = 156,7 mm) tuvo éxito en 7 de 7
   ejecuciones de hoy con percepción (5 lanzamientos de repetición, contrafácticos y auditoría) y en 5/5 con los
   demás modos; falló solo en la verificación previa. Los resultados por semilla con
   percepción cambian entre lanzamientos de la simulación porque la fase entre el render de 10 Hz y el paso de control
   (antigüedad de la imagen 0,009–0,105 s) es distinta en cada lanzamiento; con el oráculo (sin imágenes en la
   observación) dos lanzamientos dan trayectorias idénticas. **Ninguna conclusión puede apoyarse en una semilla.**
4. El guardián de compromiso de rama en tiempo de ejecución (sin reentrenar) **no mejora** (cinemático, 200 semillas
   nuevas por nivel) y se deja desactivado como experimento negativo.

## Fase A — cámara de cabeza (`scripts/audit_cameras.py static`)

Tópicos: `/camera/camera/color/image_raw`, `/camera/camera/aligned_depth_to_color/image_raw`,
`/camera/camera/color/camera_info` (frame `camera_color_optical_frame`); muñeca `/wrist_camera/image_raw` +
`camera_info` (`wrist_sim_camera_optical_frame`). Perfil `train`.

| | cabeza (D435 sim.) | muñeca (virtual) |
|---|---|---|
| resolución / FOV (de K) | 160×120, HFOV 69,0°, VFOV 54,5°, sin distorsión | 96×96, 80,2°×80,2° |
| frecuencia medida (tiempo sim.) | 10,0 Hz; profundidad 10,0 Hz, mismo sello que RGB (Δ = 0) | 10,0 Hz |
| antigüedad respecto del reloj tras un paso | 0,009–0,105 s según el lanzamiento (fase render/paso) | ídem |

**Pose de la tarea (pan 1,20, tilt 0,78):** la cámara queda en (−0,014; 0,042; 1,345) m, 2,4 cm sobre la placa
superior de la columna (z 1,321). La placa ocupa el **93 %** de la imagen y está más cerca que el plano near (0,105 m):
profundidad −inf. El cubo real se proyecta en el centro de la imagen (84, 62 px) a 0,81 m, pero allí la profundidad es
−inf → **tapado**. Confirma con imágenes la advertencia de `HALLAZGOS.md` §1.

**Barrido pan/tilt** (36 poses, límites ±1,57 / ±0,785 rad): en ninguna se detecta el cubo. Con tilt ≥ 0,4 la placa
(17–100 % de −inf) o la caja superior tapan la proyección; con tilt ≤ 0,2 el cubo queda fuera del campo y el brazo
Piper (montado del mismo lado) ocupa la imagen. **Mapa de la mesa** (cabeza apuntada a cada una de 40 posiciones,
x −0,21…0,49, y 0,41…0,81): **0/40** visibles, todas `occluded`.

**Control positivo** (la cadena RGB-D + TF + CameraInfo funciona cuando hay línea de vista): cubo en el aire a
0,5–0,8 m delante de la cámara → 4/4 detectados, centroide de la máscara a 1,0–1,2 px de la proyección de la pose real
(calibración y TF coherentes), error 3D 16–21 mm (sesgo esperado del centroide de la cara visible de un cubo de 4 cm).

**Geometría (estimación, no se modificó nada):** para que la recta cámara→cara superior del cubo pase por encima del
borde de la placa, el centro óptico tendría que estar a ≈ 1,40 m (posición nominal) – 1,42 m (esquina cercana de
`var_l`), es decir **≈ 6–7 cm más alto** que hoy; la caja superior exige ≈ 1,36–1,39 m. No incluye la oclusión por el
brazo. **[robot real]** verificar si la D435 real ve la placa al mirar abajo-izquierda.

## Fase B — cobertura por cámara (`scripts/audit_cameras.py episodes`)

Mismas políticas congeladas, la sonda solo se suscribe a tópicos (prueba: `test_on_step_hook_does_not_change_rollout`).

| corrida | episodios | pasos | muñeca válida | cabeza detecta | solo muñeca | solo cabeza | ambas | ninguna |
|---|---|---|---|---|---|---|---|---|
| bc_v2w `reach_var_m` (902002–003 + 903000–009 nuevas) | 12 | 516 | 54,3 % | **0,0 %** | 54,3 % | 0,0 % | 0,0 % | 45,7 % |
| bc_v1 `reach` (902000–002) | 3 | 183 | 52,5 % | **0,0 %** | 52,5 % | 0,0 % | 0,0 % | 47,5 % |

Estado de la cabeza en los 699 pasos: `occluded` 699 (placa: −inf en el 93 % de la imagen en todos los pasos; el
cabezal no se mueve durante la tarea). **Las dos cámaras no aportan información complementaria: en ningún paso la
cabeza cubre una pérdida de la muñeca.** Por semilla, cobertura de la muñeca entre 24 % y 100 %.

Error de la muñeca cuando detecta (bc_v2w): 12,4 mm de media (máx. 41,6). Recalculando la proyección con la FK en el
instante de la IMAGEN (historial de `/joint_states`): 8,7 mm (máx. 32,1); bc_v1: 10,0 → 7,5 mm. **La latencia
imagen–estado articular (0–0,13 s) explica ≈ 30 % del error de la muñeca**; es corregible sin hardware, pero cambia la
distribución que vio el BC (no se activó). El error que ve la política es mayor (17–22 mm de media con percepción) porque durante las pérdidas la
observación conserva la ÚLTIMA detección válida, típicamente tomada con la muñeca en movimiento y el cubo en el borde
de la imagen; congelar la PRIMERA detección (desde home, cubo centrado) da 5–7 mm.

Imágenes de ambas cámaras en la misma trayectoria: `frames_902002.png`, `frames_902003.png`; auditoría estática:
`head_audit_sheet.png`, `head_table_visibility.png`.

## Fase F — separar visión y control

### Contrafácticos de la observación del objeto (`scripts/diag_obs_counterfactual.py`)

Solo cambia, del lado de la política, `obj_rel`/`obj_valid`; el entorno y la recompensa no cambian.
`perception` = tal cual; `hold_valid` = bandera 1 si hubo alguna detección; `first_estimate` = primera detección
congelada (sin verdad); `oracle` = pose real (PRIVILEGIADO, solo diagnóstico, nunca un modo de percepción).

| política / nivel / semillas | perception | hold_valid | first_estimate | oracle (privilegiado) |
|---|---|---|---|---|
| bc_v2w `reach` 901000–019 | 16/20 | — | 20/20 | 20/20 |
| bc_v2w `reach_var_m` 902000–019 | 18/20 (902009, 902017) | 20/20 | 18/20 (902010, 902017) | 17/20 (902008, 902010, 902017) |
| bc_v2w `reach_var_s` 901100–129 | 22/30 | — | — | 25/30 |
| bc_v2w `reach_var_s` 905000–059 (nuevas) | 49/60 | — | 52/60 | — |
| bc_v1 `reach_var_l` 900300–329 | 20/30 | — | — | **12/30** |

Pareado contra `perception` (b = recupera, c = pierde, McNemar exacto): `reach` first_estimate y oracle 4/0
(p = 0,125); `var_m` hold_valid 2/0 (0,50), first_estimate 1/1 (1,0), oracle 1/2 (1,0); `var_s` oracle 4/1 (0,375);
`var_s` 905k first_estimate 7/4 (0,55); bc_v1 `var_l` oracle 1/9 (**0,021**, empeora). Error medio de la estimación
vista por la política: perception 16–22 mm, first_estimate ≈ 6 mm, oracle 0.

Semilla 902017 (`trace_bc_v2w_var_m_902017.png`): en los cuatro modos arranca por la rama + (acción de joint4 → +1).
Con `perception`, `first_estimate` y **`oracle`** revierte joint4 a −1 entre los pasos 19 y 25 (el oráculo, primero),
la pinza se inclina y la distancia sube a 120–340 mm; solo con `hold_valid` sigue por la rama + y tiene éxito. La
política está sobre la frontera entre ramas y diferencias mínimas de la entrada deciden el lado: es la
multimodalidad del BC (ya identificada en `REACH_BC_SAC.md`), no falta de información visual.

### Rama de muñeca frente a pérdida de detección (`scripts/analyze_branch.py`, 330 episodios previos)

| | episodios | éxitos | con cambio de rama | cambio iniciado con detección válida |
|---|---|---|---|---|
| bc_v2w + bc_v1, todas las evaluaciones previas | 330 | 281 | 107 | **102/107** |

En `902003` (verificación previa) la acción de joint4 deja la saturación (−0,99) en el paso 12 con detección válida,
cruza 0 en el 16 y llega a +0,85 en el 18; vuelve a −1 en los pasos 33–36: es una oscilación progresiva entre ramas,
no un salto instantáneo. El TCP pierde inclinación (z −0,94 → −0,12) y recién entonces el cubo sale del campo.

### Backend cinemático (detector continuo con ruido de 1 cm y 5 % de pérdidas, sin campo visual ni física)

bc_v2w: `var_m` 194/200, `var_s` 199/200, `reach` 20/20; bc_v1 `var_m` 185/200. Los fallos 902021 y 902112 de bc_v2w
repiten la firma (cambio de rama de joint4, TCP casi horizontal) **con detección continua**. La brecha con Gazebo
(`var_s` 49/60 – 25/30) no se cierra dándole a la política la posición exacta: depende de la dinámica física y de la
fase de las imágenes, no de la cantidad de información visual.

### Mecanismo de selección de rama: guardián (`imitation/branch_guard.py`, `scripts/eval_branch_guard.py`)

Sin reentrenar ni información privilegiada: cuando joint4 se alejó 0,15 rad de su valor inicial, limita a 0,2 la
acción de joint4 que empuja hacia la otra rama. Cinemático, 200 semillas nuevas (904000+), base / guardián:
`var_m` 196 / 195, `var_s` 200 / 194, `var_l` 188 / 185, `reach` 20 / 20 → **no mejora; desactivado** (no se llevó a
Gazebo). Restringir solo joint4 no alcanza: la decisión de rama está repartida en joint4–joint6.

## Recomendaciones

1. **No integrar fusión de dos cámaras con este montaje.** Se necesita una decisión de hardware (cámara más alta o
   adelantada respecto de la placa, o mirar la mesa de frente con la base) y luego repetir `audit_cameras.py static`.
2. **Control (la causa dominante):** política multimodal (mezcla de gaussianas / diffusion) o etiquetado con rama
   consistente, como propone `REACH_BC_SAC.md`; ahora con evidencia de que la percepción continua no lo resuelve.
   No hace falta DAgger para demostrar esto, y no se modificó el experto.
3. **Percepción con una sola cámara (sin hardware), solo junto con reentrenamiento:** (a) proyectar con la FK en el
   instante de la imagen (−30 % de error de la muñeca); (b) un estimador temporal del cubo estático que no reemplace
   una buena detección por la última, sesgada, y lo marque como "recordado" con un indicador propio. Con la política
   congelada, `first_estimate` no mostró mejora significativa (52/60 frente a 49/60) y una entrada exacta llegó a
   empeorar bc_v1: cualquier cambio de percepción tiene que entrar en los datos de entrenamiento.
4. **Evaluación:** fijar o aleatorizar explícitamente la fase render/paso y reportar tasas sobre muchas semillas y
   varios lanzamientos; una semilla recuperada no es evidencia.

## Limitaciones

* Solo el cubo rojo de 4 cm, una mesa, perfil de cámaras `train` (160×120 / 96×96) y física DART.
* Las conclusiones de fase F usan 20–30 semillas por grupo en Gazebo: ninguna diferencia pareada alcanzó p < 0,05.
* El oráculo es un diagnóstico privilegiado; `first_estimate` y `hold_valid` cambian la semántica de `obj_valid`
  respecto del entrenamiento (distribución distinta: bandera 1 en el 100 % de los pasos frente a ~50 %).
* El análisis geométrico de altura necesaria usa las cajas envolventes de `geometry_v3.yaml`, no las mallas.
* La reproducibilidad entre lanzamientos se midió en 3 semillas × 5 lanzamientos (percepción) y 2 × 2 (oráculo).

## Reproducción

```bash
source tools/wsl/sim_env.sh 1 >/dev/null; python3 -m pytest -q solverbot_rl/tests          # 54 pruebas
J=solverbot_rl/experiments/dual_camera_jobs
tools/headless/queue.sh 1 $J/audit.txt      # fases A/B
tools/headless/queue.sh 2 $J/cf_a.txt       # y cf_b … cf_f2, repro.txt (fase F)
M=solverbot_rl/docs/resultados/reach_bc_v1/modelos; O=runs/dual_camera
python3 solverbot_rl/scripts/eval_reach.py --backend kinematic --policy $M/bc_v2w.zip --level reach_var_m --seeds 902000-902199 --out $O/kin/bc_v2w_reach_var_m
python3 solverbot_rl/scripts/eval_branch_guard.py --backend kinematic --policy $M/bc_v2w.zip --level reach_var_m --seeds 904000-904199 --out $O/guard/kin_guard_bc_v2w_reach_var_m
python3 solverbot_rl/scripts/analyze_branch.py --runs runs/reach_bc_v1/eval runs/reach_bc_v1/eval_confirm --out $O/branch_analysis
python3 solverbot_rl/scripts/report_dual_camera.py --root $O --out solverbot_rl/docs/resultados/dual_camera --trace-seeds 902003,902017
tools/headless/run_experiment.sh 1 validate --seeds 40000-40004                                # regresión del agarre
```

## Regresión

* `pytest solverbot_rl/tests`: **54/54** (47 previas + 7 nuevas en `test_dual_camera.py`).
* Agarre físico (experto validado, `validate_grasp.py`, semillas 40000–40004, Gazebo): **5/5** levantamientos
  sostenidos, elevación 84,8–99,9 mm, error de mimic máx. 1,98 mm (igual que la referencia).
* Único cambio en código existente: `run_episode(..., on_step=None)` (gancho de diagnóstico; sin él, idéntico).
