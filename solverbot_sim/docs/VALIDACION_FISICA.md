# Validación física de los desafíos HRI, GPSR y Doing Laundry — iteración 2

Este documento complementa [DESAFIOS_SPEC.md](DESAFIOS_SPEC.md). Registra:

- la auditoría de integración con Pick and Place;
- la seguridad en puertas;
- las correcciones de HRI y GPSR;
- el estudio de alcance del brazo para Doing Laundry;
- la prueba del pipeline de entrenamiento en Gazebo.

La entrega es una **primera versión funcional de los entornos Gymnasium**. No equivale a tres desafíos físicamente
completados. Los porcentajes del experto guionado miden los entornos y las habilidades de referencia, **no** una
política aprendida.

Evidencia: `solverbot_rl/docs/resultados/desafios/`. Aislamiento de todas las corridas:

- `ROS_DOMAIN_ID` 44–47 (nunca 10);
- `GZ_PARTITION` propia por simulación;
- una sola simulación por dominio.

## 1. Auditoría de integración y regresión de Pick and Place

**Diff contra `fix/solverbot-grasp-headless`.** Archivos fuera de `challenges/` que esta rama toca:

| Archivo | Cambio | ¿Lo usa P&P? | Efecto en P&P |
|---|---|---|---|
| `solverbot_sim/config/controllers_rl.yaml` | Solo `diff_drive_controller`: 6 claves de límite de aceleración (`linear.x` ±0.4 m/s², `angular.z` ±1.2 rad/s²) | Sí, el mismo archivo carga los controladores del brazo, la pinza, el elevador y la cabeza | Ninguno (ver abajo) |
| `solverbot_sim/tools/make_worlds.py`, `worlds/home_arena.sdf`, `config/arena_spec.yaml` | Mobiliario de la casa: profundidad de la heladera según el plano 10, se quitó una silla que bloqueaba una puerta | No: P&P usa `pick_place_training.sdf` | `pick_place_training.sdf` se regenera **idéntico** (test) |
| `solverbot_sim/launch/sim.launch.py` | Pose de aparición solo si `world == home_challenges` | Sí | Rama nueva, no se ejecuta con el mundo de P&P |
| `tools/headless/*` | Modos nuevos `ctrain` y `cexpert` | Sí (`validate`, `train`) | Los modos de P&P conservan el mundo y los argumentos (`world:=pick_place_training cameras:=false`) |

**Por qué los límites de aceleración no afectan a P&P.**

- El límite solo actúa cuando cambia la velocidad comandada de la base.
- Ningún archivo de P&P (`envs/pick_place.py`, `backends/gazebo.py`, `backends/kinematic.py`, `expert.py`,
  `gripper.py`, `task.py`) publica `cmd_vel` ni `Twist` (test).

**Prueba A/B en Gazebo** (`regresion_pick_place_ab.json`). Condiciones: `validate_grasp.py`, nivel 4 (pick), semillas
40000–40004, misma simulación (dominio 46).

| Configuración | Éxitos | Métricas |
|---|---|---|
| B: `controllers_rl.yaml` de la base | 5/5 | Iguales a A |
| A: esta rama | 5/5 | Iguales a B |

- Iguales significa: subida, error de mimic, deriva y pasos.
- El volcado de parámetros confirma que los límites solo existen en A.
- **Hallazgo previo, no causado por esta rama:** la base se desplaza unos 30 mm durante un episodio de P&P aunque
  nunca se comanda. Es idéntico en A y B.

**Tests nuevos de regresión** (`tests/test_regression_pickplace.py`, 4 tests):

- los parámetros de los controladores del brazo, la pinza, el elevador y la cabeza son iguales a los de la base;
- el `diff_drive` solo **agrega** las 6 claves;
- P&P no comanda la base;
- `pick_place_training.sdf` es reproducible y no contiene entidades de los desafíos.

**Independencia.** No se tocan `pick_place.py`, `expert.py`, `gripper.py`, la pinza dual ni los controladores del
robot. La PR #3 (BC/SAC) es independiente: no se mezcla ni se mergea.

## 2. Seguridad y validez física

### 2.1 Qué es verdad física y qué es aproximación

| Aspecto | Backend cinemático | Gazebo |
|---|---|---|
| Movimiento de la base | Unicycle 2D con los mismos límites de velocidad y aceleración | `diff_drive_controller` con DART: ruedas, fricción, inercia |
| Contacto con el mundo | Holgura **exacta** (eje separador) de la huella contra las primitivas del SDF, con la pose real. El backend no deja entrar en contacto | La misma verificación exacta sobre la pose **real** de Gazebo, más las colisiones físicas reales del motor |
| Personas | Guionadas, cuerpos de colisión estáticos que se reubican (teletransporte) | Igual. No hay modelo de marcha, se mueven por pasos |
| Percepción | Detector simulado: campo visual, línea de vista, ruido y pérdidas | Segmentación por color en la imagen RGB-D real |
| Localización | Scan-matching LiDAR + odometría, igual en ambos | Igual |
| Éxito | Calculado con la **verdad** del simulador (posición real, mirada real, contacto), nunca con la creencia del robot | Igual, mismo código |

- **Huella de colisión.** Es un prisma conservador de 0.545 × 0.669 m que contiene toda la geometría de colisión del
  robot con el brazo plegado, verificado contra las mallas del URDF.
- **Sin falsos negativos.** Holgura < 0 significa contacto.
- **Posibles falsos positivos.** Pueden aparecer, a lo sumo, por la diferencia entre el prisma y la forma real.
- **Las colisiones nunca se desactivan.** La holgura mínima de cada episodio se registra en `info['min_clearance_m']`
  y el obstáculo más cercano en `info['contact_with']`.

### 2.2 Puertas: tolerancias y condiciones de viabilidad

**Geometría (no se modificó).**

- La puerta mide 0.70 m (regla) y el robot 0.669 m con el brazo plegado: quedan **15.5 mm por lado**.
- Presupuesto de error:
  - el error lateral consume holgura 1:1;
  - cada grado de rumbo consume unos 3 mm (`0.545 m · sin 1°` repartido entre las esquinas).

**Maniobra.**

- Pausa de 2 s antes del vano para que converja la localización.
- Alineación al eje con tolerancia de 0.5°.
- Cruce recto con una corrección de rumbo de hasta 1.5°.
- Dentro del vano no se replanifica ni se gira.
- La hoja de la puerta de entrada abierta forma parte del mapa de planificación.

**Validación** (`scripts/validate_doors.py`). Ruta: afuera → sala → cocina → dormitorio → cocina → sala → lavadero →
sala, ida y vuelta por las 4 puertas transitables. Se usa la navegación de producción con la pose **estimada**.

| Backend | Tramos | Cruces | Contactos | Holgura mínima | Error lateral máx. | Error de rumbo máx. | Error de localización medio / máx. |
|---|---|---|---|---|---|---|---|
| Gazebo (`puertas_gz.json`, 2 vueltas) | 14/14 | 13 | 0 | **3.6 mm** | 15.1 mm | 1.7° | 6.8 / 40.7 mm |
| Cinemático (`puertas_kin.json`, 3 vueltas) | 21/21 | 19 | 0 | **4.8 mm** | 13.5 mm | 1.6° | 5.4 / 45.2 mm |

- El error lateral máximo incluye los bordes de entrada y salida del vano.
- El error de localización máximo ocurre lejos de las puertas.

**Conclusión.**

- El cruce es físicamente viable sin contacto, pero **ajustado**: la holgura mínima medida es de 3.6 mm.
- El factor limitante es el error de localización en el vano (4–9 mm de media por cruce).
- Experimento descartado: bajar la corrección de rumbo a 0.75° **empeoró** la holgura mínima, a 2.2 mm en Gazebo y
  0.3 mm en el backend cinemático.

**Condiciones de viabilidad.**

1. El brazo debe estar plegado (STOW); con otra pose la huella es otra.
2. El error de localización en el vano debe ser menor a unos 10 mm. Para eso hacen falta un LiDAR que vea las jambas
   y la pausa de convergencia.
3. Velocidad de cruce ≤ 0.15 m/s con límites de aceleración.
4. Puerta libre de personas: el planificador no cruza si hay alguien en el vano.

**Mejora propuesta, no aplicada.**

- Una pose de **traslado** del brazo encontrada por optimización sobre las mallas reales reduce el ancho máximo de
  0.3616 a unos 0.337 m. Esto daría unos +12 mm por lado.
- Requiere validar:
  - auto-colisión con el cubo de montaje (el chequeo burdo de cajas da falsos positivos incluso para STOW);
  - que no tape el LiDAR ni la cámara de cabeza;
  - el comportamiento en Gazebo.
- No reemplazaría el `DEFAULT_HOME` de P&P: sería exclusiva de los entornos de los desafíos.

**Puertas no transitables con esta huella:**

- lavadero–dormitorio: el lavarropas queda a 5 cm del eje;
- exterior del lavadero: cerrada por las reglas.

## 3. HRI

### Fallas del nivel 5 (antes 18/20: semillas 1000 y 1012)

| Semilla | Causa raíz | Corrección |
|---|---|---|
| 1000, aproximación a la mano | La pose final de entrega se validaba con un margen de 5 cm a la persona. El freno por personas de la aproximación recta usaba 8 cm y predecía hasta 1.2 s, **más allá** de la pose final. El robot se detenía para siempre antes de llegar (`tiempo_agotado`, luego `bloqueado`) | - Freno de la aproximación recta a 3 cm, con la predicción acotada a lo que falta recorrer<br>- Plan con margen de 8 cm, porque la posición percibida tiene unos 3–4 cm de ruido<br>- Si la persona se mueve hacia el tramo, el freno sigue actuando |
| 1000 y 1012, seguir al anfitrión | - Después de la entrega el robot quedaba a 5–8 cm del invitado: cualquier giro barría su cuerpo y el gobernador por LiDAR bloqueaba (con razón) todo movimiento<br>- Además, cuando el anfitrión cruzaba una puerta y se perdía de vista, el robot quedaba girando en el vano, donde el giro está bloqueado | - **Retirada recta** por el mismo tramo validado al terminar la entrega<br>- Si la búsqueda falla, ir a la habitación de destino anunciada en la instrucción, o al último punto visto, y buscar desde ahí |

**Correcciones adicionales en esta iteración:**

- el plan de entrega usa la última posición conocida de todas las personas, más los puntos LiDAR que el mapa no
  explica (un invitado sentado que la cámara no identificó dejaba la pose previa a 0.6 m de él);
- el freno por LiDAR ahora solo frena lo que **acerca** la huella al obstáculo, igual que el freno por personas.
  Antes congelaba al robot junto a un invitado.

**Resultado cinemático del nivel 5** (`kin_hri_L5_expert.json`): **18/20, 0 contactos**.

- Las semillas 1000 y 1012 se corrigieron.
- Fallan 1006 y 1019: aproximación a la mano `bloqueado`. La ruta hacia la pose previa pasa junto a un invitado
  sentado y el gobernador frena. No se fuerza el paso.

El porcentaje es el mismo que antes, pero ahora con verificación de contacto **exacta**.

**Gazebo:** los niveles 3–5 de HRI **no se validaron en Gazebo en esta iteración**. Pendiente.

### Interacción estructurada frente a percepción real

| Aspecto | Qué está simulado | Qué NO está |
|---|---|---|
| Voz | Interfaz **estructurada**: el robot pregunta y recibe `(nombre, bebida)` como datos | ASR, TTS, diálogo |
| Identidad | Color de la camiseta como proxy | Reconocimiento facial o de personas |
| Mirada / contacto visual | Ángulo real entre el eje de la cámara de cabeza y la cabeza de la persona | Detección de mirada de la persona |
| Entregar la mano | La pinza (TCP) a < 10 cm de la mano real del invitado, con IK y el brazo real en Gazebo | Bolsa, agarre de la bolsa, transporte |
| Personas | Guionadas, se reubican por pasos | Marcha, intención, reacciones |

La **manipulación de la bolsa queda pendiente** mientras no haya evidencia física de agarre de ese objeto.

## 4. GPSR

### Hallazgo de percepción (Gazebo, nivel 1, semilla 2000)

**Diagnóstico corregido.**

- La iteración anterior atribuyó el error a la madera de los muebles.
- La traza en Gazebo muestra que la detección falsa de `person_c` (camiseta naranja) estaba en (4.11, 0.53,
  **0.96 m**), 6 cm por encima de la mesada de la cocina. Ahí está la **naranja (fruta)**, un objeto de GPSR del mismo
  color.
- La persona real estaba en (6.89, 2.41).
- Además, la madera (0.55, 0.36, 0.20) tiene casi el mismo tono que la camiseta (0.95, 0.50, 0.05): ~27° contra ~30°.
  La cromaticidad no las separa.

**Corrección: percepción más robusta, sin cambiar colores.** Compuertas para personas en `ColorHeadDetector`:

1. **Saturación** ≥ 0.75 × la saturación nominal de la camiseta: naranja 0.95 → 0.71; madera 0.64.
2. **Mapa estático:** se descartan los puntos 3D que caen sobre una primitiva del mapa. Una persona está en espacio
   libre.
3. **Tamaño físico:** el grupo debe abarcar ≥ 0.20 m en altura (torso ~0.5 m; fruta ~0.08 m).

Tests: `tests/test_perception_gates.py`.

**Verificación en Gazebo con la misma semilla** (`gz_gpsr_L1_s2000_compuertas.json`):

- La **detección falsa desapareció**: la creencia ya no reporta a la persona sobre la mesada.
- En una corrida de diagnóstico, `search_room` detectó a la `person_c` real a 19 cm de su posición verdadera.
- El episodio completo **sigue fallando (0/1)**: el robot repite `search_room` desde la entrada de la cocina sin volver
  a detectarla.
- Queda pendiente analizar la cobertura de la búsqueda y la pérdida de píxeles con la cámara de 160 × 120 (perfil
  `train`). No se declara resuelto.

### Fallas del nivel 3

**Antes:** 13/20 con verificación de contacto por grilla. Al pasar a la verificación **exacta** aparecieron 2
contactos reales, que se corrigieron:

| Semilla | Contacto | Corrección |
|---|---|---|
| 1019 | Tapa de la mesa al girar en el lugar al empezar `guide` (el robot quedó junto a la mesa tras `approach_person`) | - Holgura de giro exacta (`rotation_clearance`)<br>- `rot_free` conservador en la resolución de la grilla<br>- Antes de navegar, salir en recta de una pose no girable, verificando mapa y personas |
| 1002 | Extremo de una pared interior a 0.35 m/s con rumbo oblicuo durante `follow` (la grilla supone la huella orientada a la ruta) | Gobernador contra el mapa estático con holgura exacta de la huella predicha |

**Ahora** (`kin_gpsr_L3_expert.json`): **13/20, 0 contactos**. Fallas restantes y causas:

| Semilla | Comandos | Causa |
|---|---|---|
| 1001, 1003 | 0/3 | `approach_person` → `sin_ruta`: no hay una pose libre junto a la persona con el inflado social (persona en una esquina o pegada a un mueble) |
| 1012, 1017 | 1/3 | `go_target` / `go_instruction` → `bloqueado`: una persona quedó en un paso angosto después de `guide`/`follow` y el gobernador no la atraviesa (correcto) |
| 1002 | 2/3 | El tercer comando no se completa; queda esperando (sin contacto) |
| 1008 | 2/3 | `object_yesno` responde "no" cuando el objeto está: falla de percepción del detector simulado |
| 1010 | 1/3 | `search_room` no encuentra a la persona |

**Mejora pendiente:** pedir paso o esperar a que la persona libere el pasillo. Lo que no se hace es atravesar
personas ni ignorar colisiones.

**Separación.** El reconocimiento del comando (gramática cerrada con paráfrasis + intérprete) se evalúa aparte de la
ejecución física. `progress.instruction` y `progress.complete` se registran por separado.

## 5. Doing Laundry: alcance real del brazo

`scripts/reachability_study.py` → `alcance_brazo.json`. Método:

- IK sobre el URDF real: 6 articulaciones del Piper más el elevador `torso_lift_joint` en [-0.19, 0.068] m, con sus
  límites;
- restricción de orientación de la pinza;
- chequeo de colisión burdo contra el piso, el prisma de la base, la pared del cesto y la boca del lavarropas.

Todo se expresa en el frame `base_footprint`. El brazo está montado a la izquierda (+y).

**z mínima alcanzable con la pinza hacia abajo, por distancia lateral y.**

| y (m) | 0.40 | 0.45 | 0.50 | 0.55 | 0.60 | 0.65 | 0.70 | 0.75 | 0.80 | 0.85 |
|---|---|---|---|---|---|---|---|---|---|---|
| Elevador libre | 0.14 | 0.18 | 0.18 | 0.22 | 0.22 | 0.26 | 0.30 | 0.38 | 0.46 | — |
| Elevador en 0 | 0.34 | 0.34 | 0.38 | 0.38 | 0.42 | 0.46 | 0.50 | 0.54 | 0.66 | — |

**Cesto de piso.** Medidas 0.45 × 0.36 × 0.30 m. Centro a y = 0.62 (mínimo geométrico 0.58); pared cercana en
y = 0.44.

| Altura de la prenda (m) | y = 0.52 | y = 0.62 | y = 0.72 |
|---|---|---|---|
| 0.05 (fondo) | ✗ (100 mm) | ✗ (134 mm) | ✗ (178 mm) |
| 0.10 | ✗ (54 mm) | ✗ (92 mm) | ✗ (138 mm) |
| 0.20 | ✓ | ✗ (14 mm) | ✗ (65 mm) |
| 0.30 (borde) | ✓ | ✓ | ✓ |

Entre paréntesis, el error de posición del IK: el elevador queda en su límite inferior y no es una colisión.

**Tambor del lavarropas frontal.** Boca de 0.36 m a z = 0.38. Robot de costado, a 5 cm del frente. Aproximación
horizontal con el brazo pasando por la boca: **1/12 puntos**. El único alcanzable está a 0.30 m de profundidad, en el
centro, en el límite de tolerancia (9.5 mm, elevador en su límite inferior). Cerca de la boca, la muñeca debería estar
dentro de la base.

**Mesa de doblado.** Superficie a 0.74 m. **12/15 puntos**: alcanzable hasta 0.35 m desde el borde; a 0.45 m, no.

**Clasificación de los límites.**

| Límite | Tipo | Detalle |
|---|---|---|
| Fondo del cesto de piso inalcanzable | **Definitivo del robot** | Altura de montaje del brazo + largo del Piper + carrera del elevador. Con el elevador libre, la z mínima es 0.18–0.22 m a 0.45–0.60 m de distancia lateral |
| Altura del cesto (en el piso) y su posición | **De la escena** (supuestas: el plano no las da) | Ver alternativas |
| Tambor del lavarropas | **Definitivo del robot**, para este lavarropas frontal y esta altura | La boca está demasiado baja y cerca de la base para una entrada horizontal |
| Mesa de doblado | Viable hasta 0.35 m de profundidad | Ubicar las prendas cerca del borde |
| Cámara de cabeza que no ve el fondo del cesto | **De la escena** (borde de 0.30 m) | Se usa la cámara de muñeca; sus especificaciones reales no están confirmadas |
| Prendas | **De la simulación** | Proxies rígidos. No hay simulación de tela: no se declara doblado |

**Alternativas físicamente justificadas, sin modificar el robot:**

1. **Cesto elevado** (sobre un banco o estante) con el fondo a z ≥ 0.20 m. Con el elevador, el fondo queda alcanzable
   a y ≤ 0.55. Requiere confirmar que la regla lo permite.
2. **Usar el elevador** en el entorno (hoy fijo en 0) para el pre-agarre y para mirar dentro del cesto: baja la z
   mínima unos 0.15–0.20 m. Es un cambio del entorno, no del robot.
3. **Lavarropas:** solo prendas en la boca o un lavarropas de carga superior. La extracción profunda no es viable.
4. **No** se implementan agarres falsos, teletransportes de prendas ni doblado con cuerpos rígidos.

**Laundry nivel 3, cinemático** (`kin_laundry_L3_expert.json`): **19/20, 0 contactos**. Con la verificación exacta,
la aproximación recta al cesto rozaba la pared del cesto (se aceptaba con la grilla de 2 cm). Ahora se valida con
holgura exacta ≥ 5 cm. La falla restante (semilla 1017, `sin_pose_de_manipulacion`) es una prenda sin pose de base
alcanzable.

## 6. Entrenamiento

**Prueba corta del pipeline de PPO en Gazebo**, una por desafío (`ppo_gz_smoke/`). Configuración:

- `train_challenge.py --backend gazebo --steps 32 --n-steps 16 --checkpoint-every 16`;
- luego **reanudación** hasta 48 pasos;
- dominio 47, `GZ_PARTITION solverbot_val`.

| Desafío | Entrenar (32) | Reanudar (32 → 48) | Checkpoints | Actualizaciones de PPO | Tiempo de reloj |
|---|---|---|---|---|---|
| HRI | ✓ | ✓ (`reanudando ... en t=32`) | 16, 32, 48 + `latest` | Sí (`n_updates` 10, pérdidas finitas) | ~1 min por tramo |
| GPSR | ✓ | ✓ | 16, 32, 48 + `latest` | Sí | 0.04–1.2 pasos/s |
| Laundry | ✓ | ✓ | 16, 32, 48 + `latest` | Sí | ~15 min por tramo |

Verificado:

- observaciones y acciones (`MultiDiscrete`) válidas en Gazebo;
- recompensas finitas;
- guardado atómico de checkpoints y estado;
- reanudación;
- `model_final.zip` y `train_summary.json`.

**No verificado:**

- Ningún episodio terminó en 48 pasos de política aleatoria. No hay métricas de episodio en Gazebo durante el
  entrenamiento.
- **No hay aprendizaje demostrado**: no se entrenó ni se evaluó ninguna política. El experto guionado no es una política
  aprendida.

**Semántica de éxito, cinemático frente a Gazebo:**

- Es el mismo código de entorno: éxito, recompensas y verificación de contacto exacta sobre la pose real.
- Diferencias:
  - física de la base (DART frente a unicycle con los mismos límites);
  - percepción (imagen RGB-D real frente a un detector simulado con ruido);
  - en Gazebo, además, las colisiones físicas del motor.
- Las personas están guionadas en ambos.

## 7. Resumen del estado

| Categoría | Contenido |
|---|---|
| **Entornos Gymnasium implementados** | HRI (6 niveles), GPSR (4), Doing Laundry (4): API de Gymnasium, `reset` reproducible, verdad fuera de la observación, curriculum, experto de referencia |
| **Verificado en Gazebo** | - Navegación y puertas: 13 cruces sin contacto, holgura mínima de 3.6 mm<br>- HRI N0–N1<br>- GPSR N1, 2/3 en la iteración anterior: `go` y `object_yesno`<br>- Laundry hasta la localización de prendas con la cámara de muñeca, y el pre-agarre (2/3, iteración anterior)<br>- Regresión A/B de P&P<br>- Pipeline de PPO (entrenar + reanudar) en los 3 desafíos |
| **Verificado solo en cinemático** | HRI N2–N5 (N5: 18/20), GPSR N2–N3 (N3: 13/20), Laundry N3 (19/20). Todos sin contactos con la verificación exacta |
| **Pendiente por límites físicos** | - Agarre y transporte de la bolsa (HRI)<br>- `bring` (GPSR)<br>- Fondo del cesto de piso y tambor del lavarropas (Laundry, alcance)<br>- Doblado (no hay simulación de tela) |
| **Entrenamiento** | **Probado:** pipeline de PPO en cinemático (iteración anterior) y en Gazebo (esta). **No probado:** aprendizaje, es decir, política entrenada y evaluada |
| **Fallas restantes** | - HRI N5: 1006 y 1019 (persona sentada junto a la ruta)<br>- GPSR N3: 7, por sin ruta, personas en pasos angostos, búsqueda y percepción<br>- GPSR Gazebo, semilla 2000: falso positivo resuelto, pero el episodio no completa<br>- Laundry N3: 1017 |
| **Regresiones verificadas** | - P&P: A/B en Gazebo 5/5 contra 5/5 y 4 tests<br>- Suite completa: 80 tests pasan<br>- Los cambios de navegación (gobernador contra el mapa, giro exacto, freno LiDAR) se verificaron con los barridos de 20 semillas de los tres desafíos |
