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

Resultado cinemático del nivel 5 después de las correcciones: **PENDIENTE_HRI_L5**.

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

Tests: `tests/test_perception_gates.py`. Verificación en Gazebo con la misma semilla: **PENDIENTE_GZ_2000**.

### Fallas del nivel 3

- Antes: 13/20.
- Después de las correcciones comunes (aproximación, seguimiento, búsqueda): **PENDIENTE_GPSR_L3**.

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

PENDIENTE_ALCANCE

## 6. Entrenamiento

PENDIENTE_PPO

## 7. Resumen del estado

PENDIENTE_RESUMEN
