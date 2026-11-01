# Desafíos HRI, GPSR y Doing Laundry — especificación de simulación

> Validación física, auditoría de integración y estado actualizado (iteración 2): [VALIDACION_FISICA.md](VALIDACION_FISICA.md).

Fuente de reglas: `docs/referencias_desafios/` (imágenes 01, 02, 04, 05 y 12). Esta especificación separa en cada
desafío:

- **[R]** lo que dicen las referencias;
- **[S]** las decisiones de simulación tomadas aquí, que no son reglas;
- **[P]** lo que no está implementado.

Las referencias no traen un puntaje oficial detallado. Las recompensas de cada entorno son **hitos verificados** de
diseño propio, no un reglamento.

Pick and Place está **fuera de alcance** de esta rama. Su código (`envs/pick_place.py`, `expert.py`, `gripper.py`,
controladores) no se modificó.

## Arquitectura común (`solverbot_rl/challenges/`)

| Pieza | Archivo | Qué hace |
|---|---|---|
| Mapa 2D | `arena.py` | Grillas de ocupación desde las mismas primitivas que el mundo SDF (`make_worlds.py`). Planificador A* con costo por holgura. Las puertas son aristas explícitas (punto previo → alinear → cruce recto). |
| Navegación | `nav.py` | Seguimiento de ruta para el eje de ruedas, que es el centro de giro real. Localizador por scan-matching del LiDAR, robusto a personas. |
| Backends | `backend.py`, `gazebo_mobile.py` | **Cinemático:** sin física, rápido; sirve para tests y entrenamiento de alto nivel. **Gazebo:** física, LiDAR, cámara RGB-D de cabeza, cámara de muñeca; avance sincrónico. |
| Personas | `humans.py` | Comportamiento **guionado**: caminar, sentarse, seguir, guiar. Ceden el paso al robot. |
| Percepción | `sensing.py` | **Cinemático:** detector simulado (campo visual, línea de vista, ruido, pérdidas). **Gazebo:** segmentación por color con compuertas espaciales sobre la imagen real. |
| Entorno base | `base_env.py` | Ciclo del episodio y habilidades físicas (navegar, girar, buscar, hablar mirando, seguir, alcanzar). Incluye el gobernador de seguridad. |
| Entornos | `hri_env.py`, `gpsr_env.py`, `laundry_env.py` | Un entorno Gymnasium por desafío, con su experto de referencia. |

### Interfaz Gymnasium

IDs registrados:

- `SolverBotHRI-v0`
- `SolverBotGPSR-v0`
- `SolverBotLaundry-v0`

Acción y paso:

- La acción es `MultiDiscrete([n_habilidades, n_argumentos])`: la política elige **qué habilidad** ejecutar.
- La habilidad corre físicamente: muchos pasos de 0.1 s con controladores de referencia.
- La navegación usa siempre la **pose estimada** (LiDAR + odometría).

Observación y verdad:

- La observación tiene solo lo que el robot sabe: su pose estimada, sus detecciones con antigüedad, su memoria de
  diálogo y de tareas, y sus creencias sobre el éxito de lo que hizo.
- La **verdad** del simulador se usa solo para la recompensa, `is_success` e `info`. Nunca entra en la observación.
  Hay un test que lo verifica: `test_observation_does_not_read_truth`.

Episodio:

- `reset(seed)` es reproducible. Fija las semillas del escenario, del ruido de odometría y del detector, y estaciona
  todas las entidades.
- `terminated` ocurre por éxito, colisión, vuelco o falla de sensor.
- `truncated` ocurre al llegar al máximo de habilidades o de tiempo simulado.
- `info` lleva `contact`, `tipped`, `fault`, el error de localización (medio y máximo), `stale`, `backend` y
  `physical`.

### Seguridad y verificación física

- **Colisión real.**
  - Gazebo: se usa la huella contra el mapa y la pose real.
  - Cinemático: el backend se niega a mover la base a una pose en contacto.
- **Vuelco.** Inclinación mayor a 15° según el PosePublisher de Gazebo. Fue necesario agregar límites de aceleración
  al `diff_drive_controller`: sin ellos el robot volcaba.
- **Gobernador de seguridad.** Frena si la huella predicha (0.3–1.2 s) se acerca a una persona percibida o a puntos
  LiDAR no explicados por el mapa. Alejarse sigue permitido, para no generar bloqueos.
- **Margen social.** El planificador mantiene 0.15 m extra a las personas.
- **Puertas.** Nunca se abandona una ruta dentro de una puerta. Si hay que replanificar dentro del vano, el robot sale
  en recta.
- **Velocidad.** Máximo 0.35 m/s, por debajo de los 6 km/h de la regla (imagen 05).
- **Límites.** NaN en odometría, LiDAR o pose, sensores vencidos (`stale`), timeouts y límites de pan/tilt.

### Hallazgos de geometría (afectan a los tres desafíos)

- **Ancho del robot contra la puerta.** El robot mide 0.669 m de ancho con el brazo plegado y la puerta 0.70 m
  (regla): quedan unos 1.5 cm por lado.
  - El cruce se hace recto, alineado a menos de 1° y con desviación de rumbo de hasta 1.5°.
  - El robot gira alrededor del eje de ruedas (x = 0.084 m). Al girar 90°, `base_footprint` se desplaza 129 mm.
- **Puerta lavadero–dormitorio.** Es inutilizable con esta huella: el lavarropas del plano queda a 5 cm del eje de la
  puerta.
- **Puerta exterior del lavadero.** Queda cerrada en los desafíos **[S]**: las reglas solo mencionan la apertura de la
  puerta de entrada.

## HRI — Human Robot Interaction (imagen 04)

**[R] Requisitos de las referencias**

Recepción:

- Detectar el timbre (opcional).
- Abrir la puerta (opcional).
- Saludar y preguntar nombre y bebida favorita.
- Mantener contacto visual y seguir a la persona con la mirada.

Guiado y asiento:

- Guiar a cada invitado a la sala con la mirada hacia la dirección de navegación.
- Identificar un asiento libre e indicarlo. Los invitados pueden cambiar de asiento.

Presentación:

- Presentarlos mutuamente con nombre y bebida, mirando a cada uno.
- Describir visualmente al primer invitado (opcional).

Bolsa:

- Recibirla de la mano del 2.º invitado, con el manipulador a menos de 10 cm de la mano.
- Informar al anfitrión y seguirlo hasta el lugar indicado.
- Depositar la bolsa en el suelo.

**[S] Decisiones de simulación**

- Personas simuladas guionadas.
- La identidad de cada persona es el color de su camiseta. Es un proxy de reconocimiento, no reconocimiento facial.
- El diálogo es una interfaz **estructurada**. No hay audio ni reconocimiento de voz.
- El timbre es un evento del escenario.
- "Contacto visual" significa que el eje de la cámara de cabeza queda a menos de 12° de la cabeza real.
- "Indicar asiento" significa mirarlo a menos de 15°.
- Mano a la bolsa: la FK de las articulaciones medidas, contra la mano de la persona, debe dar menos de 10 cm.
- El anfitrión guía y espera si el robot queda a más de 1.8 m.
- Curriculum de 6 niveles: `recepcion_1`, `acomodar_1`, `acomodar_2`, `presentar`, `cambio_asiento` y `bolsa`.

**[P] No implementado**

- Tomar la bolsa y depositarla: no hay modelo de bolsa y depende del agarre.
- Timbre real y apertura de puerta.
- Descripción visual del invitado.

## GPSR — General Purpose Service Robot (imagen 02)

**[R] Requisitos de las referencias**

- El robot empieza afuera. Al abrirse la puerta, navega al Instruction Point.
- El operador da 3 comandos. El robot decide si los recibe juntos o de a uno:
  - de a uno: vuelve al operador después de cada tarea;
  - juntos: puede combinarlos.
- Interpretar instrucciones formuladas de distintas maneras y planificar una secuencia.
- Las tareas pueden ser buscar, reconocer y transportar objetos; identificar personas; navegar; entregar; manipular el
  entorno.
- Al final, volver al Instruction Point.

**[S] Decisiones de simulación** (`gpsr_grammar.py`, `gpsr_env.py`)

- **Texto en lugar de voz.** La **gramática es CERRADA**: plantillas con sinónimos y variantes de orden. No es
  comprensión de lenguaje natural.
  - El intérprete reconoce exactamente esa gramática: 8000/8000 comandos generados se interpretan bien, con unas
    3800 formulaciones distintas.
  - Lo que queda fuera de la gramática devuelve `None`.
- **Separación de responsabilidades:**
  - **Interpretación:** `CommandParser`, de texto a `Command`. El robot solo recibe el texto.
  - **Planificación:** `GPSRPlanner`, el experto de referencia. En modo "juntos" reordena los comandos por cercanía.
  - **Ejecución:** las habilidades físicas.
  - **Evaluación:** con la verdad. La posición real, la mirada real y la respuesta correcta se evalúan según el estado
    real.
- **Verbos ejecutables:** `go`, `greet`, `where_person`, `object_yesno`, `count_people`, `follow` y `guide`.
- **Variación sin cambiar código.** Las tareas varían por semilla: comandos, personas, objetos y la hora de apertura
  de la puerta.
- **Escenario.** Las personas se ubican sin bloquear puertas, ni poses de aproximación, ni el acceso a ninguna
  habitación.
- **Objetos.** Tienen colores distinguibles. Se apoya a lo sumo uno por mueble y la detección se restringe al mueble
  inspeccionado.
- **Curriculum:**

  | Nivel | Comandos | Verbos |
  |---|---|---|
  | `ir_1` | 1 | `go` |
  | `simple_1` | 1 | 3 verbos |
  | `simple_3` | 3 | 4 verbos |
  | `completo_3` | 3 | 7 verbos |

**[P] No implementado**

- `bring` (traer objetos): se interpreta y se marca `requires_grasp`, pero no se ejecuta porque depende del agarre.
- Manipular elementos del entorno.
- Características de personas más allá del color.
- Voz.

## Doing Laundry (imagen 01) — etapa intermedia

**[R] Requisitos de las referencias**

- El robot empieza afuera, navega al lavadero al abrirse la puerta e identifica lavarropas, mesa de doblado y cesto.
- Debe localizar prendas en el cesto o en el lavarropas, tomar al menos una camiseta, llevarla a la mesa y depositarla.
- Debe doblarla: identificar posición y orientación, extenderla, alinear bordes y plegar.
- Opcionales:
  - abrir el lavarropas;
  - extraer prendas;
  - usar el cesto;
  - doblar varias;
  - apilar.

**[S] Decisiones de simulación**

- **Prendas.** Son **PROXIES RÍGIDOS**, placas delgadas de color (`challenges.yaml: model: rigid_proxy`). **No
  modelan tela** y así se reportan en `info` (`garment_model`).
- **Cesto.** Va en el piso, de 0.45 × 0.36 × 0.30 m. Las medidas y la posición son supuestas: el plano no las da.
- **Identificar.** Se mira cada mueble (a menos de 15°) y su posición estimada por percepción debe tener un error
  menor a 0.25 m. Es verificación visual de un mapa semántico conocido.
- **Localizar.** La **cámara de muñeca**, sobre el cesto y apuntando hacia abajo, estima cada prenda con un error menor
  a 6 cm.
- **Pre-agarre.** La pinza apunta hacia abajo sobre la prenda, a menos de 5 cm en xy, entre 2 y 15 cm por encima del
  borde.
- **Curriculum:** `llegar`, `identificar`, `localizar_prenda` y `pre_agarre`.

**HALLAZGOS** (por qué no se puede avanzar más con este robot y este cesto):

1. **La cámara de cabeza no ve el fondo del cesto.**
   - La cámara está a 1.37 m, centrada en la base, con tilt máximo de 45° y VFOV de unos 42°. El piso visible más
     cercano queda a unos 0.6 m.
   - El borde de 0.30 m tapa el fondo, salvo que la cámara esté a menos de unos 0.5 m.
   - Por eso se usa la cámara de muñeca. Sus especificaciones reales **no están confirmadas**: el frame es provisional
     (README de referencias).
2. **El brazo no llega al fondo del cesto.** Estudio completo en [VALIDACION_FISICA.md](VALIDACION_FISICA.md) §5.
   - Con el elevador en 0, la z mínima con la pinza hacia abajo es de 0.34–0.42 m.
   - Con el elevador en su límite inferior (-0.19 m) baja a 0.18–0.22 m, a 0.45–0.60 m de distancia lateral.
   - El fondo de un cesto de piso de 0.30 m sigue siendo inalcanzable: límite definitivo del robot.
   - El tambor del lavarropas frontal tampoco es viable: 1/12 puntos, en el límite de tolerancia.
   - Alternativas: un cesto elevado o usar el elevador en el entorno.
3. **Dónde poner la base.** El cesto debe quedar a 0.58 m o más al costado de la base: medio ancho del cesto más la
   huella izquierda. Junto al cesto no se puede girar en el lugar, así que la aproximación final es recta, desde una
   pose previa donde sí se puede girar.

**[P] No implementado**

- Recoger, transportar y depositar: dependen del agarre y del alcance (hallazgo 2).
- Doblar y apilar: requieren simulación de tela, que Gazebo Harmonic no tiene en este stack.
- Abrir el lavarropas y extraer prendas.

## Entrenamiento

`solverbot_rl/scripts/train_challenge.py` entrena la política de **alto nivel** con PPO MultiDiscrete. Incluye:

- curriculum por tasa de éxito real (ventana de 20, promoción al 80 %);
- semillas;
- checkpoints atómicos;
- reanudación automática;
- evaluación independiente periódica con semillas fijas, en un entorno aparte;
- `failures.jsonl` y `episodes.jsonl`;
- parada limpia con SIGINT/SIGTERM;
- `--device cpu|cuda` y `--threads`;
- pre-entrenamiento opcional por imitación del experto (`--bc-episodes`), registrado en `train_summary.json`.

PPO es on-policy: no tiene replay buffer que persistir. **El experto guionado NO es una política aprendida**:
`evaluate_challenge.py` registra `policy_is_learned`.

`solverbot_rl/scripts/evaluate_challenge.py` evalúa con semillas fijas y escribe un reporte JSON: resumen más un
registro por episodio, con la traza de cada falla.

## Ejecución headless (`tools/headless/`)

Se selecciona el desafío con:

- `ctrain <hri|gpsr|laundry> <run> [args]`: entrenamiento;
- `cexpert <hri|gpsr|laundry> [args]`: experto con reporte JSON.

Ambos usan el mundo `home_challenges` con cámaras. Comandos disponibles: `start.sh`, `status.sh`, `stop.sh`
(parada limpia con checkpoint) y `solverbot-challenge@.service` (reanuda tras un reinicio). Cada instancia aísla
`ROS_DOMAIN_ID = 42 + instancia` y su `GZ_PARTITION`. El dominio 10 del robot real se rechaza.

## Estado de las funcionalidades

Leyenda:

- ✅ implementado y verificado con la verdad del simulador (backend cinemático; ver la columna Gazebo para la física);
- 🟡 parcial o con salvedades;
- ❌ pendiente.

| Desafío | Funcionalidad | Estado | Notas |
|---|---|---|---|
| Común | Mapa, planificador con puertas explícitas, localización LiDAR | ✅ | Gazebo: rutas por las 5 puertas transitables sin contacto; error medio de localización ~1 cm |
| Común | Gobernador de seguridad (personas + LiDAR no mapeado), detección de vuelco | ✅ | Contadores en `info['safety_halts']` |
| Común | Interfaz Gymnasium, reset reproducible, verdad fuera de la observación | ✅ | Tests de interfaz, reproducibilidad y separación |
| Común | Entrenamiento PPO + curriculum + checkpoints + reanudación + evaluación | ✅ | Probado en el backend cinemático (corrida corta y reanudación). En Gazebo, mismo script, sin corrida larga |
| HRI | Recepción: mirar, preguntar nombre y bebida (estructurado) | ✅ | Gazebo: niveles 0 y 1 con éxito |
| HRI | Guiar con la mirada hacia adelante, ofrecer asiento libre | ✅ | Gazebo: nivel 1, 3/3 |
| HRI | Presentar a los invitados mirándolos; cambios de asiento | ✅ | Solo cinemático |
| HRI | Manipulador a < 10 cm de la mano del 2.º invitado | 🟡 | Solo cinemático; no hay modelo de bolsa |
| HRI | Seguir al anfitrión hasta el destino | ✅ | Solo cinemático |
| HRI | Tomar y depositar la bolsa | ❌ | Depende del agarre |
| HRI | Timbre real, abrir la puerta, describir al invitado | ❌ | Opcionales |
| HRI | Voz / audio | ❌ | Interfaz estructurada (decisión de simulación) |
| GPSR | Puerta, Instruction Point, recibir de a uno o juntos, volver | ✅ | Gazebo: nivel 1, 2/3 (`go` y `object_yesno` con éxito; `greet` falló por el color, ver Resultados) |
| GPSR | Gramática cerrada con paráfrasis + intérprete | ✅ | No es comprensión de lenguaje natural |
| GPSR | go, greet, where_person, object_yesno, count_people | ✅ | Cinemático: N1 y N2 19/20 |
| GPSR | follow, guide (personas que se mueven) | 🟡 | Cinemático: N3 (los 7 verbos) 13/20; ver las fallas en Resultados |
| GPSR | bring (traer o entregar objetos), manipular el entorno | ❌ | Depende del agarre |
| Laundry | Puerta, navegar al lavadero, identificar lavarropas, mesa y cesto | ✅ | Gazebo: 3/3 en esta parte |
| Laundry | Localizar prendas (proxies rígidos) con la cámara de muñeca | ✅ | Gazebo: error de 0.8 a 5.7 cm |
| Laundry | Pre-agarre sobre la prenda | 🟡 | Gazebo: 2/3. Limitado por el alcance (hallazgo 2) |
| Laundry | Recoger, transportar, depositar | ❌ | Agarre + alcance |
| Laundry | Doblar, apilar, lavarropas | ❌ | Requiere simulación de tela |

## Resultados del experto de referencia (no son políticas aprendidas)

Reportes JSON en `solverbot_rl/docs/resultados/desafios/`: semillas, progreso, tiempo simulado, colisiones, error
de localización y traza de cada falla.

### Backend cinemático (sin física), 20 semillas por nivel (1000–1019)

| Desafío | N0 | N1 | N2 | N3 | N4 | N5 | Colisiones |
|---|---|---|---|---|---|---|---|
| HRI | 20/20 | 20/20 | 20/20 | 20/20 | 20/20 | 18/20 | 0 |
| GPSR | 20/20 | 19/20 | 19/20 | 13/20 | – | – | 0 |
| Laundry | 20/20 | 20/20 | 20/20 | 19/20 | – | – | 0 |

### Gazebo Harmonic (física, LiDAR y cámaras simuladas; render por CPU con llvmpipe)

| Ensayo | Resultado | Notas |
|---|---|---|
| HRI nivel 1, semillas 2000–2002 | 3/3 | Sin contacto; error medio de localización 4.7 mm |
| HRI nivel 0, por `tools/headless` (`cexpert`, instancia 3, dominio 45) | 1/1 | Prueba de punta a punta del orquestador |
| Doing Laundry nivel 3, semillas 2000–2002 | 2/3 | La falla (semilla 2001) es el límite de alcance del pre-agarre (hallazgo 2) |
| GPSR nivel 1, semillas 2000–2002 | 2/3 | Ver el hallazgo de color de la camiseta naranja más abajo |

Notas sobre la cobertura:

- **Fallas de GPSR N3 (7/20).**
  - Después de `follow` o `guide`, una persona queda en un paso angosto y la navegación se bloquea (4).
  - Aproximación sin ruta a una persona en una esquina (2).
  - Saludo no verificado (1).
  - Mejora pendiente: que las personas que terminan de guiar o de seguir liberen los pasos (o que el robot les pida
    paso).
- **No se corrieron 20 semillas en Gazebo por nivel.** Con render por CPU, cada episodio tarda de 1 a 30 minutos de
  reloj. Los comandos para correrlo en WSL o con GPU están en `tools/headless/README.md`.
- **Correcciones posteriores a algunas corridas.** El guardia anti-bucle de `humans._advance_along` se agregó después
  de las corridas HRI y GPSR N0–N2 y de Laundry. Afecta solo a casos degenerados de detención de una persona que
  sigue al robot: los episodios completados no pasan por ese caso.

**Hallazgo en Gazebo (GPSR, semilla 2000).**

- La camiseta **naranja** (`person_c`) se confunde con la **madera** de los muebles en la segmentación por
  cromaticidad: distancia 0.21, por debajo de la tolerancia de 0.30.
- El robot cree que mira a la persona (creencia 1.0), pero la mirada real no la apunta (verdad 0.0).
- **Actualización (iteración 2):** la detección falsa era la **naranja (fruta)** sobre la mesada. No se cambió el
  color: se agregaron compuertas de saturación, mapa estático y altura física para personas. El falso positivo
  desapareció, pero el episodio todavía no completa. Ver [VALIDACION_FISICA.md](VALIDACION_FISICA.md) §4.

### Entrenamiento

`train_challenge.py` se validó en el backend cinemático:

- pre-entrenamiento por imitación;
- PPO;
- promoción de curriculum;
- checkpoints atómicos;
- evaluación independiente;
- reanudación desde t = 1024.

No se hizo una corrida larga: el presupuesto de Cloud se reservó para la validación física.

En Gazebo, el mismo script corre por `tools/headless/start.sh <i> ctrain ...`, pero **no se ejecutó** en esta
sesión. No hay política aprendida evaluada.
