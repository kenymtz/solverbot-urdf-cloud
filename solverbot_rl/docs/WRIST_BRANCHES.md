# Ramas de muñeca en `reach` — Fase A (diagnóstico)

> La Fase B (etiquetado consistente, factibilidad corregida, BC nuevos y evaluación) está en
> [`WRIST_BRANCHES_FASE_B.md`](WRIST_BRANCHES_FASE_B.md). Corrige la afirmación "B factible en 620/620" de §8 (contaba
> 40°, que nunca cumple el criterio de éxito).

Rama `exp/solverbot-wrist-branch-consistency`, desde `feat/solverbot-reach-bc-sac` (PR #3, `a38c2e2`).

**Solo diagnóstico.** No se entrenó ningún modelo. No se modificaron:

- `ScriptedExpert`;
- BC v1 / v2w (congelados);
- física, URDF, pinza, cámaras, recompensas ni criterios de éxito;
- los espacios de observación (31) y acción (7).

El etiquetador consistente que se propone **no está implementado ni activado**. La verdad del simulador se usa solo para
**medir** (soluciones de IK de referencia de cada episodio); las políticas ven únicamente la observación.

## Herramientas y reproducción

| Archivo | Rol |
|---|---|
| `solverbot_rl/imitation/branch_analysis.py` | Familias de IK, costes, rama ejecutada, intención, cambios de signo, contradicciones |
| `scripts/diagnose_wrist_branches.py` | `ikmap`, `rollouts`, `analyze`, `labelrules`, `plots` |
| `tests/test_branch_analysis.py` | 9 pruebas de la clasificación (datos sintéticos) |
| `docs/resultados/wrist_branches/` | `summary.json`, `episodes.jsonl` (un registro por episodio), `label_rules.json`, figuras |

**Semillas de diagnóstico: 915000–915799.** Es un rango propio, disjunto de:

- demostraciones: 100000–113029;
- evaluaciones previas: 900000–905xxx y 950000+;
- futuras evaluaciones: 920000+;
- confirmación: 930000+.

| Escenario | Semillas | Cinemático | Gazebo |
|---|---|---|---|
| `reach` (home exacto, cambia el yaw del cubo) | 915000–915019 | 20 × 3 políticas | 915000 × 3 políticas |
| `reach_var_s` (±3 cm, ±0,05 rad) | 915200–915399 | 200 × 3 | — |
| `reach_var_m` (±5 cm, ±0,10 rad) | 915400–915599 | 200 × 3 | 8 episodios elegidos (abajo) |
| `reach_var_l` (±7 cm, ±0,15 rad; nunca entrenado) | 915600–915799 | 200 × 3 | — |

```bash
W="python3 solverbot_rl/scripts/diagnose_wrist_branches.py"; R=solverbot_rl/runs/wrist_branches
$W ikmap --out $R/ikmap --grid 11 --seeds-s '' --seeds-m '' --seeds-l ''
$W rollouts --policy scripted --level reach_var_m --seeds 915400-915599 --out $R/rollouts   # idem bc_v1, bc_v2w y niveles
tools/headless/run_experiment.sh 2 py $R diagnose_wrist_branches.py rollouts --backend gazebo --out $R/rollouts \
    --policy bc_v2w --level reach_var_m --seeds 915407,915425,915405                         # 1 instancia, dominio 44
$W analyze --root $R --out solverbot_rl/docs/resultados/wrist_branches
$W labelrules --root $R --analysis solverbot_rl/docs/resultados/wrist_branches
$W plots --root $R --out solverbot_rl/docs/resultados/wrist_branches
```

**Versiones.**

- ROS 2 Jazzy, Gazebo Sim 8.10.0 (DART), render por CPU (llvmpipe), 4 núcleos.
- numpy 2.5.4, scipy 1.18.1, torch 2.14.1, SB3 2.9.0, gymnasium 1.3.0.
- Política determinista. Experimento `reach_bc_v1`, sin cambios.

**Backend cinemático frente a Gazebo.**

| Aspecto | Cinemático | Gazebo |
|---|---|---|
| Articulaciones | Siguen la orden exactamente | Servos con dinámica |
| Detector del cubo | Verdad + ruido gaussiano de 1 cm + 5 % de pérdidas, sin campo visual (`obj_valid` ≈ 95 %) | Cámara de muñeca real, con pérdidas por campo visual (`obj_valid` 29–81 % por episodio) y latencia imagen/estado |
| Repetibilidad | Misma semilla → misma trayectoria | Misma semilla ≠ misma trayectoria: la pose medida tras el reset y la fase cámara/paso varían |

## 1. Las dos familias de soluciones de IK

`ScriptedExpert._ik` en `reach` funciona así:

- resuelve la IK del punto de pre-agarre con el eje de aproximación inclinado 20/30/40° y el yaw libre;
- usa 4 reinicios aleatorios con **semilla fija** (`seed=0`), así que la elección es determinista dada la pose inicial;
- se calcula **una vez**, al inicio. Después interpola en espacio articular y usa IK local en los últimos 4 cm.

Estructura medida del conjunto de soluciones (`ikmap`):

- **joint1–joint3** quedan fijados por la inclinación.
- **joint6** es un continuo libre (giro alrededor del eje de aproximación).
- **(joint4, joint5)** tienen **dos familias discretas**, un volteo de muñeca (joint4 ± π, −joint5):
  - **A:** joint4 ≈ +1,5;
  - **B:** joint4 ≈ −1,5.
- La etiqueta de familia es el signo de joint4 **de la solución objetivo**. El signo de joint5 depende de la inclinación.

**Cómo elige el experto.** La regularización hacia la pose inicial domina el coste de las soluciones convergidas, así que
el experto elige la familia de **menor distancia L2 desde la pose inicial** entre las que encuentran sus 5 arranques:

| Escenarios con ambas familias | Coincide con mínimo L2 | Coincide con mínimo tiempo (pasos) |
|---|---|---|
| `var_s` (200) | 200 | 119 |
| `var_m` (195) | 187 | 125 |
| `var_l` (187) | 156 | 92 |

Los desacuerdos con el mínimo L2 son soluciones que sus 5 arranques no encuentran.

**El coste está empatado.** En el mapa de pose inicial (perturbación ±0,15 rad de joint4 y joint6 con el cubo nominal):

- la diferencia L2(A) − L2(B) tiene mediana **0,04 rad**;
- la elección es **no monótona en joint4**: Δ = −0,15 → B, −0,12…−0,03 → A, 0…+0,12 → B, +0,15 → A;
- **joint6 no influye**.

Un cambio de 0,03 rad en la pose inicial invierte la rama (`map_q0_j4_j6.png`). En cambio, **B es siempre más rápida**:
A necesita **5,5–11,5 pasos más** desde `home`.

**La posición del cubo determina qué familia existe** (`map_cube_home.png`, home exacto, ±7 cm):

- arriba a la izquierda (dy > 0, dx < 0) solo existe **A** a 20°;
- abajo (dy < −4 cm) solo existe **B** (abajo a la derecha, recién a 30°);
- en la banda central existen ambas, y ahí la elección del experto aparece en parches.

En los escenarios de nivel:

| Nivel | Escenarios | A infactible | B infactible | B solo a 30° | A solo a 30° |
|---|---|---|---|---|---|
| `reach` | 20 | 0 | 0 | 0 | 0 |
| `reach_var_s` | 200 | 0 | 0 | 0 | 0 |
| `reach_var_m` | 200 | 5 | 0 | 3 | 2 |
| `reach_var_l` | 200 | 13 | 0 | 27 | 10 |

**Consecuencia:** fijar siempre el signo de joint4 no es posible en todo el espacio de la tarea. La tarea exige A o B
según la región, y la frontera atraviesa `var_m`/`var_l`. Esto explica el fracaso del etiquetador "canónico" anterior.

**Margen a los límites articulares en la solución** (`ikmap`, mediana [mínimo]):

| Familia | joint4 | joint5 |
|---|---|---|
| A | 0,13–0,19 [0,00] rad | 0,32 [0,17] rad |
| B | 0,22–0,24 [0,00] rad | 0,32 [0,17] rad |

Ninguna familia es intrínsecamente más frágil.

## 2. Distribución de ramas por escenario (cinemático, 20/200 semillas por nivel)

Definiciones (sin confundir acción y rama):

- **Rama ejecutada:** familia a la que la configuración medida (joint4, joint5) se ha **acercado ≥ 0,25 rad** y de la que
  está más cerca, con histéresis.
- **Cambio real de rama:** pasar de comprometido con una familia a comprometido con la otra.
- **Cambio de signo de la acción de joint4:** con |a4| ≥ 0,3 a ambos lados. **No implica cambio de rama.**
- **Cambio de intención:** la orden de muñeca apunta a la otra familia (coseno > 0,5) durante ≥ 3 pasos, en la fase
  lejana (d > 45 mm).

| Política · nivel | Éxito [IC 95 %] | Ejecuta A / B | Cambios reales | Cambios de signo de a4 | Cambios de intención | Éxito si ejecuta A | Éxito si ejecuta B |
|---|---|---|---|---|---|---|---|
| Experto · `reach` | 20/20 [84, 100] | 0 / 20 | 0 | 20 | 0 | — | 20/20 |
| Experto · `var_s` | 200/200 | 128 / 72 | 0 | 50 | 0 | 128/128 | 72/72 |
| Experto · `var_m` | 200/200 | 123 / 77 | 0 | 43 | 0 | 123/123 | 77/77 |
| Experto · `var_l` | 200/200 | 132 / 68 | 0 | 35 | 0 | 132/132 | 68/68 |
| bc_v1 · `reach` | 20/20 | 20 / 0 | 0 | 0 | 0 | 20/20 | — |
| bc_v1 · `var_s` | 193/200 [93, 98] | 193 / 7 | 0 | 8 | 0 | 193/193 | **0/7** |
| bc_v1 · `var_m` | 172/200 [81, 90] | 165 / 33 | 2 | 33 | 7 | 163/165 | **7/33** |
| bc_v1 · `var_l` | 169/200 [79, 89] | 164 / 32 | 1 | 49 | 10 | 150/164 | **18/32** |
| bc_v2w · `reach` | 20/20 | 20 / 0 | 0 | 1 | 0 | 20/20 | — |
| bc_v2w · `var_s` | 197/200 [96, 99] | 197 / 3 | 1 | 82 | 35 | 196/197 | 1/3 |
| bc_v2w · `var_m` | 193/200 [93, 98] | 176 / 23 | 0 | 75 | 32 | 174/176 | 19/23 |
| bc_v2w · `var_l` | 189/200 [90, 97] | 168 / 32 | 0 | 71 | 26 | 163/168 | 26/32 |

**Lectura.**

1. **El experto nunca cambia de rama** (0 cambios reales, 0 cambios de intención, objetivo constante). Aun así cambia
   el **signo** de la acción de joint4 en 20–100 % de los episodios, al pasar a IK local cerca del objetivo. El cambio de
   signo **no es** un indicador de cambio de rama.
2. **Las políticas BC están sesgadas a la familia mayoritaria A** (82–99 % de los episodios frente a 62 % del experto)
   y **fallan sobre todo cuando ejecutan B**: bc_v1 en `var_m` acierta 99 % en A y 21 % en B.
3. **Cambios reales de rama: casi nunca** (0–2 por 200). El fallo no es "saltar de A a B" sino no ejecutar
   limpiamente la rama:
   - **bc_v1, modo de fallo "B deformada":** joint4 se queda corto (−1,0…−1,3 contra −1,5 de la solución) y joint5
     **sobrepasa la solución** (−0,85) hasta su **límite de −1,22**. Se estanca a 40–90 mm. En `var_m`, 28/28 fallos
     terminan con una muñeca en el límite (joint5 en 27). La solución B tiene 0,32 rad de margen en joint5: es la
     política la que la sobrepasa.
   - **bc_v2w, modo de fallo "indecisión":** 6/7 fallos en `var_m` y 8/11 en `var_l` tienen cambios de intención. joint4
     sube hacia A y vuelve (oscilación), con desvíos de 100–300 mm y pérdida de inclinación del TCP. Solo 1 fallo
     termina en un límite.

Figuras: `traces_j4_j5_var_m.png` (60 episodios por política, fallos en naranja) y `map_executed_family.png` (rama
ejecutada según la posición del cubo; las políticas usan A donde el experto usa B, y los fallos se concentran en la zona
B).

## 3. Contradicciones entre etiquetas del experto (cinemático, 620 episodios, 23 522 pasos)

Se buscan pares de pasos de **episodios distintos** con entrada del actor cercana. La entrada es la observación sin la
acción previa, que el actor BC enmascara, estandarizada. El radio es el cuantil de la distancia al vecino más cercano de
otro episodio.

| Radio | Pares | Con familias opuestas | \|Δa4\| opuestos | \|Δa4\| misma familia |
|---|---|---|---|---|
| p10 (0,35) | 4 158 | 0,17 % | 2,00 | 0,004 |
| p25 (0,44) | 11 231 | 0,62 % | 2,00 | 0,008 |
| p50 (0,58) | 41 792 | 1,14 % | 2,00 | 0,017 |
| **Primer paso** (p50) | 689 | **38,5 %** | **2,00** | 0,000 |

- **Las contradicciones se concentran en el instante de decisión.** En el primer paso, el 38 % de los pares de estados
  iniciales casi iguales tienen etiquetas opuestas y totalmente incompatibles (+1 contra −1 en joint4).
- **Después desaparecen**, porque la propia configuración articular separa las ramas.

## 4. ¿Promedian BC v1 / v2w acciones incompatibles? (predicciones sobre los estados del experto)

| | bc_v1 | bc_v2w |
|---|---|---|
| \|a4\| predicha, estados en conflicto / sin conflicto | 0,91 / 0,92 | **0,50 / 0,88** |
| Acción intermedia (\|a4\| < 0,5) donde el experto satura, en conflicto / sin conflicto | 1 % / 7 % | **51 % / 10 %** |
| Signo opuesto a la etiqueta, en conflicto / sin conflicto | **57 %** / 6 % | 57 % / 7 % |
| **t = 0:** predice A / el experto elige A | **89 %** / 62 % | 69 % / 62 % |
| **t = 0:** acción intermedia (todos / en conflicto) | 11 % / 4 % | **34 % / 51 %** |

Dos patrones distintos del mismo problema (regresión por MSE sobre etiquetas multimodales):

- **bc_v2w promedia.** En la región en conflicto, la acción de joint4 cae a la mitad (|a4| 0,50 frente a 0,88) y es
  intermedia la mitad de las veces. En ejecución, eso es la indecisión y oscilación de la §2. El histograma
  `bc_a4_hist.png` muestra la masa alrededor de 0 solo en la región en conflicto.
- **bc_v1 colapsa al modo mayoritario.** Predice A en el 89 % de los estados iniciales, con acciones saturadas. Cuando el
  estado lo arrastra a B, la ejecuta deformada (§2).

**Efecto en la trayectoria y la orientación** (§2):

- desvío medio de 30–34 mm (experto 34–59 mm, que también aleja el TCP al girar la muñeca);
- fallos con desvío > 30 mm: bc_v1 27/28 y bc_v2w 7/7 en `var_m`;
- en las oscilaciones de bc_v2w, el eje z del TCP pasa de −0,9 a −0,4…−0,6;
- `episodes_tool_tilt_lost` (z > −0,6 en la fase lejana) **no discrimina**: el experto también inclina la pinza al
  girar la muñeca.

## 5. Coste de movimiento entre las familias

| | Desde `home` (mapa de pose inicial) | Escenarios `var_*` |
|---|---|---|
| L2(A) − L2(B) | Mediana +0,04, rango [−0,43, +0,36] rad | Rango [−1,5, +0,8] (`map_levels.png`) |
| Pasos(A) − pasos(B) | **+5,5…+11,5** (B siempre más rápida) | Mediana +6,7 (mapa de cubo) |
| Familia más rápida | B | B en ~75 % de los escenarios |

## 6. ¿Aparece el problema con percepción válida?

- **Cinemático** (detector continuo, 5 % de pérdidas): en el paso del cambio de intención, la detección es válida en el
  **92–97 %** de los casos, y válida en los 5 pasos previos en el 77–88 %. El experto, con la misma percepción, no cambia
  nunca. **El problema aparece con percepción válida.**
- **Gazebo** (cámara de muñeca real; validez por episodio 29–81 %): solo 3 cambios de intención en 11 episodios, de los
  cuales 1 con detección válida. Muestra insuficiente; el análisis de PR #5 sobre 330 episodios dio 102/107 con
  detección válida.

## 7. Validación física en Gazebo (11 episodios, una instancia, dominio 44)

| Política | Semilla | Gazebo | Cinemático (misma semilla) | Observación |
|---|---|---|---|---|
| Experto | `reach` 915000 | Éxito, B | Éxito, B | Igual |
| Experto | `var_m` 915406 | Éxito, **A** | Éxito, **B** | **La misma semilla elige otra rama:** empate de coste + pose inicial medida distinta tras el reset |
| Experto | `var_m` 915481, 915407 | Éxito, A | Éxito, A | Igual |
| bc_v1 | `reach` 915000 | Éxito, A | Éxito, A | Igual |
| bc_v1 | `var_m` 915402 | **Fallo:** B deformada, joint5 en el límite, 88 mm | Fallo idéntico | Modo de fallo reproducido |
| bc_v1 | `var_m` 915453 | Éxito, A | Fallo: va hacia A, vuelve, se estanca en B | Frontera: el lado lo decide una diferencia mínima |
| bc_v2w | `reach` 915000 | Éxito, A | Éxito, A | Igual |
| bc_v2w | `var_m` 915407 | **Fallo:** oscilación de joint4, 300 mm | Fallo (oscilación) | Modo de fallo reproducido |
| bc_v2w | `var_m` 915425 | Éxito, B directo | Éxito tras revertir joint4 (−1,1 → +1,7, desvío de 290 mm) | Otra rama |
| bc_v2w | `var_m` 915405 | Éxito, A | Éxito, A | Igual |

Figura: `gazebo_vs_kinematic.png`.

- Los dos modos de fallo de las políticas se reproducen físicamente.
- La elección de rama, **incluida la del experto**, cambia entre backends en los escenarios cercanos a la frontera.
- No se amplió a 60 episodios: los 11 confirman el diagnóstico cinemático sin contradicciones.

## 8. ¿Etiquetado consistente o política multimodal?

**Prueba directa de viabilidad del etiquetado** (`labelrules`, `label_rules.json`). Para cada regla de elección de
familia (solo entre las factibles), se midió la tasa de pares de estados iniciales cercanos con familias distintas. Es
la contradicción que ve la regresión. **No se ejecutó ningún etiquetador nuevo.**

| Regla | Episodios con A | Contradicciones (radio p50 / p75 / p90) | Pasos extra frente a la familia más rápida (media / máx.) |
|---|---|---|---|
| Experto actual (IK global) | 62 % | **49 % / 47 % / 47 %** | 3,9 / 31 |
| Mínimo L2 (todas las soluciones) | 58 % | 49 % / 47 % / 47 % | 2,6 / 14 |
| Mínimo tiempo | 25 % | 28 % / 31 % / 33 % | 0 / 0 |
| Preferir A salvo infactible | 97 % | **0 % / 0,2 % / 0,3 %** | 7,2 / 31 |
| **Preferir B salvo infactible** | 0 % (B es factible en 620/620) | **0 % / 0 % / 0 %** | 5,0 / 26 |

- Las reglas "naturales" (L2, tiempo) **no** resuelven la contradicción: su frontera cae en medio de la distribución.
- Una **preferencia fija con respaldo por factibilidad la elimina** en el estado inicial.

**Recomendación: etiquetado consistente (alternativa A) primero.**

**Por qué.**

- La contradicción es localizada (decisión inicial) y desaparece con una regla de elección determinista.
- Una regresión unimodal es suficiente para etiquetas sin contradicción: una MLP con MSE no tiene nada que promediar.
- Mantiene el actor compatible con SAC y la infraestructura de PR #3.
- La arquitectura multimodal resolvería un problema que las etiquetas pueden evitar, a cambio de complejidad y de perder
  la compatibilidad directa con SAC.

**Propuesta concreta para la Fase B** (a implementar solo con autorización):

1. **Preferencia por B**:
   - es factible en el 100 % de los escenarios medidos;
   - es la más rápida desde `home` (5,5–11,5 pasos menos);
   - tiene más margen en joint4;
   - en `var_l`, 27/200 escenarios necesitan 30° de inclinación.

   Alternativa a decidir: preferir A, más parecida a lo que ya aprendieron las políticas, pero infactible en 18/620.
2. **Continuidad con histéresis.** Una vez que la muñeca **medida** está comprometida con una familia (la misma
   definición observable de §2), la etiqueta sigue esa familia; nunca fuerza el regreso. Así se evita el fallo del
   etiquetador "canónico" anterior (forzaba rotaciones completas en DAgger). La regla es una función determinista de
   observables: articulaciones medidas y estimación del cubo.
3. **Transición a 30° donde B no existe a 20°**, sin cambiar de familia.
4. Validación: primero las etiquetas, con 0 contradicciones y 0 cambios de rama del nuevo experto, y BC en cinemático.

**Riesgos.**

- **Frontera de compromiso.** Un estado de DAgger que deriva hacia A recibe la etiqueta B hasta cruzar el umbral de
  compromiso y A después: una discontinuidad estrecha en el espacio articular. Si la regresión la difumina, reaparecería
  la indecisión. Es el criterio de paso a la alternativa B.
- **Cambio de distribución** respecto de bc_v1/v2w. Hacen falta demostraciones nuevas (el experto original elige A en
  62 %).
- **El tiempo extra** de la regla (5 pasos de media) no afecta al éxito en el presupuesto del episodio, que el experto
  completa en 35–40 pasos.

**Cuándo pasar a la política multimodal (alternativa B):** si con etiquetas consistentes el BC en cinemático sigue
mostrando cambios de intención o acciones intermedias junto a la frontera de compromiso, o fallos por debajo de bc_v2w.
En ese caso, la opción más simple es una mezcla de dos gaussianas con el modo elegido en t = 0 y mantenido durante el
episodio. Diffusion policy no está justificada por la evidencia actual.

## Limitaciones

- La mayor parte de la evidencia es cinemática. Gazebo se usó para confirmar con 11 episodios elegidos, no para estimar
  tasas.
- Las familias de referencia se calculan con 12 arranques por inclinación más la búsqueda del volteo. La elección del
  experto difiere del mínimo L2 del análisis en 39 de los 582 escenarios con ambas familias (8 en `var_m`, 31 en
  `var_l`): sus 5 arranques fijos no encuentran la solución mínima.
- El análisis de contradicciones usa los estados que visita el experto, no los de las políticas (DAgger).
- `labelrules` mide la coherencia de las etiquetas, no el desempeño de un BC entrenado con ellas.
