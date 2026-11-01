# Ramas de muñeca en `reach` — Fase B (etiquetado consistente)

Rama `exp/solverbot-wrist-branch-consistency` (PR borrador #6 hacia `feat/solverbot-reach-bc-sac`). Continúa
[`WRIST_BRANCHES.md`](WRIST_BRANCHES.md) (Fase A, commit `2698473`, sin cambios).

**Resultado en una línea.** Las etiquetas consistentes eliminan la contradicción y la alternancia de rama, pero **ningún
BC nuevo supera a bc_v2w en conjunto**:

- **bc_c2** generaliza mejor en el espacio (`var_l` 198/200 frente a 184/200, p = 0,001), pero colapsa con arranques
  de muñeca girada (45/200 frente a 191/200).
- **bc_d1** recupera esos arranques (182/200), pero pierde frente a bc_v2w en el total (836 frente a 856 de 890,
  p = 0,033).

bc_v2w sigue siendo la referencia. No se declara éxito.

## Qué no se tocó

- `ScriptedExpert`, sigue siendo el experto por defecto (`collect_demos.py --labeler scripted`).
- BC v1 y BC v2w, congelados; no se reentrenaron ni se sobrescribieron.
- URDF, física, pinza, cámaras, recompensas, criterios de éxito, ni los espacios de observación (31) y acción (7).
- Las PR #4 y #5.

Las políticas nuevas ven solo la observación de 31 dimensiones; el actor BC enmascara `prev_action`, como v1/v2w. La
verdad del simulador se usa solo en el etiquetador (como el experto original) y para medir.

## Código nuevo

| Archivo | Qué hace |
|---|---|
| `imitation/branch_feasibility.py` | Factibilidad física por familia × inclinación. Comprueba:<ul><li>IK precisa;</li><li>orientación de éxito;</li><li>márgenes articulares;</li><li>colisión con el cuerpo, autocolisión y mesa a lo largo de la trayectoria del experto (nubes de puntos de los STL y cajas del URDF);</li><li>muñeca libre de límites;</li><li>cubo visible en la cámara de muñeca (marco verificado contra el xacro).</li></ul> |
| `imitation/branch_label_expert.py` | `BranchConsistentLabelExpert(ScriptedExpert)`, opcional.<ul><li>Preferencia B@20 → B@30 → A@20 → A@30, solo entre opciones viables.</li><li>Continuidad: si la muñeca medida ya está comprometida, sigue con esa familia sin forzar el regreso.</li><li>El objetivo es siempre la solución verificada.</li><li>Nunca cambia de familia en silencio.</li><li>En `reach` da el paso acotado hacia ese objetivo.</li></ul> |
| `imitation/start_override.py` | Arranques con joint4 girado (solo cinemático), fijos o sorteados con la semilla. |
| `scripts/diagnose_wrist_branches.py` | Subcomandos nuevos `feasibility` y `rollouts --policy consistent / --start-j4`; figura `feasibility_map`. |
| `scripts/collect_demos.py` | `--labeler consistent`, `--start-j4-range lo,hi`. |
| `scripts/train_bc.py` | Tolerancia de la verificación de carga 1e-6 → 1e-5 (redondeo float32; bc_c1 falló por 1,07e-6). |
| `scripts/summarize_wrist_phase_b.py` | Tablas versionadas de este informe (solo lee resultados). |
| `tests/test_branch_label_expert.py` | 7 pruebas, entre ellas una de regresión del fallo 121128. |

Resultados versionados en [`resultados/wrist_branches/fase_b/`](resultados/wrist_branches/fase_b/).

## B0 — Factibilidad corregida (`feasibility_summary.json`, `feasibility_map.png`)

**Corrección de la Fase A.** "B factible en 620/620" contaba 40° de inclinación y no exigía margen de muñeca. Pero 40°
**nunca** es viable: el éxito exige `tcp_z[2] < −0,8`, es decir, una inclinación menor de 36,9°.

**Grilla del cubo alrededor de home** (11 × 11, paso 14 mm, brazo en home exacto):

| Opción elegida por la regla | Celdas | Dónde |
|---|---|---|
| B@20 | 88 / 121 | centro y la mayor parte de la grilla |
| B@30 (B no existe a 20°) | 27 / 121 | borde izquierdo (dx −7…−4 cm, dy 0…+7 cm) y esquina inferior derecha (dx +1,4…+7 cm, dy −4…−7 cm) |
| **Solo A** (B no viable a ninguna inclinación) | **6 / 121** | arriba al centro (dx −2,8…0 cm, dy +4,2…+7 cm) |
| Ninguna | 0 | — |

**Por qué hacen falta 30°.** En las 27 celdas B@30, B a 20° falla por:

- **IK sin solución precisa** (25 celdas): el cubo queda fuera del alcance de esa orientación;
- **muñeca en el límite** (2 celdas).

Inclinar a 30° acerca el TCP y abre margen. B@30 es, de hecho, la opción con más margen y mejor visibilidad (tabla
siguiente).

**Celdas que necesitan A.** En las 6 celdas de solo A, B falla a 20° (3 por límite de muñeca, 3 por IK) y a 30° (6 por
IK). **A es el respaldo necesario. B no es viable en todo el espacio.**

| Opción viable (todas las filas) | n | Margen mín. medio (rad) | Margen j4 | Margen j5 | Fracción con cubo visible | Colisión cuerpo / auto / mesa |
|---|---|---|---|---|---|---|
| A@20 | 249 | 0,22 | 0,23 | 0,38 | 0,45 | 0 / 0 / 0 |
| A@30 | 171 | 0,38 | 0,46 | 0,78 | 0,72 | 0 / 0 / 0 |
| B@20 | 245 | 0,20 | 0,23 | 0,34 | 0,64 | 0 / 0 / 0 |
| B@30 | 228 | 0,41 | 0,49 | 0,74 | **0,87** | 0 / 0 / 0 |

**Ninguna inviabilidad se debió a colisión, mesa o visibilidad**; todas fueron por IK o por límite de muñeca. El
detector no es trivialmente permisivo: como control positivo, marca 47 de 300 configuraciones aleatorias (15,7 %).

**Elección de la regla en `reach_var_l`** (200 escenarios, semillas 915600+): B@20 157, B@30 35, A@20 8.

## B1 — Etiquetador consistente y verificación de etiquetas (`labels_consistent_v3.json`)

**Verificación** (semillas 917xxx, 320 episodios):

- 320/320 éxitos;
- **0 cambios de familia** del objetivo dentro de un episodio;
- 0 cambios de rama ejecutados;
- 0 cambios de intención.

| Nivel | Éxito | Familia ejecutada A / B / indefinida | Pasos medios |
|---|---|---|---|
| reach | 20/20 | 0 / 20 / 0 | 34,0 |
| var_s | 100/100 | 0 / 100 / 0 | 33,9 |
| var_m | 100/100 | 3 / 94 / 3 | 31,3 |
| var_l | 100/100 | 4 / 93 / 3 | 29,5 |

**Contradicciones** entre episodios distintos (misma métrica que la Fase A):

| Radio | Etiquetador consistente | Experto original |
|---|---|---|
| p10 | 0 % | 49 % (a radio p50) |
| p25 | 0 % | — |
| p50 | 0,05 % (10 de 20 018 pares) | 49 % |
| primer paso | 2,2 % | 38,5 % |

Los residuos son estados iniciales en la frontera B/A, del respaldo por factibilidad.

**Arranques girados** (joint4 inicial fijado en −0,3 / 0 / 0,25 / 0,5, semillas 918500+): el etiquetador obtiene
**200/200**. Sin cambios de muñeca, el experto original obtiene 28/50 con joint4 = 0,5.

### Fallos del etiquetador encontrados y corregidos (también son resultados)

1. **Semilla 121128.** B@30 era viable, pero la búsqueda barata del objetivo no encontraba esa solución, así que el
   etiquetador volvía a la IK original (familia A) sin registrarlo y se estancaba.

   *Corrección:*

   - el objetivo es la solución verificada;
   - nunca hay respaldo silencioso después de elegir;
   - prueba de regresión.

   Las demostraciones se recolectaron de nuevo (`demos_v2`).
2. **Continuidad no verificada y fase cercana inestable.** Con muñeca girada, la continuidad tomaba objetivos sin
   verificar. Además, la IK local de la fase cercana oscilaba junto al límite de joint4, y un intento de enclavamiento
   no lo resolvió. Así se produjo el colapso de bc_c2 con arranques girados (ver B3).

   *Corrección:*

   - la continuidad verifica su familia a 20°/30°;
   - en `reach` la etiqueta siempre es el paso hacia el objetivo verificado;
   - se añadieron arranques girados en las demostraciones (`demos_v3`).

## B2 — Datos y modelos

**Demostraciones** del etiquetador consistente (todas en `runs/wrist_b2/`, no versionadas).

- No se mezclaron las demostraciones antiguas, que eran contradictorias.
- Semillas de demostración: 120000–122039 y 124000+ (girados). Semillas de DAgger: 123000+ y 125000+.

| Conjunto | Episodios | Contenido |
|---|---|---|
| demos_v2 | 440 | reach, var_s, var_m (etiquetador tras la corrección 1) |
| demos_v2 + DAgger(bc_c1) | 610 | + 170 estados de bc_c1 etiquetados |
| demos_v3 | 640 | demos_v2 con el etiquetador tras la corrección 2, **+ 200 con joint4 inicial sorteado en [−0,6, 0,6]** |
| demos_v3 + DAgger(bc_d1) | 910 | + 270 estados de bc_d1 etiquetados (incluye girados) |

**Modelos.** Todos usan el mismo actor SB3 SAC (256, 256), `prev_action` oculto y la misma receta de `train_bc.py`.
Por eso son compatibles con SAC.

| Modelo | Datos | Val. MSE | Observación |
|---|---|---|---|
| bc_c1 | demos_v2 | 0,018 | — |
| bc_c1w | demos_v2, `weight_near` 3 | 0,019 | **resultado negativo**: peor que c1 en `var_m` (87 frente a 96) |
| bc_c2 | demos_v2 + DAgger(c1) | 0,108 | mejor en espacio; sin cobertura de arranques girados |
| bc_d1 | demos_v3 | 0,0097 | — |
| bc_d2 | demos_v3 + DAgger(d1) | 0,079 | **resultado negativo**: `var_m` 74/100, `var_s` 91/100 |

El primer bc_c1 falló la verificación de carga por 1,07e-6 (redondeo float32). Se guardó en `bc_c1_failed_loadcheck`
y no se usó.

## B3 — Evaluación cinemática (`eval_val.json`, `eval_final.json`, `eval_confirm.json`)

**Diseño.**

- Mismas semillas para todas las políticas; política autónoma, sin experto en el lazo; cubo por percepción.
- **Validación** (918xxx) para elegir; **final** (920000+) y **confirmación** (930000+) reservadas, sin usar antes.
- Escenarios:
  - `reach` (20);
  - `var_s`, `var_m`, `var_l` (200 cada uno);
  - **zonas de frontera de factibilidad**, 30 cada una: `A_respaldo` (B inviable), `B30` (B solo a 30°) y
    `frontera_B20`. Se tomaron de un banco de semillas aparte (921000+ / 931000+) clasificadas con B0;
  - **arranques en la frontera de compromiso**: `var_m` con joint4 inicial = −0,3 / 0 / 0,25 / 0,5 (50 cada uno).

### Confirmación (930000+): éxitos

| Escenario | bc_v1 | bc_v2w | bc_c2 | bc_d1 |
|---|---|---|---|---|
| reach | 20/20 | 20/20 | 20/20 | 20/20 |
| var_s | 193 | 198 | **200** | **200** |
| var_m | 185 | 195 | **200** | 195 |
| var_l | 170 | 184 | **198** | 181 |
| zona A_respaldo | 28/30 | 30/30 | 30/30 | 27/30 |
| zona B30 | 24/30 | 29/30 | 29/30 | 25/30 |
| zona frontera_B20 | 26/30 | 29/30 | 30/30 | 26/30 |
| joint4₀ = −0,3 | 47/50 | **50/50** | 41/50 | 48/50 |
| joint4₀ = 0 | 46/50 | **50/50** | **4/50** | 49/50 |
| joint4₀ = 0,25 | 43/50 | **48/50** | **0/50** | 46/50 |
| joint4₀ = 0,5 | 43/50 | 43/50 | **0/50** | 39/50 |
| **Total sin `reach`** | 805/890 | **856/890** | 732/890 | 836/890 |
| Sin arranques girados (incl. `reach`) | 646/710 | 685/710 | **707/710** | 674/710 |

### Final (920000+): mismo patrón

- **bc_v2w 857, bc_d1 833, bc_c1 797, bc_v1 786, bc_c2 732.**
- bc_c2: 199/200 en `var_l` y 200/200 en `var_s`/`var_m`, con arranques girados 39 / 4 / 1 / 0.

### McNemar exacto pareado contra bc_v2w (confirmación)

`cand / ref` = episodios ganados solo por la candidata / solo por bc_v2w.

| Candidata | Total sin `reach` | var_l | var_m | Arranques girados (0 / 0,25 / 0,5) |
|---|---|---|---|---|
| bc_c2 | 25 / 149, p ≈ 1e-22 (**peor**) | 16 / 2, **p = 0,001 (mejor)** | 5 / 0, p = 0,06 | 0/46, 0/48, 0/43, p < 1e-12 (**peor**) |
| bc_d1 | 30 / 50, **p = 0,033 (peor)** | 12 / 15, p = 0,70 | 5 / 5, p = 1 | 0/1, 2/4, 7/11 (n.s.) |
| bc_v1 | 24 / 75, p < 1e-6 (peor) | 11 / 25, p = 0,03 | 4 / 14, p = 0,03 | n.s. |

En la evaluación final, bc_d1 frente a bc_v2w da 30/54 (p = 0,012). La conclusión se repite en dos conjuntos de
semillas reservadas independientes.

### Otras métricas (confirmación, todos los escenarios)

| Política | Familia final A / B / indef. | Episodios con cambio de intención | Cambios de rama reales | Fallos atascados en límite de muñeca | Fallos con cambio de intención | Estancados | Colisiones | Tasa de acción media (reach / var_m) |
|---|---|---|---|---|---|---|---|---|
| bc_v1 | 823 / 77 / 10 | 27 | 0 | 57 | 20 | 82 | 0 | 0,20 / 0,27 |
| bc_v2w | 839 / 69 / 2 | **257** | 7 | 10 | 30 | 31 | 0 | 0,61 / 0,53 |
| bc_c2 | 158 / 670 / 82 | 30 | 4 | **141** | 18 | 60 | **98** | 0,28 / 0,27 |
| bc_d1 | 59 / 813 / 38 | 21 | 2 | 46 | 7 | 54 | 0 | **0,07 / 0,07** |

Comunes a todas las políticas:

- detección del cubo de 0,93–0,96;
- margen articular mínimo (p05) de 0–0,05 rad (todas tocan algún límite en algún episodio);
- distancia final media del TCP de 21–38 mm en los éxitos de `reach`/`var_l`.

Detalle por escenario en `eval_confirm.json`.

### Lectura

1. **Las etiquetas consistentes hacen lo que debían.**
   - Las políticas nuevas ejecutan B de forma estable.
   - Casi no alternan: 21–30 episodios con cambio de intención frente a 257 de bc_v2w.
   - El movimiento es mucho más suave: la tasa de acción baja de 0,53–0,61 a 0,07 con bc_d1.
   - El fallo de la Fase A (indecisión en la frontera de compromiso) prácticamente desaparece.
2. **El fallo dominante cambió: ahora es la saturación de la muñeca, no la alternancia.**
   - bc_d1: 46 de 54 fallos tienen joint5 (47) o joint4 en su límite; solo 7 tienen cambio de intención.
   - bc_c2 con arranques girados ejecuta A (sin datos de ese régimen) y se atasca en el límite de muñeca en 141 de 158
     fallos.
   - **bc_c2 además colisiona** en 98 episodios, todos con arranque girado. Es un riesgo físico que bc_v1/v2w no
     presentan.
3. **La generalización espacial mejora con etiquetas consistentes y DAgger** (bc_c2: `var_l`, zonas de frontera). En la
   zona `A_respaldo`, bc_c2 obtiene 30/30, así que la regla no perjudica las posiciones donde A es necesaria. Pero esto
   solo vale sin arranques girados.
4. **Añadir arranques girados (bc_d1) cubre ese régimen pero cuesta generalización espacial** (`var_l` 181,
   zonas 78/90). Además, **DAgger sobre bc_d1 (bc_d2) empeora**:
   - `var_m` cae a 74/100;
   - joint6 deriva y joint3 va a su límite.

   Hipótesis, apoyada por el código: el etiquetador fija `q_goal` en la primera decisión del episodio y luego lo reutiliza.
   joint6 es libre (rotación alrededor del eje de aproximación) y su valor en la solución depende del arranque. Por eso dos
   estados observados iguales pueden recibir objetivos distintos según de dónde vino el episodio: **las etiquetas no son
   función de la observación**. DAgger amplifica ese ruido en los estados que visita la política. **No está corregido.**

## B4 — Gazebo

- **Regresión de agarre del experto original** (`grasp_regression_gazebo.json`):
  - ejecutada con `tools/headless/run_experiment.sh 2 validate`, una sola instancia (ROS_DOMAIN_ID 44,
    GZ_PARTITION `solverbot_sim_2`), semillas 40000–40004;
  - **5/5** levantamientos sostenidos;
  - subida mínima 84,8 mm;
  - error máximo del mimic de la pinza 1,93 mm.

  El experto y la física de agarre siguen intactos.
- **No se llevó ninguna política nueva a Gazebo.** El criterio era pasar *claramente* la evaluación cinemática, y ninguna
  lo hace:
  - bc_c2 pierde 149 episodios pareados;
  - bc_d1 es significativamente peor que bc_v2w en dos conjuntos reservados.

  No se declara robustez física de nada nuevo.

## Recomendación

1. **Mantener bc_v2w como política de referencia.** Ninguna candidata de la Fase B la supera sin regresiones mayores.
2. **No pasar todavía a una política de mezcla.** El criterio acordado era "etiquetas consistentes, pero la política
   sigue promediando o alternando".
   - La primera parte se cumple: 0 contradicciones.
   - La segunda **no**: las políticas nuevas alternan 10 veces menos que bc_v2w, y sus fallos son de saturación de
     muñeca y de cobertura de arranques, no de elección de modo.

   Una mezcla con el modo fijado en t = 0 no ataca ninguno de esos dos fallos.
3. **Siguiente paso propuesto (requiere autorización): etiquetas markovianas.** Recalcular en cada paso el objetivo a partir de la
   articulación **medida**, en lugar de reutilizar el `q_goal` del inicio:
   - la familia se mantiene por continuidad, como ahora;
   - joint6 se toma lo más cerca posible del valor medido;
   - la elección de la solución penaliza la proximidad a los límites de joint4/joint5.

   Después: demostraciones `v4` con arranques girados, un BC (receta de bc_d1) y **una** ronda de DAgger, evaluados con
   el mismo protocolo sobre semillas reservadas nuevas.

   Criterio de parada: si no iguala a bc_v2w en arranques girados **y** a bc_c2 en `var_l`/zonas, la familia B
   consistente no aporta lo suficiente frente a bc_v2w con este actor, y se cierra la línea.
4. **No** se propone diffusion ni SAC prolongado: no hay evidencia de que el cuello de botella sea la expresividad del
   actor o la falta de exploración.

## Limitaciones

- Toda la comparación de políticas es cinemática. En Gazebo solo se verificó el experto original (5 episodios de
  regresión).
- Las zonas de frontera tienen 30 episodios cada una; las diferencias de ±3 no son significativas.
- Las semillas de validación (918xxx) se usaron para elegir entre c1/c1w/c2/d1/d2. Las conclusiones se apoyan en final y
  confirmación, que no se usaron para ninguna decisión de entrenamiento.
- Con 30 episodios por zona y sin Gazebo, la afirmación de que B@30 da más visibilidad (0,87) se apoya en el modelo
  virtual de la cámara de B0, no en imágenes reales.
