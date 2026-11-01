# Ramas de muñeca en `reach` — Fase C (objetivo de IK recalculado en cada paso)

Rama `exp/solverbot-wrist-branch-consistency` (PR borrador #6), desde `43013d7`. Continúa
[`WRIST_BRANCHES.md`](WRIST_BRANCHES.md) (Fase A) y [`WRIST_BRANCHES_FASE_B.md`](WRIST_BRANCHES_FASE_B.md).

## Conclusión

**Sí: recalcular el objetivo de IK en cada paso desde el estado articular medido produjo una política BC más
robusta que las referencias, de forma reproducible en cinemático.** El candidato es **bc_e2**: BC sobre las
demostraciones del etiquetador nuevo más una sola ronda de corrección supervisada.

En la confirmación, con semillas nuevas (960000+ / 961000+) que nunca se usaron para entrenar, diagnosticar ni elegir:

| Criterio acordado | Referencia | bc_e2 | ¿Cumple? |
|---|---|---|---|
| Igualar o superar a bc_v2w en el total (sin `reach`) | 858/890 (856 en la Fase B) | **880/890**; McNemar 32 frente a 10, p = 0,0009 | sí |
| Acercarse a bc_c2 en `reach_var_l` | 198/200 | **197/200**; frente a bc_c2, 2 frente a 3, p = 1 | sí |
| Igualar a bc_v2w con muñeca girada (joint4₀ = 0 / 0,25 / 0,5) | 50 / 48 / 43 | **50 / 49 / 48** | sí |
| Conservar `reach` desde home | 20/20 | 20/20 | sí |
| Sin aumento de colisiones ni de saturaciones | v2w: 0 colisiones, 10 atascos en límite de muñeca | 0 colisiones, 3 atascos | sí |

Regresiones honestas:

- **zona `B30`:** 28/30 frente a 30/30 de bc_v2w. La diferencia pareada es 0 frente a 2, p = 0,5, no significativa.
- **Orientación final:** con el etiquetador nuevo los episodios terminan más cerca del límite de orientación de éxito
  (ver C1). En la confirmación no produjo ningún fallo de bc_e2, pero es el margen más estrecho del candidato.

**Gazebo:** ver C4.

## Qué no se tocó

- `ScriptedExpert` (experto por defecto) y `BranchConsistentLabelExpert` (Fase B).
- BC v1, v2w, c2 y d1, congelados. Se evalúan con sus archivos sin cambios.
- URDF, física, pinza, límites articulares, recompensas, criterios de éxito, ni los espacios de observación (31) y
  acción (7).
- Las PR #3, #4 y #5. La PR #6 sigue en borrador y sin merge.

La política ve solo la observación de 31 dimensiones (el actor oculta `prev_action`, como v1/v2w). La pose
verdadera del cubo solo la usa el etiquetador, igual que el experto original.

## C0 — Diseño del etiquetador (`imitation/dynamic_label_expert.py`, `MeasuredIKLabelExpert`)

En cada paso de la fase de pre-agarre de `reach`:

1. **Candidatas.**
   - Se calculan con IK local, partiendo de varias semillas:
     - la articulación **medida**;
     - su volteo de muñeca;
     - semillas canónicas si falta una familia;
     - la solución del paso anterior, solo como semilla. La elección es siempre por coste.
   - Inclinaciones de 20° y 30°. 40° nunca cumple el criterio de éxito.
2. **joint6 = joint6 medido.** joint6 no mueve el TCP ni el eje de aproximación (verificado), así que la etiqueta nunca
   lo gira. Era la dependencia del arranque que se sospechaba en la Fase B.
3. **Familia.**
   - B si la muñeca medida no está comprometida con A.
   - A si lo está: continuidad, sin forzar el regreso.
   - La otra familia como respaldo si la elegida no tiene candidata válida.
   - Una banda de histéresis evita que el ruido alterne la familia en la frontera.
4. **Coste de selección** (no es la recompensa de Gymnasium): distancia articular a q (joint1..joint5) más una
   penalización suave por proximidad a los límites de joint4/joint5.
5. **Muñeca singular** (|joint5| < 0,12): A y B son el mismo continuo de soluciones. Se toma el punto del continuo más
   cercano a la articulación medida: SLSQP con el TCP en el objetivo y el eje a menos de 0,06 (~3,4°).
6. **Refinamiento de límites** de la candidata elegida. Mismo TCP, eje a menos de ~3,4° de la inclinación nominal, y
   se minimiza la distancia a la solución sin refinar más la penalización de límites.
7. **Validez.**
   - Del objetivo: IK < 3 mm, orientación de éxito, muñeca a más de 0,02 rad de sus límites y cubo visible en la
     cámara de muñeca.
   - Del camino desde q: sin colisión con el cuerpo, autocolisión, mesa ni muñeca en el límite (con
     `branch_feasibility`).
8. **Etiqueta:** paso acotado hacia el objetivo del paso. Sin candidata válida, IK global del experto original (respaldo
   registrado).

### Problemas encontrados durante el diseño (resultados negativos de C0)

| Versión | Problema | Corrección |
|---|---|---|
| v1 | Dos soluciones locales de la misma clave alternaban según la semilla del optimizador (saltos de 0,4 rad). | La solución previa como semilla extra. |
| v1 | **Muñeca singular**: con joint5 ≈ 0, joint4 queda en un continuo y el optimizador se detenía en cualquier punto (saltos de 0,9 rad, objetivo con joint4 = 1,33 con el brazo medido en −0,63). | Familia neutra y proyección SLSQP al punto más cercano. |
| v2 | Penalización de límites solo en la elección discreta: **no cambió nada** (pesos 1, 3 y 6 dieron los mismos márgenes); no había alternativas mejores. | Refinamiento continuo del objetivo. |
| v2 | El refinamiento medido respecto de q deslizaba el óptimo por el borde del cono de holgura (saltos de 0,15–0,4 rad). | Distancia a la solución sin refinar. |
| descartado | Preferencia por 20° (coste 0,3 por 10°): no bajó la inclinación media del objetivo (27°) y reintrodujo saltos de 0,2–0,96 rad. | Revertido. |
| descartado | Holgura del eje solo hacia inclinaciones menores: provocó cambios de inclinación a mitad de episodio (saltos de 0,96 rad). No mejoró la orientación final. | Revertido. |

Las corridas descartadas se conservan en `runs/` (no versionadas): `grid_dynamic_v1_descartado`,
`grid_dynamic_v2_descartado`, `wrist_c1/descartado_v1` y `wrist_c2_descartado_v2`.

**Pruebas:** `tests/test_dynamic_label_expert.py` (9). Cubren:

- B desde home y joint6 sin girar;
- objetivo como función del estado medido: misma q, historia distinta, mismo objetivo;
- continuidad con A;
- robustez a ruido de 0,005 rad;
- sin saltos en la frontera singular (semilla 940401);
- penalización solo en el coste;
- etiquetadores anteriores intactos.

### Verificación sobre la grilla de B0 (`c0_grid.json`)

Brazo en home exacto, cubo en cada una de las 121 celdas.

| Celdas | Etiquetador | Éxito | Familia final | Inclinación | Margen mín. de muñeca | Salto máx. del objetivo | Pasos |
|---|---|---|---|---|---|---|---|
| 88 B@20 | experto original | 88 | A 56 / B 32 | — | 0,071 | — | 38,5 |
| | etiquetador B (Fase B) | 88 | B 88 | 20° | 0,022 | 0 | 33,6 |
| | **nuevo** | **88** | **B 88** | 20° 28 / 30° 60 | **0,151** | 0,057 | 26,7 |
| 27 B@30 (B solo a 30°) | experto original | 27 | A 15 / B 12 | — | 0,304 | — | 32,0 |
| | etiquetador B | 27 | B 27 | 30° | 0,304 | 0 | 20,2 |
| | **nuevo** | **27** | **B 27** | **30°** | 0,304 | 0,011 | 20,3 |
| **6 solo A** | experto original | 6 | A 6 | — | 0,271 | — | 38,0 |
| | etiquetador B | 6 | A 6 | 20° | 0,304 | 0 | 37,3 |
| | **nuevo** | **6** | **A 6** | 30° | 0,304 | 0,015 | 16,7 |

Ninguna colisión ni atasco con ninguno de los tres. El etiquetador nuevo:

- usa A exactamente en las 6 celdas donde B no es viable;
- mantiene B a 30° en las 27 donde B solo existe a 30°;
- sube el margen mínimo de muñeca de 0,022 a 0,151 rad.

## C1 — Calidad de las etiquetas (`c1_labels.json`)

Semillas nuevas 943000–943449: `reach` 20, `var_s` / `var_m` / `var_l` 100 cada uno, y muñeca girada (joint4₀ =
−0,3 / 0 / 0,25 / 0,5) 50 cada una. Son 520 episodios por etiquetador, con el etiquetador en el lazo.

| Métrica | Experto original | Etiquetador B | **Nuevo** |
|---|---|---|---|
| Éxito | 470/520 (joint4₀ = 0,5: **22/50**; 0,25: 32/50) | 519/520 | **520/520** |
| Colisiones | 1 | 1 | **0** |
| Episodios atascados en el límite de muñeca | 2 | 0 | 0 |
| Episodios con muñeca a < 0,05 rad del límite | 13 | 9 | **0** |
| Objetivos con joint4/joint5 a < 0,1 rad del límite (home / girados) | 21,6 % / 25,2 % | 9,9 % / 11,4 % | **0 % / 0 %** |
| Salto máximo del objetivo entre pasos | 0,21 rad | **2,19 rad** | **0,068 rad** |
| Cambios reales de familia ejecutada | — | 1 episodio | 1 episodio |
| Contradicciones entre episodios (radio p50 / primer paso) | — | 0,03 % / 0,4 % | 0,09 % / 0 % |
| Ruido de 0,005 rad: cambio de familia / de inclinación | — | 0 % / 0 % | 0 % / 0,26 % |
| Ruido: cambio de la acción etiquetada (mediana / p95) | — | 0,095 / 0,225 | **0,027 / 0,188** |
| Pasos medios (`var_m`) | 37,6 | 31,7 | **24,1** |
| `tcp_z[2]` final > −0,82, cerca del límite de éxito (home) | 2,5 % | 4,7 % | **47 %** |

**Puerta de C1: superada.** El etiquetador conserva el rendimiento del experto (lo supera con muñeca girada) y elimina
los objetivos y trayectorias junto a los límites de muñeca.

**Riesgo identificado en C1:** la orientación final. El etiquetador nuevo elige 30° con más frecuencia (inclinación
media del objetivo 28°, máximo 33,4°, frente a 20° del etiquetador B). Además, como alcanza la posición antes, el
episodio suele terminar justo cuando la inclinación del camino vuelve por debajo de 36,9°.

- Que la inclinación del camino articular supere 36,9° a mitad de camino es común a ambos etiquetadores (78 % y 83 % de
  los episodios de `var_m`).
- Las dos correcciones probadas no lo resolvieron y se revirtieron (ver C0).

Se dejó como riesgo a vigilar. En la confirmación no apareció ningún fallo por orientación.

## C2 — Datos y modelos

**Demostraciones** (`runs/wrist_c2/demos`, no versionadas). Misma composición que las de bc_d1 para aislar el
efecto del etiquetador:

- 640 episodios, todos con éxito: `reach` 40, `var_s` 150, `var_m` 250, `var_m` girada 150 y `var_s` girada 50;
- joint4 inicial sorteado en [−0,6, 0,6];
- semillas 126000–126849;
- no se mezcló ningún dato de etiquetadores anteriores.

**bc_e1.**

- Receta exacta de bc_d1: actor SB3 SAC (256, 256), `prev_action` oculto, mismos hiperparámetros, sin búsqueda.
- Val. MSE 0,0124.

**Una sola ronda de corrección supervisada** sobre bc_e1 (DAgger, β = 0, el etiquetador nuevo etiqueta los estados
de la política):

- 270 episodios, semillas 127000–127619;
- misma composición que la ronda de bc_d2 en la Fase B;
- la política tuvo éxito en 262 de ellos;
- los episodios fallidos de la política también se usan, como en la Fase B.

**bc_e2.** Demostraciones más esa ronda. Val. MSE 0,064.

No se entrenó ningún otro modelo. Las semillas de selección y confirmación se fijaron antes de entrenar.

## C3 — Evaluación

### Selección (944000+ y banco 945000+, 465 episodios sin `reach`; `eval_sel.json`)

| Escenario | bc_v2w | bc_c2 | bc_d1 | bc_e1 | bc_e2 |
|---|---|---|---|---|---|
| Total sin `reach` | 454 | 364 | 443 | 441 | **460** |
| `var_l` (/100) | 96 | 96 | 90 | 93 | **99** |
| Girados 0,25 / 0,5 (/30) | 30 / 26 | 0 / 0 | 29 / 26 | 25 / 20 | **30 / 26** |

Se eligió **bc_e2** y la decisión se registró (`selection_decision.json`) **antes** de ejecutar la confirmación. bc_e1
también se evaluó en la confirmación como contraste.

### Confirmación (960000+ y banco de zonas 961000+, semillas nuevas; `eval_confirm.json`)

| Escenario | bc_v1 | bc_v2w | bc_c2 | bc_d1 | bc_e1 | **bc_e2** | IC95 de bc_e2 |
|---|---|---|---|---|---|---|---|
| reach | 20/20 | 20/20 | 20/20 | 20/20 | 20/20 | **20/20** | 0,84–1,00 |
| var_s | 192 | 197 | 200 | 200 | 200 | **200** | 0,98–1,00 |
| var_m | 184 | 195 | 200 | 195 | 200 | **200** | 0,98–1,00 |
| var_l | 162 | 188 | **198** | 180 | 186 | **197** | 0,96–0,99 |
| zona A_respaldo | 30 | 30 | 30 | 28 | 28 | **30** | 0,89–1,00 |
| zona B30 | 25 | **30** | **30** | 25 | 29 | 28 | 0,79–0,98 |
| zona frontera_B20 | 24 | 27 | **30** | 25 | 27 | 28 | 0,79–0,98 |
| joint4₀ = −0,3 | 47 | **50** | 42 | 48 | **50** | **50** | 0,93–1,00 |
| joint4₀ = 0 | 48 | **50** | 7 | 49 | **50** | **50** | 0,93–1,00 |
| joint4₀ = 0,25 | 42 | 48 | **0** | 45 | 44 | **49** | 0,90–1,00 |
| joint4₀ = 0,5 | 42 | 43 | **0** | 32 | 33 | **48** | 0,87–0,99 |
| **Total sin `reach`** | 796 | 858 | 737 | 827 | 847 | **880** | 0,979–0,994 |

IC95 del total de bc_v2w: 0,950–0,974.

**McNemar exacto pareado** (episodios ganados solo por la candidata / solo por la referencia):

- **bc_e2 frente a bc_v2w:**
  - total 32 / 10, **p = 0,0009**;
  - `var_l` 12 / 3, p = 0,035;
  - `var_m` 5 / 0, p = 0,06;
  - joint4₀ = 0,5: 7 / 2, p = 0,18;
  - zona B30 0 / 2, p = 0,5;
  - ningún escenario significativamente peor.
- **bc_e2 frente a bc_c2:**
  - total 150 / 7, p < 1e-30;
  - `var_l` 2 / 3, p = 1 (equivalentes);
  - girados 148 / 0.
- **bc_e1 frente a bc_v2w:** 29 / 40, p = 0,23. Sin la ronda de corrección, el etiquetador nuevo **no** basta: bc_e1
  pierde con muñeca girada a 0,5 (33/50).

### Otras métricas y causas de fallo (confirmación, 910 episodios por política; `c3_failure_modes.json`)

| Política | Fallos | Atascado en límite de muñeca | Alternancia de rama | Colisión | Estancado (otro) | Episodios con cambio de intención | Estancados (total) | Tasa de acción media |
|---|---|---|---|---|---|---|---|---|
| bc_v1 | 94 | 62 | 21 | 0 | 11 | 42 | 90 | 0,35 |
| bc_v2w | 32 | 10 | 21 | 0 | 1 | **255** | 32 | 0,55 |
| bc_c2 | 153 | 32 | 11 | **109** | 1 | 32 | 44 | 0,27 |
| bc_d1 | 63 | 55 | 7 | 0 | 1 | 30 | 63 | 0,08 |
| bc_e1 | 43 | 33 | 10 | 0 | 0 | 12 | 42 | 0,07 |
| **bc_e2** | **10** | **3** | 5 | **0** | 2 | 18 | **10** | 0,18 |

- Ningún fallo de bc_e2 se debe a orientación fuera de éxito.
- Fallos de bc_e2 por escenario:
  - `var_l`: 3;
  - zona B30: 2;
  - frontera_B20: 2;
  - joint4₀ = 0,25: 1;
  - joint4₀ = 0,5: 2.
- Pasos medios de bc_e2: 25–28, frente a 44–50 de bc_v2w.
- Margen articular mínimo (p05) de bc_e2: 0,02 rad, frente a 0,0 de bc_v2w.
- Detección del cubo: 0,95 en todas las políticas.
- La alternancia de bc_v2w (255 episodios con cambio de intención) casi desaparece (18).

## C4 — Gazebo

*(pendiente: se completa con los resultados de la corrida física)*

## Recomendación

*(se completa tras C4)*

## Reproducción

| Paso | Comando |
|---|---|
| C0 | `diagnose_wrist_branches.py gridlabel --out runs/wrist_c0/grid_<etiquetador> --labelers <etiquetador>` |
| C1 | `diagnose_wrist_branches.py rollouts --policy dynamic …` (semillas 943xxx) y luego `labelquality`, `noisetest` y `analyze --labeler-prefix kinematic_dynamic` |
| C2 | `collect_demos.py --backend kinematic --labeler dynamic …`; `train_bc.py` (receta por defecto); DAgger con `--policy runs/wrist_c2/bc_e1/model.zip --beta 0` |
| C3 | `boundary_seeds.py --pool 945000-945999 --per-zone 15` y `--pool 961000-961999`; `diagnose_wrist_branches.py rollouts …`; `summarize_wrist_phase_b.py` |

Semillas de la Fase C (todas nuevas):

| Uso | Semillas |
|---|---|
| Grilla | 942000 |
| Verificación de etiquetas | 943000–943449 |
| Demostraciones | 126000–126849 |
| DAgger | 127000–127619 |
| Selección | 944000–944829 y 945000+ |
| Confirmación | 960000–960849 y 961000+ |
| Gazebo | 962000–962329 |
