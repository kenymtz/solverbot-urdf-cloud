# Reach por imitación (BC) + SAC — experimento `reach_bc_v1`

Rama `feat/solverbot-reach-bc-sac` (base `04a0dbc`). Solo simulación. Sin cambios en física, URDF, pinza, fricción,
DART, recompensas, criterios de éxito ni experto guionado.

## Fase A — Auditoría (evidencia)

### Por qué `sac_reach_baseline_v1` no aprendió

Descomposición de la recompensa de sus 84 episodios (`runs/sac_reach_baseline_v1/episodes.jsonl`, WSL):

| cuartil | retorno | pen_action_rate | pen_limit | pen_time | pen_no_progress | shaping |
|---|---|---|---|---|---|---|
| 1 | −4,27 | −1,95 | −0,55 | −0,58 | −1,0 | −0,18 |
| 2 | −4,42 | −2,07 | −0,59 | −0,59 | −1,0 | −0,18 |
| 3 | −4,59 | −2,24 | −0,57 | −0,64 | −1,0 | −0,14 |
| 4 | −4,40 | −1,84 | −0,87 | −0,56 | −1,0 | −0,13 |

1. **La política seguía siendo ruido al final.** `pen_action_rate` por paso = 0,0340; el valor esperado para acciones
   uniformes en [−1, 1] es 0,05·E[(a−a′)²] = 0,05·2/3 = 0,0333. `ent_coef` todavía 0,30 en el paso 5000.
2. **Presupuesto de muestras ~12× insuficiente.** El mismo entorno en backend cinemático necesitó 60 000 pasos de PPO
   para 50/50 en `reach` (`docs/resultados/kin_reach_model_final.json`); la política aleatoria allí obtiene −4,62
   (`kin_reach_random.json`), igual que el SAC baseline (−4,40). 5 000 pasos (≈ 40 min a 2 pasos/s) no distinguen SAC
   del azar: no prueban que la tarea sea inalcanzable por RL.
3. **Señal hacia el objetivo pequeña frente a las penalizaciones.** El shaping máximo de un episodio exitoso es
   ≈ w_reach·d0 = 2·0,17 ≈ 0,34 (+10 de bono, nunca observado). La penalización por variación de acciones de una
   política exploratoria vale ≈ 2 por episodio: domina el retorno (≈ 45 %). No se cambió la recompensa (restricción);
   se evita el problema de exploración con demostraciones.
4. **Truncado por falta de progreso:** 84/84 episodios (40 pasos sin mejorar el máximo del potencial en 1e-3). Con una
   exploración que no se acerca, el episodio dura ~40–60 pasos; no es la causa primaria sino un síntoma de 1–3.
   El experto nunca dispara esta condición (0/100).
5. **Escala de observaciones.** En los datos de Gazebo: desvío de `obj_rel` (objeto − TCP) = 0,048 m frente a 0,22 de
   las articulaciones normalizadas y 0,37 de la acción previa. La característica decisiva para `reach` tiene la menor
   escala; SB3 `MlpPolicy` no normaliza. El BC usa normalización fija dentro del modelo (`NormalizedObs`).
6. **Percepción (nuevo, medido en Gazebo con el experto, 100 episodios):** detección válida solo en el **56 %** de los
   pasos y error medio 10,5 mm (≈ 5 mm en reposo, hasta ~25 mm con la muñeca en movimiento rápido). Causa: la
   trayectoria del experto interpola en espacio articular hacia una solución de IK global y primero **aleja** el TCP
   (p. ej. 126 → 162 mm) girando la muñeca; el cubo sale del campo de la cámara de muñeca durante ~20 pasos. Durante
   esos pasos la observación conserva la última estimación válida (en el frame base, sigue siendo útil) y la bandera
   `obj_valid = 0`. La precisión es suficiente para `reach` (radio de éxito 30 mm) pero la disponibilidad no es 100 %.
7. **Currículo.** El nivel 0 tiene `object_xy_range: 0` (una sola geometría; las semillas solo cambian el yaw del
   cubo) y promueve con 80 % de éxito en 50 episodios: un SAC que nunca tiene éxito nunca recibe el bono de 10 y no
   hay un nivel intermedio más fácil (p. ej. arranque más cercano). Para imitación se agregaron niveles con variación
   real (abajo); los originales no se tocaron.

### Riesgos de imitación identificados en el código
* El experto devuelve acciones **saturadas** (39 % de las componentes del brazo en ±1) y **ceros** (55 %): `atanh(±1)`
  es infinito → no se puede ajustar la media del actor gaussiano por máxima verosimilitud ingenua.
* La observación incluye la acción previa y las acciones del experto están muy autocorrelacionadas: riesgo de
  "política copiona" (repetir a_{t−1}).
* Multimodalidad de la IK: verificado que la configuración final varía suavemente con la posición del cubo en las
  demostraciones (misma rama).
* SB3 usa acciones **uniformes aleatorias** mientras `num_timesteps < learning_starts` y un modelo nuevo tiene
  `ent_coef = 1`: ambos destruirían una inicialización BC.

## Diseño

| Componente | Archivo |
|---|---|
| Experimento (niveles agregados, semillas disjuntas) | `solverbot_rl/experiments/reach_bc_v1.yaml`, `imitation/experiment.py` |
| Perturbación inicial opcional por nivel (`start_q_noise`, por defecto ausente) | `envs/pick_place.py` |
| Episodios con registro completo | `imitation/rollout.py` |
| Dataset (npz + index.jsonl, validación) | `imitation/dataset.py` |
| BC sobre el actor SAC de SB3 | `imitation/bc.py`, `scripts/train_bc.py` |
| Recolección (experto / DAgger) | `scripts/collect_demos.py` |
| Evaluación autónoma | `scripts/eval_reach.py` |
| Fine-tuning SAC | `imitation/finetune.py`, `scripts/finetune_sac.py` |
| Cola headless | `tools/headless/queue.sh`, modo `py` de `run_experiment.sh` |

Niveles experimentales (agregados al final; índices 0–5 intactos y nivel 0 con reset idéntico, probado):

| nivel | cubo (±m, cuadrado) | pose inicial (±rad por articulación) |
|---|---|---|
| `reach` (original, 0) | 0 | 0 |
| `reach_var_s` | 0,03 | 0,05 |
| `reach_var_m` | 0,05 | 0,10 |

Semillas: demostraciones 100000+, evaluación 900000+ (nunca usadas para entrenar), fine-tuning: episodios 500000+,
evaluación periódica 950000+.

**BC compatible con SAC por construcción.** Se crea un `stable_baselines3.SAC` real con `BCSACPolicy` y se entrena
por supervisión `model.actor` (la misma red que optimiza SAC). Media: MSE entre `tanh(mu(s))` y la acción del experto
(espacio normalizado [−1, 1], sin atanh de etiquetas saturadas). Dispersión: NLL gaussiana de
`atanh(clip(a, ±0,995))` con `mu` detenida, que con el término constante es la verosimilitud exacta de la gaussiana
comprimida por tanh de SAC → `log_std` inicial coherente. Normalización (media/desvío fijos del conjunto de
entrenamiento) como buffers del `features_extractor`: el `.zip` acepta la observación cruda. El actor no ve la acción
previa (máscara); el crítico ve todo. Separación train/val por **episodios**; early stopping por MSE de validación.

Interfaz: `from solverbot_rl.imitation.bc import load_policy; act = load_policy('model.zip'); a = act(obs31)`.

## Resultados (Gazebo Harmonic + DART, WSL, 2026-10-27)

Todo lo de esta sección es **físico** (Gazebo, no backend cinemático), con `object_source: perception` (cámara de
muñeca), política determinista y, para los modelos aprendidos, **sin el experto instanciado**. Semillas de evaluación
900000+ (nunca usadas para entrenar). Tabla completa: `runs/reach_bc_v1/report/table.md` (generada por
`report_reach.py`; los `runs/` no se versionan).

### Demostraciones
* Experto guionado (`ScriptedExpert`, sin modificar) en `reach_var_m`: **100/100 éxitos**, 0 colisiones, 0 NaN, estructura
  válida en todos; 40,9 pasos y 26,7 mm finales de media; detección válida 56 % de los pasos, error de percepción 10,5 mm.
* DAgger 1 (BC v1 ejecuta, `ReachLabelExpert` etiqueta): 40 episodios `var_m` (política 33/40) + 20 `var_s` (17/20).
* DAgger 2 (BC v2 ejecuta): 30 `var_s` (25/30) + 30 `var_m` (27/30).

### Behavior Cloning

| modelo | datos | episodios / muestras | MSE val | MSE train | MSE de predecir 0 |
|---|---|---|---|---|---|
| bc_v1 | experto | 100 / 4094 | 0,023 | 0,022 | 0,466 |
| bc_v2 | experto + DAgger 1 (`var_m`) | 140 / 6135 | 0,110 | 0,033 | 0,614 |
| bc_v2w | ídem, peso extra cerca del objetivo | 140 / 6135 | 0,112 | 0,034 | 0,614 |
| bc_v3 | experto + DAgger 1 + DAgger 2 | 220 / 10001 | 0,117 | 0,044 | 0,610 |

La MSE de validación **no predice** el desempeño físico (bc_v1 tiene la menor y es la peor en `var_m`): la evaluación
en Gazebo es imprescindible.

### Evaluación autónoma (éxitos / episodios, IC 95 % de Wilson)

| política | `reach` (nivel 0 original) | `reach_var_s` | `reach_var_m` | `reach_var_l` (extrapolación) |
|---|---|---|---|---|
| experto guionado (referencia, usa verdad) | 20/20 | 30/30 | 30/30 | 29/30 |
| aleatoria | 0/20 | — | 0/30 | — |
| SAC desde cero, 5k pasos (`sac_reach_baseline_v1`) | 0/20 (150 pasos, 144,8 mm) | — | 0/30 | — |
| **bc_v1** | **20/20** [84 %, 100 %] | 22/30 | 24/30 | 16/30 |
| bc_v2 | 4/20 | 25/30 | 30/30 | 30/30 |
| **bc_v2w** | **20/20** [84 %, 100 %] | 24/30 | **30/30** [89 %, 100 %] | — |
| bc_v3 | 0/20 | 29/30 | 30/30 | 24/30 |

0 colisiones en todas las corridas. Cuidado al leer `reach`: con `object_xy_range: 0` sus 20 semillas son la misma
geometría (solo cambia el yaw del cubo); los niveles `var_*` son los escenarios realmente distintos.

### Confirmación en semillas nuevas (901000+, después de elegir modelos)

Elegir el "mejor" modelo mirando las semillas 900000+ introduce sesgo de selección; los modelos elegidos se volvieron a
evaluar en semillas nunca vistas (`docs/resultados/reach_bc_v1/confirmacion/`):

| política | `reach` | `reach_var_s` | `reach_var_m` | `reach_var_l` (extrapolación) |
|---|---|---|---|---|
| **bc_v1** | **20/20** [84 %, 100 %] | — | — | — |
| bc_v2w | 15/20 | 23/30 | **30/30** [89 %, 100 %] | **28/30** [79 %, 98 %] |

* **bc_v1 resuelve el nivel original `reach` 40/40** sumando ambas rondas (IC 95 % [91 %, 100 %]): 28,7 mm finales,
  Z del TCP −0,93, 47 pasos, sin colisiones ni truncados, experto no instanciado, percepción real.
* El sesgo de selección era real en `reach` para bc_v2w (20/20 → 15/20).
* 12 de los 14 fallos de bc_v2w en semillas nuevas son el cambio de rama de muñeca (joint4 −1 y después +1); los 2
  restantes son de extrapolación (`var_l`).

### Diagnóstico de fallos (con evidencia)
1. **bc_v1 — se queda en el borde.** Los 6 fallos de `var_m` alcanzan 30,3–33,1 mm (umbral 30) con buena orientación y
   luego derivan hasta saturar una articulación de muñeca. Causa: el experto termina el episodio apenas cruza 30 mm,
   las demostraciones no contienen "seguir acercándose"; con ~5 mm de error de percepción, quedar en el borde es
   frágil. DAgger 1 lo corrige: bc_v2 resuelve esos 6 escenarios (30/30; 6 pares discordantes, todos a favor,
   McNemar exacto p ≈ 0,031).
2. **bc_v2 / bc_v3 — cambio de rama de muñeca.** La muñeca es redundante (yaw libre) y la IK global del experto elige
   entre dos ramas equivalentes de forma arbitraria: joint4 arranca positivo en 74/100 demostraciones y negativo en
   26/100 (configuraciones finales joint4/joint5 ≈ +1,53/+0,93 frente a −1,39/−0,82 rad). Trazas: la política arranca
   por una rama, a mitad de camino cambia a la otra (joint4 de −1 a +1), la pinza se inclina hasta la horizontal, la
   cámara pierde el cubo y el episodio se trunca. Desde `home` exacto la política queda sobre la frontera entre ramas:
   bc_v2 resuelve las semillas 900000–003 y falla 004–019 de forma idéntica (misma geometría, solo cambia el yaw del
   cubo). Cada ronda de DAgger con el etiquetador local mejora `var_s` (73 % → 83 % → 97 %) y empeora `home` exacto
   (20 → 4 → 0 de 20), porque el etiquetador local también es bimodal cerca de `home`.
3. **Intentos descartados con evidencia** (cinemático, rápido): (a) etiquetador con IK local en lugar del experto
   original — igual de bimodal (joint4 inicial 46/49), MSE peor; (b) etiquetador "canónico" (IK sembrada en una
   configuración fija de la rama mayoritaria) — unimodal (160/160 éxitos del experto, joint4 inicial 40/0) pero BC
   36/32/29 de 40 y, tras una ronda de DAgger, 10/12/17 de 40 (truncado por falta de progreso al forzar la rotación
   completa de la muñeca desde la otra rama). No se llevó a Gazebo.
4. **SAC desde cero estaba ciego.** Su acción de pinza determinista es −0,04 → apertura ordenada 16,6 mm < 25 mm, umbral
   por debajo del cual el backend no acepta detecciones: detección válida 4 % de los pasos en `reach` (0 % con la
   pinza cerrada, 96 % de los pasos). Un actor SAC nuevo produce ≈ 0 en todas las dimensiones: arranca ciego y la
   recompensa de `reach` no da señal sobre la pinza. BC aprende −1,00 del experto.

### Fine-tuning SAC (Fase F)

Procedimiento ajustado primero en cinemático (`/tmp`, no versionado): con SAC puro sobre el actor BC, la política
pasa de 10/10 a 0/10 tras ~50 actualizaciones del actor (Adam mueve todos los pesos ~lr por paso; el crítico, entrenado
casi solo con acciones del experto, sobreestima Q fuera de distribución). Con regularización BC fuerte (λ = 30),
normalización TD3+BC (α = 1), lr del actor ×0,1, 5000 actualizaciones offline del crítico y 2000 online con el actor
congelado, se sostiene en 18–20/20 durante 5000 pasos, **sin mejora**.

En Gazebo, desde bc_v2, 4000 pasos (`runs/reach_bc_v1/ft_v1`):

| paso | actualizaciones del actor | éxitos (10 escenarios fijos 950000+) | dfinal medio |
|---|---|---|---|
| 0 (= bc_v2) | 0 | 10/10 | 28,1 mm |
| 1000 | 0 | 10/10 | 28,7 mm |
| 2000 | 0 | 10/10 | 27,8 mm |
| 3000 | 1000 | 10/10 | 28,4 mm |
| 4000 | 2000 | 9/10 (1 sin progreso) | 29,6 mm |

Verificado: actor inicial idéntico al BC (77 582 parámetros, |Δa| = 0), actor intacto durante el calentamiento,
el actor recibe gradiente después (cambio máx. 0,033), checkpoints numerados y replay buffer reanudables (probado
300 → 450 en cinemático y por `test_finetune_freeze_update_and_resume`). Episodios de entrenamiento: 14/20, 17/20,
15/20, 17/20 por cuartil (política estocástica). **Conclusión: el fine-tuning es estable pero no mejora el alcance
en 2000 actualizaciones; la evaluación del paso 4000 registra un fallo (no significativo con n = 10). No se
recomienda extenderlo hasta resolver la multimodalidad de las etiquetas**, que es la limitación dominante.

## Regresión
* `pytest solverbot_rl/tests`: 47/47 (37 previas + 10 nuevas en `test_imitation.py`: estructura y corrupción del
  dataset, ausencia de información privilegiada, separación por episodios, normalización y máscara, entrenamiento,
  guardado/carga/inferencia, congelamiento del actor, actualización del crítico y reanudación del fine-tuning, y
  reset del nivel 0 idéntico con la configuración experimental).
* Espacios: observación `Box(31,)` y acción `Box(7,)` sin cambios; `start_q_noise` ausente = comportamiento original.
* Agarre físico (experto validado, `validate_grasp.py`, semillas 40000–40004, Gazebo): **5/5** levantamientos
  sostenidos, elevación 84,8–99,9 mm, error mimic máx. 1,93 mm. No se tocó física, URDF, pinza ni recompensas.
* Checkpoints existentes: `runs/sac_reach_baseline_v1/model_final.zip` se cargó y evaluó sin cambios (0/20, 0/30).

## Conclusión y próximos pasos

**Hito cumplido para el nivel original:** una política aprendida (bc_v1), sin el experto en el lazo y con observaciones
de percepción, resuelve físicamente `reach` en Gazebo en 40/40 episodios de evaluación. Con variación controlada,
bc_v2w (BC + 1 ronda de DAgger) resuelve 30/30 en `reach_var_m` y 28/30 en extrapolación (semillas nuevas).

**No resuelto:** ningún modelo es robusto a la vez en `home` exacto y en `reach_var_s`. Causa identificada: la muñeca es
redundante y el experto elige rama de forma arbitraria cerca de `home`; la regresión por MSE promedia ramas. SAC
fine-tuning no lo corrige (estable, sin mejora). Opciones con justificación, en orden:
1. Política multimodal (mezcla de gaussianas / diffusion policy) para el BC, o
2. etiquetado con rama consistente que no fuerce rotaciones completas (p. ej. elegir la rama por la acción del
   paso anterior de la política dentro de una histéresis), validado primero en cinemático, o
3. eliminar la redundancia en la definición de la tarea (fijar el yaw de aproximación en `reach`) — exige una
   configuración experimental nueva y revisar que no cambie el criterio de éxito.
Para SAC desde cero: arrancar con la acción de pinza en −1 (o excluirla en `reach`) para no cegar la percepción, y
presupuestos ≥ 60k pasos (≈ 8 h a 2 pasos/s).

## Reproducción en WSL / PC dedicada (headless)

Requisitos: los de `tools/headless/README.md` (ROS 2 Jazzy, `~/solverbot_ws` con `solverbot_sim`, venv con SB3 ≥ 2.3
y PyTorch). Todo desde la raíz del repositorio. Cada línea de los archivos `solverbot_rl/experiments/reach_bc_v1_jobs/*.txt`
es un trabajo con su propia simulación headless limpia (`tools/headless/queue.sh`); las instancias 1, 2, 3 usan
`ROS_DOMAIN_ID` 43, 44, 45 (nunca 10). Todos los pasos son reanudables (semillas ya registradas se saltean).

```bash
source tools/wsl/sim_env.sh 1 >/dev/null; python3 -m pytest -q solverbot_rl/tests      # 47+ pruebas, sin simulador

# 1. Demostraciones del experto en Gazebo (2 instancias en paralelo, ~13 min cada una)
tools/headless/run_experiment.sh 1 py runs/reach_bc_v1 collect_demos.py --backend gazebo --experiment reach_bc_v1 \
    --level reach_var_m --seeds 100000-100049 --out runs/reach_bc_v1/demos/expert_var_m
tools/headless/run_experiment.sh 2 py runs/reach_bc_v1 collect_demos.py --backend gazebo --experiment reach_bc_v1 \
    --level reach_var_m --seeds 100050-100099 --out runs/reach_bc_v1/demos/expert_var_m_b

# 2. Behavior Cloning (CPU, < 1 min)
python3 solverbot_rl/scripts/train_bc.py --data runs/reach_bc_v1/demos/expert_var_m runs/reach_bc_v1/demos/expert_var_m_b \
    --out runs/reach_bc_v1/bc_v1

# 3. Evaluación autónoma + líneas base (semillas 900000+)
setsid nohup tools/headless/queue.sh 1 solverbot_rl/experiments/reach_bc_v1_jobs/eval_round1_a.txt > ~/.solverbot/queue_1.out 2>&1 &
setsid nohup tools/headless/queue.sh 2 solverbot_rl/experiments/reach_bc_v1_jobs/eval_round1_b.txt > ~/.solverbot/queue_2.out 2>&1 &

# 4. DAgger ronda 1 (BC v1 ejecuta, el experto local etiqueta) y BC v2 / v2w
tools/headless/queue.sh 3 solverbot_rl/experiments/reach_bc_v1_jobs/dagger_round1.txt
D=runs/reach_bc_v1/demos
python3 solverbot_rl/scripts/train_bc.py --data $D/expert_var_m $D/expert_var_m_b $D/dagger1_var_m --out runs/reach_bc_v1/bc_v2
python3 solverbot_rl/scripts/train_bc.py --data $D/expert_var_m $D/expert_var_m_b $D/dagger1_var_m --weight-near 3 \
    --out runs/reach_bc_v1/bc_v2w
tools/headless/queue.sh 4 solverbot_rl/experiments/reach_bc_v1_jobs/eval_round2_bc_v2.txt     # y eval_round2_bc_v2w.txt

# 5. DAgger ronda 2 (BC v2 ejecuta) y BC v3
tools/headless/queue.sh 1 solverbot_rl/experiments/reach_bc_v1_jobs/dagger_round2_a.txt       # y dagger_round2_b.txt
python3 solverbot_rl/scripts/train_bc.py --data $D/expert_var_m $D/expert_var_m_b $D/dagger1_var_m $D/dagger1_var_s \
    $D/dagger2_var_s $D/dagger2_var_m --out runs/reach_bc_v1/bc_v3
tools/headless/queue.sh 1 solverbot_rl/experiments/reach_bc_v1_jobs/eval_round4_a.txt         # y eval_round4_b.txt

# 6. Extrapolación, confirmación en semillas nuevas y fine-tuning SAC (reanudable)
tools/headless/queue.sh 1 solverbot_rl/experiments/reach_bc_v1_jobs/eval_round3_ood.txt
tools/headless/queue.sh 1 solverbot_rl/experiments/reach_bc_v1_jobs/confirm_a.txt             # confirm_b, confirm_bc_v1
tools/headless/queue.sh 3 solverbot_rl/experiments/reach_bc_v1_jobs/finetune_v1.txt

# 7. Tablas y gráficas
python3 solverbot_rl/scripts/report_reach.py --eval runs/reach_bc_v1/eval --out runs/reach_bc_v1/report \
    --bc-curves runs/reach_bc_v1/bc_v1 runs/reach_bc_v1/bc_v2 runs/reach_bc_v1/bc_v3
python3 solverbot_rl/scripts/report_reach.py --eval runs/reach_bc_v1/eval_confirm --out runs/reach_bc_v1/report_confirm
```

Modelos versionados (1,5 MB c/u): `solverbot_rl/docs/resultados/reach_bc_v1/modelos/bc_v1.zip` y `bc_v2w.zip`.
Evaluar uno directamente: `tools/headless/run_experiment.sh 1 py runs/reach_bc_v1 eval_reach.py --policy
solverbot_rl/docs/resultados/reach_bc_v1/modelos/bc_v1.zip --level reach --seeds 902000-902019 --out runs/mi_eval`.

Con varias instancias simultáneas en una máquina de 16 hilos se observaron dos timeouts del backend
(`/joint_states` atrasado; reloj detenido en el arranque). El backend aborta en vez de seguir con datos viejos; los
trabajos se reanudan por semilla. Recomendación: ≤ 3–4 instancias en WSL.

Seguimiento: `tail -f ~/.solverbot/queue_<n>.out`, `runs/reach_bc_v1/logs/`. Detener: `tools/headless/stop.sh <n>`
(para `queue.sh` en segundo plano, además `pkill -f "queue.sh <n>"`).
