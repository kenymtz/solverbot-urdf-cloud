# Ejecución headless (PC Linux dedicada o WSL)

Solo simulación. Cada instancia `N` usa `ROS_DOMAIN_ID=42+N` y `GZ_PARTITION=solverbot_sim_N`
(`tools/wsl/sim_env.sh`). El dominio 10 (robot real) está bloqueado. **Empezar con UNA instancia.**

## 1. Preparación (una vez)

Requisitos: Ubuntu 24.04, ROS 2 Jazzy en `/opt/ros/jazzy`, repo clonado en `~/solverbot-urdf-cloud`.
```bash
cd ~/solverbot-urdf-cloud
tools/headless/setup.sh          # apt (ros-gz, gz_ros2_control...), workspace ~/solverbot_ws, venv, pytest
# SKIP_APT=1 tools/headless/setup.sh   si las dependencias ya están
```
Sin GPU: los sensores se renderizan por EGL con Mesa (llvmpipe). Verificado en Claude Cloud (4 CPU, sin GPU):
~0.35–0.4 s de reloj por paso de RL (0.1 s simulados) con el perfil de cámaras `train`. Con GPU se espera más rápido
(sin medir). `glxinfo -B` muestra el renderer.

## 2. Validación física (obligatoria antes de entrenar)

```bash
tools/headless/start.sh 0 validate --seeds 40000-40004          # en segundo plano
tools/headless/status.sh 0                                      # estado y logs
cat runs/validate_*/summary.json                                # por semilla + tasa con denominador
```
Criterio: cubo ≥ 30 mm sobre su altura apoyada durante ≥ 10 pasos (1 s), a < 4 cm del TCP; se registran q7/q8,
error de mimic, deriva del cubo antes del cierre y apertura al sostener (`steps.csv`). Más exigente:
`--seeds 40005-40019 --hold-steps 40 --steps 200`.

## 3. Entrenamiento autónomo

```bash
tools/headless/start.sh 0 train sac_gz_0 --algo sac --steps 300000 --checkpoint-every 5000
tools/headless/status.sh 0      # progreso (pasos/s, nivel, éxito), último checkpoint, fin del log
tools/headless/stop.sh 0        # SIGTERM: guarda checkpoint, cierra la simulación y no deja procesos
tools/headless/start.sh 0 train sac_gz_0 --algo sac --steps 300000   # REANUDA desde runs/sac_gz_0/checkpoints/latest.zip
```
* Checkpoints atómicos (`latest.zip`, `latest_replay.pkl`, `latest_state.json` + numerados), reanudación automática
  (modelo, replay buffer, pasos y nivel del curriculum); `--fresh` empieza de cero.
* `runs/<run>/`: `config.yaml` (y `config_resume_*.yaml`), `episodes.jsonl`, `progress.jsonl`, `curriculum.jsonl`,
  `logs/sim_*.log`, `logs/train_*.log`, `train_summary.json` (`completed` | `stopped` | `error`).
* Observación: por defecto `object_source: perception` (cámara de muñeca). El backend aborta si las imágenes o el
  CameraInfo no llegan. `--object-source privileged` queda marcado en `config.yaml` y en cada episodio: una política
  así no se presenta como basada en visión.
* Duplicados: `start.sh` rechaza una instancia en marcha (archivo PID) y `run_experiment.sh` rechaza arrancar si ya
  hay Gazebo en esa partición o un `controller_manager` en ese dominio.

Tras un reinicio sin systemd: volver a ejecutar el mismo `start.sh ... train <run>` (reanuda). Con systemd de
usuario: ver `solverbot-train@.service` (reinicio automático y `loginctl enable-linger`). En primer plano (tmux):
`tools/headless/run_experiment.sh 0 train <run> ...`.

**No iniciar entrenamientos largos hasta que la validación física dé una tasa aceptable** en las semillas propias.
