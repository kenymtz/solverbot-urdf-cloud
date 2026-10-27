# SolverBot v3 — Simulación doméstica 3D y Pick and Place con RL

Gazebo Harmonic (gz-sim 8) · ROS 2 Jazzy · gz_ros2_control · ros_gz_bridge · Gymnasium · Stable-Baselines3.
**Solo simulación.** Ningún archivo de esta etapa habla con el hardware real (CAN, ESP32, Dynamixel, hoverboard).

| Paquete | Contenido |
|---|---|
| `robot_v3_description` | URDF v3 sin cambios cinemáticos. Nuevo modo `ros2_control:=gz` (Harmonic) + `urdf/gazebo_harmonic.xacro` (sensores) + `config/sim_cameras.yaml`. El modo Fortress (`gazebo`) queda intacto. |
| `solverbot_sim` | Arena de 4 habitaciones y mundo de entrenamiento (generados desde YAML), launch, puente, controladores RL. |
| `solverbot_rl` | Entorno Gymnasium, recompensas, curriculum, backends (Gazebo / cinemático), entrenamiento y evaluación. |

Documentos: [especificación y supuestos de la arena](docs/ARENA_SPEC.md) · [hallazgos de ingeniería](docs/HALLAZGOS.md) ·
evidencias en `docs/evidencias/` y resultados en `../solverbot_rl/docs/resultados/`.

---

## 1. Instalación en WSL2 (Ubuntu 24.04 + ROS 2 Jazzy ya instalado)

```bash
# 1) dependencias de simulación + venv de Python para RL (torch CPU)
cd ~/solverbot-urdf-cloud            # este repositorio, rama feat/solverbot-rl-arena
tools/wsl/install_deps.sh

# 2) workspace colcon con los dos paquetes ROS
mkdir -p ~/solverbot_ws/src && cd ~/solverbot_ws/src
ln -s ~/solverbot-urdf-cloud/robot_v3_description .
ln -s ~/solverbot-urdf-cloud/solverbot_sim .
cd ~/solverbot_ws && source /opt/ros/jazzy/setup.bash && colcon build --symlink-install

# 3) paquete Python de RL (editable)
source ~/solverbot_venv/bin/activate && pip install -e ~/solverbot-urdf-cloud/solverbot_rl
```

Todo lo que sigue se ejecuta desde la raíz del repositorio (`cd ~/solverbot-urdf-cloud`).

## 2. Demostración: arena + SolverBot + dos cámaras simultáneas

```bash
tools/wsl/run_sim.sh 0                      # GUI de Gazebo + 2 ventanas rqt_image_view (cabeza y pinza)
tools/wsl/run_sim.sh 0 rviz:=true           # además RViz
```
SolverBot aparece en la cocina con la mesa del comedor a su izquierda (el Piper va montado en el costado
izquierdo). Mover cabeza y brazo para ver que cada cámara sigue a su soporte (en otra terminal):
```bash
source tools/wsl/sim_env.sh 0
ros2 topic pub --once /head_position_controller/commands std_msgs/msg/Float64MultiArray "{data: [0.6, 0.3]}"
ros2 topic pub --once /arm_position_controller/commands std_msgs/msg/Float64MultiArray \
  "{data: [1.753, 0.046, -0.671, -0.626, 0.918, 1.128]}"
ros2 topic pub --once /gripper_position_controller/commands std_msgs/msg/Float64MultiArray "{data: [0.0]}"   # una orden
ros2 topic pub --once /lift_position_controller/commands std_msgs/msg/Float64MultiArray "{data: [-0.05]}"
ros2 topic pub --once /diff_drive_controller/cmd_vel geometry_msgs/msg/TwistStamped "{twist: {angular: {z: 0.3}}}"
python3 solverbot_rl/scripts/capture_cameras.py --out capturas/   # PNG de ambas cámaras sin GUI
```
Tópicos: `/camera/camera/color/image_raw`, `/camera/camera/aligned_depth_to_color/image_raw`,
`/camera/camera/color/camera_info`, `/camera/camera/depth/color/points`, `/wrist_camera/image_raw`,
`/wrist_camera/camera_info`, `/scan`, `/joint_states`, `/clock`.

Sin GPU o por SSH: `headless:=true egl:=true` (render de sensores por EGL; en Claude Cloud funcionó con Mesa llvmpipe).
Si la GUI de WSLg falla: `export LIBGL_ALWAYS_SOFTWARE=1` antes del launch.

## 3. Validación en tiempo de ejecución (articulaciones, cámaras, percepción, agarre físico)

```bash
# terminal 1
tools/wsl/run_sim.sh 1 world:=pick_place_training headless:=true egl:=true cameras:=false
# terminal 2
source tools/wsl/sim_env.sh 1
python3 solverbot_rl/scripts/validate_sim.py --out validacion/     # report.json + PNG; sale != 0 si algo falla
```

### Validación física del agarre por semilla

```bash
python3 solverbot_rl/scripts/validate_grasp.py --backend gazebo --seeds 40000-40019 --hold-steps 40 --out runs/grasp_val
```
Criterio independiente de la recompensa: cubo ≥ 30 mm sobre su altura apoyada durante ≥ N pasos, a < 4 cm del TCP.
`steps.csv` registra por paso q7/q8, error de mimic, objetivo de la pinza, TCP y cubo; `summary.json` da la tasa con
denominador. Ejecución headless y entrenamiento autónomo en otra PC: [`tools/headless/README.md`](../tools/headless/README.md).

**Pinza en simulación** (`config/pick_place.yaml`, sección `robot`): precierre a 32 mm (nunca el límite de 35 mm),
rampa de cierre de 1.5 mm/paso, y con los controladores RL `gripper_servo:=dual` + acoplamiento emulado de los dedos
(`solverbot_rl/gripper.py`). El modo del robot real y la cinemática no cambian. Ver `docs/HALLAZGOS.md` (13–19).

## 4. Entrenamiento y evaluación

```bash
# Pruebas rápidas sin simulador (recompensas, anti reward hacking, API Gymnasium): ~1 s
python3 -m pytest -q solverbot_rl/tests

# Sanidad del pipeline sin física (backend cinemático, ~3 min)
python3 solverbot_rl/scripts/train.py --backend kinematic --algo ppo --steps 60000 --run ppo_kin
python3 solverbot_rl/scripts/evaluate.py --policy runs/ppo_kin/model_final.zip --algo ppo --level 0 --episodes 50

# Gazebo (física): simulación de la sección 3 en marcha en la instancia 1, luego
source tools/wsl/sim_env.sh 1
python3 solverbot_rl/scripts/train.py --backend gazebo --algo sac --steps 200000 --run sac_gz --checkpoint-every 10000
python3 solverbot_rl/scripts/evaluate.py --backend gazebo --policy runs/sac_gz/model_final.zip --algo sac --level 0 --episodes 30
# líneas base: --policy random | --policy scripted (experto con IK y estado privilegiado = cota superior)
# continuar desde un checkpoint: train.py ... --resume runs/sac_gz/checkpoints/sac_100000_steps.zip --level <n>
```
Cada corrida guarda en `runs/<run>/`: `config.yaml`, `episodes.jsonl` (un registro por episodio con semilla,
retorno, éxito, nivel y términos de recompensa), `curriculum.jsonl` (promociones de nivel), `checkpoints/`,
`model_final.zip`. La evaluación usa semillas 10000+i (distintas de las de entrenamiento) y política determinista.

**Instancias en paralelo**: cada instancia `N` usa `ROS_DOMAIN_ID=42+N` y `GZ_PARTITION=solverbot_sim_N`
(`tools/wsl/sim_env.sh N`); lanzar la simulación y el entrenamiento con el mismo `N`.

## 5. Tarea, observaciones, recompensas y curriculum

Configuración única: `config/pick_place.yaml` (mundo, robot, control, observación, curriculum, pesos de recompensa).

* **Acción** (7): incrementos articulares del brazo (±0.05 rad/paso a 10 Hz) + pinza (una sola orden, joint7).
* **Observación**: articulaciones, pinza, TCP por cinemática directa, objeto **estimado** (cámara de muñeca:
  segmentación + rayo al plano de la mesa), zona objetivo (dato de la tarea), acción previa, meta.
  `object_source: privileged` (pose real) existe solo para depuración/maestro y queda registrado en cada episodio.
  `images: [wrist]` agrega la imagen de la cámara de la pinza (política `MultiInputPolicy`).
* **Recompensa**: shaping basado en potencial (los ciclos no pagan) + bonos de evento una sola vez (agarre,
  levantamiento) + éxito verificable; penalizaciones por tiempo, brusquedad, límites, colisión con la mesa,
  caída del objeto y falta de progreso. Pruebas: `solverbot_rl/tests/test_rewards.py`.
* **Éxito** (medido con la pose real, nunca observado por la política): reach (TCP a 3 cm del pre-agarre con
  aproximación desde arriba), grasp (orden de cierre + dedos detenidos por el objeto + objeto junto al TCP),
  lift (sujeto y ≥ 6 cm sobre la mesa durante 5 pasos), place (levantado antes, en la zona, apoyado, quieto,
  soltado y con la pinza alejada).
* **Curriculum**: reach → reach aleatorio → grasp (arranque cerca) → lift → pick desde el inicio → place.
  Se asciende por **tasa de éxito** (≥ 80 % en 50 episodios), no por recompensa.

## 6. Qué se verificó realmente y qué falta (ver resultados)

Ejecutado en Claude Cloud con Gazebo Harmonic 8.10 + gz_ros2_control 1.2 (RoboStack Jazzy, render por CPU):
arena y mundo de entrenamiento cargan; 6 controladores activos; mimic de la pinza; ambas cámaras publican y siguen
a su soporte; percepción con cámara de muñeca (error 3 mm); **agarre físico** con levantamiento de 10 cm;
experto a través del entorno Gym: lift 3/3, place 1/2; PPO (cinemático) reach 50/50 vs aleatorio 0/50;
SAC 400 pasos sobre Gazebo (integración, sin aprendizaje: 0/3).

Pendiente en WSL: GUI de Gazebo y ventanas de cámaras con WSLg, entrenamiento largo sobre Gazebo, ajuste de
fricción/contactos del agarre y aprendizaje de los niveles grasp → place.
