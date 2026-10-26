# robot_v3_description

Modelo URDF/Xacro **simplificado y articulado** de SolverBot / Base@Home para **ROS 2 Humble**:
base diferencial, **elevador prismático**, **brazo AgileX Piper de 6 GDL con pinza** (cinemática oficial),
cabezal **pan-tilt** con **RealSense D435** y LiDAR **Hokuyo UST-05LX**.
Sirve para RViz, para simulación con `ros2_control` (mock, sin física) y para **Ignition Gazebo Fortress**.

Es un paquete nuevo, independiente de `robot_v2_description`, que no se modificó.

| | v2 | v3 |
|---|---|---|
| Brazo | un link rígido escaneado | Piper articulado: joint1–6 + pinza joint7/8, cinemática oficial |
| Elevador | prismático, sin controlador | prismático + controlador de trayectoria (mock y Gazebo) |
| Cabezal | soporte rígido | pan_joint + tilt_joint (nombres de base_athome) |
| Cámara | `camera_link` + 1 frame óptico | D435 completa: rosca, `camera_link`, depth/infra1/infra2/color con sus frames ópticos |
| Visual | mallas escaneadas con texturas (37 MB) | estructura reconstruida con primitivas medidas sobre el escaneo + Piper con su malla oficial a color (12.4 MB en total, sin texturas) |
| Simulación | no | ros2_control mock, Gazebo Fortress con cámara RGB-D y LiDAR |

---

## 1. Instalación en la NUC

Requisitos: Ubuntu 22.04, ROS 2 Humble en `/opt/ros/humble`. **No compilar en los workspaces existentes** (`~/base_athome_ws`, `~/limoatwork`, …): usar uno nuevo.

```bash
mkdir -p ~/robot_v3_ws/src && cd ~/robot_v3_ws/src
git clone -b claude/pensive-fermat-3miexh https://github.com/kenymtz/solverbot-urdf-cloud.git
cd ~/robot_v3_ws
source /opt/ros/humble/setup.bash

# Dependencias (pide sudo: hacerlo solo con autorización). Mínimo para RViz + mock:
sudo apt install ros-humble-xacro ros-humble-robot-state-publisher ros-humble-joint-state-publisher-gui \
  ros-humble-rviz2 ros-humble-ros2-control ros-humble-ros2-controllers liburdfdom-tools
# Solo para Gazebo Fortress:
sudo apt install ros-humble-ros-gz ros-humble-ign-ros2-control
# (alternativa: rosdep install --from-paths src --ignore-src -y)

colcon build --packages-select robot_v3_description
source install/setup.bash
```

`colcon` encuentra los dos paquetes del repositorio (`robot_v2_description` y `robot_v3_description`); `--packages-select` compila solo el v3.

## 2. Uso

> **Dominio DDS.** El robot real usa `ROS_DOMAIN_ID=10` (NUC + Jetson). Los launch que publican estados
> **simulados** se niegan a arrancar en ese dominio (salvo `allow_robot_domain:=true`). Para simular:
> `export ROS_DOMAIN_ID=42`.

| Launch | Qué hace | Hardware |
|---|---|---|
| `display.launch.py` | RViz + sliders (`joint_state_publisher_gui`); el brazo arranca en la pose de reposo de la Jetson (`arm_pose:=home`, o `zero`) | ninguno |
| `sim_mock.launch.py` | `ros2_control` con `mock_components/GenericSystem` + controladores de trayectoria + RViz | ninguno |
| `gazebo.launch.py` | Ignition Gazebo Fortress + `ign_ros2_control` + cámara RGB-D y LiDAR simulados + RViz | ninguno |
| `real_state.launch.py` | TF del robot real a partir de los `/joint_states` que publica la Jetson | **solo lee** |

```bash
export ROS_DOMAIN_ID=42
ros2 launch robot_v3_description display.launch.py                       # sliders
ros2 launch robot_v3_description sim_mock.launch.py                      # controladores simulados
ros2 run robot_v3_description demo_motion.py                             # mueve elevador, cabezal, pinza y brazo
ros2 launch robot_v3_description gazebo.launch.py                        # Gazebo con GUI
ros2 launch robot_v3_description gazebo.launch.py headless:=true egl:=true rviz:=false   # por SSH
```

Argumentos comunes (todos los launch):

| Argumento | Valores | Por defecto | Efecto |
|---|---|---|---|
| `head_mount` | `column`, `lift` | `column` | Cabezal sobre la columna fija (lo que muestra el escaneo) o sobre el carro del elevador (árbol de base_athome) |
| `arm_limits` | `official`, `base_athome` | `official` | Límites del Piper (la cinemática es siempre la oficial). Ver §5 |
| `camera_frames` | `true`, `false` | `true` (`false` en `real_state`) | Frames nominales de la D435. `false` cuando el driver real publica los suyos |
| `wrist_camera` | `true`, `false` | `true` | Frames de la cámara Orbbec de muñeca |

### Comandar articulaciones en simulación

Controladores (`config/ros2_controllers.yaml`), todos `joint_trajectory_controller` salvo la base:

| Controlador | Articulaciones | Acción / tópico |
|---|---|---|
| `lift_controller` | `torso_lift_joint` | `/lift_controller/follow_joint_trajectory` |
| `arm_controller` | `joint1`…`joint6` | `/arm_controller/follow_joint_trajectory` |
| `gripper_controller` | `joint7`, `joint8` (dedos, ±0.035 m) | `/gripper_controller/follow_joint_trajectory` |
| `head_controller` | `pan_joint`, `tilt_joint` | `/head_controller/follow_joint_trajectory` |
| `diff_drive_controller` (solo Gazebo) | ruedas | `/diff_drive_controller/cmd_vel_unstamped` |

```bash
ros2 action send_goal /lift_controller/follow_joint_trajectory control_msgs/action/FollowJointTrajectory \
  "{trajectory: {joint_names: [torso_lift_joint], points: [{positions: [0.05], time_from_start: {sec: 3}}]}}"
```

Ningún nombre coincide con los del robot real (la Jetson escucha `/arm/joint_cmd`, `/arm/enable`, …), y la base simulada no usa `/cmd_vel`.

## 3. Árbol cinemático

```
base_footprint
└─ base_link                                   chasis (caja medida)
   ├─ left_wheel_joint / right_wheel_joint     continuous, eje +Y
   ├─ rear_left_caster / rear_right_caster     fijas
   ├─ laser_mount ─ laser                      Hokuyo; "laser" = punto focal (urg_node, Nav2)
   └─ torso_column                             perfil del elevador (fijo)
      ├─ torso_lift_joint (PRISMÁTICO +Z) ─ torso_lift_link        carro
      │  └─ mounting_cube_link                 caja (bandeja + monitor)
      │     ├─ screen_link
      │     └─ arm_mount_joint ─ arm_base_link  Piper, z hacia +Y (montaje lateral izquierdo)
      │        └─ joint1 … joint6 ─ link6 ─ gripper_base ─ joint7/link7, joint8/link8
      │                             ├─ tcp_link (+0.1358 m)
      │                             └─ arm_camera_link ─ arm_camera_optical_frame
      └─ pan_tilt_base_link                    (head_mount:=lift → cuelga de torso_lift_link)
         └─ pan_joint (+Z) ─ pan_link ─ tilt_joint (+Y) ─ tilt_link
            └─ camera_bottom_screw_frame ─ camera_link
               └─ camera_{depth,infra1,infra2,color}_frame ─ camera_*_optical_frame
```

38 links, 37 joints; 13 articulaciones móviles (2 ruedas, elevador, 6 del brazo, 2 dedos, pan, tilt).
Con el elevador se mueven: `torso_lift_link`, `mounting_cube_link`, `screen_link` y todo el brazo
(y el cabezal solo con `head_mount:=lift`).

## 4. Procedencia de los datos

Todos los valores están en `config/` con su fuente al lado. Resumen:

| Elemento | Fuente |
|---|---|
| Chasis, ruedas, ruedas locas, LiDAR, columna, caja, pantalla | Escaneo métrico de v2 (`robot_v2_description/config/robot_geometry.yaml`) |
| Cinemática, inercias y mallas del Piper | Descripción oficial `agx_arm_urdf` (submódulo de `agx_arm_ros`, rama `ros2`) |
| Pinza (joint7/joint8) | `piper_ros` rama `humble`: los nombres que publica la Jetson |
| Posición del brazo | **Ajuste**: cinemática oficial en la pose `DEFAULT_HOME` de la Jetson contra la malla escaneada del brazo (`tools/fit_arm_mount.py`), error medio 12.6 mm |
| Orientación del brazo, cabezal (ejes, palanca de la cámara), cámara de muñeca, nombres de frames | `ShiiSuii/base_athome` @ `6edb8a4` (`robot_piper.urdf.xacro`, `piper_kinematics.py`) |
| Altura del eje de tilt | Derivada: carcasa de la cámara medida en v2 − palanca de base_athome |
| Frames de la D435 | `realsense2_description/_d435.urdf.xacro` (Intel), extrínsecos **nominales** |
| Límites del elevador, masas, parámetros de sensores simulados | **Provisionales**, solo simulación: `config/sim_provisional.yaml` |

### Variante del Piper

Los parámetros que usa el equipo en `piper_kinematics.py` (d4 = 250.75 mm, d6 = 91 mm, a3 = 285.03 mm), que según su autor
coinciden a 0.02 mm con `GetArmEndPoseMsgs()` del firmware, son los del **Piper estándar** (`piper` en `piper_sdk`).
Descartan Piper H (d4 = 255.75, d6 = 96.3), Piper L (a3 = 338.23) y Piper X (otra cadena).
La validación compara la cadena del URDF con los MDH de `piper_sdk` para el Piper estándar: coinciden con un error máximo de 0.09 mm en 202 poses.

## 4b. Geometría visual

La geometría visual es una versión intermedia entre el escaneo y un modelo de cajas. Solo cambia lo que se ve: las colisiones, inercias, joints y frames son los mismos (la única transformada que cambió es la de `screen_link`, que es fija y solo visual, ver abajo).

* **Fuente:** `config/visual_v3.yaml`, con cada pieza medida sobre el escaneo métrico de v2 (`tools/measure_v2_scan.py`; ±1–2 cm y ±3 % de escala).
* **Chasis:** marco de perfiles 20×20 en dos niveles (z ≈ 0.175 y 0.303 m), esquinas achaflanadas, montantes, placa inferior, traviesas bajo la columna, volumen de electrónica entre niveles y placa del Hokuyo.
* **Columna:** módulo lineal (paredes, alma, dos guías frontales, husillo, placas de base y superior) más el **carro** que se mueve con `torso_lift_joint`.
* **Parte superior:** bandeja, carcasa, **monitor inclinado 12.8°** (ajuste lineal de su cara frontal en el escaneo) con pantalla azul, y bloque superior.
* **Ruedas:** neumático con radio y ancho calibrados, llanta y un radio marcador para ver el giro. Las ruedas locas tienen horquilla y color naranja, como en el escaneo.
* **Cabezal:** servos XL430 y soporte. El micrófono está en la posición escaneada, suponiendo que va montado en el soporte de la cámara.
* **D435:** carcasa de 90×25×25 mm con extremos redondeados, frente negro y lentes en las posiciones de los extrínsecos nominales de Intel. La malla oficial de Intel pesa 15.8 MB y se descartó.
* **Piper:** malla visual oficial (DAE de `agx_arm_urdf`) con sus colores: eslabones blancos, articulaciones negras y detalles rojos. Está simplificada con un desvío p95 < 1 mm respecto de la oficial (`tools/make_piper_visuals.py`). La escala es la real (`link2` mide 0.348 m) y en la pose de reposo calza con el brazo escaneado.

Cada link tiene **un solo `<visual>` con un DAE** (`meshes/structure/`, generado con `tools/make_structure_visuals.py`). RViz2 aplica a todas las `<visual>` de un link el material de la primera, así que con varias primitivas por link la columna se veía negra y las ruedas no mostraban la llanta. El validador comprueba que cada DAE coincide (±1 mm) con sus primitivas del YAML.

**`screen_link`:** antes estaba en la cara frontal de la caja envolvente (x = 0.283), sin inclinación. Ahora está en el centro de la cara del monitor real (x = 0.1815, z = 1.09) e inclinado −12.8°. Es un frame fijo, hijo de `mounting_cube_link`, sin hijos ni joints móviles. No lo usa el software de base_athome, y no afecta a la cinemática ni a las colisiones.

Comparación con v2 desde 5 vistas: `docs/evidencias/visual/` (v3, v2 y ambos superpuestos).

### "No Image" en RViz

El panel de imagen que aparecía en la configuración anterior de RViz mostraba "No Image" porque **nadie publicaba** `/camera/camera/color/image_raw`. Con `display.launch.py` no hay simulador ni driver, y el tópico no existe (`docs/evidencias/no_image_diagnostico.txt`). La cámara está bien representada en el URDF (frames y geometría), pero eso no significa que haya una cámara funcionando. Con `gazebo.launch.py` el tópico tiene 1 publicador y entrega imágenes `rgb8` en `camera_color_optical_frame`. En el robot real lo publica el driver de la Jetson. El panel se quitó de la configuración por defecto.

## 5. Brazo Piper: límites y estados reales

* **Límites `official`** (por defecto): los de `agx_arm_urdf`, iguales a los de `piper_sdk` salvo joint6 (el SDK usa ±120° por firmware antiguo).
* **Límites `base_athome`**: los del URDF del equipo, más amplios. **Las poses que usa la Jetson los necesitan**:
  `DEFAULT_READY` tiene joint2 = 182.68° (> 180° oficial) y `home` de `arm_poses_and_limits.yaml` tiene joint4 = 102.62° (> 100°).
  `robot_state_publisher` no recorta, así que en RViz el modelo sigue al robot igual; pero un controlador o planificador
  con los límites oficiales rechazaría esas poses. Verificar en el robot qué permite el firmware antes de planificar con `base_athome`.
* **Estados reales:** la Jetson (`piper_arm_controller.py`) publica `/joint_states` con `joint1..joint6` (rad) y
  `joint7 = +apertura/2`, `joint8 = −apertura/2` (m), igual que este URDF. `real_state.launch.py` los combina con valores por
  defecto para lo que nadie publica y alimenta a `robot_state_publisher`:

```bash
# En la NUC, con el robot encendido (dominio 10). Solo lee /joint_states; no envía comandos.
ros2 launch robot_v3_description real_state.launch.py rviz:=true
```

  **No** lanzarlo junto con `hoverboard_base/bringup.launch.py` sin quitar el `robot_state_publisher` de ese launch:
  habría dos URDF publicando TF de `base_link`, las ruedas y `laser`.

## 6. Elevador

* `torso_lift_joint`: prismático en +Z, de `torso_column` a `torso_lift_link`. **Cero = altura del escaneo.**
* El controlador real (`stepper_controller.py` + firmware `nema23_esp32.ino`) **se analizó pero no se ejecutó**:
  * consola serie (115200 baudios) con comandos `calibrate`, `top`, `bottom`, `move <pasos>`; dos finales de carrera.
  * la carrera se **mide en cada calibración** (`max_travel_steps`) y no se guarda; el cero del firmware es el final de carrera inferior.
  * no se documenta el paso del husillo: solo hay un comentario "250 pasos (~1 mm)" con microstepping 8 (1600 pasos/vuelta).
  * **no publica ninguna posición en ROS**: hoy no se puede reflejar la altura real en RViz.
* Límites usados: **provisionales para simulación** (−0.19 / +0.068 m, envolvente geométrica del escaneo; velocidad
  0.05 m/s; esfuerzo 1000 N solo para Gazebo), en `config/sim_provisional.yaml`.

**Las trayectorias del elevador deben respetar su velocidad límite.** En Gazebo, `ign_ros2_control` sigue la posición
con un comando de velocidad (ganancia 0.1 × frecuencia de actualización); si ese comando supera el límite del URDF, DART lo
descarta y el carro cae por gravedad al tope inferior y queda trabado allí. Se observó al pedir 0.2 m en 3 s (≈0.1 m/s de pico).
`demo_motion.py` y `runtime_checks.py` calculan las duraciones a partir de los límites del URDF. En el robot real la
velocidad también estará limitada, así que conviene planificar igual (por ejemplo con la parametrización temporal de MoveIt).

Para integrar el elevador real hace falta (pendiente, ver §9): carrera en pasos (`max_travel_steps` tras `calibrate`), pasos por metro,
altura del final de carrera inferior respecto del cero del URDF y un nodo que publique `torso_lift_joint` en `/joint_states`.

## 7. RealSense D435 y sensores simulados

* Estructura de frames idéntica a la de Intel (`camera_bottom_screw_frame`, `camera_link`, `camera_{depth,infra1,infra2,color}_frame`
  y sus `_optical_frame`, REP-103: z adelante, x derecha, y abajo). Extrínsecos **nominales**; no se inventó ninguna calibración.
* Con el driver real (`realsense2_camera` en la Jetson, `camera_name:=camera`, `publish_tf:=true`) el driver publica
  `camera_link → *_frame/*_optical_frame` con la calibración de fábrica: usar `camera_frames:=false` (por defecto en `real_state.launch.py`).
  `camera_link` siempre lo publica el URDF.
* **Gazebo elegido: Ignition Fortress.** Es el simulador emparejado oficialmente con Humble (binarios `ros-humble-ros-gz`,
  `ros-humble-ign-ros2-control`), con sensores RGB-D y LiDAR por GPU y soporte de `ros2_control` mantenido. Gazebo Classic 11
  terminó su vida útil en enero de 2025.
* Tópicos simulados con los **mismos nombres que el driver real**, para probar el software de base_athome:
  `/camera/camera/color/image_raw`, `/camera/camera/color/camera_info`, `/camera/camera/aligned_depth_to_color/image_raw`,
  `/camera/camera/depth/color/points`, `/scan`. Resolución/FOV nominales (640×480, FOV horizontal 69°), en `sim_provisional.yaml`.

## 8. Validación

### Ejecutada en Claude Cloud (26/10/2026)

Claude Cloud no tiene ROS. Para probar con **ROS 2 Humble real** se instalaron en un contenedor efímero los paquetes de
RoboStack (conda-forge, `ros-humble-*`: ros2_control 2.54, ros2_controllers 2.53, ign_ros2_control 0.7.19, ros_gz 0.244,
Ignition Gazebo 6.16). No se tocó la NUC, la Jetson ni el robot. Evidencias: [`docs/VALIDACION.md`](docs/VALIDACION.md).

| Prueba | Resultado |
|---|---|
| `colcon build` + `colcon test` | OK (2 tests, 0 fallas) |
| `tools/validate_v3.py`: xacro en 24 combinaciones (sin salida por stderr), `check_urdf`, árbol, límites, cinemática (brazo contra MDH de `piper_sdk`: 0.088 mm), D435, recursos, inercias, DAE visuales = primitivas, ros2_control, v2 sin cambios | 62/62 OK |
| `sim_mock.launch.py` + `runtime_checks.py` (TF real, controladores reales), también con `head_mount:=lift` | 22/22 y 22/22 OK |
| `demo_motion.py` y guardia de dominio (aborta con `ROS_DOMAIN_ID=10`) | OK |
| `real_state.launch.py` con `/joint_states` en el formato de la Jetson (`check_real_state.py`) | 9/9 OK |
| `gazebo.launch.py` (Fortress, headless) + `runtime_checks.py --sensors` (física, RGB-D, nube, LiDAR) | 28/28 OK (dos corridas seguidas) |
| `tools/nuc_validate.sh` completo (sin `--gazebo`) | 9/9 etapas OK |

Durante la validación aparecieron y se corrigieron errores reales (remapeo en `real_state`, `set -u` en el script de
la NUC) y un comportamiento de Gazebo con el elevador (§6). El detalle está en `docs/VALIDACION.md`.

### Para repetir en la NUC

```bash
cd ~/robot_v3_ws
bash src/solverbot-urdf-cloud/robot_v3_description/tools/nuc_validate.sh ~/robot_v3_ws            # estático + mock
bash src/solverbot-urdf-cloud/robot_v3_description/tools/nuc_validate.sh ~/robot_v3_ws --gazebo   # + Gazebo
```

Corre en `ROS_DOMAIN_ID=42` con `ROS_LOCALHOST_ONLY=1` y deja los registros en `~/robot_v3_ws/log/robot_v3_validation/`.
Comprobación visual recomendada: `display.launch.py`, mover cada slider y verificar que el elevador arrastra caja y brazo.

## 9. Pendientes (requieren datos del robot físico)

1. **Elevador:** carrera real, pasos por metro, altura del final de carrera inferior y un publicador de `torso_lift_joint`. Hasta entonces, los límites son provisionales.
2. **Brazo:** confirmar con cinta la posición de montaje (el ajuste tiene ±2 cm en X) y los límites que permite el firmware (perfil `base_athome`).
3. **Pinza:** confirmar el modelo de pinza (estándar de 70 mm vs. AGX Gripper de 100 mm, `agx_gripper`); el equipo limita a 105 mm en el comando.
4. **Cabezal:** confirmar dónde se apoya (columna fija o carro: `head_mount`), la altura del eje de tilt (derivada: 38 mm sobre la placa) y que el cero del tilt (3932 ticks) sea horizontal.
5. **Cámara de muñeca:** con la pose de base_athome, `arm_camera_link` mira hacia +X de `gripper_base` (de costado), no hacia el eje de la pinza. Se conservó por compatibilidad, pero conviene verificarlo.
6. **Escala del escaneo v2** (±3 %): medir con cinta una cota, por ejemplo la placa superior del perfil (1.321 m).
7. **Masas reales** (todas las de `sim_provisional.yaml` son marcadores).
8. **Visión del cabezal:** con la geometría medida, la caja con el monitor tapa la vista de la cámara si baja más de unos 26° (ver la imagen simulada en `docs/VALIDACION.md`). Verificar en el robot.
9. **Geometría visual no verificada en el robot:** ubicación del motor NEMA23 (no se distingue en el escaneo, no se modeló), diámetro del husillo, montaje del micrófono y contenido del volumen de electrónica.
10. Diferencias del README de base_athome: indica ROS 2 Jazzy en la NUC; este paquete apunta a Humble (CONTEXTO_SOLVERBOT.md).

## 10. Estructura

```
config/   geometry_v3.yaml (medido/derivado) · visual_v3.yaml (geometría visual) · sim_provisional.yaml (solo simulación) · piper_limits.yaml
          ros2_controllers.yaml · gz_bridge.yaml
urdf/     robot_v3.urdf.xacro · common · base · torso · piper · head · realsense_d435 · ros2_control · gazebo
launch/   display · real_state · sim_mock · gazebo (+ robot_v3_common.py)
meshes/   piper/collision (STL oficiales) · piper/visual (DAE oficiales a color, simplificados) · structure (DAE generados del YAML)
scripts/  demo_motion.py
tools/    validate_v3.py · runtime_checks.py · check_real_state.py · nuc_validate.sh · fit_arm_mount.py
          measure_v2_scan.py · make_structure_visuals.py · make_piper_visuals.py
docs/     VALIDACION.md · evidencias/ (informes, capturas de RViz y Gazebo, árbol TF)
test/     prueba de colcon test
worlds/   robot_v3_test.sdf (mesa y objetos frente al robot)
```

Licencias de terceros: [`THIRD_PARTY.md`](THIRD_PARTY.md).
