| # | Comprobación | Resultado | Detalle |
|---|---|---|---|
| 1 | xacro: 24/24 combinaciones de argumentos generan URDF | OK |  |
| 2 | xacro rechaza argumento inválido {'ros2_control': 'foo'} | OK |  |
| 3 | xacro rechaza argumento inválido {'arm_limits': 'foo'} | OK |  |
| 4 | xacro rechaza argumento inválido {'head_mount': 'foo'} | OK |  |
| 5 | check_urdf ('none', 'column', 'official', 'true') | OK | robot name is: robot_v3 |
| 6 | check_urdf ('gazebo', 'lift', 'base_athome', 'true') | OK | robot name is: robot_v3 |
| 7 | check_urdf ('none', 'column', 'official', 'false') | OK | robot name is: robot_v3 |
| 8 | una sola raíz = base_footprint | OK | raíces: ['base_footprint'] |
| 9 | todos los joints conectan links existentes | OK |  |
| 10 | árbol sin ciclos y conexo (links = joints + 1) | OK | 38 links, 37 joints |
| 11 | cada link tiene a lo sumo un padre | OK |  |
| 12 | frames de compatibilidad (base_athome, urg_node, Jetson, realsense) presentes | OK | 36 frames |
| 13 | joints de compatibilidad presentes | OK |  |
| 14 | joints que publica la Jetson existen y son móviles | OK |  |
| 15 | articulaciones móviles | OK | joint1, joint2, joint3, joint4, joint5, joint6, joint7, joint8, left_wheel_joint, pan_joint, right_wheel_joint, tilt_joint, torso_lift_joint |
| 16 | torso_lift_joint prismático en +Z de la columna | OK |  |
| 17 | límites del elevador = valores provisionales separados (sim_provisional.yaml) | OK | {'lower': -0.19, 'upper': 0.068, 'effort': 1000.0, 'velocity': 0.05} |
| 18 | todas las revolutas/prismáticas: lower < upper, effort > 0, velocity > 0, eje unitario | OK |  |
| 19 | límites del Piper = perfil "official" de piper_limits.yaml | OK |  |
| 20 | límites del Piper = perfil "base_athome" de piper_limits.yaml | OK |  |
| 21 | pose DEFAULT_HOME de la Jetson dentro de los límites oficiales | OK |  |
| 22 | posición de left_wheel = medida/derivada | OK | error 0.000 mm |
| 23 | posición de right_wheel = medida/derivada | OK | error 0.000 mm |
| 24 | posición de laser = medida/derivada | OK | error 0.000 mm |
| 25 | posición de torso_lift_link = medida/derivada | OK | error 0.000 mm |
| 26 | posición de mounting_cube_link = medida/derivada | OK | error 0.000 mm |
| 27 | posición de arm_base_link = medida/derivada | OK | error 0.000 mm |
| 28 | posición de pan_tilt_base_link = medida/derivada | OK | error 0.000 mm |
| 29 | centro de la carcasa D435 (tilt del escaneo) = posición medida en v2 | OK | error 0.05 mm |
| 30 | base del brazo: z del Piper hacia +Y (montaje lateral izquierdo) | OK |  |
| 31 | elevador (column) -0.190 m: suben/bajan 6 frames exactamente, 8 quietos | OK |  |
| 32 | elevador (column) +0.050 m: suben/bajan 6 frames exactamente, 8 quietos | OK |  |
| 33 | elevador (column) +0.068 m: suben/bajan 6 frames exactamente, 8 quietos | OK |  |
| 34 | elevador (lift) -0.190 m: suben/bajan 9 frames exactamente, 5 quietos | OK |  |
| 35 | elevador (lift) +0.050 m: suben/bajan 9 frames exactamente, 5 quietos | OK |  |
| 36 | elevador (lift) +0.068 m: suben/bajan 9 frames exactamente, 5 quietos | OK |  |
| 37 | brazo: posición de link6 (URDF) = MDH oficial de piper_sdk en 202 poses | OK | máx 0.088 mm |
| 38 | brazo: orientación de link6 (URDF) = MDH oficial de piper_sdk | OK | máx ‖ΔR‖ 1.34e-07 |
| 39 | brazo: cada joint1..6 gira 0.3 rad sobre su eje sin mover a su padre | OK |  |
| 40 | pinza: joint7=+0.035 y joint8=-0.035 separan los dedos 70 mm en direcciones opuestas | OK | 70.00 mm |
| 41 | tcp_link a 0.1358 m sobre el eje de link6 | OK |  |
| 42 | pan sobre +Z y tilt sobre +Y (positivo = cámara hacia abajo) | OK |  |
| 43 | D435: 4 frames ópticos con convención REP-103 (z adelante, x derecha, y abajo) | OK |  |
| 44 | D435: extrínsecos nominales de Intel (color +15 mm, infra2 -50 mm en Y) | OK |  |
| 45 | D435: camera_link mira hacia adelante del robot con el cabezal en cero | OK |  |
| 46 | camera_frames:=false: quedan camera_link y la rosca; se quitan los 8 frames del driver | OK |  |
| 47 | todas las mallas referenciadas existen dentro del paquete | OK | 34 referencias |
| 48 | sin mallas sin usar en el paquete | OK |  |
| 49 | paquete liviano: mallas < 15 MB, sin texturas de imagen | OK | 12.9 MB |
| 50 | rutas de mallas con package:// (portables) | OK |  |
| 51 | inercias: masa > 0, tensor definido positivo y desigualdad triangular | OK |  |
| 52 | todo link con geometría tiene inercia (requisito de Gazebo) | OK | [] |
| 53 | visual: los 12 DAE de estructura coinciden con sus primitivas (±1 mm) y colores del YAML | OK |  |
| 54 | visual: todos los grupos de visual_v3.yaml se usan en el URDF | OK | [] |
| 55 | visual: un solo <visual> por link (RViz2 aplica el material de la primera a todas) | OK | [] |
| 56 | visual: neumático con radio y ancho calibrados (0.084 / 0.055) | OK |  |
| 57 | ros2_control mock: todas las articulaciones móviles declaradas (mock_components/GenericSystem) | OK |  |
| 58 | ros2_control gazebo: todas las articulaciones móviles declaradas (ign_ros2_control/IgnitionSystem) | OK |  |
| 59 | controladores de trayectoria cubren brazo, pinza, elevador y cabezal | OK |  |
| 60 | Gazebo: plugin ign_ros2_control, cámara RGB-D y LiDAR presentes | OK |  |
| 61 | modo none: sin etiquetas ros2_control ni gazebo (descripción pura) | OK |  |
| 62 | robot_v2_description idéntico al commit 9401da9 (git diff + status vacíos) | OK |  |

**62/62 comprobaciones OK**
