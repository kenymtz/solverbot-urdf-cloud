# Material de terceros incluido en robot_v3_description

| Archivos | Origen | Licencia | Cambios |
|---|---|---|---|
| `meshes/piper/collision/{base_link,link1..link6}.stl` | [agilexrobotics/agx_arm_urdf](https://github.com/agilexrobotics/agx_arm_urdf) @ `680fa925` (submódulo de [agx_arm_ros](https://github.com/agilexrobotics/agx_arm_ros), rama `ros2` @ `286633e0`), `piper/meshes/` | MIT, © 2026 aalicecc | Ninguno (copia byte a byte; idénticas a las de `base_athome/src/piper_description/meshes`) |
| `meshes/piper/collision/link7.stl`, `link8.stl` | agx_arm_urdf @ `680fa925`, `agx_gripper/meshes/gripper_link1.stl`, `gripper_link2.stl` | MIT | Solo cambio de nombre (idénticas a `link7.STL`/`link8.STL` de piper_ros) |
| `meshes/piper/collision/gripper_base.stl` | [agilexrobotics/piper_ros](https://github.com/agilexrobotics/piper_ros), rama `humble` @ `017ffefa`, `src/piper_description/meshes/gripper_base.STL` | MIT, © 2024 RosenYin | Solo cambio de nombre |
| `meshes/piper/visual/*.stl` | Generadas desde las anteriores con `tools/simplify_piper_meshes.py` | MIT (obra derivada) | Simplificación de malla |
| Cinemática, inercias y límites del Piper en `urdf/piper.xacro` y `config/piper_limits.yaml` | agx_arm_urdf `piper/urdf/piper_description.urdf`; pinza: piper_ros `humble` `piper_description.urdf` | MIT | Ver comentario al inicio de `urdf/piper.xacro` |
| Estructura de frames y dimensiones nominales de `urdf/realsense_d435.xacro` | [IntelRealSense/realsense-ros](https://github.com/IntelRealSense/realsense-ros) `ros2-master` @ `7a87097c`, `realsense2_description/urdf/_d435.urdf.xacro` | Apache-2.0, © 2023 RealSense, Inc. | Reescrito con geometría de caja, sin malla ni tapón USB |
| Referencias MDH en `tools/validate_v3.py` (`PIPER_MDH`) | [agilexrobotics/piper_sdk](https://github.com/agilexrobotics/piper_sdk) @ `b847e539`, `piper_sdk/piper_param/constants.py` | MIT | Solo los valores numéricos |

Datos del robot tomados del repositorio del equipo [ShiiSuii/base_athome](https://github.com/ShiiSuii/base_athome) @ `6edb8a4`
(nombres de frames, montaje del brazo, cabezal, cámara de muñeca, límites del perfil `base_athome`) y de `robot_v2_description`
(este repositorio): ver la procedencia de cada valor en `config/geometry_v3.yaml`.

Avisos de copyright y licencia originales: `licenses/`. Apache-2.0: <https://www.apache.org/licenses/LICENSE-2.0>.
