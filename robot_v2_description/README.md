# robot_v2_description

Descripción URDF/Xacro del robot @Home (base hoverboard + perfil elevador + monitor + brazo + cámara + Hokuyo),
reconstruida a partir de:

| Fuente | Qué aporta |
|---|---|
| `nuevo.glb` (escaneo iPhone 15) | Apariencia, posición de todas las piezas, centros de rueda, cámara, LiDAR |
| Repo `UruBots/base_athome` @ `d40e006` (`hoverboard_base`, `urg_node`) | Trocha y ancho de rueda calibrados, nombres de frames, convención `base_footprint`/`base_link`, malla CAD del Hokuyo, altura focal del láser |
| Modelo anterior `~/robot_scan_ws` | Posición de `upper_box_link` (re-escalada) |
| Fotos `este.jpeg`, `este2.jpeg` | Identificación de componentes y comparación visual (ver *Limitaciones*) |

ROS 2 Humble. No depende de Jazzy ni modifica otros paquetes.

## Uso

```bash
cd ~/robot_scan_v2_ws && source /opt/ros/humble/setup.bash
colcon build --packages-select robot_v2_description && source install/setup.bash
ros2 launch robot_v2_description display.launch.py              # RViz + sliders (estados simulados)
ros2 launch robot_v2_description display.launch.py gui:=false   # joint_state_publisher en ceros
# opciones: use_estimated_inertials:=true   with_camera_optical_frame:=false   rviz:=false
```

El launch **no** se conecta a hardware, no publica `cmd_vel` y no activa motores. Los estados de las articulaciones
son simulados (joint_state_publisher[_gui]).

## Árbol cinemático

```
base_footprint
└─ base_joint (fixed) ─ base_link                     centro del marco superior del chasis, sobre el piso
   ├─ left_wheel_joint  (continuous, eje +Y) ─ left_wheel     origen = eje real de la rueda
   ├─ right_wheel_joint (continuous, eje +Y) ─ right_wheel
   ├─ rear_left_caster_joint  (fixed) ─ rear_left_caster      rueda loca (pivote pasivo no modelado)
   ├─ rear_right_caster_joint (fixed) ─ rear_right_caster
   ├─ laser_mount_joint (fixed) ─ laser_mount ─ laser_joint (fixed, +0.0474 z) ─ laser
   └─ column_joint (fixed) ─ column_link                      perfil vertical
      ├─ lift_joint (prismatic, eje +Z) ─ upper_box_link       monitor + bandeja + carro
      │  └─ arm_mount_joint (fixed) ─ arm_link                 brazo (rígido, pose escaneada)
      └─ camera_mount_joint (fixed) ─ camera_mount_link        torreta + cámara + micrófono
         └─ camera_joint (fixed, pitch 5.55°) ─ camera_link
            └─ camera_optical_joint (fixed, -90°/0/-90°) ─ camera_optical_frame
```

14 links, 13 joints: 2 continuos (ruedas), 1 prismático (elevador), 10 fijos.

## Posiciones (base_link, metros) y procedencia

| Frame | xyz | Cómo se obtuvo |
|---|---|---|
| left_wheel | 0.0815, 0.2835, 0.0837 | círculo LSQ al arco superior del neumático (MAD 0.1 mm) + cara exterior del flanco − ancho/2 |
| right_wheel | 0.0865, −0.2786, 0.0820 | ídem |
| rear_left_caster | −0.1475, 0.2228, 0.0398 | centro del aro naranja (captura parcial, ±1–2 cm) |
| rear_right_caster | −0.1780, −0.2459, 0.0406 | ídem |
| laser_mount | 0.2429, 0.0082, 0.3165 | centro del cuerpo del Hokuyo; base apoyada en la placa negra |
| laser | 0.2429, 0.0082, 0.3639 | laser_mount + 0.0474 (punto focal, `urg_node/hokuyo_ust10.urdf.xacro`) |
| column_link | −0.0350, −0.0026, 0.3300 | centro de la sección del perfil; z = arranque sobre el chasis |
| upper_box_link | 0.0817, 0.0018, 0.9787 | modelo anterior re-escalado; coincide con el borde inferior de la pantalla del escaneo nuevo (0.980) |
| arm_link | 0.0007, 0.1850, 1.0763 | unión del brazo con el costado izquierdo de la caja (aprox.) |
| camera_mount_link | −0.0350, −0.0026, 1.3211 | placa superior del perfil |
| camera_link | 0.0394, −0.0029, 1.3707 | centro de la carcasa; orientación = normal de la cara frontal (pitch 5.55° abajo, yaw 0.34°) |

Valores completos y generados: `config/robot_geometry.yaml`.

## Escala del escaneo (hallazgo importante)

**El escaneo no está en escala métrica.** En bruto, el robot mide 1.09 m y las ruedas tienen r ≈ 0.0585 m.
La escala se fijó en **s = 1.4322** haciendo coincidir la distancia entre caras exteriores de los neumáticos
(0.4308 u.) con trocha + ancho del repo (0.562 + 0.055 = 0.617 m). Contraste independiente:

| Referencia | Resultado con s | Esperado |
|---|---|---|
| Cuerpo del Hokuyo | ≈ 0.050 m | 50 mm (CAD) |
| Radio geométrico del neumático | 0.0840 m | 0.0825 (6.5") / 0.0853 (calibrado) |
| Ancho del neumático | ≈ 0.055 m | 0.055 (repo) |
| Pantalla | 0.345 × 0.198 m | monitor 15.6" 16:9 = 0.345 × 0.194 |
| Mesas de la sala | ≈ 0.74 m | altura estándar |
| Altura total del robot | **1.45 m** | — |

El modelo anterior (`robot_scan_ws`) tenía el mismo problema: medía 1.09 m en vez de ~1.45 m.
**Verificación pendiente:** medir con cinta una dimensión (p. ej. la altura de la placa superior del perfil, 1.321 m según el modelo).

## Diferencias con el URDF del repo (`hoverboard_base/robot.urdf.xacro`)

| Elemento | Repo | Robot real (escaneo) | Decisión |
|---|---|---|---|
| Chasis | caja 0.48×0.48×0.05 a 4 cm del piso | marco de perfiles de dos niveles, ≈0.53 × 0.47 m, desde ≈0.11 hasta ≈0.37 m del piso | malla escaneada; colisión = caja medida |
| Trocha | 0.562 (calibrada) | coincide (por construcción de la escala) | se usa |
| Radio de rueda | 0.0853 (efectivo, odometría) | 0.0840 geométrico; eje a 0.083 m del piso | visual 0.0840; colisión = altura del eje (0.0829) |
| Eje de ruedas en X | +0.095 | +0.084 (media de ambas ruedas) | se usa lo medido |
| Rueda loca | 1, centrada, x = −0.20 | **2**, en las esquinas traseras | dos links fijos |
| LiDAR (`laser`) | (0.22, 0, 0.18) | (0.243, 0.008, 0.364) | se usa lo medido; nombre `laser` conservado |
| Perfil, caja, brazo, cámara | no existen | presentes | agregados |

Nombres conservados para compatibilidad: `base_footprint`, `base_link`, `base_joint`, `left_wheel`, `right_wheel`,
`left_wheel_joint`, `right_wheel_joint`, `laser` (usado por urg_node, Nav2 y slam_toolbox).

## Mallas y texturas

* Visuales: COLLADA (`.dae`) con textura JPG propia por link. Se verificó con assimp (el cargador de RViz) y en RViz.
* Proceso por pieza (`~/robot_scan_v2_ws/tools`):
  1. Segmentación por regiones medidas.
  2. Eliminación de islas chicas y fragmentos flotantes, y del "faldón" de piso bajo el chasis.
  3. **Aplanado de superficies planas** (RANSAC: planos ≥ 40 cm², tolerancia 2 mm; solo proyecta vértices de esos planos).
  4. Simplificación controlada (quadric; 590k caras de escaneo → 285k en total).
  5. UV nuevas con xatlas.
  6. **Horneado de textura** proyectando cada texel al triángulo más cercano de la malla original de alta resolución (mediana < 0.3 mm).

  Así cada pieza tiene un atlas dedicado de hasta 4096 px. En los modelos anteriores, el robot ocupaba una fracción de un atlas de toda la sala, y por eso se veía borroso.
* Ruedas: **cilindros limpios** (r 0.084, ancho 0.055) con textura horneada desde el escaneo (llanta de 5 rayos y neumático visibles).
  La geometría de rueda del escaneo se excluyó de `base` (0 vértices duplicados), así que no hay duplicación al girar.
* LiDAR: malla CAD `hokuyo_ust10.stl` de `urg_node` (BSD; misma carcasa que el UST-05LX). El Hokuyo escaneado se excluyó de `base`.
* Colisiones: primitivas simples (cajas, cilindros, esferas). Nunca la malla escaneada.

## Parámetros estimados (separados)

`config/estimated_parameters.yaml`, usado solo por `urdf/estimated.xacro`:
* Límites de `lift_joint` [−0.19, +0.068] m: **envolvente geométrica** del escaneo (choque brazo–rueda / tope del perfil), no la carrera real. effort/velocity = 0 (desconocidos).
* Inercias opcionales (`use_estimated_inertials:=true`): masas de marcador (excepto 12 kg del chasis, tomado del repo y no verificado), con inercias de sólidos homogéneos. Solo para simulación preliminar.

## Validación ejecutada (`tools/validate_v2.py`, 20/20 OK)

* xacro + `check_urdf` con dos combinaciones de argumentos.
* Todos los recursos existen (mallas y texturas referenciadas por los DAE).
* Árbol: una raíz, sin ciclos, 14 links conectados.
* TF en ejecución (robot_state_publisher) igual a las posiciones medidas (error ≤ 1e-16 m).
* Ruedas con estados simulados (1.1 rad): centro fijo, radio conservado, ángulo exacto, eje = +Y de base_link.
* `lift_joint` +0.05: caja y brazo suben 0.05; cámara y perfil no se mueven.
* `camera_optical_frame`: Z adelante, X derecha, Y abajo (REP-103), pitch 5.55°.
* Sin duplicación: 0 vértices de `base` dentro de las ruedas o del Hokuyo.
* RViz: carga sin errores, con texturas; verificado con capturas en estado cero y con el elevador movido.

## Integración con el robot real: qué falta y cuidados

1. **Medir con cinta** para confirmar la escala (±3 %).
2. **No lanzar junto con `hoverboard_base/bringup.launch.py` sin cambios:** ese launch levanta su propio
   robot_state_publisher con `robot.urdf`, y habría TF duplicadas de `base_link`, las ruedas y `laser`. Hay que usar uno u otro.
3. El driver del hoverboard publica `odom → base_footprint`, pero **no publica `/joint_states`**. Sin un
   joint_state_publisher, robot_state_publisher no publica las ruedas ni `upper_box_link`/`arm_link` (cuelgan del prismático).
   Hace falta un joint_state_publisher (ceros) o un publicador real de ruedas y elevador.
4. **Odometría y `base_footprint`:** el repo pone `base_footprint` en el centro del chasis, pero el centro de giro
   diferencial está en el eje de ruedas (+0.084 m en X). Se mantuvo la convención del repo para no cambiar Nav2 ni el LiDAR. Revisar si
   afecta la odometría al girar en el lugar.
5. **Cámara:** el driver Orbbec publica `camera_link → camera_*_optical_frame`. Este URDF termina en `camera_link`;
   `camera_optical_frame` usa otro nombre (no colisiona). Confirmar el modelo de cámara y su punto de referencia.
6. **Pendientes de datos:** carrera real y velocidad del elevador; articulaciones del brazo (modelo, ejes, límites);
   si la torreta de la cámara es un pan; pivote de las ruedas locas; masas reales.
7. Con `laser` 18 cm más alto que en el URDF del repo: si el mapa se hizo con el URDF viejo, la altura del láser no
   afecta a un scan 2D, pero el desplazamiento en x/y (+2.3 cm / +0.8 cm) sí afecta levemente.

## Limitaciones

* `este.jpeg` y `este2.jpeg` **no son fotografías de cámara**: son capturas del render del propio escaneo (fondo blanco,
  artefactos y una persona fusionada). Sirvieron para identificar componentes y comparar, pero no aportan texturas nuevas.
  Las texturas provienen solo del escaneo. Para mejorarlas haría falta fotos reales, frontales y bien iluminadas.
* Zonas ocultas en el escaneo: la cara interior de las ruedas (textura difusa), las ruedas locas (captura parcial) y la parte inferior del chasis.
* El brazo es un único link rígido en la pose escaneada: no hay información de sus articulaciones.
* Los cables y piezas finas conservan el ruido del escaneo.
