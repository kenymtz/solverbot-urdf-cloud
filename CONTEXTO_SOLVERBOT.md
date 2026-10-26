# SolverBot - Continuación de desarrollo

## Entorno original
- NUC Ubuntu
- ROS 2 Humble
- Jetson como controladora del brazo
- Modelo anterior: robot_v2_description
- Nuevo proyecto objetivo: robot_v3_description

## Repositorio actualizado
https://github.com/ShiiSuii/base_athome

## Estado anterior
El URDF v2 funciona en RViz y superó 20/20 validaciones
con estados articulares simulados.

El workspace v3 fue creado, pero todavía no contiene
el paquete robot_v3_description.

## Objetivo del nuevo URDF
Crear una versión simplificada, liviana y funcional
basada en el repositorio y el modelo escaneado.

Componentes:
- Base diferencial con ruedas.
- Elevador prismático.
- Brazo Piper completamente articulado.
- RealSense D435 RGB-D.
- Pan-tilt si corresponde.
- Hokuyo UST-05LX.

## Últimos hallazgos
- Falta el wrapper ROS de RealSense en el entorno inspeccionado.
- Falta piper_sdk en ese entorno.
- El brazo Piper se controla desde la Jetson.
- La NUC dispone de Gazebo Classic 11 e Ignition Fortress 6.18.
- Estaba pendiente auditar cinemática, límites, poses y frames.

## Restricciones
No activar motores ni actuadores reales.
No modificar firmware ni controladores existentes.
No modificar otros workspaces.
No utilizar sudo sin autorización.
No tocar ~/limoatwork.

## Objetivo
Generar un único URDF/Xacro cinemáticamente correcto
para RViz, simulación y futura integración con hardware real.

Los estados articulares reales deben poder actualizar RViz.
Los controladores reales y simulados deben estar separados.

No inventar dimensiones, límites ni características mecánicas.
