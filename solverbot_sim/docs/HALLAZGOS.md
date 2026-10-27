# Hallazgos de ingeniería (fase 1: Pick and Place)

Todos surgieron de ejecuciones reales en Gazebo Harmonic 8.10 (no de suposiciones). Los que afectan al robot real
están marcados **[robot real]**: conviene verificarlos físicamente.

1. **[robot real] La D435 de cabeza no ve la mesa de costado.** Con `head_mount:=column`, la placa superior de la
   columna (z 1.311–1.321 m, x ≤ 0.03 m) queda 2–5 cm por debajo de la cámara. Al girar a la izquierda (pan ≈ 1.2)
   e inclinar > ~20°, la cámara queda sobre la placa y esta llena la imagen (más cerca que el plano near de 0.105 m:
   profundidad −inf). Hacia adelante la vista es libre. Consecuencia: la localización del objeto para la manipulación
   lateral se hace con la **cámara de muñeca**; la de cabeza sirve para vista frontal/global.
   Verificar en el robot real si la D435 ve la placa al mirar abajo-izquierda.
2. **[robot real] Montaje lateral del Piper y aproximación vertical.** El eje del brazo apunta a +Y (izquierda).
   Con el elevador en 0 la pinza puede bajar vertical sobre la mesa de 0.74 m en el centro de la zona; con el
   elevador ≤ −0.10 ya no (joint5 llega a su límite de ±70°). Aproximación usada: 20° respecto de la vertical.
3. **[robot real] Con el elevador abajo el brazo en reposo (DEFAULT_HOME) choca con la rueda** (coincide con la nota
   de `sim_provisional.yaml`): plegar el brazo con el elevador arriba.
4. **[robot real] Bloque de la cámara de muñeca vs. altura del objeto.** El bloque Orbbec (geometría de base_athome)
   queda 4 cm detrás de las puntas de los dedos: un objeto que sobresale más de ~4 cm por encima del punto de agarre
   choca con el bloque y la pinza se traba. El cubo de entrenamiento es de 4 cm; objetos altos (latas, botellas)
   requerirán agarre lateral o tomarlos más arriba.
5. **`tcp_link` está en las puntas de los dedos** (las mallas de los dedos ocupan z ∈ [−0.076, 0] en ese frame).
6. **Deriva de la base.** Con ruedas locas sin fricción, la reacción del brazo desplaza/gira la base (hasta 3.5 cm y
   6° observados). Las recompensas se calculan en el frame de la base (pose real del modelo) y el reset la devuelve a
   su pose si derivó > 1 cm o 1°.
7. **Agarre en Gazebo.** Validado: dedos detenidos por el objeto (j7 = 0.023 m, media anchura 0.020), objeto
   levantado 10 cm y sostenido. En algunos episodios el cubo rota dentro de la pinza (agarre por la arista, j7 ≈ 7 mm)
   y se mantiene sujeto; el criterio "sujeto" lo contempla (dedos no cerrados del todo + objeto junto al TCP +
   elevado). Ajuste pendiente de contacto/fricción si se buscan agarres más firmes.
8. **gz_ros2_control (Jazzy 1.2) no tolera el reset completo del mundo** (`/world/<w>/control reset`): los
   controladores dejan de actualizar. Se usa reset suave (controladores a la pose inicial + teletransporte de base y
   objeto); ante un fallo de reset el entorno lanza excepción y se debe reiniciar el proceso de simulación.
9. **Arranque en pausa**: con el mundo pausado los spawners no activan controladores. El mundo arranca corriendo y el
   entorno RL lo pausa al conectarse; luego avanza con `multi_step`.
10. **`/joint_states` va algunos ciclos detrás de `/clock`** (hasta ~36 ms): la sincronización exige un estado
    articular nuevo posterior a cada paso, no una tolerancia absoluta.
11. **Render por CPU**: Gazebo Harmonic funciona sin GPU con Mesa llvmpipe (EGL o Xvfb). ~0.65 s por paso de RL
    (0.1 s simulados) con cámaras de 160×120/96×96; con GPU en WSL debería ser bastante más rápido.
12. **El tilt positivo mira hacia abajo** (verificado por FK e imagen).

No verificado todavía: conducción de la base con `diff_drive_controller` en Harmonic (el controlador se activa;
no se probó navegación), LiDAR (publica, sin análisis), GUI de Gazebo y rqt_image_view en WSLg.
