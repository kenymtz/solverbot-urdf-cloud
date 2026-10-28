## Fase A — cámara de cabeza (auditoría estática)

| cámara | frame | resolución | HFOV (de K) | VFOV | distorsión | frecuencia (tiempo sim.) | antigüedad tras un paso |
|---|---|---|---|---|---|---|---|
| head | `camera_color_optical_frame` | 160×120 | 69.0° | 54.5° | plumb_bob [0.0, 0.0, 0.0, 0.0, 0.0] | 10.0 Hz (49 cuadros) | 0.099 s |
| wrist | `wrist_sim_camera_optical_frame` | 96×96 | 80.2° | 80.2° | plumb_bob [0.0, 0.0, 0.0, 0.0, 0.0] | 10.0 Hz (49 cuadros) | 0.099 s |

Profundidad alineada: 10.0 Hz; desfase de sello RGB−profundidad en la pose configurada: 0.0 s.

Pose configurada pan=1.2, tilt=0.78: cámara en [-0.014, 0.042, 1.345] (base), eje óptico [0.258, 0.663, -0.703]; proyección del cubo real en (84.4, 62.2) px a 0.814 m; profundidad medida allí: -inf; píxeles con profundidad −inf (más cerca que near = 0,105 m): 93 %; estado: **occluded**.

Barrido pan/tilt (cubo en la posición nominal, brazo en home): estado / píxeles de la máscara / % −inf

| pan \ tilt | -0.30 | +0.00 | +0.20 | +0.40 | +0.60 | +0.79 |
|---|---|---|---|---|---|---|
| +0.00 | behind / 0 / 0 | out_of_fov / 0 / 0 | out_of_fov / 0 / 0 | out_of_fov / 0 / 0 | out_of_fov / 0 / 0 | out_of_fov / 0 / 0 |
| +0.60 | out_of_fov / 0 / 0 | out_of_fov / 0 / 0 | out_of_fov / 0 / 0 | too_small / 0 / 1 | occluded / 0 / 12 | occluded / 0 / 40 |
| +0.90 | out_of_fov / 4 / 0 | out_of_fov / 0 / 0 | out_of_fov / 2 / 1 | occluded / 0 / 17 | occluded / 1 / 45 | occluded / 0 / 77 |
| +1.20 | out_of_fov / 0 / 0 | out_of_fov / 2 / 0 | out_of_fov / 2 / 6 | occluded / 2 / 32 | occluded / 0 / 62 | occluded / 2 / 93 |
| +1.40 | out_of_fov / 2 / 0 | out_of_fov / 2 / 0 | out_of_fov / 0 / 11 | occluded / 4 / 39 | occluded / 3 / 68 | occluded / 0 / 98 |
| +1.57 | out_of_fov / 3 / 0 | out_of_fov / 4 / 0 | out_of_fov / 2 / 15 | occluded / 3 / 42 | occluded / 2 / 71 | occluded / 0 / 100 |

Mapa de visibilidad de la mesa (cabeza apuntada a cada posición, 40 posiciones): occluded: 40; detecciones: 0.

| control positivo (cubo en el aire, vista libre) | estado | píxeles | |proyección − máscara| (px) | error 3D (mm) |
|---|---|---|---|---|
| [0.55, 0.0, 1.38] | visible | 90 | 0.97 | 20.6 |
| [0.65, 0.25, 1.3] | visible | 70 | 1.20 | 15.9 |
| [0.75, -0.25, 1.45] | visible | 49 | 1.03 | 17.0 |
| [0.5, 0.1, 1.25] | visible | 123 | 1.17 | 17.7 |

## Fase B — cobertura por cámara (misma trayectoria, registro independiente)

| corrida | episodios | pasos | muñeca válida | cabeza detecta | solo muñeca | solo cabeza | ambas | ninguna | err. muñeca (mm) | err. muñeca con FK en el instante de la imagen (mm) | antigüedad imagen (s) | estado cabeza (pasos) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bc_v1_reach | 3 | 183 | 52.5 % | 0.0 % | 52.5 % | 0.0 % | 0.0 % | 47.5 % | 10.0 (máx 42.3) | 7.5 (máx 26.2) | 0.070 (máx 0.070) | occluded 183 |
| bc_v2w_var_m | 12 | 516 | 54.3 % | 0.0 % | 54.3 % | 0.0 % | 0.0 % | 45.7 % | 12.4 (máx 41.6) | 8.7 (máx 32.1) | 0.062 (máx 0.127) | occluded 516 |

| corrida | semilla | pasos | muñeca válida | cabeza detecta | ninguna |
|---|---|---|---|---|---|
| bc_v1_reach | 902000 | 47 | 38 % | 0 % | 62 % |
| bc_v1_reach | 902001 | 88 | 66 % | 0 % | 34 % |
| bc_v1_reach | 902002 | 48 | 42 % | 0 % | 58 % |
| bc_v2w_var_m | 902002 | 47 | 49 % | 0 % | 51 % |
| bc_v2w_var_m | 902003 | 39 | 49 % | 0 % | 51 % |
| bc_v2w_var_m | 903000 | 47 | 43 % | 0 % | 57 % |
| bc_v2w_var_m | 903001 | 37 | 89 % | 0 % | 11 % |
| bc_v2w_var_m | 903002 | 50 | 40 % | 0 % | 60 % |
| bc_v2w_var_m | 903003 | 41 | 100 % | 0 % | 0 % |
| bc_v2w_var_m | 903004 | 37 | 73 % | 0 % | 27 % |
| bc_v2w_var_m | 903005 | 16 | 100 % | 0 % | 0 % |
| bc_v2w_var_m | 903006 | 51 | 37 % | 0 % | 63 % |
| bc_v2w_var_m | 903007 | 50 | 24 % | 0 % | 76 % |
| bc_v2w_var_m | 903008 | 46 | 48 % | 0 % | 52 % |
| bc_v2w_var_m | 903009 | 55 | 51 % | 0 % | 49 % |

## Fase F — contrafácticos de la observación del objeto (misma política congelada, mismas semillas)

| política / nivel | modo | privilegiado | éxitos | IC 95 % | dmin medio (mm) | dfinal medio (mm) | TCP z final | pasos | bandera válida vista por la política | error de la estimación vista (mm) | fallos |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bc_v1_var_l | perception | no | 20/30 | [49 %, 81 %] | 32.4 | 52.1 | -0.858 | 53.0 | 60 % | 19.8 | [900302, 900311, 900312, 900313, 900315, 900316, 900318, 900321, 900324, 900328] |
| bc_v1_var_l | oracle | sí | 12/30 | [25 %, 58 %] | 35.5 | 56.3 | -0.894 | 64.2 | 100 % | 0.0 | [900300, 900302, 900304, 900305, 900307, 900308, 900311, 900312, 900313, 900314, 900315, 900316, 900318, 900319, 900320, 900321, 900324, 900327] |
| bc_v2w_reach | perception | no | 16/20 | [58 %, 92 %] | 38.5 | 46.5 | -0.884 | 58.8 | 38 % | 17.2 | [901003, 901010, 901012, 901018] |
| bc_v2w_reach | first_estimate | no | 20/20 | [84 %, 100 %] | 27.8 | 27.8 | -0.935 | 64.6 | 100 % | 6.3 | [] |
| bc_v2w_reach | oracle | sí | 20/20 | [84 %, 100 %] | 26.7 | 26.7 | -0.930 | 69.0 | 100 % | 0.0 | [] |
| bc_v2w_var_m | perception | no | 18/20 | [70 %, 97 %] | 31.8 | 43.8 | -0.880 | 42.3 | 52 % | 21.5 | [902009, 902017] |
| bc_v2w_var_m | hold_valid | no | 20/20 | [84 %, 100 %] | 25.3 | 25.3 | -0.912 | 42.0 | 100 % | 24.3 | [] |
| bc_v2w_var_m | first_estimate | no | 18/20 | [70 %, 97 %] | 29.6 | 38.9 | -0.912 | 48.8 | 100 % | 5.9 | [902010, 902017] |
| bc_v2w_var_m | oracle | sí | 17/20 | [64 %, 95 %] | 29.1 | 33.1 | -0.919 | 50.7 | 100 % | 0.0 | [902008, 902010, 902017] |
| bc_v2w_var_s | perception | no | 22/30 | [56 %, 86 %] | 36.3 | 62.4 | -0.842 | 52.8 | 48 % | 16.4 | [901101, 901111, 901113, 901114, 901118, 901120, 901126, 901129] |
| bc_v2w_var_s | oracle | sí | 25/30 | [66 %, 93 %] | 33.8 | 48.5 | -0.881 | 56.9 | 100 % | 0.0 | [901113, 901114, 901120, 901128, 901129] |
| bc_v2w_var_s_905k | perception | no | 49/60 | [70 %, 89 %] | 31.1 | 44.5 | -0.879 | 48.7 | 53 % | 17.4 | [905002, 905017, 905020, 905023, 905026, 905027, 905030, 905035, 905048, 905049, 905052] |
| bc_v2w_var_s_905k | first_estimate | no | 52/60 | [76 %, 93 %] | 29.2 | 42.3 | -0.891 | 48.8 | 100 % | 6.2 | [905007, 905014, 905030, 905035, 905046, 905049, 905052, 905054] |

Comparación PAREADA contra `perception` (mismas semillas): b = falla con percepción y tiene éxito con el modo, c = al revés; McNemar exacto bilateral.

| grupo | modo | semillas comunes | b (recupera) | c (pierde) | p |
|---|---|---|---|---|---|
| bc_v1_var_l | oracle | 30 | 1 | 9 | 0.021 |
| bc_v2w_reach | first_estimate | 20 | 4 | 0 | 0.125 |
| bc_v2w_reach | oracle | 20 | 4 | 0 | 0.125 |
| bc_v2w_var_m | hold_valid | 20 | 2 | 0 | 0.500 |
| bc_v2w_var_m | first_estimate | 20 | 1 | 1 | 1.000 |
| bc_v2w_var_m | oracle | 20 | 1 | 2 | 1.000 |
| bc_v2w_var_s | oracle | 30 | 4 | 1 | 0.375 |
| bc_v2w_var_s_905k | first_estimate | 60 | 7 | 4 | 0.549 |

## Reproducibilidad por semilla entre lanzamientos de la simulación

| lanzamiento | modo | antigüedad mediana de la imagen de muñeca (s) | resultado por semilla (éxito, dfinal mm, pasos) |
|---|---|---|---|
| launch1 | oracle | 0.105 | 902003: ✓ 27 / 50; 902017: ✗ 341 / 56 |
| launch2 | oracle | 0.078 | 902003: ✓ 27 / 50; 902017: ✗ 341 / 56 |
| launch1 | perception | 0.079 | 902003: ✓ 28 / 48; 902009: ✗ 253 / 51; 902017: ✓ 25 / 54 |
| launch2 | perception | 0.051 | 902003: ✓ 28 / 48; 902009: ✗ 40 / 101; 902017: ✓ 29 / 37 |
| launch3 | perception | 0.009 | 902003: ✓ 25 / 49; 902009: ✓ 27 / 42; 902017: ✓ 29 / 36 |
| launch4 | perception | 0.075 | 902003: ✓ 28 / 48; 902009: ✗ 253 / 51; 902017: ✓ 25 / 36 |
| launch5 | perception | 0.095 | 902003: ✓ 29 / 43; 902009: ✗ 197 / 51; 902017: ✗ 368 / 56 |

## Backend cinemático (detector simulado continuo, sin campo visual ni física)

| corrida | éxitos | IC 95 % | dmin medio | dfinal medio | TCP z final | fallos |
|---|---|---|---|---|---|---|
| bc_v1_reach_var_m | 185/200 | [88 %, 95 %] | 26.1 | 28.9 | -0.928 | 15 semillas |
| bc_v2w_reach | 20/20 | [84 %, 100 %] | 25.6 | 25.6 | -0.932 | [] |
| bc_v2w_reach_var_m | 194/200 | [94 %, 99 %] | 26.5 | 27.7 | -0.916 | [902021, 902098, 902112, 902115, 902127, 902192] |
| bc_v2w_reach_var_s | 199/200 | [97 %, 100 %] | 25.7 | 26.9 | -0.925 | [902340] |

## Guardián de compromiso de rama (cinemático, mismas semillas base / guardián)

| corrida | éxitos | IC 95 % | dmin medio | dfinal medio | TCP z final | fallos |
|---|---|---|---|---|---|---|
| kin_base_bc_v2w_reach | 20/20 | [84 %, 100 %] | 26.3 | 26.3 | -0.931 | [] |
| kin_base_bc_v2w_reach_var_l | 188/200 | [90 %, 97 %] | 26.4 | 34.0 | -0.901 | 12 semillas |
| kin_base_bc_v2w_reach_var_m | 196/200 | [95 %, 99 %] | 25.6 | 27.8 | -0.922 | [904036, 904051, 904078, 904168] |
| kin_base_bc_v2w_reach_var_s | 200/200 | [98 %, 100 %] | 25.3 | 25.3 | -0.930 | [] |
| kin_guard_bc_v2w_reach | 20/20 | [84 %, 100 %] | 26.0 | 26.0 | -0.930 | [] |
| kin_guard_bc_v2w_reach_var_l | 185/200 | [88 %, 95 %] | 27.1 | 41.4 | -0.873 | 15 semillas |
| kin_guard_bc_v2w_reach_var_m | 195/200 | [94 %, 99 %] | 26.2 | 30.1 | -0.908 | [904005, 904026, 904054, 904091, 904132] |
| kin_guard_bc_v2w_reach_var_s | 194/200 | [94 %, 99 %] | 26.4 | 29.5 | -0.916 | [904064, 904083, 904092, 904120, 904136, 904148] |
