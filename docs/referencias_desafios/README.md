# SolverBot — Referencias visuales para arena de desafíos y RL

Este paquete reúne las **12 capturas suministradas por el usuario** para que Claude Code Cloud pueda inspeccionarlas directamente. Se incluyen intactas.

## Orden e interpretación

1. `01_desafio_doing_laundry.png` — Doing Laundry: entrar por la puerta, navegar a laundry, encontrar prendas, tomar al menos camiseta, trasladarla, extenderla y doblarla. Extras abrir lavarropas, extracción, cesta y apilado.
2. `02_desafio_gpsr.png` — General Purpose Service Robot: entrar hacia Instruction Point, recibir 3 órdenes juntas o separadas, voz, navegación, búsqueda, traslado y retorno.
3. `03_desafio_pick_and_place.png` — Pick and Place: retirar seis objetos de mesa; cubierto, plato, taza/vaso, basura, otros dos. Ubicarlos en lavavajillas/basurero/armario; preparar desayuno con bol, cuchara, leche y cereales.
4. `04_desafio_human_robot_interaction.png` — HRI: dos invitados, saludo, nombre y bebida, mirar y seguir, ubicarlos, presentarlos, tomar una bolsa y depositarla donde indique el anfitrión. Timbrar/abrir puerta optativos.
5. `05_restricciones_seguridad.png` — Puerta de referencia de 2,00 m alto x 0,70 m ancho; e-stop rojo accesible, cables protegidos, velocidad máxima 6 km/h entre personas, autonomía, reinicios y tiempos de prueba. Ver imagen para frases exactas.
6. `06_objetos_bebidas_vajilla_limpieza.png` — Ejemplos fotografiados: jugos, leche, latas y bebidas, limpieza, cubiertos, taza, bol y esponja.
7. `07_objetos_despensa_vajilla.png` — Ejemplos fotografiados: gelatina, conservas, crema/canister, mostaza, cuchillo, cuchara, plato, vino. Productos de referencia, no exige reproducción de marca.
8. `08_objetos_frutas_latas.png` — Ejemplos fotografiados: frutas, banana, limon, naranja, manzana, latas de comida y azúcar; variabilidad de forma y agarre.
9. `09_objetos_snacks_y_juguetes.png` — Ejemplos fotografiados: snacks, cajas, pelotas, dado, cubo rompecabezas y frutas. Incluir formas diversas.
10. `10_plano_mobiliario_18_ubicaciones.png` — Plano esquemático de 4 habitaciones: Laundry (arriba izquierda), Bedroom (arriba derecha), Living room (abajo izquierda), Kitchen (abajo derecha). Leyenda numerada 1-18 (lavadora, mesa de ropa, sofá, lavavajillas, etc.). Escala métrica NO suministrada.
11. `11_plano_categorias_objetos.png` — Categorías coloridas en el plano: Snacks, Dishes, Cleaning Supplies, Fruits, Drinks, Foods, Fabrics, Toys. La ubicación corresponde a un esquema y puede variar.
12. `12_plano_zonas_evaluacion.png` — Cuatro diagramas de zonas de pruebas: DL, HRI, GPSR y P&P; muestran entrada, áreas objetivo e Instruction Point. Respetar layout cualitativo; no inferir dimensiones físicas de pixels.

## Datos y límites de interpretación

- Las imágenes corresponden a **cuatro desafíos distintos**: Doing Laundry (DL), General Purpose Service Robot (GPSR), Pick and Place (P&P) y Human Robot Interaction (HRI).
- El plano del hogar es esquemático: **NO ofrece escala métrica verificable** ni dimensiones de muebles/habitaciones. Preservar relaciones y zonas, parametrizar medidas provisionales y documentar supuestos.
- Restricción explícita: puerta de referencia de 200 cm de alto × 70 cm de ancho; no deducir automáticamente de la foto otras cotas.
- Realizar puntuación de RL por *hitos verificados*, no inventar sistema de puntos oficiales donde la captura no lo detalla.
- El robot objetivo es SolverBot v3 con base diferencial, elevador, brazo Piper, gripper, RealSense D435 en cabeza. La segunda cámara próxima a la pinza **no tiene especificaciones confirmadas**; representar un frame provisional y evitar inventar extrínsecos o intrínsecos de un sensor real.
- Para doblar camisetas de manera física hace falta una simulación de telas deformables, no basta un rectángulo rígido en Gazebo.

## Cómo usar

Copiar las imágenes y este README al repositorio de trabajo en `docs/referencias_desafios/`, y pedir a Claude que examine *todos* los PNG antes de implementar el escenario.
