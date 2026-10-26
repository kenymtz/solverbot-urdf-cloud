#!/usr/bin/env bash
# =====================================================================================================
#  Validación reproducible de robot_v3_description en la NUC (Ubuntu 22.04 + ROS 2 Humble).
#  NO usa hardware: no abre puertos serie, CAN ni USB, y corre en un dominio DDS propio (por defecto 42),
#  distinto del robot (10). No modifica otros workspaces: compila solo en el workspace indicado.
#
#  Uso:
#    bash tools/nuc_validate.sh <workspace> [--gazebo] [--domain N]
#      <workspace>   workspace colcon que contiene src/.../robot_v3_description (p. ej. ~/robot_v3_ws)
#      --gazebo      además prueba Ignition Gazebo Fortress en modo headless (sensores incluidos)
#  Salida: resumen en pantalla y registros en <workspace>/log/robot_v3_validation/
# =====================================================================================================
set -u
WS="${1:?Indique el workspace, p. ej. ~/robot_v3_ws}"
shift
GAZEBO=0
DOMAIN=42
while [ $# -gt 0 ]; do
  case "$1" in
    --gazebo) GAZEBO=1 ;;
    --domain) DOMAIN="$2"; shift ;;
    *) echo "Argumento desconocido: $1"; exit 2 ;;
  esac
  shift
done

if [ "$DOMAIN" = "10" ]; then echo "El dominio 10 es el del robot real. Use otro."; exit 2; fi
export ROS_DOMAIN_ID="$DOMAIN"
export ROS_LOCALHOST_ONLY=1

# shellcheck disable=SC1091
source /opt/ros/humble/setup.bash || { echo "No se encuentra ROS 2 Humble en /opt/ros/humble"; exit 2; }
cd "$WS" || exit 2
OUT="$WS/log/robot_v3_validation"
mkdir -p "$OUT"
declare -a RESULTS
run_step() {  # nombre, comando...
  local name="$1"; shift
  echo -e "\n===== $name ====="
  if "$@"; then RESULTS+=("OK     $name"); else RESULTS+=("FALLA  $name"); fi
}

stop_launch() {  # termina un launch en segundo plano y sus hijos
  local pid="$1"
  kill -INT "$pid" 2>/dev/null
  for _ in $(seq 1 15); do kill -0 "$pid" 2>/dev/null || return 0; sleep 1; done
  pkill -9 -P "$pid" 2>/dev/null; kill -9 "$pid" 2>/dev/null
}

run_step "colcon build" colcon build --packages-select robot_v3_description
# shellcheck disable=SC1091
source "$WS/install/setup.bash"
PKG_SHARE="$(ros2 pkg prefix robot_v3_description)/share/robot_v3_description"
SRC_DIR="$(dirname "$(find "$WS/src" -name package.xml -path '*robot_v3_description/*' | head -1)")"

run_step "xacro + check_urdf" bash -c "xacro '$PKG_SHARE/urdf/robot_v3.urdf.xacro' > '$OUT/robot_v3.urdf' && check_urdf '$OUT/robot_v3.urdf' > '$OUT/check_urdf.txt' && head -3 '$OUT/check_urdf.txt'"
REPO_ARG=()
if git -C "$SRC_DIR/.." rev-parse --git-dir > /dev/null 2>&1; then REPO_ARG=(--repo "$(git -C "$SRC_DIR/.." rev-parse --show-toplevel)"); fi
run_step "validación estática (validate_v3.py)" python3 "$SRC_DIR/tools/validate_v3.py" --package "$SRC_DIR" "${REPO_ARG[@]}" --markdown "$OUT/static.md"
run_step "colcon test" bash -c "colcon test --packages-select robot_v3_description > '$OUT/colcon_test.txt' 2>&1; colcon test-result --verbose | tail -3"

# --- ros2_control mock: estados simulados, controladores reales
ros2 launch robot_v3_description sim_mock.launch.py rviz:=false > "$OUT/sim_mock_launch.log" 2>&1 &
LP=$!
sleep 15
run_step "ros2_control mock: chequeos en ejecución" bash -c "ros2 run robot_v3_description runtime_checks.py | tee '$OUT/runtime_mock.txt' | tail -3; exit \${PIPESTATUS[0]}"
run_step "ros2_control mock: demo_motion.py" ros2 run robot_v3_description demo_motion.py
run_step "TF: view_frames" bash -c "cd '$OUT' && ros2 run tf2_tools view_frames > /dev/null 2>&1; ls frames*.gv frames*.pdf 2>/dev/null | grep -q frames"
stop_launch $LP

# --- head_mount:=lift y límites de base_athome (mismo chequeo, otra configuración)
ros2 launch robot_v3_description sim_mock.launch.py rviz:=false head_mount:=lift arm_limits:=base_athome > "$OUT/sim_mock_lift_launch.log" 2>&1 &
LP=$!
sleep 15
run_step "mock con head_mount:=lift arm_limits:=base_athome" bash -c "ros2 run robot_v3_description runtime_checks.py > '$OUT/runtime_mock_lift.txt'; r=\$?; tail -1 '$OUT/runtime_mock_lift.txt'; exit \$r"
stop_launch $LP

# --- real_state.launch.py con estados que imitan a la Jetson (dominio de prueba, nunca el 10)
ros2 launch robot_v3_description real_state.launch.py > "$OUT/real_state_launch.log" 2>&1 &
LP=$!
sleep 8
run_step "real_state.launch.py con /joint_states tipo Jetson" bash -c "ros2 run robot_v3_description check_real_state.py | tee '$OUT/real_state.txt' | tail -1; exit \${PIPESTATUS[0]}"
stop_launch $LP

if [ "$GAZEBO" = "1" ]; then
  ros2 launch robot_v3_description gazebo.launch.py headless:=true rviz:=false > "$OUT/gazebo_launch.log" 2>&1 &
  LP=$!
  sleep 45
  run_step "Gazebo Fortress: chequeos en ejecución con sensores" bash -c "ros2 run robot_v3_description runtime_checks.py --use-sim-time --tolerance 0.01 --sensors | tee '$OUT/runtime_gazebo.txt' | tail -3; exit \${PIPESTATUS[0]}"
  stop_launch $LP
fi

echo -e "\n===================== RESUMEN ====================="
printf '%s\n' "${RESULTS[@]}" | tee "$OUT/summary.txt"
echo "Registros: $OUT"
! grep -q '^FALLA' "$OUT/summary.txt"
