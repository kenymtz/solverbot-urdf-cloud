#!/bin/bash
# Orquestador en PRIMER PLANO (lo usan start.sh, tmux o systemd): simulación headless -> trabajo -> apagado limpio.
#   run_experiment.sh <instancia> validate [args de validate_grasp.py]
#   run_experiment.sh <instancia> train <run> [args de train.py]      (reanuda solo si existe runs/<run>/checkpoints)
#   run_experiment.sh <instancia> py <dir> <script.py> [args]          (cualquier script de solverbot_rl/scripts;
#                                                                       logs en <dir>/logs, p. ej. runs/reach_bc_v1)
# Variables: SIM_ARGS (extra para sim.launch.py), SIM_READY_TIMEOUT (s, def. 240).
# sin "set -u": los setup.bash de ROS/colcon usan variables no definidas
INSTANCE="$1"; MODE="$2"; shift 2
source "$(dirname "$0")/common.sh" "$INSTANCE"
cd "$REPO"
STAMP=$(date +%Y%m%d_%H%M%S)
if [ "$MODE" = train ]; then RUN="$1"; shift; LOGDIR="runs/$RUN/logs"
elif [ "$MODE" = py ]; then PYDIR="$1"; PYSCRIPT="$2"; shift 2; LOGDIR="$PYDIR/logs"
else LOGDIR="runs/validate_$STAMP/logs"; fi
mkdir -p "$LOGDIR"
SIM_LOG="$LOGDIR/sim_${STAMP}_i$INSTANCE.log"; JOB_LOG="$LOGDIR/${MODE}_${PYSCRIPT:+$(basename "$PYSCRIPT" .py)_}${STAMP}_i$INSTANCE.log"
echo "[run] instancia=$INSTANCE ROS_DOMAIN_ID=$ROS_DOMAIN_ID GZ_PARTITION=$GZ_PARTITION modo=$MODE logs=$LOGDIR"

if timeout 8 gz topic -l 2>/dev/null | grep -q "^/clock" || \
   timeout 8 ros2 node list 2>/dev/null | grep -q controller_manager; then
  echo "[run] ERROR: ya hay una simulación/controller_manager en ROS_DOMAIN_ID=$ROS_DOMAIN_ID. Ejecutar stop.sh $INSTANCE" >&2
  exit 3
fi

JOB_PID=""; SIM_PID=""
cleanup() {
  [ -n "$JOB_PID" ] && kill -0 "$JOB_PID" 2>/dev/null && { kill -TERM "$JOB_PID"; wait "$JOB_PID" 2>/dev/null; }
  if [ -n "$SIM_PID" ] && kill -0 "$SIM_PID" 2>/dev/null; then
    kill -INT -- "-$SIM_PID" 2>/dev/null; for _ in $(seq 1 20); do kill -0 "$SIM_PID" 2>/dev/null || break; sleep 1; done
    kill -KILL -- "-$SIM_PID" 2>/dev/null
  fi
  echo "[run] apagado $(date)"
}
trap 'echo "[run] señal recibida"; cleanup; exit 130' INT TERM

# Simulación sin GUI. Sin GPU, los sensores se renderizan por EGL con Mesa (llvmpipe).
setsid ros2 launch solverbot_sim sim.launch.py world:=pick_place_training headless:=true egl:=true cameras:=false \
  rviz:=false camera_profile:=train ${SIM_ARGS:-} > "$SIM_LOG" 2>&1 &
SIM_PID=$!
echo "$SIM_PID" > "$STATE_DIR/instance_${INSTANCE}.sim.pid"
T0=$(date +%s)
until grep -q "activated head_position_controller\|activated diff_drive_controller" "$SIM_LOG"; do
  if ! kill -0 "$SIM_PID" 2>/dev/null || [ $(( $(date +%s) - T0 )) -gt "${SIM_READY_TIMEOUT:-240}" ]; then
    echo "[run] ERROR: la simulación no quedó lista (ver $SIM_LOG)" >&2; cleanup; exit 4
  fi
  sleep 2
done
echo "[run] simulación lista en $(( $(date +%s) - T0 )) s"

if [ "$MODE" = validate ]; then
  python3 solverbot_rl/scripts/validate_grasp.py --backend gazebo --out "${LOGDIR%/logs}" "$@" > "$JOB_LOG" 2>&1 &
elif [ "$MODE" = py ]; then
  python3 "solverbot_rl/scripts/$(basename "$PYSCRIPT")" "$@" > "$JOB_LOG" 2>&1 &
else
  python3 solverbot_rl/scripts/train.py --backend gazebo --run "$RUN" "$@" > "$JOB_LOG" 2>&1 &
fi
JOB_PID=$!
wait "$JOB_PID"; RC=$?
JOB_PID=""
echo "[run] trabajo terminado con código $RC"; tail -n 3 "$JOB_LOG"
cleanup
exit $RC
