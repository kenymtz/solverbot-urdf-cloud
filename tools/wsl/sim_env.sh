#!/bin/bash
# source tools/wsl/sim_env.sh <instancia>  -> entorno ROS/Gazebo aislado para esa instancia de simulación
INSTANCE=${1:-0}
ROS_SETUP=${SOLVERBOT_ROS_SETUP:-/opt/ros/jazzy/setup.bash}
if [ -f "$ROS_SETUP" ]; then source "$ROS_SETUP"
elif ! command -v ros2 >/dev/null; then echo "[sim_env] ROS 2 no encontrado ($ROS_SETUP)" >&2; return 1 2>/dev/null || exit 1; fi
WS=${SOLVERBOT_WS:-$HOME/solverbot_ws}
[ -f "$WS/install/setup.bash" ] && source "$WS/install/setup.bash"
[ -f "$HOME/solverbot_venv/bin/activate" ] && source "$HOME/solverbot_venv/bin/activate"
export ROS_DOMAIN_ID=$((42 + INSTANCE))          # 42, 43, ... (el robot real usa 10)
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
export GZ_PARTITION="solverbot_sim_${INSTANCE}"
if [ "$ROS_DOMAIN_ID" = "10" ]; then echo "ROS_DOMAIN_ID=10 es el del robot real" >&2; return 1 2>/dev/null || exit 1; fi
echo "[sim_env] instancia=$INSTANCE ROS_DOMAIN_ID=$ROS_DOMAIN_ID GZ_PARTITION=$GZ_PARTITION"
