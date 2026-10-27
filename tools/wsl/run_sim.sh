#!/bin/bash
# Lanza la simulación AISLADA del robot real:
#   * ROS_DOMAIN_ID propio (nunca 10, el del robot real) y solo localhost.
#   * GZ_PARTITION propia: varias instancias de Gazebo no se mezclan.
# Uso:  tools/wsl/run_sim.sh [instancia] [args de sim.launch.py ...]
#   tools/wsl/run_sim.sh 0                                   # arena + GUI + dos cámaras (demo)
#   tools/wsl/run_sim.sh 1 world:=pick_place_training headless:=true cameras:=false camera_profile:=train
# Para el entorno RL de esa instancia, exportar las MISMAS variables:  source tools/wsl/sim_env.sh 1
set -e
INSTANCE=${1:-0}; shift || true
source "$(dirname "$0")/sim_env.sh" "$INSTANCE"
exec ros2 launch solverbot_sim sim.launch.py "$@"
