#!/bin/bash
# Preparación reproducible de una PC Ubuntu 24.04 + ROS 2 Jazzy (ya instalado) para simulación/entrenamiento headless.
# Verifica dependencias, crea el workspace colcon (enlaces a este repo), el venv y corre las pruebas rápidas.
set -e
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
WS="${SOLVERBOT_WS:-$HOME/solverbot_ws}"
[ -f /opt/ros/jazzy/setup.bash ] || { echo "Falta ROS 2 Jazzy en /opt/ros/jazzy" >&2; exit 1; }
[ "${SKIP_APT:-0}" = 1 ] || "$REPO/tools/wsl/install_deps.sh"
mkdir -p "$WS/src"
for p in robot_v3_description solverbot_sim; do ln -sfn "$REPO/$p" "$WS/src/$p"; done
source /opt/ros/jazzy/setup.bash
(cd "$WS" && colcon build --symlink-install)
source "$HOME/solverbot_venv/bin/activate"
pip install -e "$REPO/solverbot_rl"
python3 -m pytest -q "$REPO/solverbot_rl/tests"
echo "--- render (para cámaras): "; (glxinfo -B 2>/dev/null | grep -E "renderer|version string" ) || echo "glxinfo no disponible (apt install mesa-utils)"
gz sim --versions
echo "Listo. Validación física: tools/headless/start.sh 0 validate --seeds 40000-40004"
