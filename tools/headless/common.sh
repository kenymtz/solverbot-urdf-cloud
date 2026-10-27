#!/bin/bash
# Variables comunes de los scripts headless. Uso: source common.sh <instancia>
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
INSTANCE="${1:-0}"
STATE_DIR="${SOLVERBOT_STATE_DIR:-$HOME/.solverbot}"
mkdir -p "$STATE_DIR"
PIDFILE="$STATE_DIR/instance_${INSTANCE}.pid"
INFOFILE="$STATE_DIR/instance_${INSTANCE}.info"
# shellcheck source=/dev/null
source "$REPO/tools/wsl/sim_env.sh" "$INSTANCE" >/dev/null
alive() { [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; }
