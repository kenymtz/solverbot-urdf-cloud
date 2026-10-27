#!/bin/bash
# Inicia un experimento en segundo plano (sobrevive al cierre de la terminal). Una instancia a la vez por número.
#   tools/headless/start.sh <instancia> train <run> [args de train.py]
#   tools/headless/start.sh <instancia> validate [args de validate_grasp.py]
set -e
INSTANCE="$1"
source "$(dirname "$0")/common.sh" "$INSTANCE"
if alive; then echo "Instancia $INSTANCE ya en marcha (pid $(cat "$PIDFILE")). Ver status.sh / stop.sh" >&2; exit 1; fi
mkdir -p "$STATE_DIR"
LOG="$STATE_DIR/instance_${INSTANCE}.out"
setsid nohup "$REPO/tools/headless/run_experiment.sh" "$@" > "$LOG" 2>&1 < /dev/null &
echo $! > "$PIDFILE"
echo "$*" > "$INFOFILE"
echo "Iniciado: instancia $INSTANCE pid $(cat "$PIDFILE"). Log del orquestador: $LOG"
