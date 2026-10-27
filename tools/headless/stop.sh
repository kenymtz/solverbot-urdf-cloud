#!/bin/bash
# Detención limpia: SIGTERM al orquestador -> el entrenamiento guarda checkpoint -> se apaga la simulación.
INSTANCE="${1:-0}"
source "$(dirname "$0")/common.sh" "$INSTANCE"
if alive; then
  PID=$(cat "$PIDFILE"); kill -TERM "$PID"
  for _ in $(seq 1 90); do kill -0 "$PID" 2>/dev/null || break; sleep 1; done
  kill -0 "$PID" 2>/dev/null && { echo "forzando cierre del grupo $PID"; kill -KILL -- "-$PID" 2>/dev/null; }
fi
SP="$STATE_DIR/instance_${INSTANCE}.sim.pid"
[ -f "$SP" ] && kill -KILL -- "-$(cat "$SP")" 2>/dev/null
rm -f "$PIDFILE" "$SP"
echo "Instancia $INSTANCE detenida."
