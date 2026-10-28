#!/bin/bash
# Ejecuta en orden los trabajos de un archivo (una línea = argumentos de run_experiment.sh SIN la instancia), cada uno
# con su propia simulación headless limpia. Líneas vacías y que empiezan con # se ignoran. Se detiene en el primer fallo.
#   tools/headless/queue.sh <instancia> <archivo_de_trabajos>
# En segundo plano (sobrevive al cierre de la terminal):
#   setsid nohup tools/headless/queue.sh 1 jobs.txt > ~/.solverbot/queue_1.out 2>&1 < /dev/null &
INSTANCE="$1"; JOBS="$2"
HERE="$(cd "$(dirname "$0")" && pwd)"
[ -f "$JOBS" ] || { echo "no existe $JOBS" >&2; exit 2; }
n=0
while IFS= read -r line || [ -n "$line" ]; do
  case "$line" in ''|\#*) continue ;; esac
  n=$((n + 1))
  echo "[queue] $(date +%H:%M:%S) trabajo $n: $line"
  # shellcheck disable=SC2086
  "$HERE/run_experiment.sh" "$INSTANCE" $line < /dev/null || { echo "[queue] falló el trabajo $n" >&2; exit 1; }
done < "$JOBS"
echo "[queue] $(date +%H:%M:%S) $n trabajos completados"
