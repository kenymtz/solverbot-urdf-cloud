#!/bin/bash
# Estado de una instancia: proceso, últimos checkpoints/progreso y fin de los logs.
INSTANCE="${1:-0}"
source "$(dirname "$0")/common.sh" "$INSTANCE"
if alive; then echo "EN MARCHA  pid $(cat "$PIDFILE")  ($(cat "$INFOFILE" 2>/dev/null))"; else echo "DETENIDA"; fi
OUT="$STATE_DIR/instance_${INSTANCE}.out"
[ -f "$OUT" ] && { echo "--- orquestador"; tail -n 8 "$OUT"; }
RUN=$(awk '$2=="train"{print $3} $2=="ctrain"{print $4}' "$INFOFILE" 2>/dev/null)
if [ -n "$RUN" ] && [ -d "$REPO/runs/$RUN" ]; then
  echo "--- runs/$RUN"
  [ -f "$REPO/runs/$RUN/progress.jsonl" ] && tail -n 3 "$REPO/runs/$RUN/progress.jsonl"
  [ -f "$REPO/runs/$RUN/curriculum.jsonl" ] && tail -n 2 "$REPO/runs/$RUN/curriculum.jsonl"
  [ -f "$REPO/runs/$RUN/evals.jsonl" ] && tail -n 1 "$REPO/runs/$RUN/evals.jsonl"
  [ -f "$REPO/runs/$RUN/checkpoints/latest_state.json" ] && cat "$REPO/runs/$RUN/checkpoints/latest_state.json" | head -3
  L=$(ls -t "$REPO/runs/$RUN"/logs/*train_*.log 2>/dev/null | head -1); [ -n "$L" ] && { echo "--- $L"; tail -n 5 "$L"; }
fi
