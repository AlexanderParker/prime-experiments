#!/bin/bash
# One SAT probe, HONESTLY labelled. Distinguishes:
#   SAT / UNSAT      - the solver decided
#   TIMEOUT <TB>s    - ONLY when timeout(1) returns exit 124
#   DIED rc=<rc>     - any other non-zero exit (OOM, fork exhaustion, kill)
# stderr is PRESERVED in a sibling .err file - never discarded.
# usage: probe_one.sh <logpath> <TB> <cmd...>
log=$1; TB=$2; shift 2
err=${log%.log}.err
start=$(date +%s)
timeout "$TB" "$@" > "$log" 2> "$err"
rc=$?
el=$(( $(date +%s) - start ))
if grep -qE " SAT| UNSAT|True|False" "$log" 2>/dev/null; then
  echo "  [elapsed ${el}s rc=$rc]" >> "$log"
elif [ $rc -eq 124 ]; then
  echo "TIMEOUT ${TB}s (elapsed ${el}s)" >> "$log"
else
  echo "DIED rc=$rc after ${el}s - NOT a timeout; see $(basename $err)" >> "$log"
fi
