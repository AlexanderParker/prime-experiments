#!/bin/bash
# FORMALIST round 28 - the single resumable driver for the increment-width case
# modules.  ONE driver only (round-27 verdict 30: two builders over one .olean
# tree is a correctness problem, not just a mess).
#
#   bash research/inc_build_r28.sh <width> <module...>
#
# Skips any module whose .olean already exists, keeps at most <width> lake
# invocations alive, and logs one line per module with its wall time.
set -u
W=${1:-2}
shift
cd /c/dev/primes/proofs || exit 1
LOG=/c/dev/primes/research/data/r28/inc_build.log
mkdir -p /c/dev/primes/research/data/r28

run_one() {
  local m=$1
  local t0=$(date +%s)
  ~/.elan/bin/lake.exe build "$m" >/tmp/inc_$m.out 2>&1
  local rc=$?
  local t1=$(date +%s)
  echo "$(date +%H:%M:%S) $m rc=$rc $((t1-t0))s" >>"$LOG"
}

for m in "$@"; do
  if [ -f ".lake/build/lib/lean/$m.olean" ]; then
    echo "$(date +%H:%M:%S) $m SKIP (built)" >>"$LOG"
    continue
  fi
  while [ "$(jobs -rp | wc -l)" -ge "$W" ]; do sleep 5; done
  run_one "$m" &
done
wait
echo "$(date +%H:%M:%S) DRIVER DONE" >>"$LOG"
