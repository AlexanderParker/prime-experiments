#!/bin/bash
# FORMALIST round 29 - the single resumable driver for the 31->37 case modules.
# Same design as research/inc_build_r28.sh (round-27 verdict 30: ONE driver
# only), with the round-29 log.  Skips any module whose .olean already exists,
# keeps at most <width> lake invocations alive, and logs one line per module
# with its return code and wall time.  Priority boosting (round-28 verdict 35,
# worth 8.9x) is done by a separate PowerShell poll loop, not from here.
#
#   bash research/case37_build_r29.sh <width> <first> <last>
set -u
W=${1:-4}
FIRST=${2:-0}
LAST=${3:-384}
cd /c/dev/primes/proofs || exit 1
LOG=/c/dev/primes/research/data/r29/case37_build.log
mkdir -p /c/dev/primes/research/data/r29

run_one() {
  local m=$1
  local t0=$(date +%s)
  ~/.elan/bin/lake.exe build "$m" >/tmp/c37_$m.out 2>&1
  local rc=$?
  local t1=$(date +%s)
  echo "$(date +%H:%M:%S) $m rc=$rc $((t1-t0))s" >>"$LOG"
}

for i in $(seq "$FIRST" "$LAST"); do
  m="CaseCert37C$i"
  if [ -f ".lake/build/lib/lean/$m.olean" ]; then
    echo "$(date +%H:%M:%S) $m SKIP (built)" >>"$LOG"
    continue
  fi
  while [ "$(jobs -rp | wc -l)" -ge "$W" ]; do sleep 5; done
  run_one "$m" &
done
wait
echo "$(date +%H:%M:%S) DRIVER DONE" >>"$LOG"
