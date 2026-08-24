#!/bin/bash
cd C:/dev/primes
PY=.venv-sat/Scripts/python.exe
TB=36000
for S in $(seq 156 190); do
  log=research/data/asc/q6_47_S$S.log
  grep -q " SAT\| UNSAT" $log 2>/dev/null && continue
  timeout $TB $PY research/f3_one.py 47 6 $S 18 > $log 2>&1 || { echo "S=$S TIMEBOX ${TB}s" > $log; break; }
  grep -q " UNSAT" $log && break
done
echo Q6EXT_DONE
