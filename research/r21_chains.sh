#!/bin/bash
cd C:/dev/primes
PY=.venv-sat/Scripts/python.exe
TB=36000
( log=research/data/gaptails/m47_v119.log
  timeout $TB $PY research/cov_sat.py one 47 119 > $log 2>&1 || echo "v=119 TIMEBOX ${TB}s" >> $log
  echo CHAIN_DONE_m47_119 ) &
( for S in 150 151 152 153 154 155; do
    log=research/data/asc/q6_47_S$S.log
    grep -q " SAT\| UNSAT" $log 2>/dev/null && continue
    timeout $TB $PY research/f3_one.py 47 6 $S 18 > $log 2>&1 || { echo "S=$S TIMEBOX ${TB}s" > $log; break; }
    grep -q " UNSAT" $log && break
  done
  echo CHAIN_DONE_q6 ) &
( log=research/data/asc/q3_47_S141.log
  timeout $TB $PY research/f3_one.py 47 3 141 18 > $log 2>&1 || echo "S=141 TIMEBOX ${TB}s" > $log
  echo CHAIN_DONE_q3 ) &
wait
echo R21_CHAINS_DONE
