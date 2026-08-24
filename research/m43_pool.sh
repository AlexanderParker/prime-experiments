#!/bin/bash
cd C:/dev/primes
for v in $(seq 107 118); do
  while [ $(ls research/data/gaptails/m43b_running_* 2>/dev/null | wc -l) -ge 4 ]; do sleep 60; done
  log=research/data/gaptails/m43_v$v.log
  grep -q "True\|False" $log 2>/dev/null && continue
  touch research/data/gaptails/m43b_running_$v
  ( timeout 14400 .venv-sat/Scripts/python.exe research/cov_sat.py one 43 $v > $log 2>&1 || echo "v=$v TIMEBOX 14400" >> $log
    rm -f research/data/gaptails/m43b_running_$v ) &
  sleep 2
done
wait; echo M43_POOL_DONE
