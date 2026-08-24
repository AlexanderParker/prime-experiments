#!/bin/bash
# Round 21 unified gap-tail pool: runs cov_sat "one y v" jobs from a queue,
# skipping values whose log already holds a result; width-bounded; per-value
# logs in research/data/gaptails/. Queue built for: m47 119..140 (F(47) for
# the q'=53 margin), m43 107..119 (tail), m53 137..145 (F(53) exact).
cd C:/dev/primes
W=${1:-8}
QUEUE=""
for v in $(seq 140 -1 119); do QUEUE="$QUEUE 47:$v"; done
for v in $(seq 119 -1 107); do QUEUE="$QUEUE 43:$v"; done
for v in $(seq 145 -1 137); do QUEUE="$QUEUE 53:$v"; done
for job in $QUEUE; do
  y=${job%%:*}; v=${job##*:}
  log=research/data/gaptails/m${y}_v${v}.log
  grep -q "True\|False" $log 2>/dev/null && continue
  while [ $(ls research/data/gaptails/p21_running_* 2>/dev/null | wc -l) -ge $W ]; do sleep 60; done
  touch research/data/gaptails/p21_running_${y}_${v}
  ( .venv-sat/Scripts/python.exe research/cov_sat.py one $y $v > $log 2>&1
    rm -f research/data/gaptails/p21_running_${y}_${v} ) &
  sleep 2
done
wait
echo POOL21_DONE
