#!/bin/bash
# Round 21 close-out pool (relaunch after the premature kill at 17:25).
# Timebox 36000s = 10h, ABOVE the observed hard-refutation max of 9.1h at m43,
# so an expiry here is evidence of genuine intractability, not of impatience.
# Width 3 for m43 + 3 single chains = 6 concurrent (the 13-wide m47 batch that
# decided nothing in 8h is the sizing lesson being applied).
cd C:/dev/primes
PY=.venv-sat/Scripts/python.exe
D=research/data/gaptails
A=research/data/asc
TB=36000

# --- m43 tail: the undecided values, width 3 -------------------------------
for v in 110 113 115 117 118 119; do
  while [ $(ls $D/r21f_running_* 2>/dev/null | wc -l) -ge 3 ]; do sleep 60; done
  log=$D/m43_v$v.log
  grep -q "True\|False" $log 2>/dev/null && continue
  touch $D/r21f_running_$v
  ( timeout $TB $PY research/cov_sat.py one 43 $v > $log 2>&1 \
      || echo "v=$v TIMEBOX ${TB}s" >> $log
    rm -f $D/r21f_running_$v ) &
  sleep 2
done

# --- m47 v=119: the value that most constrains F(47) ------------------------
( log=$D/m47_v119.log
  if ! grep -q "True\|False" $log 2>/dev/null; then
    timeout $TB $PY research/cov_sat.py one 47 119 > $log 2>&1 \
      || echo "v=119 TIMEBOX ${TB}s" >> $log
  fi ) &

# --- Q_j(47;18) upper-bound chains: what actually decides q'=53 -------------
# q6 ascends from 150 (SAT through 149 already); q3 retries its 2700s timebox.
( for S in $(seq 150 200); do
    log=$A/q6_47_S$S.log
    grep -q "SAT\|UNSAT" $log 2>/dev/null && continue
    timeout $TB $PY research/f3_one.py 47 6 $S 18 > $log 2>&1 \
      || { echo "S=$S TIMEBOX ${TB}s" >> $log; break; }
    grep -q " UNSAT" $log && break
  done
  echo CHAIN_DONE_q6 ) &

( log=$A/q3_47_S141.log
  rm -f $log
  timeout $TB $PY research/f3_one.py 47 3 141 18 > $log 2>&1 \
    || echo "S=141 TIMEBOX ${TB}s" >> $log
  echo CHAIN_DONE_q3 ) &

wait
echo R21_FINISH_DONE
