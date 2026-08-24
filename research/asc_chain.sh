#!/bin/bash
# Ascending probe chain: SATs are cheap, first hard instance stops the chain.
# asc_chain.sh gap47      -> v=119.. gap probes at m47
# asc_chain.sh qj47 J     -> Q_J(47;18) span probes S=141..
cd C:/dev/primes
mode=$1; J=$2
if [ "$mode" = gap47 ]; then
  for v in $(seq 119 144); do
    log=research/data/asc/gap47_v$v.log
    grep -q "True\|False" $log 2>/dev/null && continue
    timeout 2700 .venv-sat/Scripts/python.exe research/cov_sat.py one 47 $v > $log 2>&1
    rc=$?
    if [ $rc -eq 124 ]; then echo "v=$v TIMEBOX 2700s" >> $log; echo "chain gap47 stopped at v=$v (timebox)"; break; fi
    grep -q "False" $log && { echo "chain gap47 stopped at v=$v (False)"; break; }
  done
else
  for S in $(seq 141 200); do
    log=research/data/asc/q${J}_47_S$S.log
    grep -q "SAT\|UNSAT" $log 2>/dev/null && continue
    timeout 2700 .venv-sat/Scripts/python.exe research/f3_one.py 47 $J $S 18 > $log 2>&1
    rc=$?
    if [ $rc -eq 124 ]; then echo "S=$S TIMEBOX 2700s" >> $log; echo "chain q$J stopped at S=$S (timebox)"; break; fi
    grep -q " UNSAT" $log && { echo "chain q$J stopped at S=$S (UNSAT)"; break; }
  done
fi
echo CHAIN_DONE_${mode}_${J}
