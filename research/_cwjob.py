"""round-24 job runner: one (machine, width) cell of the full composition."""
import sys, time, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from cw_consistent import decideCF, gears_of, budget, F_EXACT

y = int(sys.argv[1]); W = int(sys.argv[2])
rec = (len(sys.argv) < 4) or sys.argv[3] != 'norec'
g = gears_of(y)
t0 = time.time()
print("machine %d  width %d  recursion=%s  F=%d budget=%d"
      % (y, W, rec, F_EXACT[y], budget(y)), flush=True)
feas, info = decideCF(g, W, 2, use_recursion=rec, verbose=True, maxrounds=400)
dt = time.time() - t0
if feas:
    print("RESULT: no certificate (rows %d, cols %d, its %d)  [%.0fs]"
          % (info['rows'], info['cols'], info['its'], dt), flush=True)
else:
    print("RESULT: CERTIFIED  %s < %s   support %d  ops %d  rows %d cols %d"
          "  [%.0fs]" % (info['lhs'], info['rhs'], info['support'],
                         info['ops'], info['rows'], info['cols'], dt),
          flush=True)
