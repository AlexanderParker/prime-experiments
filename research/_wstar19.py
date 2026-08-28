"""pin the adaptive block-independent degree-2 W* at machine 19 by bisection."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lp_degree_range import decide, gears_of
g = gears_of(19)
lo, hi = 24, 37          # lo verified feasible (exact) in the r24 scan
# verify hi infeasible first
t0=time.time(); f,info = decide(g, hi, 2)
print("W=37:", "feasible" if f else "INFEASIBLE (cert %d ops)"%info['ops'],
      "[%.0fs]"%(time.time()-t0), flush=True)
assert not f
while hi - lo > 1:
    mid = (lo+hi)//2
    t0=time.time(); f,info = decide(g, mid, 2)
    print("W=%d:"%mid, "feasible" if f else "INFEASIBLE",
          "[%.0fs]"%(time.time()-t0), flush=True)
    if f: lo = mid
    else: hi = mid
print("RESULT: adaptive degree-2 W*(19) =", hi, flush=True)
