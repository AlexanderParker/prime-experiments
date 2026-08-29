"""round 26 LP thread: the case-split's certificate WIDTH at machine 23.
F(23) = 34, so width 33 must NOT certify (a fully blocked window of width 33
exists inside the maximal gap) and every width >= 34 is a true statement."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from star_case import RelaxStar, decide_star
from lp_degree_range import gears_of
g = gears_of(23)
for W in range(34, 49):
    vs, t = [], time.time()
    for w in range(5):
        R = RelaxStar(g, W, (5,), (w,))
        v, info = decide_star(R, verbose=False, maxrounds=80, time_budget=25)
        vs.append(v)
        del R
    ok = all(v == 'CERTIFIED' for v in vs)
    print("  m23 W=%2d: %s  %s  [%.1fs]"
          % (W, "ALL CERTIFIED" if ok else vs, "", time.time() - t), flush=True)
    if ok:
        print("  => W*_case(23) <= %d  (budget 48, F = 34)" % W, flush=True)
        break
