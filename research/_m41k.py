"""round 26: scouting the machine-41 case split at k = 2 and k = 3 -
how far does the LP maximum of the recursion row fall below |pos|?"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from star_case import RelaxStar, decide_star
from lp_degree_range import gears_of, budget
g = gears_of(41); W = budget(41)
CASES = [((5, 7), (0, 0)), ((5, 7), (3, 1)), ((5, 7, 11), (0, 0, 0)),
         ((5, 7, 11), (3, 1, 5))]
for held, ws in CASES:
    R = RelaxStar(g, W, held, ws)
    print("m41 W=%d hold %s at %s: cols=%d |pos|=%d frhs=%s"
          % (W, held, ws, len(R.cols), len(R.pos), R.frhs), flush=True)
    t = time.time()
    v, info = decide_star(R, verbose=True, maxrounds=400, time_budget=600)
    print("  -> %s  %s  [%.0fs]\n"
          % (v, {k: x for k, x in info.items() if k != 'R'}, time.time() - t),
          flush=True)
    del R
