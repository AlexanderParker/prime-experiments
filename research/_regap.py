"""round 26: re-run the budget-limited undecided cells with a large budget."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from star_case import RelaxStar, decide_star, two_gap_geometry
from lp_degree_range import gears_of
g = gears_of(19)
for (W, a) in ((28, 2), (28, 26), (26, 1), (26, 25)):
    A, op = two_gap_geometry(W, a)
    R = RelaxStar(g, A, (), (), op)
    t = time.time()
    v, info = decide_star(R, verbose=False, maxrounds=1200, time_budget=900)
    print("  m19 span %d split (%d,%d): %s  its=%s rows=%s ops=%s [%.1fs]"
          % (W, a, W - a, v, info.get('its'), info.get('rows'),
             info.get('ops'), time.time() - t), flush=True)
