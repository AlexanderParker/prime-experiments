"""round 26 LP thread: decide one case of the machine-41 case split."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from star_case import RelaxStar, decide_star
from lp_degree_range import gears_of, budget

w = int(sys.argv[1]) if len(sys.argv) > 1 else 0
tb = float(sys.argv[2]) if len(sys.argv) > 2 else 2400.0
R = RelaxStar(gears_of(41), budget(41), (5,), (w,))
print("case w=%d: %d cols, %d links, |pos|=%d, frhs=%s"
      % (w, len(R.cols), len(R.links), len(R.pos), R.frhs), flush=True)
v, info = decide_star(R, maxrounds=200, verbose=True,
                      tag="m41_w129_h%d" % w, time_budget=tb)
print("VERDICT", v, {k: x for k, x in info.items() if k != 'R'}, flush=True)
