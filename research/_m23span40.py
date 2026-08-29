"""round 26: machine 23, span 40 - the cells the tight-budget ladder left STUCK."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from star_case import RelaxStar, decide_star, two_gap_geometry
from lp_degree_range import gears_of
g = gears_of(23); W = 40
nc = nr = nu = 0
for a in range(1, W):
    A, op = two_gap_geometry(W, a)
    R = RelaxStar(g, A, (), (), op)
    if R.dead:
        continue
    t = time.time()
    v, info = decide_star(R, verbose=False, maxrounds=600, time_budget=90)
    if v == 'CERTIFIED':
        nc += 1
    elif v == 'REFUTED':
        nr += 1
        print("  m23 span 40 split (%d,%d): REFUTED slack=%s [%.0fs]"
              % (a, W - a, info.get('row_slack'), time.time() - t), flush=True)
    else:
        nu += 1
        print("  m23 span 40 split (%d,%d): %s lp=%s its=%s [%.0fs]"
              % (a, W - a, v, info.get('lp_max'), info.get('its'),
                 time.time() - t), flush=True)
    del R
print("  m23 span 40: %d certified, %d refuted, %d undecided" % (nc, nr, nu),
      flush=True)
