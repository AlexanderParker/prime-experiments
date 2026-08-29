"""round 26: the machine-19 cells the windowed vehicle could not certify.
Each is an UNREALISED adjacent gap pair (full-period scan) - so an exact
in-polytope witness here is a genuine INTEGRALITY GAP of the vehicle, not a
budget artefact."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from star_case import RelaxStar, decide_star, two_gap_geometry, reverify
from lp_degree_range import gears_of
g = gears_of(19)
CELLS = [(30, 2), (30, 4), (30, 7), (30, 15), (30, 23), (30, 26), (30, 28)]
res = {}
for (W, a) in CELLS:
    A, op = two_gap_geometry(W, a)
    R = RelaxStar(g, A, (), (), op)
    tag = "m19_gap_s%d_a%d" % (W, a)
    t = time.time()
    v, info = decide_star(R, verbose=False, maxrounds=2000, time_budget=180,
                          tag=tag)
    res[(W, a)] = v
    print("  m19 (%2d,%2d) span %2d: %-9s how=%s slack=%s its=%s [%.1fs]"
          % (a, W - a, W, v, info.get('how'), info.get('row_slack'),
             info.get('its'), time.time() - t), flush=True)
    if v == 'REFUTED':
        reverify(tag)
print(res)
