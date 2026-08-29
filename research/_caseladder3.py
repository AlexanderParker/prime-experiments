"""round 26 LP thread: the (D) rung 31->37 for the k=3 case split
(hold 5, 7 and 11 - 385 cases).  Every case must certify."""
import os, sys, time
from itertools import product
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from star_case import RelaxStar, decide_star, reverify_cert
from lp_degree_range import gears_of, budget

a, b = 31, 37
tb = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0
g = gears_of(b); W = budget(b)
vs, ops, t0, bad = [], 0, time.time(), []
for ws in product(range(g[0]), range(g[1]), range(g[2])):
    R = RelaxStar(g, W, g[:3], ws)
    tag = "rung3_m%d_w%d_h%d_%d_%d" % (b, W, ws[0], ws[1], ws[2])
    v, info = decide_star(R, verbose=False, maxrounds=400, tag=tag,
                          time_budget=tb)
    vs.append(v); ops += info.get('ops') or 0
    if v != 'CERTIFIED':
        bad.append((ws, v, info.get('lp_max'), len(R.pos)))
        print("   case %s: %-9s lp=%s |pos|=%d" % (ws, v, info.get('lp_max'),
                                                   len(R.pos)), flush=True)
    if len(vs) % 55 == 0:
        print("   ... %d/%d cases, %d not certified  [%.0fs]"
              % (len(vs), 385, len(bad), time.time() - t0), flush=True)
    del R
ok = not bad
print("  RUNG %d->%d (hold 5,7,11; %d cases): %s   ops %d  [%.1fs]"
      % (a, b, len(vs), "CASE-SPLIT CERTIFIED (all cases)" if ok
         else "NOT certified; %d failures, first %s" % (len(bad), bad[:3]),
         ops, time.time() - t0), flush=True)
if ok:
    reverify_cert("rung3_m%d_w%d_h0_0_0" % (b, W))
