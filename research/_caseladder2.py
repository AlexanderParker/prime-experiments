"""round 26 LP thread: the (D) rungs for the k=2 case split (hold 5 and 7,
35 cases).  Every case must certify."""
import os, sys, time
from itertools import product
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from star_case import RelaxStar, decide_star, reverify_cert
from lp_degree_range import gears_of, budget

steps = [(int(x.split('-')[0]), int(x.split('-')[1])) for x in sys.argv[1:2][0].split(',')] \
    if len(sys.argv) > 1 else [(23, 29)]
tb = float(sys.argv[2]) if len(sys.argv) > 2 else 300.0
for (a, b) in steps:
    g = gears_of(b); W = budget(b)
    vs, ops, t0, worst = [], 0, time.time(), None
    for ws in product(range(g[0]), range(g[1])):
        R = RelaxStar(g, W, g[:2], ws)
        tag = "rung2_m%d_w%d_h%d_%d" % (b, W, ws[0], ws[1])
        v, info = decide_star(R, verbose=False, maxrounds=400, tag=tag,
                              time_budget=tb)
        vs.append(v); ops += info.get('ops') or 0
        if v != 'CERTIFIED':
            worst = (ws, v, info.get('lp_max'), info.get('its'))
            print("   %d->%d case %s: %-9s lp=%s its=%s |pos|=%d"
                  % (a, b, ws, v, info.get('lp_max'), info.get('its'),
                     len(R.pos)), flush=True)
        del R
    ok = all(v == 'CERTIFIED' for v in vs)
    print("  RUNG %d->%d (hold 5,7; %d cases): %s   ops %d  [%.1fs]"
          % (a, b, len(vs),
             "CASE-SPLIT CERTIFIED (all cases)" if ok
             else "NOT certified; first failure %s" % (worst,), ops,
             time.time() - t0), flush=True)
    if ok:
        reverify_cert("rung2_m%d_w%d_h0_0" % (b, W))
