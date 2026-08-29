"""round 26 LP thread: the (D) rung ladder for the CASE-SPLIT vehicle.

Round 25 closed the composed level-2 vehicle's ladder at four rungs
(7->11 .. 17->19), with 19->23, 23->29 REFUTED by exhibited exact witnesses
and 37->41 refuted by the uniform product measure.  This runs the same rungs
for the case split (gear 5's phase held, five cases, all five must certify).
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from star_case import RelaxStar, decide_star, reverify_cert
from lp_degree_range import gears_of, budget

STEPS = [(19, 23), (23, 29), (29, 31), (31, 37)]
tb = float(sys.argv[1]) if len(sys.argv) > 1 else 900.0
for (a, b) in STEPS:
    g = gears_of(b)
    W = budget(b)
    vs, ops, t0 = [], 0, time.time()
    for w in range(g[0]):
        R = RelaxStar(g, W, (5,), (w,))
        tag = "rung_m%d_w%d_h%d" % (b, W, w)
        v, info = decide_star(R, verbose=True, maxrounds=400, tag=tag,
                              time_budget=tb)
        vs.append(v)
        ops += info.get('ops') or 0
        print("   %d->%d W=%d case w=%d: %-9s cols=%d |pos|=%d its=%s"
              " ops=%s lp=%s [%.1fs]"
              % (a, b, W, w, v, len(R.cols), len(R.pos), info.get('its'),
                 info.get('ops'), info.get('lp_max'), time.time() - t0),
              flush=True)
        del R
    ok = all(v == 'CERTIFIED' for v in vs)
    print("  RUNG %d->%d: %s   total certificate ops %d  [%.1fs]\n"
          % (a, b, "CASE-SPLIT CERTIFIED (all 5 cases)" if ok else vs,
             ops, time.time() - t0), flush=True)
    if ok:
        reverify_cert("rung_m%d_w%d_h0" % (b, W))
