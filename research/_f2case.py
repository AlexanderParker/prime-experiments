"""round 26 LP thread: THE TWO RESTRICTIONS COMPOSE.
The windowed statement (prescribed open positions) and the case split (a held
gear's phase) are the same construct with different arguments, so they can be
applied AT ONCE: for each split, hold gear 5 at each of its five phases and
certify every case.  Cases where the held phase blocks a required-open point
are vacuous (the configuration is impossible outright)."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from star_case import RelaxStar, decide_star, two_gap_geometry
from lp_degree_range import gears_of

y = int(sys.argv[1]); lo = int(sys.argv[2]); hi = int(sys.argv[3])
tb = float(sys.argv[4]) if len(sys.argv) > 4 else 10.0
g = gears_of(y)
tot = dict(spans=0, splits=0, dead_split=0, cert=0, bad=0)
ops = 0
for W in range(lo, hi + 1):
    t0, kinds, bad = time.time(), dict(cert=0, dead=0, bad=0), []
    wops = 0
    for a in range(1, W):
        A, op = two_gap_geometry(W, a)
        vs = []
        for w in range(g[0]):
            R = RelaxStar(g, A, (g[0],), (w,), op)
            if R.dead:
                vs.append('DEAD')
                del R
                continue
            v, info = decide_star(R, verbose=False, maxrounds=300,
                                  time_budget=tb)
            vs.append(v)
            wops += info.get('ops') or 0
            del R
        if all(v in ('DEAD',) for v in vs):
            kinds['dead'] += 1
        elif all(v in ('DEAD', 'CERTIFIED') for v in vs):
            kinds['cert'] += 1
        else:
            kinds['bad'] += 1
            bad.append((a, vs))
    ops += wops
    tot['spans'] += 1
    tot['splits'] += W - 1
    tot['dead_split'] += kinds['dead']
    tot['cert'] += kinds['cert']
    tot['bad'] += kinds['bad']
    print("  m%d span %3d: %2d dead, %2d certified, %2d not  ops %d  [%.0fs]%s"
          % (y, W, kinds['dead'], kinds['cert'], kinds['bad'], wops,
             time.time() - t0, "  " + str(bad[:4]) if bad else ""), flush=True)
print("\n  m%d spans %d..%d with gear %d held: %s ; ops %d"
      % (y, lo, hi, g[0], tot, ops))
if tot['bad'] == 0:
    print("  => LP PROOF (case split x windowed): machine %d has NO two-gap"
          " window of span in [%d, %d]" % (y, lo, hi))
