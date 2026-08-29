"""ROUND 27, LP THREAD - score the round-26 pre-registered predictions E2/E3.

E3  "THE VEHICLE IS TIGHT ON F AT EVERY MACHINE ONCE k IS LARGE ENOUGH:
     F(31) <= 58 (which fails at k = 2, 19/35) certifies at k = 3."
E2  "THE CASE-SPLIT LADDER IS MONOTONE IN k: no rung certified at k fails at
     k+1."  Tested on the 29->31 rung (budget width 74), certified at k = 2, now
     re-run at k = 3 - these are DIFFERENT LPs, not refinements of one, so it is
     a real test.

    python research/_predscore_r27.py <E2|E3> <worker> <nworkers>
"""
import os
import sys
import time
from itertools import product

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import star_case                                       # noqa: E402
from star_case import RelaxStar, decide_star           # noqa: E402
from lp_degree_range import gears_of, budget           # noqa: E402

R27 = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'r27')


def main():
    which = sys.argv[1].upper()
    wi = int(sys.argv[2])
    nw = int(sys.argv[3])
    tb = float(sys.argv[4]) if len(sys.argv) > 4 else 180.0
    y = 31
    W = 58 if which == 'E3' else budget(31)      # 58 = F(31) ; 74 = budget
    g = gears_of(y)
    held = g[:3]
    star_case.OUT = R27
    os.makedirs(R27, exist_ok=True)
    allc = list(product(*[range(q) for q in held]))
    todo = [w for i, w in enumerate(allc) if i % nw == wi]
    print("%s: machine %d width %d hold %s, %d of %d cases"
          % (which, y, W, list(held), len(todo), len(allc)), flush=True)
    t0, ops, bad = time.time(), 0, []
    for ws in todo:
        tag = "%s_m%d_w%d_h%d_%d_%d" % (which.lower(), y, W, *ws)
        R = RelaxStar(g, W, held, ws)
        v, info = decide_star(R, verbose=False, maxrounds=400, tag=tag,
                              time_budget=tb)
        ops += info.get('ops') or 0
        if v != 'CERTIFIED':
            bad.append((ws, v, info.get('lp_max')))
            print("  case %s -> %s lp=%s" % (str(ws), v, info.get('lp_max')),
                  flush=True)
        del R
    print("%s worker %d: %d/%d certified, %d ops, %d failures %s [%.0fs]"
          % (which, wi, len(todo) - len(bad), len(todo), ops, len(bad),
             bad[:5], time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
