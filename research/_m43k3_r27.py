"""ROUND 27, LP THREAD.  Finish the 41->43 rung at k = 3 (hold 5, 7, 11).

Round 26 stopped at 163/385 with six stalls at a 45 s/case budget.  This is
the same job, SIZED: a resumable striped worker that skips any case whose
certificate is already on disk, with a per-case budget passed in.

    python research/_m43k3_r27.py <worker> <nworkers> <time_budget>
"""
import os
import sys
import time
from itertools import product

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from star_case import RelaxStar, decide_star, OUT           # noqa: E402
from lp_degree_range import gears_of, budget                # noqa: E402

R27 = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'r27')


def main():
    wi = int(sys.argv[1])
    nw = int(sys.argv[2])
    tb = float(sys.argv[3]) if len(sys.argv) > 3 else 240.0
    y = 43
    g = gears_of(y)
    W = budget(y)
    held = g[:3]
    allc = list(product(*[range(q) for q in held]))
    todo = []
    for i, ws in enumerate(allc):
        tag = "rung3_m%d_w%d_h%d_%d_%d" % (y, W, ws[0], ws[1], ws[2])
        # round-26 certificates live in r26; new ones land in r27
        if os.path.exists(os.path.join(OUT, 'cert_%s.pkl' % tag)) or \
           os.path.exists(os.path.join(R27, 'cert_%s.pkl' % tag)):
            continue
        if i % nw == wi:
            todo.append(ws)
    print("worker %d/%d: %d cases, budget %.0f s" % (wi, nw, len(todo), tb),
          flush=True)
    os.makedirs(R27, exist_ok=True)
    import star_case
    star_case.OUT = R27                    # new certificates land in r27
    t0 = time.time()
    for ws in todo:
        tag = "rung3_m%d_w%d_h%d_%d_%d" % (y, W, ws[0], ws[1], ws[2])
        t1 = time.time()
        R = RelaxStar(g, W, held, ws)
        v, info = decide_star(R, verbose=False, maxrounds=400, tag=tag,
                              time_budget=tb)
        print("  case %s -> %-9s lp=%s |pos|=%d its=%s [%.0fs / %.0fs]"
              % (str(ws), v, info.get('lp_max'), len(R.pos), info.get('its'),
                 time.time() - t1, time.time() - t0), flush=True)
        del R
    print("worker %d DONE [%.0fs]" % (wi, time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
