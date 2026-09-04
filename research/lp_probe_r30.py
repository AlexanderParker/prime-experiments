"""ROUND 30, LP-DUALITY THREAD - PHASE-TIMED PRICING PROBE FOR ONE CELL.

Times the three phases of a plain-loop certificate separately - the
`RelaxStar` build (columns, links, the recursion row's max-cover cells), the
float LP `_solve_max`, and the exact certificate `certificate_star` - and
reports the exact op count and the peak working set.  A pricing instrument,
nothing here is a verdict; the cell's verdict is written by lp_tree_r30.

    uv run python research/lp_probe_r30.py <y> <W> <k> <ws> [log]
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    y, W, k = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    ws = tuple(int(x) for x in sys.argv[4].split(','))
    log = sys.argv[5] if len(sys.argv) > 5 else None
    out = open(log, 'a') if log else sys.stdout

    def say(s):
        out.write(s + "\n")
        out.flush()

    try:
        import psutil
        psutil.Process().nice(psutil.HIGH_PRIORITY_CLASS)
    except Exception:                                        # noqa: BLE001
        pass
    from lp_degree_range import gears_of
    from star_case import RelaxStar, certificate_star
    g = gears_of(y)
    say("PROBE m%d W=%d k=%d ws=%s  (n=%d free)" % (y, W, k, ws, len(g) - k))
    t0 = time.time()
    R = RelaxStar(g, W, g[:k], ws)
    tb = time.time() - t0
    say("  build: %.1fs  |pos|=%d cols=%d links=%d inexact=%d"
        % (tb, len(R.pos), len(R.cols), len(R.links), R.inexact))
    t1 = time.time()
    val, z, res = R._solve_max()
    tl = time.time() - t1
    say("  LP _solve_max: %.1fs  status=%s  max row = %s vs |pos| = %s"
        % (tl, res.status, val, R.frhs))
    if val is None or val < float(R.frhs) - 1e-7:
        nb = len(R.subsets)
        if val is None:
            # status 2 = INFEASIBLE: the base cuts alone make the polytope
            # EMPTY (the strongest reading); decide_star then takes the
            # common-slack LP's duals - timed here the same way
            t15 = time.time()
            _t, _z, res = R._solve_float()
            say("  base-cut polytope EMPTY (status 2); common-slack LP for"
                " the duals: %.1fs" % (time.time() - t15))
            yv = list(-res.ineqlin.marginals)
            yff = yv.pop()
            nu = res.eqlin.marginals[nb:]
        else:
            yv = list(-res.ineqlin.marginals)
            yff = 1.0
            nu = res.eqlin.marginals[nb:]
        t2 = time.time()
        ok, lhs, rhs, yq, yffq, nuq, ops = certificate_star(R, yv, yff, nu)
        tc = time.time() - t2
        say("  certificate_star: %.1fs  ok=%s  lhs=%s rhs=%s ops=%s"
            % (tc, ok, lhs, rhs, ops))
    else:
        say("  the LP does not close at iteration zero (max row >= |pos|)")
    try:
        import psutil
        mi = psutil.Process().memory_info()
        say("  peak working set %.0f MB" % (mi.peak_wset / 1e6))
    except Exception:                                        # noqa: BLE001
        pass
    say("  total %.1fs" % (time.time() - t0))


if __name__ == '__main__':
    main()
