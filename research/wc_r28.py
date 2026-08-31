"""ROUND 28, LP-DUALITY THREAD - THE EXACT CROSSING WIDTH W_c.

The frontier map's grid answers "converge or not" at sampled widths.  This file
answers the sharper question by BISECTION on the lifted value: for a machine y
and a held-gear count k, the cell function

    G(y, k, W)  =  V*(y, k, W, case 0)  -  |pos(y, k, W)|

is measured at each width, and

    W_c(y, k)  =  min { W : G(y, k, W) < 0 }

is the exact width at which the cut loop stops being able to converge - the
CONVERGENCE FRONTIER, the second frontier of `product-measure-frontier.md`
section 5, as a number rather than as a symptom.

Bisection is only valid if G changes sign once; it is not assumed.  After the
bisection the neighbourhood [W_c - 4, W_c + 4] is swept width by width and the
sign pattern is asserted to be "all >= 0 then all < 0".

    python research/wc_r28.py <y> <k> <lo> <hi>
    python research/wc_r28.py TABLE
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cutlimit_r28                                       # noqa: E402
from star_case import RelaxStar                           # noqa: E402
from lp_degree_range import gears_of, budget, F_EXACT     # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
R28 = os.path.join(HERE, 'data', 'r28')

CACHE = {}


def G(y, k, W, ws=None):
    """V* - |pos| at one cell; None means the limit polytope is EMPTY, which
    is the strongest CERTIFIABLE reading there is (-infinity)."""
    ws = tuple(ws if ws is not None else [0] * k)
    key = (y, k, W, ws)
    if key in CACHE:
        return CACHE[key]
    g = gears_of(y)
    R = RelaxStar(g, W, g[:k], ws)
    L = cutlimit_r28.lifted(R)
    v = (None if L['val'] is None else L['val'] - float(R.frhs))
    CACHE[key] = (v, len(R.pos), L.get('secs'))
    return CACHE[key]


def wc(y, k, lo, hi, ws=None, verbose=True):
    """min{W : G < 0}, by bisection, then the sign pattern asserted around it"""
    t0 = time.time()
    glo = G(y, k, lo, ws)[0]
    ghi = G(y, k, hi, ws)[0]
    assert glo is not None and glo >= 0, \
        "lower end already certifiable - widen the bracket (G=%s)" % (glo,)
    assert ghi is None or ghi < 0, \
        "upper end not certifiable - widen the bracket (G=%s)" % (ghi,)
    a, b = lo, hi
    while b - a > 1:
        m = (a + b) // 2
        v = G(y, k, m, ws)[0]
        if verbose:
            print("    W=%-4d G=%s" % (m, 'EMPTY' if v is None
                                       else '%+.5f' % v), flush=True)
        if v is None or v < 0:
            b = m
        else:
            a = m
    # sign pattern around the crossing, width by width - monotonicity ASSERTED
    band = {}
    for W in range(max(lo, b - 4), min(hi, b + 4) + 1):
        band[W] = G(y, k, W, ws)[0]
    sgn = [(W, (band[W] is None or band[W] < 0)) for W in sorted(band)]
    ok = all(not s for (_W, s) in sgn if _W < b) and \
        all(s for (_W, s) in sgn if _W >= b)
    return dict(machine=y, k=k, wc=b, bracket=[lo, hi],
                band={str(W): band[W] for W in band},
                single_crossing=ok, secs=time.time() - t0,
                F=F_EXACT.get(y), budget=budget(y))


def main():
    a = sys.argv[1:]
    os.makedirs(R28, exist_ok=True)
    if a and a[0].upper() == 'TABLE':
        print("  machine  k    W_c   F(y)  budget  W_c/F   single crossing")
        for f in sorted(os.listdir(R28)):
            if f.startswith('wc_') and f.endswith('.json'):
                d = json.load(open(os.path.join(R28, f)))
                print("  %5d   %2d  %5d  %5s  %6s  %5.3f   %s"
                      % (d['machine'], d['k'], d['wc'], d['F'], d['budget'],
                         d['wc'] / float(d['F']), d['single_crossing']))
        return
    y, k, lo, hi = int(a[0]), int(a[1]), int(a[2]), int(a[3])
    d = wc(y, k, lo, hi)
    p = os.path.join(R28, 'wc_m%d_k%d.json' % (y, k))
    with open(p, 'w') as fh:
        json.dump(d, fh, indent=1)
    print("  m%d k=%d : W_c = %d  (F = %s, budget = %s, W_c/F = %.3f)"
          "  single-crossing %s  [%.0fs]"
          % (y, k, d['wc'], d['F'], d['budget'], d['wc'] / float(d['F']),
             d['single_crossing'], d['secs']))


if __name__ == '__main__':
    main()
