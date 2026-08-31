"""Round 28 (mechanic): Y_5(M) - the mechanism candidate for the inflation
onset.

X_5(M) (onset_ladder_r28.py) is the min span of a 5-walk that the order-4
closure admits and the machine does not.  It is 9 at every machine, with the
universal witness (1,2,3,2,1) - which is PHASE-SATURATED at gear 5, i.e. zero
by theorem at every machine.  So X_5 explains the UNSCREENED onset (also 9 at
every step) and nothing else.

Y_5(M) removes exactly that free part:

    Y_5(M) = min span of a 5-walk whose two 4-windows are realised at M, which
             is NOT phase-saturation-obstructed (K9), and which is NOT realised
             at M.

Y_5 is a one-machine quantity, needs no target gear and no solver.  If the
inflation onset is a CLOSURE fact, Y_5 tracks it; if it is a target-gear fact,
it does not.

Usage:  <venv>/python research/y5_r28.py
"""
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from onset_ladder_r28 import gaps_cyclic, ktuples, primes_upto   # noqa: E402
from dict_transfer import load_dict                              # noqa: E402

DATA = os.path.join(HERE, "data")
ONSET = {13: 15, 17: 17, 19: 25, 23: 31, 29: 41, 31: 53, 37: 68}


def saturated(t, M):
    """K9: some gear q <= M has no phase avoiding the walk's exposed set."""
    X = [0]
    for g in t:
        X.append(X[-1] + g)
    for q in primes_upto(M):
        if q < 5:
            continue
        if q >= 2 * len(X):
            break
        s = (-2 * pow(6, -1, q)) % q
        bad = set()
        for x in X:
            bad.add(x % q)
            bad.add((x - s) % q)
        if len(bad) == q:
            return q
    return None


def y5(d4, d5, M):
    by_pref = defaultdict(list)
    for b in d4:
        by_pref[b[:3]].append(b[3])
    bestX = bestY = None
    witX = witY = None
    nsat = nclosure = 0
    for a in d4:
        for last in by_pref.get(a[1:], ()):
            nclosure += 1
            t = a + (last,)
            if t in d5:
                continue
            s = sum(t)
            if bestX is None or s < bestX:
                bestX, witX = s, t
            if saturated(t, M) is not None:
                nsat += 1
                continue
            if bestY is None or s < bestY:
                bestY, witY = s, t
    return (bestX, witX), (bestY, witY), nclosure, nsat


def main():
    print("X_5 (any unrealised closure 5-walk) vs Y_5 (unrealised AND not "
          "phase-saturated)\n")
    print("  machine   closure    saturated    X_5  witness            "
          "Y_5  witness            onset(M->q')")
    for y in (13, 17, 19, 23):
        g = gaps_cyclic(y)
        d4, d5 = ktuples(g, 4), ktuples(g, 5)
        (xv, xw), (yv, yw), ncl, nsat = y5(d4, d5, y)
        print("     %2d     %7d      %7d   %4s  %-18s %4s  %-18s  %4s"
              % (y, ncl, nsat, xv, str(xw), yv, str(yw), ONSET[y]))
    # machines 29, 31, 37: exact 4-tuple dictionaries exist, exact 5-tuple ones
    # do NOT - so Y_5 is not computable there without a new full-period scan.
    print("\n  machines 29, 31, 37: Y_5 needs the exact 5-TUPLE dictionary, "
          "which does not exist on disk\n  (the scanned dictionaries are "
          "arity 4).  NOT computed - stated, not guessed.")


if __name__ == "__main__":
    main()
