"""ROUND 26, LP-DUALITY THREAD - THE WINDOWED VEHICLE AGAINST THE TRUTH.

The windowed composed LP decides "machine M has no configuration with
positions 0, a, W open and every position of (0,W) minus {a} blocked" - i.e.
"the ADJACENT GAP PAIR (a, W-a) is not realised by M".  That is exactly
membership in the level-2 gap dictionary the chain (Constructor R60/R61) and
the merge law (Mechanic) consume, decided by LP DUALITY with an exact
rational certificate instead of by a CSP search or a period scan.

This file measures the vehicle against the TRUTH: the realised adjacent gap
pair set of machine M, computed by sieving the full period.  Soundness demands
CERTIFIED => not realised; the interesting question is the converse - is the
vehicle EXACT on this family, or does it have an integrality gap?

Run:  python research/window_dict.py <y> <lo> <hi>
"""
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from star_case import RelaxStar, decide_star, two_gap_geometry   # noqa: E402
from lp_degree_range import gears_of, teeth                      # noqa: E402


def true_pairs(y):
    import numpy as np
    g = gears_of(y)
    P = 1
    for q in g:
        P *= q
    a = np.ones(P, dtype=bool)
    for q in g:
        for t in teeth(q):
            a[(t % q)::q] = False
    idx = np.flatnonzero(a)
    d = np.diff(np.concatenate([idx, [idx[0] + P]]))
    return set(zip(d.tolist(), np.roll(d, -1).tolist())), int(d.max())


def compare(y, lo, hi, maxrounds=80, tb=30.0):
    g = gears_of(y)
    T, F = true_pairs(y)
    print("machine %d: F = %d, %d realised adjacent gap pairs (period scan)"
          % (y, F, len(T)), flush=True)
    tally = dict(ok_cert=0, ok_ref=0, dead=0, GAP=0, UNSOUND=0, undec=0)
    gaps, unsound = [], []
    for W in range(lo, hi + 1):
        line = []
        for a in range(1, W):
            real = (a, W - a) in T
            A, op = two_gap_geometry(W, a)
            R = RelaxStar(g, A, (), (), op)
            if R.dead:
                if real:
                    tally['UNSOUND'] += 1
                    unsound.append((W, a, 'dead'))
                else:
                    tally['dead'] += 1
                continue
            v, info = decide_star(R, verbose=False,
                                  maxrounds=maxrounds, time_budget=tb)
            if v == 'CERTIFIED':
                if real:
                    tally['UNSOUND'] += 1
                    unsound.append((W, a, 'certified but realised'))
                else:
                    tally['ok_cert'] += 1
            elif v == 'REFUTED':
                if real:
                    tally['ok_ref'] += 1
                else:
                    tally['GAP'] += 1
                    gaps.append((W, a, v))
                    line.append((a, 'refuted, not realised'))
            else:
                tally['undec'] += 1
                if not real:
                    tally['GAP'] += 1
                    gaps.append((W, a, v))
                line.append((a, v))
            del R
        print("  span %3d: %s %s" % (W, tally, line if line else ""),
              flush=True)
    print("\nRESULT machine %d spans %d..%d: %s" % (y, lo, hi, tally))
    assert tally['UNSOUND'] == 0, ("SOUNDNESS FAILURE", unsound)
    print("  SOUND: every CERTIFIED cell is genuinely unrealised.")
    if tally['GAP'] == 0:
        print("  EXACT: the windowed vehicle decides the level-2 dictionary"
              " with NO integrality gap over this range.")
    else:
        print("  integrality/undecided cells: %s" % gaps[:20])
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'data', 'r26',
                           'windict_m%d_%d_%d.pkl' % (y, lo, hi)), 'wb') as fh:
        pickle.dump(dict(y=y, lo=lo, hi=hi, tally=tally, gaps=gaps), fh)
    return tally, gaps


if __name__ == '__main__':
    compare(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))
