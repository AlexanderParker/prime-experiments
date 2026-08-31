"""Round 28 (mechanic): THE INFLATION-ONSET LADDER, AND ITS MECHANISM.

PART 1 - EXTEND THE LADDER DOWNWARD.  research/onset_r28.py measures the onset
at the four steps whose exact 4-tuple dictionaries already exist (23->29,
29->31, 31->37, 37->41).  Machines 13, 17, 19 have periods of 5,005 / 85,085 /
1,616,615 slots, so their exact 4- AND 5-tuple dictionaries are a few seconds of
numpy - three more steps for free, and seven points beat four.

PART 2 - THE MECHANISM, NOT A FIT.  A transfer emission of span s comes from a
WALK of M-gaps of total span s in which some interiors are deleted by q'.  With
ZERO deletions the emission IS a realised M 4-tuple, and its phase is free by
CRT, so it is realised at M + q' - no refutation can come from depth 0.  Hence

    a REFUTED emission needs an M-walk of >= 5 gaps that the order-4 closure
    admits, and the cheapest way for the closure to be wrong is a 5-walk whose
    two 4-windows are both realised while the 5-walk itself is not.

Define  X_5(M) = min span of such a 5-walk  ("the span at which order-4 closure
stops determining machine M's 5-tuple dictionary").  X_5 is a property of ONE
machine, needs no target gear, and is a LOWER BOUND on the onset by the
argument above.  Measuring X_5 alongside the onset says whether the onset is a
closure fact (X_5 tracks it) or a target-gear fact (it does not).

Usage:  <venv>/python research/onset_ladder_r28.py
"""
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(DATA, "r28")
sys.path.insert(0, HERE)

from dict_transfer import transfer                     # noqa: E402
from onset_r28 import screen, onset_of                 # noqa: E402

F1 = {13: 11, 17: 18, 19: 25, 23: 34, 29: 43, 31: 58, 37: 88, 41: 91}
F4 = {13: 26, 17: 33, 19: 38, 23: 58, 29: 70, 31: 90, 37: 105, 41: 118}
SMALL = [13, 17, 19, 23]
STEPS = [(13, 17), (17, 19), (19, 23)]


def primes_upto(n):
    return [p for p in range(2, n + 1)
            if all(p % d for d in range(2, int(p ** 0.5) + 1))]


def openings(y):
    gears = [p for p in primes_upto(y) if p >= 5]
    P = 1
    for g in gears:
        P *= g
    ex = np.zeros(P, bool)
    for g in gears:
        u = pow(6, -1, g)
        ex[u % g::g] = True
        ex[(-u) % g::g] = True
    return np.flatnonzero(~ex).astype(np.int64), P


def gaps_cyclic(y):
    """The period's gap sequence, CYCLICALLY CLOSED (rule 25): N gaps for N
    openings, the wrap gap included and asserted."""
    op, P = openings(y)
    g = np.diff(np.concatenate([op, [op[0] + P]]))
    assert len(g) == len(op), (len(g), len(op))
    assert int(g.sum()) == P, (int(g.sum()), P)
    # C26's closed form: slot 0 is always an opening and the opening set is
    # mirror-symmetric, so the wrap gap P - x_{N-1} equals x_1 = the FIRST gap.
    assert int(op[0]) == 0 and int(g[-1]) == int(g[0]), "cyclic close"
    return g.astype(np.int64)


def ktuples(g, k):
    """Every contiguous k-window of the cyclic gap sequence, as a set."""
    gg = np.concatenate([g, g[:k - 1]])
    st = np.stack([gg[i:len(g) + i] for i in range(k)], axis=1)
    return set(map(tuple, np.unique(st, axis=0).tolist()))


def x5(d4, d5):
    """(min span, witness, #closure 5-walks) - closure admits, machine does
    not."""
    by_pref = defaultdict(list)
    for b in d4:
        by_pref[b[:3]].append(b[3])
    best, wit, ncl = None, None, 0
    for a in d4:
        for last in by_pref.get(a[1:], ()):
            ncl += 1
            t = a + (last,)
            if t in d5:
                continue
            s = sum(t)
            if best is None or s < best:
                best, wit = s, t
    return best, wit, ncl


def main():
    os.makedirs(OUT, exist_ok=True)
    print("PART 1 - EXACT SMALL-MACHINE DICTIONARIES (cyclically closed)\n")
    d4, d5 = {}, {}
    for y in SMALL:
        g = gaps_cyclic(y)
        d4[y] = ktuples(g, 4)
        d5[y] = ktuples(g, 5)
        mx = int(g.max())
        assert mx == F1[y], ("F mismatch", y, mx, F1[y])
        assert max(sum(t) for t in d4[y]) == F4[y], ("F_4 mismatch", y)
        print("  machine %2d: %9s gaps, F = %2d and F_4 = %3d both ASSERTED, "
              "%6d exact 4-tuples, %7d exact 5-tuples"
              % (y, "{:,}".format(len(g)), mx, F4[y], len(d4[y]), len(d5[y])))
        if y in (13, 17, 19):
            p = os.path.join(OUT, "gap_tuples_%d_4.csv" % y)
            with open(p, "w") as f:
                f.write("g1,g2,g3,g4\n")
                for t in sorted(d4[y]):
                    f.write(",".join(map(str, t)) + "\n")

    print("\nPART 2 - THE ONSET AT THREE MORE STEPS\n")
    onsets = {}
    for M, qp in STEPS:
        src = sorted(d4[M])
        truth = d4[qp]
        sup, _, _ = transfer(src, qp, F4[qp], F1[qp], verbose=False)
        assert not (truth - sup), ("SUPERSET VIOLATED", M, qp)
        scr, _ = screen(sup, qp)
        assert not (truth - set(scr)), "SCREEN REMOVED A REALISED TUPLE"
        print("  %d -> %d: source %d, exact target %d, superset %d "
              "(inflation %.4fx), screened %d"
              % (M, qp, len(src), len(truth), len(sup),
                 len(sup) / len(truth), len(scr)))
        onset_of(sorted(sup), truth, "unscreened superset")
        o, tot, ref = onset_of(scr, truth, "screened superset")
        onsets[(M, qp)] = o
        if o:
            print("      span:  " + " ".join("%4d" % s
                                             for s in range(o, o + 10)))
            print("      refut: " + " ".join("%4d" % ref.get(s, 0)
                                             for s in range(o, o + 10)))
            print("      cands: " + " ".join("%4d" % tot.get(s, 0)
                                             for s in range(o, o + 10)))
        print()

    print("\nPART 3 - X_5(M): WHERE ORDER-4 CLOSURE STOPS DETERMINING THE "
          "5-TUPLE DICTIONARY\n")
    X = {}
    for y in SMALL:
        v, w, ncl = x5(d4[y], d5[y])
        X[y] = v
        print("  machine %2d: closure admits %8d 5-walks vs %7d exact "
              "5-tuples (inflation %.4fx);  X_5 = %s   witness %s"
              % (y, ncl, len(d5[y]), ncl / len(d5[y]), v, w))

    print("\n  THE COMPARISON  (X_5 is a LOWER BOUND on the onset by the "
          "depth-0 argument)")
    print("    step        onset   X_5(M)   onset - X_5")
    known = dict(onsets)
    known[(23, 29)] = 31          # research/onset_r28.py, this round
    for (M, qp) in sorted(known):
        o, v = known[(M, qp)], X.get(M)
        print("    %2d -> %2d      %4s    %5s       %s"
              % (M, qp, o, v, (o - v) if (o and v) else "-"))


if __name__ == "__main__":
    main()
