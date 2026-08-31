"""Round 28 (mechanic): Y_5(29) - the onset's residual, at the first machine
where the exact 5-tuple dictionary is not already on disk.

WHY.  C39 leaves the onset's mechanism half-explained: X_5(M) = 9 with the
universal phase-saturated witness (1,2,3,2,1) accounts for the UNSCREENED onset
exactly, and Y_5(M) - the same minimum with the phase-saturated walks removed -
is a lower bound on the screened onset that was tight at one machine of four
(10, 17, 18, 22 against onsets 15, 17, 25, 31 at m13..m23).  The open question
is whether the gap grows, and the named construct for it is Y_5 at a bigger
machine.  Machine 29's exact 5-tuple dictionary is not on disk and the arity-4
censuses do not contain it.

WHAT MAKES IT AFFORDABLE.  Machine 29's period is 1,078,282,205 slots with
214,708,725 openings.  Holding a (N, 5) int64 array is 8.6 GB and the box has
nowhere near that, so the pass is STREAMED: sieve in blocks, keep only the gap
sequence as int8 (215 MB), then walk it in blocks packing each 5-window into a
single int64 (gaps are < 64 at m29, so 5 x 6 bits fit) and accumulating the
distinct keys in a set.  The cyclic close is done explicitly (rule 25) and both
period identities are asserted.

usage: <venv>/python research/y5_m29_r28.py
"""
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
sys.path.insert(0, HERE)
from dict_transfer import load_dict                    # noqa: E402
from y5_r28 import saturated                           # noqa: E402

GEARS = [5, 7, 11, 13, 17, 19, 23, 29]
P = 1
for q in GEARS:
    P *= q
NOPEN = 1
for q in GEARS:
    NOPEN *= (q - 2)
BLOCK = 1 << 27
SH = 6                     # gaps at m29 are <= 43 < 64


def gap_stream():
    """The cyclically closed gap sequence of machine 29, as int8."""
    outs = []
    tail = None
    first = None
    for lo in range(0, P, BLOCK):
        hi = min(lo + BLOCK, P)
        ex = np.zeros(hi - lo, bool)
        for q in GEARS:
            u = pow(6, -1, q)
            for r in (u % q, (-u) % q):
                s = (r - lo) % q
                ex[s::q] = True
        op = np.flatnonzero(~ex).astype(np.int64) + lo
        if first is None:
            first = int(op[0])
        if tail is not None:
            op = np.concatenate([[tail], op])
        outs.append(np.diff(op).astype(np.int8))
        tail = int(op[-1])
    outs.append(np.array([P - tail + first], np.int8))   # the wrap gap
    g = np.concatenate(outs)
    assert len(g) == NOPEN, (len(g), NOPEN)
    assert int(g.astype(np.int64).sum()) == P
    assert int(g[-1]) == int(g[0]), "wrap gap must equal the first gap"
    assert int(g.max()) == 43, int(g.max())
    return g


def dict5(g):
    """distinct 5-tuples, packed into int64 keys, streamed."""
    n = len(g)
    gg = np.concatenate([g, g[:4]]).astype(np.int64)
    keys = set()
    step = 1 << 24
    for lo in range(0, n, step):
        hi = min(lo + step, n)
        k = np.zeros(hi - lo, np.int64)
        for t in range(5):
            k = (k << SH) | gg[lo + t:hi + t]
        keys.update(np.unique(k).tolist())
    return keys


def pack(t):
    k = 0
    for v in t:
        k = (k << SH) | v
    return k


def main():
    print("machine 29: period %d, openings %d - STREAMED" % (P, NOPEN),
          flush=True)
    g = gap_stream()
    print("  cyclic close asserted (N gaps, sum = P, wrap = first, max = 43)",
          flush=True)
    D5 = dict5(g)
    print("  exact 5-tuple dictionary: %d distinct 5-tuples" % len(D5),
          flush=True)
    D4 = set(load_dict(os.path.join(DATA, "gap_tuples_29_4.csv")))
    # control: the induced 4-tuple dictionary of D5 must equal D4 exactly
    ind = set()
    M6 = (1 << SH) - 1
    for k in D5:
        v = [(k >> (SH * (4 - i))) & M6 for i in range(5)]
        ind.add(tuple(v[:4]))
        ind.add(tuple(v[1:]))
    assert ind == D4, ("induced 4-tuple dictionary != the scanned one",
                       len(ind), len(D4))
    print("  CONTROL: its induced 4-tuple dictionary is EXACTLY the round-25 "
          "full-period census (%d) - two independent scans agree" % len(D4),
          flush=True)

    # PERSIST IT - the exact m29 5-tuple dictionary is a new object and no
    # other lane can rebuild it without repeating the full-period pass.
    out = os.path.join(DATA, "r28", "gap_tuples_29_5.csv")
    rows = []
    for k in D5:
        rows.append(tuple((k >> (SH * (4 - i))) & M6 for i in range(5)))
    rows.sort()
    with open(out, "w") as fh:
        fh.write("g1,g2,g3,g4,g5\n")
        for t in rows:
            fh.write(",".join(map(str, t)) + "\n")
    rs = set(rows)
    assert all(t[::-1] in rs for t in rows), "NOT REVERSE-CLOSED"
    assert max(sum(t) for t in rows) == 85, ("F_5(29) should be 85",
                                             max(sum(t) for t in rows))
    print("  wrote %s (%d tuples, reverse-closed, max span 85 = F_5(29) "
          "asserted)" % (out, len(rows)), flush=True)

    by_pref = defaultdict(list)
    for b in D4:
        by_pref[b[:3]].append(b[3])
    bestX = bestY = None
    witX = witY = None
    ncl = nsat = 0
    for a in D4:
        for last in by_pref.get(a[1:], ()):
            ncl += 1
            t = a + (last,)
            if pack(t) in D5:
                continue
            sp = sum(t)
            if bestX is None or sp < bestX:
                bestX, witX = sp, t
            if saturated(t, 29) is not None:
                nsat += 1
                continue
            if bestY is None or sp < bestY:
                bestY, witY = sp, t
    print("\n  order-4 closure admits %d 5-walks; %d of the unrealised ones "
          "are phase-saturated" % (ncl, nsat))
    print("  X_5(29) = %s  witness %s" % (bestX, witX))
    print("  Y_5(29) = %s  witness %s" % (bestY, witY))
    print("\n  against onset(29 -> 31) = 41:   X_5 gap %s,  Y_5 gap %s"
          % (41 - bestX if bestX else "-", 41 - bestY if bestY else "-"))
    print("  the ladder so far (m13, m17, m19, m23, m29):")
    print("    X_5   9   9   9   9  %s" % bestX)
    print("    Y_5  10  17  18  22  %s" % bestY)
    print("    onset 15  17  25  31  41")


if __name__ == "__main__":
    main()
