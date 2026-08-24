"""Round 21 (mechanic): independent cross-check of the RECORD-MULTIPLICITY
ladder - how many times does the maximal gap F(M) occur per full period?

The ladder (4, 2, 4, 2, 4 at machines 23, 29, 31, 37, 41) was obtained from
single COV-SAT runs and was flagged as measured-once.  This re-derives the
reachable entries by DIRECT full-period segmented scan, a completely
independent method, and asserts F against the known exact values.

Cyclic convention: the period wraps, so the gap list is closed up by carrying
the first opening around.  Usage: python record_multiplicity.py y [y ...]
"""
import sys
import time
from math import prod

import numpy as np

SEG = 64_000_000
KNOWN_F = {13: 11, 17: 18, 19: 25, 23: 34, 29: 43, 31: 58, 37: 88}


def primes_upto(n):
    s = np.ones(n + 1, bool)
    s[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]:
            s[i * i::i] = False
    return [int(p) for p in np.flatnonzero(s)]


def run(y):
    gears = [p for p in primes_upto(y) if p >= 5]
    P = prod(gears)
    uvals = [pow(6, -1, g) for g in gears]
    t0 = time.time()

    first_op = None
    last_op = None
    F = 0
    cnt = 0            # occurrences of the current best F
    addrs = []         # left endpoints of maximal gaps (first few)
    ngaps = 0

    for lo in range(0, P, SEG):
        hi = min(P, lo + SEG)
        ex = np.zeros(hi - lo, bool)
        for g, u in zip(gears, uvals):
            ex[(u - lo) % g::g] = True
            ex[(-u - lo) % g::g] = True
        op = np.flatnonzero(~ex).astype(np.int64) + lo
        if not len(op):
            continue
        if first_op is None:
            first_op = int(op[0])
        if last_op is not None:
            op = np.concatenate([[last_op], op])
        d = np.diff(op)
        ngaps += len(d)
        if len(d):
            m = int(d.max())
            if m > F:
                F, cnt, addrs = m, 0, []
            if m == F:
                w = np.flatnonzero(d == F)
                cnt += len(w)
                for i in w[:8]:
                    addrs.append(int(op[i]))
        last_op = int(op[-1])

    # close the cycle: the wrap gap from the last opening to the first + P
    wrap = first_op + P - last_op
    ngaps += 1
    if wrap > F:
        F, cnt, addrs = wrap, 1, [last_op]
    elif wrap == F:
        cnt += 1
        addrs.append(last_op)

    dt = time.time() - t0
    if y in KNOWN_F:
        assert F == KNOWN_F[y], f"F({y}) = {F}, expected {KNOWN_F[y]}"
    print(f"machine {y}: period {P:,}  gaps {ngaps:,}  F = {F}  "
          f"MULTIPLICITY = {cnt}  ({dt:.0f}s)")
    print(f"    left endpoints (first {min(len(addrs), 8)}): "
          + ", ".join(f"{a:,}" for a in addrs[:8]))
    return F, cnt


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        run(int(arg))
