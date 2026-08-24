"""Round 21 (mechanic): THE MIRROR LAW for maximal gaps, and why the
record-multiplicity ladder is always EVEN.

Each gear q blocks the two residues {u_q, -u_q} mod q, a set symmetric under
negation.  Hence k is an opening iff -k is an opening, so the whole opening set
(and therefore the gap sequence) is invariant under k -> -k mod P.  A gap
[a, a+g] maps to [-a-g, -a], so maximal gaps come in MIRROR PAIRS with left
endpoints summing to P - F (mod P).  A gap is self-mirror only if
2a + F = 0 mod P.

This script proves the symmetry numerically (openings set equals its negation)
and verifies the pairing of the maximal-gap addresses at every reachable
machine, with an explicit check for self-mirror gaps.
"""
import sys
import time
from math import prod

import numpy as np

KNOWN_F = {13: 11, 17: 18, 19: 25, 23: 34, 29: 43}


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
    ex = np.zeros(P, bool)
    for g in gears:
        u = pow(6, -1, g)
        ex[u % g::g] = True
        ex[(-u) % g::g] = True
    op = np.flatnonzero(~ex).astype(np.int64)

    # 1. the opening set is closed under negation mod P
    neg = np.sort((-op) % P)
    assert np.array_equal(op, neg), f"machine {y}: opening set not mirror-symmetric"

    # 2. cyclic gaps and the maximal-gap addresses
    d = np.diff(np.concatenate([op, [op[0] + P]]))
    F = int(d.max())
    if y in KNOWN_F:
        assert F == KNOWN_F[y]
    left = op[np.flatnonzero(d == F)]

    # 3. every maximal gap's mirror is a maximal gap, endpoints sum to P - F
    S = set(int(a) for a in left)
    selfmirror = 0
    for a in left:
        partner = (-int(a) - F) % P
        assert partner in S, f"machine {y}: mirror of {a} missing"
        if partner == int(a):
            selfmirror += 1
    assert (len(left) - selfmirror) % 2 == 0
    print(f"machine {y}: P = {P:,}  F = {F}  multiplicity = {len(left)}  "
          f"self-mirror = {selfmirror}  -> mirror pairing VERIFIED"
          + ("  (multiplicity EVEN)" if len(left) % 2 == 0 else ""))
    return len(left), selfmirror


if __name__ == "__main__":
    args = [int(a) for a in sys.argv[1:]] or [13, 17, 19, 23, 29]
    t0 = time.time()
    tot = 0
    for y in args:
        n, sm = run(y)
        tot += sm
    print(f"ALL MACHINES: mirror law holds; total self-mirror maximal gaps = {tot}"
          f"  ({time.time()-t0:.0f}s)")
