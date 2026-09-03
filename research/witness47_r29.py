"""Round 29 (mechanic): TURN THE F_6(47) TRANSFER WITNESS INTO A REAL SLOT.

j5_multi.py reports a maximiser as (machine-23 start opening k, one phase per
new gear, the surviving-interior indices).  That is a PHASE VECTOR, and this
lane's round-28 lesson (Formalist verdict 36) is that a phase vector is not a
witness until CRT has turned it into a slot of the real machine.  This file
does that for the round-29 record and then re-checks it AT MACHINE 47 from the
definition, importing nothing from j5_multi.

THE CRT.  A machine-23 slot s lifts to x = s + t*P(23).  Gear q blocks x iff
x = +-u_q (mod q), i.e. iff s = c +- u_q (mod q) with c = -t*P(23) (mod q) -
which is exactly j5_multi's phase convention (k1, k2 = c -+ u_q).  So the
phase vector fixes t modulo every new gear:  t = -c_q * P(23)^{-1} (mod q),
and CRT over the new gears gives t, hence x, modulo P(47).

usage: uv run python research/witness47_r29.py
"""
from math import prod

import numpy as np

OLD = 23
NEW = [29, 31, 37, 41, 43, 47]
# from research/data/r29/fj47_s174/*.log (seed-174 band run, 100% coverage)
K, PHASES, MARKS, SPAN = 26216680, (3, 21, 29, 26, 26, 27), (5, 10, 16, 17, 19), 177


def gears(y):
    return [p for p in range(5, y + 1)
            if all(p % d for d in range(2, int(p ** 0.5) + 1))]


def openings(y):
    G = gears(y)
    P = prod(G)
    ex = np.zeros(P, bool)
    for q in G:
        u = pow(6, -1, q)
        ex[u % q::q] = True
        ex[(-u) % q::q] = True
    return np.flatnonzero(~ex).astype(np.int64), P


def main():
    op, P23 = openings(OLD)
    i = int(np.searchsorted(op, K))
    assert op[i] == K, "start is not a machine-23 opening"
    j = int(np.searchsorted(op, K + SPAN))
    assert op[j] == K + SPAN, "end is not a machine-23 opening"
    interior = [int(v) - K for v in op[i + 1:j]]
    offs = [0] + [interior[m] for m in MARKS] + [SPAN]
    print(f"  machine-23 window: start {K:,}, span {SPAN}, "
          f"{len(interior)} interior openings")
    print(f"  surviving interiors (marks {MARKS}): "
          f"{[interior[m] for m in MARKS]}")
    # CRT for t
    t, M = 0, 1
    for q, c in zip(NEW, PHASES):
        r = (-c * pow(P23 % q, -1, q)) % q
        # merge t (mod M) with r (mod q)
        t += M * ((r - t) * pow(M % q, -1, q) % q)
        M *= q
    x = (K + t * P23) % (P23 * M)
    P47 = prod(gears(47))
    assert P23 * M == P47, (P23 * M, P47)
    print(f"  t = {t:,} (mod {M:,});  MACHINE-47 SLOT x = {x}")
    print(f"  x < P(47) = {P47:,}: {x < P47}")
    # verify at machine 47 from the definition
    T = {q: (pow(6, -1, q), (-pow(6, -1, q)) % q) for q in gears(47)}
    oset, nb = set(offs), 0
    for s in range(SPAN + 1):
        o = all((x + s) % q not in T[q] for q in T)
        assert o == (s in oset), ("machine-47 mismatch at offset", s, o)
        nb += not o
    gaps = [offs[k + 1] - offs[k] for k in range(len(offs) - 1)]
    print(f"  VERIFIED AT MACHINE 47: {len(offs)} openings at {offs}")
    print(f"  gap word {gaps}  (J = {len(gaps)} gaps), {nb} other slots blocked")
    assert len(gaps) == 6 and sum(gaps) == 177
    print(f"\n  F_6(47) >= 177 > 171 = F(47) + 53   -> the spectrum-plus-depth "
          f"certificate FAILS at 47 -> 53")
    print("\nALL ASSERTIONS PASSED")


if __name__ == "__main__":
    main()
