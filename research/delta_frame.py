"""Harvester round 22, stage 0: the DELTA REDUCTION for the paired-Jacobsthal family.

Claim (proved, verified here): for every even difference with 3 not dividing e, the
paired max-gap F_e(y) in halved coordinates depends on e ONLY through
    delta = e * 3^{-1} mod Q,      Q = prod_{5<=q<=y} q,
and equals 3 * G(delta), where

    G(delta) = max cyclic gap of  S_delta = { k in Z_Q : k != 0, -delta mod q  for all q }.

Reason: with 3 not dividing e, gear 3 kills n = 0, -e mod 3, so every survivor lies in
the single remaining class c mod 3; writing n = 3k + c the gear-q conditions become
k != -c/3, (-e-c)/3 mod q, a translate (by the single integer -c*3^{-1} mod Q) of the
tooth pair {0, -delta}.  Translation does not change the gap multiset.  Gaps in n are
3x gaps in k.

This collapses the y=19 family scan from 2,424,922 differences (round 17: "out of
reach") to 1,616,615 delta values, and - more importantly - gives the exact
prefilter used in ext_deficit19.py.

Assertions: F_e computed the round-19 way (jacobsthal_family.F_of) equals 3*G(delta)
for random e at y = 11, 13, 17; the y=13 winner set is reproduced exactly.
"""
import numpy as np
from math import prod

GEARS = {5: [3, 5], 7: [3, 5, 7], 11: [3, 5, 7, 11], 13: [3, 5, 7, 11, 13],
         17: [3, 5, 7, 11, 13, 17], 19: [3, 5, 7, 11, 13, 17, 19]}


def F_of(gears, e, P):
    """round-19 definition (research/jacobsthal_family.py), halved coordinates."""
    a = np.ones(P, bool)
    for q in gears:
        a[0::q] = False
        a[(-e) % q::q] = False
    idx = np.flatnonzero(a)
    if idx.size < 2:
        return 0
    g = np.diff(np.append(idx, idx[0] + P))
    return int(g.max())


def surv_delta(qs, delta, Q):
    a = np.ones(Q, bool)
    for q in qs:
        a[0::q] = False
        a[(-delta) % q::q] = False
    return a


def G_of(qs, delta, Q):
    idx = np.flatnonzero(surv_delta(qs, delta, Q))
    if idx.size < 2:
        return 0
    return int(np.diff(np.append(idx, idx[0] + Q)).max())


def main():
    rng = np.random.default_rng(22)
    for y in (11, 13, 17):
        gears = GEARS[y]
        P = prod(gears)
        qs = [q for q in gears if q >= 5]
        Q = prod(qs)
        inv3 = pow(3, -1, Q)
        # random e coprime-to-3 sample + the known 13 winners
        sample = [int(x) for x in rng.integers(1, P, size=40) if x % 3]
        if y == 13:
            sample += [344, 734, 839, 916, 2164]
        for e in sample:
            d = (e * inv3) % Q
            assert F_of(gears, e, P) == 3 * G_of(qs, d, Q), (y, e)
        print(f"  y={y:>2}  P={P:>7}  Q={Q:>7}  delta-reduction verified on "
              f"{len(sample)} differences")

    # full y=13 winner set, both ways
    gears, y = GEARS[13], 13
    P = prod(gears)
    qs = [q for q in gears if q >= 5]
    Q = prod(qs)
    inv3 = pow(3, -1, Q)
    Fe = np.zeros(P // 2 + 1, np.int32)
    for e in range(1, P // 2 + 1):
        Fe[e] = F_of(gears, e, P)
    bestF = int(Fe.max())
    win_e = sorted(int(e) for e in np.flatnonzero(Fe == bestF))
    Gd = np.zeros(Q, np.int32)
    for d in range(Q):
        Gd[d] = G_of(qs, d, Q)
    bestG = int(Gd.max())
    win_d = set(int(d) for d in np.flatnonzero(Gd == bestG))
    assert bestF == 3 * bestG == 75, (bestF, bestG)
    assert {(e * inv3) % Q for e in win_e} <= win_d
    # every delta winner lifts back to exactly the e winners with 3 | e excluded
    back = set()
    for d in win_d:
        e0 = (3 * d) % Q  # e = 3*delta mod Q; lift over mod 3
        for t in range(3):
            e = e0 + t * Q
            if e % 3 == 0 or e == 0:
                continue
            back.add(min(e % P, P - (e % P)))
    assert back == set(win_e), (len(back), len(win_e))
    print(f"  y=13 winners: {len(win_e)} differences e  <->  {len(win_d)} deltas "
          f"(F={bestF}, G={bestG}); e-list {win_e[:5]}...")
    print("  delta_frame: ALL ASSERTIONS GREEN")


if __name__ == "__main__":
    main()
