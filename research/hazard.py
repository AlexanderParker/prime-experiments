"""The hazard condition h(L) >= d, and the block-start cases it reduces to.

Background: `docs/covering-bound-route.md` sections 15 to 18. In halved coordinates each odd prime
`q <= y` blocks the adjacent pair `{o, o+1} mod q`. The covering bound `N(L) <= P (1-d)^L` is
equivalent to the *hazard condition*

    h(L) = G(L) / N(L) >= d    for every L

where `G(L)` counts gaps of the exposed set exceeding `L` and `N(L) = sum_{g} max(0, g - L)`.
Because `G` is constant on the blocks `L in {1,2}, {3,4,5}, {6,7,8}, ...` while `N` decreases, `h`
rises within a block, so only the block starts `L = 1, 3, 6, 9, ...` need checking.

Two mechanical laws drive everything:

* **gear 3** blocks one of any two adjacent positions, so every gap is a multiple of 3;
* **gear 5** blocks one of any three positions spaced 3 apart, so no `m, m+3, m+6` are all exposed.

The second means every *chain count* `T_j = #{m, m+3, ..., m+3j all exposed}` vanishes for `j >= 2`,
which collapses most of the inclusion-exclusion below: any position set containing three terms in
arithmetic progression of difference 3 contributes nothing.

The counts are all products of per-gear factors. For a set `S` of positions,

    #{every position of S exposed} = prod_q ( q - |W_q(S)| ),   W_q(S) = {s-1, s : s in S} mod q

since gear `q` blocks position `s` exactly when its offset is `s-1` or `s`.
"""

import itertools
import sys
from math import prod
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def odd_primes_upto(limit):
    return [p for p in range(3, limit + 1)
            if all(p % k for k in range(2, int(p**0.5) + 1))]


def exposed_count(primes, S):
    """`#{m : every position of m + S is exposed}`, as a product of per-gear factors."""
    total = 1
    for q in primes:
        forbidden = set()
        for s in S:
            forbidden.add((s - 1) % q)
            forbidden.add(s % q)
        total *= q - len(forbidden)
        if total == 0:
            return 0
    return total


def gap_count(primes, L):
    """`#{gaps = L}` by inclusion-exclusion over which interior multiples of 3 are exposed.

    A gap of `L` from `m` means `m` and `m + L` exposed with every interior multiple of 3 blocked.
    Inclusion-exclusion over the interior set gives an alternating sum of `exposed_count` values;
    terms whose position set holds three points spaced 3 apart vanish by the gear-5 law.
    """
    if L % 3:
        return 0
    interior = list(range(3, L, 3))
    total = 0
    # A term vanishes exactly when its position set holds three points spaced 3 apart -
    # `s, s+3, s+6` - since gear 5 blocks one of any such triple. Skipping those is a pruning
    # of the sum, not a change to it. Note `{0, 3, 9}` is *not* such a triple, so it does
    # contribute; an earlier version wrongly excluded any subset touching 3 or L-3.
    for r in range(len(interior) + 1):
        for T in itertools.combinations(interior, r):
            S = (0,) + T + (L,)
            sset = set(S)
            if any(s + 3 in sset and s + 6 in sset for s in S):
                continue
            total += (-1) ** r * exposed_count(primes, S)
    return total


def gap_profile(primes, Fmax):
    """`#{gaps = 3j}` for `3j <= Fmax`."""
    return {L: gap_count(primes, L) for L in range(3, Fmax + 1, 3)}


def hazard_at(primes, L, profile, P):
    """`(G(L), N(L), h(L))` computed from the gap profile."""
    G = sum(c for g, c in profile.items() if g > L)
    N = sum(c * (g - L) for g, c in profile.items() if g > L)
    return G, N, (G / N if N else None)


PROFILE_CAP = 30  # block starts beyond this are not needed for the cases under study


if __name__ == "__main__":
    import numpy as np

    print("verifying the gap profile against direct computation")
    for primes in ([3, 5, 7], [3, 5, 7, 11], [3, 5, 7, 11, 13]):
        P = prod(primes)
        alive = np.ones(P, dtype=bool)
        for q in primes:
            for t in (0, 1):
                alive[t::q] = False
        E = np.flatnonzero(alive)
        gaps = np.diff(np.concatenate([E, [E[0] + P]]))
        direct = {}
        for g in sorted(set(gaps.tolist())):
            direct[int(g)] = int((gaps == g).sum())
        Fmax = int(gaps.max())
        pred = {g: c for g, c in gap_profile(primes, Fmax).items() if c}
        print(f"  gears {primes}: predicted == direct: {pred == direct}")
        if pred != direct:
            print(f"    predicted {pred}")
            print(f"    direct    {direct}")

    print("\nchain counts T_j vanish from j = 2, by the gear-5 law")
    for primes in ([3, 5, 7], [3, 5, 7, 11], [3, 5, 7, 11, 13, 17]):
        ts = [exposed_count(primes, tuple(3 * i for i in range(j + 1))) for j in range(4)]
        print(f"  gears {primes}: T_0..T_3 = {ts}")

    print("\nhazard at the block starts, from the closed-form profile")
    for y in (7, 11, 13, 17, 19, 23, 29):
        primes = odd_primes_upto(y)
        P = prod(primes)
        d = prod(1 - 2 / q for q in primes)
        # F_h is bounded by the largest gap with a nonzero count
        prof = {g: c for g, c in gap_profile(primes, PROFILE_CAP).items() if c}
        F = max(prof) if prof else 0
        rows = []
        for L in [1] + list(range(3, F, 3)):
            G, N, h = hazard_at(primes, L, prof, P)
            if h is not None:
                rows.append((L, h, h / d))
        worst = min(rows, key=lambda r: r[2])
        print(f"  y = {y:>3}: F_h = {F:>4}, d = {d:.6f}, "
              f"min h/d = {worst[2]:.4f} at L = {worst[0]}")
