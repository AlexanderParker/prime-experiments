"""Is the hazard bound `h(L) >= d` a negative-association statement about the gears?

The remaining gap of the covering route is `h(L) >= d` for every `L`, where `d = prod (1 - 2/q)`. In
the shift picture `h(L)` is exactly a conditional probability. Gear `q` blocks the absolute positions
`= 0, 1 mod q`, so a window starting at `m` sees gear `q` at effective offset `-m mod q`, and ranging
`m` over `[0, P)` ranges over all offset vectors bijectively by CRT. With

    A(L) = { m : positions m, ..., m+L-1 all blocked }        so  |A(L)| = N(L),

the count of `m in A(L)` whose next position is exposed is one per maximal blocked run of length at
least `L`, that is `G(L)`. Hence

    h(L) = G(L)/N(L) = P( position m+L exposed | m in A(L) ),

and the target `h(L) >= d` says: **conditioning on a blocked run of length `L` does not make the next
position more likely to be blocked than it is unconditionally.**

Unconditionally the per-gear events "gear `q` blocks the next position" are independent with
probability `2/q` each, and `d` is the product of the complements. So `h(L) >= d` would follow from two
separate claims, each of which this module measures exactly:

    (M)  per-gear marginals do not rise:   P( q blocks m+L | m in A(L) )  <=  2/q   for every gear
    (N)  the conditional events are negatively associated, in the weak product sense:
         h(L)  >=  prod_q ( 1 - P( q blocks m+L | m in A(L) ) )

Neither is assumed. `(M)` is already known to fail for the *usefulness* proxy - the offsets blocking
position `L` are jointly below average at covering `[0, L)` only when `L mod q >= q/4`
(`docs/forbidden-configurations.md` section 6b) - so the point here is to find out whether the true
conditional marginal behaves better than the proxy, and if not, which of the two claims survives.

Everything is computed by exact enumeration over `[0, P)`, so the numbers are integers, not estimates.
"""

import sys
from math import prod
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from hazard import odd_primes_upto


def blocked_mask(primes):
    """`True` where the position is blocked by some gear; gear `q` blocks `= 0, 1 mod q`."""
    P = prod(primes)
    blocked = np.zeros(P, dtype=bool)
    for q in primes:
        blocked[0::q] = True
        blocked[1::q] = True
    return blocked


def run_starts(blocked, L):
    """Boolean array over `[0, P)`, `True` at `m` when `m .. m+L-1` are all blocked (cyclically)."""
    P = len(blocked)
    acc = blocked.copy()
    for k in range(1, L):
        acc &= np.roll(blocked, -k)
    return acc


def measure(primes, L):
    """Exact `(N, G, h, per-gear conditional marginals, product bound)` at run length `L`."""
    P = prod(primes)
    blocked = blocked_mask(primes)
    A = run_starts(blocked, L)
    N = int(A.sum())
    if N == 0:
        return None
    idx = np.flatnonzero(A)
    nxt = (idx + L) % P
    exposed_next = ~blocked[nxt]
    G = int(exposed_next.sum())
    h = G / N
    marg = {}
    for q in primes:
        r = nxt % q
        marg[q] = float(((r == 0) | (r == 1)).sum()) / N
    prod_bound = prod(1 - m for m in marg.values())
    return N, G, h, marg, prod_bound


if __name__ == "__main__":
    for y in (7, 11, 13, 17):
        primes = odd_primes_upto(y)
        d = prod(1 - 2 / q for q in primes)
        P = prod(primes)
        print(f"\ngears to {y}, P = {P}, d = {d:.8f}")
        print(f"  {'L':>4} {'N(L)':>10} {'h(L)':>10} {'h/d':>8} "
              f"{'M holds':>8} {'worst q':>8} {'marg/(2/q)':>11} {'N holds':>8} {'prod bd':>10}")
        m_fail = n_fail = 0
        L = 1
        while True:
            out = measure(primes, L)
            if out is None:
                break
            N, G, h, marg, pb = out
            ratios = {q: marg[q] / (2 / q) for q in primes}
            worst = max(ratios, key=lambda q: ratios[q])
            m_ok = ratios[worst] <= 1 + 1e-12
            n_ok = h >= pb - 1e-12
            m_fail += not m_ok
            n_fail += not n_ok
            if L <= 6 or not m_ok or not n_ok or L % 10 == 0:
                print(f"  {L:>4} {N:>10} {h:>10.6f} {h / d:>8.4f} "
                      f"{str(m_ok):>8} {worst:>8} {ratios[worst]:>11.6f} "
                      f"{str(n_ok):>8} {pb:>10.6f}")
            L += 1
        print(f"  claim (M) failures: {m_fail}, claim (N) failures: {n_fail}, "
              f"L range 1..{L - 1}")
