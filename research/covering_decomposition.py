"""The covering counts N(L) in a form that separates the gear set from the run length.

`N(L)` is the number of shifts `m` for which positions `m, ..., m+L-1` are all blocked - equivalently
`sum_g max(0, g - L)` over the gaps `g` of the exposed set, since a gap of `g` contains `g - L`
starting points of a blocked run of length `L`.

Three things are established here.

**The step form.** `G(L) = N(L) - N(L+1)`, because a gap contributes to `N(L) - N(L+1)` exactly once
when `g > L`. So the hazard `h(L) = G(L)/N(L)` is `1 - rho(L)` with `rho(L) = N(L+1)/N(L)`, and the
remaining gap of the covering route - `min_L h(L) = h(1)` - is exactly

    rho(L) <= rho(1)   for every L,

that the step ratio of the covering counts is largest at `L = 1`. Note this is weaker than
log-concavity of `N`, which asks `rho` to be decreasing and is false at `L = 3` (section 27a).

**The inclusion-exclusion form.** Over positions rather than gaps,

    N(L) = sum_{T subset of [0,L)} (-1)^{|T|} E(T),    E(T) = prod_q ( q - |W_q(T)| )

with `E(empty) = P`, since `E(T)` counts shifts at which all of `T` is exposed.

**The factorisation.** Write `w(T) = |{s-1, s : s in T}|` counted over the integers, so
`w(T) = 2|T| - adj(T)` with `adj(T)` the number of `s in T` with `s+1 in T`. For any gear
`q > span(T) + 1` no two of those values collide mod `q`, so `|W_q(T)| = w(T)` - independent of where
`T` sits. The threshold is `span + 1` because the extreme values `min(T) - 1` and `max(T)` differ by
`span(T) + 1`. Since `span(T) <= L-1` for `T` inside `[0, L)`, every gear `q > L` sees only `w(T)`, and

    N(L) = sum_j c_j(L) * prod_{q > L, q <= y} (q - j),
    c_j(L) = sum_{T : w(T) = j} (-1)^{|T|} prod_{q <= L} ( q - |W_q(T)| ).

The coefficients `c_j(L)` do not depend on `y`. That is why the block starts `L = 3, 6, 9` came out in
closed form (sections 20b, 20c, 24c): there the shape-carrying gears are just `{3}`, `{3,5}` and
`{3,5,7}`, and everything else is a product. It also marks the limit of that method - once `L > y`
there is no tail left and every gear carries shape.
"""

import itertools
import sys
from math import prod
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from hazard import exposed_count, odd_primes_upto


def exposed_mask(primes):
    """Boolean array over `[0, P)`, True where the position is exposed to every gear."""
    P = prod(primes)
    alive = np.ones(P, dtype=bool)
    for q in primes:
        for t in (0, 1):
            alive[t::q] = False
    return alive


def covering_counts_direct(primes, lmax):
    """`N(L)` for `L = 1..lmax` by direct enumeration, via the gap identity."""
    alive = exposed_mask(primes)
    P = len(alive)
    E = np.flatnonzero(alive)
    gaps = np.diff(np.concatenate([E, [E[0] + P]]))
    return {L: int(np.maximum(gaps - L, 0).sum()) for L in range(1, lmax + 1)}


def w_of(T):
    """`|{s-1, s : s in T}|` over the integers."""
    return len({x for s in T for x in (s - 1, s)})


def covering_count_ie(primes, L):
    """`N(L)` by inclusion-exclusion over exposed-position subsets of `[0, L)`."""
    P = prod(primes)
    total = 0
    for r in range(L + 1):
        for T in itertools.combinations(range(L), r):
            total += (-1) ** r * (P if r == 0 else exposed_count(primes, T))
    return total


def decomposition(L, small_primes):
    """`c_j(L)` keyed by `j = w(T)`, using only the shape-carrying gears `q <= L`."""
    coeffs = {}
    for r in range(L + 1):
        for T in itertools.combinations(range(L), r):
            head = prod(q - len({x % q for s in T for x in (s - 1, s)}) for q in small_primes)
            if head == 0:
                continue
            j = w_of(T) if T else 0
            coeffs[j] = coeffs.get(j, 0) + (-1) ** r * head
    return coeffs


def from_decomposition(coeffs, primes, L):
    """Reassemble `N(L)` from `c_j(L)` and the shape-blind tail `q > L`."""
    tail = [q for q in primes if q > L]
    total = 0
    for j, c in coeffs.items():
        total += c * prod(q - j for q in tail)
    return total


if __name__ == "__main__":
    print("N(L) - N(L+1) == G(L), so h(L) = 1 - N(L+1)/N(L)")
    for primes in ([3, 5, 7], [3, 5, 7, 11], [3, 5, 7, 11, 13]):
        alive = exposed_mask(primes)
        P = len(alive)
        E = np.flatnonzero(alive)
        gaps = np.diff(np.concatenate([E, [E[0] + P]]))
        N = covering_counts_direct(primes, int(gaps.max()) + 1)
        ok = all(N[L] - N[L + 1] == int((gaps > L).sum())
                 for L in range(1, int(gaps.max())))
        print(f"  gears {primes}: identity holds: {ok}")

    print("\ninclusion-exclusion over positions reproduces N(L)")
    for primes in ([3, 5, 7], [3, 5, 7, 11]):
        N = covering_counts_direct(primes, 12)
        ok = all(covering_count_ie(primes, L) == N[L] for L in range(1, 13))
        print(f"  gears {primes}: matches direct: {ok}")

    print("\nthe c_j(L) coefficients do not depend on y")
    for L in (2, 3, 4, 6, 9, 12):
        small = odd_primes_upto(L)
        base = decomposition(L, small)
        print(f"  L = {L:>2}: shape gears {small}, c_j = "
              f"{ {j: c for j, c in sorted(base.items()) if c} }")
        for y in (13, 19, 23, 31):
            primes = odd_primes_upto(y)
            got = from_decomposition(base, primes, L)
            want = covering_count_ie(primes, L)
            flag = "ok" if got == want else f"MISMATCH {got} vs {want}"
            print(f"        y = {y:>3}: {flag}")

    print("\nthe step ratio rho(L) = N(L+1)/N(L), and where it peaks")
    for y in (7, 11, 13):
        primes = odd_primes_upto(y)
        d = prod(1 - 2 / q for q in primes)
        N = covering_counts_direct(primes, 60)
        rho1 = N[2] / N[1]
        peak, peak_at = -1.0, None
        for L in range(1, 60):
            if N[L] == 0:
                break
            r = N[L + 1] / N[L]
            if r > peak:
                peak, peak_at = r, L
        print(f"  y = {y:>3}: rho(1) = {rho1:.6f} = (1-2d)/(1-d) check "
              f"{(1 - 2 * d) / (1 - d):.6f}, peak rho = {peak:.6f} at L = {peak_at}")
