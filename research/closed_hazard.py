"""Closed forms for `N(L)` at the tight block starts, by pruned enumeration.

From `covering_decomposition.py`,

    N(L) = sum_j c_j(L) * prod_{q > L} (q - j),
    c_j(L) = sum_{T subset of [0,L), w(T) = j} (-1)^{|T|} prod_{q <= L} ( q - |W_q(T)| ),

with `w(T) = |{s-1, s : s in T}|` over the integers and `W_q(T)` its reduction mod `q`. The `c_j(L)` are
independent of the gear set beyond `q <= L`, so once they are known `N(L)` is an explicit polynomial in
the gears.

Naively `c_j(L)` is a sum over `2^L` subsets, which is why section 25b judged the per-`j` recipe not to
scale. But almost every subset contributes nothing, and the vanishing is detectable early:
`prod (q - |W_q(T)|)` is zero as soon as **one** gear has `W_q(T) = Z_q`. Gear 3 alone kills every `T`
holding positions in two different classes mod 3, cutting `2^L` to `3 * 2^(L/3)`; gear 5 then forbids
three positions spaced 3 apart, cutting that to a Fibonacci-like count. A depth-first scan that prunes
on the first fully covered gear therefore visits only the surviving subsets.

That makes the tight block starts identified in `docs/forbidden-configurations.md` section 8 -
`L = 1, 3, 6, 9, 15, 21, 24, 39, 54` - all reachable, and turns `h(L) >= d` at each of them into an
explicit inequality between products over the gear set.

**Validity condition.** `c_j(L)` is built from the gears `q <= L`, so the formula describes the machine
only when the gear set actually contains all of them, that is when `y >= L`. Below that the head gears
include primes that are not in the machine and the closed form is for a different pattern - at
`y = 13, L = 21` it returns 406008 against a true `N(21) = 312`. `valid_for` encodes the condition, and
the reassembly check below reports agreement only where it applies.

The hazard at `L` needs `N(L)` **and** `N(L+1)`, so its validity condition is the stronger
`y >= L + 1`. Testing only `valid_for(L)` lets one bad term through - at `y = 29, L = 30` the formula
for `N(31)` invents a gear 31 that the machine does not have, and the reported hazard comes out
negative, `h/d = -536`. `hazard_condition` returns `None` outside its range for that reason.
"""

import sys
from math import prod
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from hazard import odd_primes_upto


def c_coefficients(L, verbose=False):
    """`c_j(L)` by pruned depth-first enumeration. Returns `(coeffs, visited, kept)`."""
    head_gears = odd_primes_upto(L)
    full = {q: (1 << q) - 1 for q in head_gears}
    coeffs = {}
    stats = [0, 0]

    def rec(pos, chosen, masks):
        stats[0] += 1
        # record the current subset
        if chosen:
            w = len({x for s in chosen for x in (s - 1, s)})
        else:
            w = 0
        head = 1
        for q in head_gears:
            head *= q - bin(masks[q]).count("1")
        if head:
            stats[1] += 1
            coeffs[w] = coeffs.get(w, 0) + (-1) ** len(chosen) * head
        if pos >= L:
            return
        for nxt in range(pos, L):
            new_masks = dict(masks)
            dead = False
            for q in head_gears:
                new_masks[q] |= (1 << ((nxt - 1) % q)) | (1 << (nxt % q))
                if new_masks[q] == full[q]:
                    dead = True
                    break
            if dead:
                continue  # this position kills the product; skip it, keep scanning
            rec(nxt + 1, chosen + (nxt,), new_masks)

    rec(0, (), {q: 0 for q in head_gears})
    if verbose:
        print(f"    L = {L}: visited {stats[0]}, contributing {stats[1]}, "
              f"vs 2^{L} = {2 ** L}")
    return {j: c for j, c in coeffs.items() if c}, stats[0], stats[1]


def valid_for(L, primes):
    """The closed form at block start `L` needs every prime `<= L` present in the gear set."""
    return all(q in primes for q in odd_primes_upto(L))


def N_from_coeffs(coeffs, primes, L):
    tail = [q for q in primes if q > L]
    return sum(c * prod(q - j for q in tail) for j, c in coeffs.items())


def N_direct(primes, L):
    """`N(L)` from the gap multiset, by direct construction of the pattern."""
    import numpy as np
    P = prod(primes)
    blocked = np.zeros(P, dtype=bool)
    for q in primes:
        blocked[0::q] = True
        blocked[1::q] = True
    E = np.flatnonzero(~blocked)
    gaps = np.diff(np.concatenate([E, [E[0] + P]]))
    return int(np.maximum(gaps - L, 0).sum())


_CACHE = {}


def c_cached(L):
    if L not in _CACHE:
        _CACHE[L] = c_coefficients(L)[0]
    return _CACHE[L]


def hazard_condition(L, primes):
    """`(N(L), N(L+1), h(L), d, holds)` from the closed forms, or None if out of range.

    Needs `N(L+1)` as well as `N(L)`, hence the condition `valid_for(L + 1, primes)`.
    """
    if not valid_for(L + 1, primes):
        return None
    cL = c_cached(L)
    cL1 = c_cached(L + 1)
    NL = N_from_coeffs(cL, primes, L)
    NL1 = N_from_coeffs(cL1, primes, L + 1)
    d = prod(1 - 2 / q for q in primes)
    h = (NL - NL1) / NL
    return NL, NL1, h, d, h >= d


if __name__ == "__main__":
    TIGHT = [1, 3, 6, 9, 15, 21, 24]

    print("pruning: how much of the 2^L subset space actually contributes")
    for L in TIGHT + [25, 30, 39]:
        coeffs, visited, kept = c_coefficients(L, verbose=True)

    print("\nclosed-form c_j(L) at the tight block starts")
    for L in TIGHT:
        coeffs, _, _ = c_coefficients(L)
        print(f"  L = {L:>3}: c_j = { {j: c for j, c in sorted(coeffs.items())} }")

    print("\nreassembly against direct construction, where the formula is valid")
    for y in (13, 17, 19, 23):
        primes = odd_primes_upto(y)
        ok, checked, skipped = True, 0, []
        for L in TIGHT:
            if not valid_for(L, primes):
                skipped.append(L)
                continue
            coeffs, _, _ = c_coefficients(L)
            got = N_from_coeffs(coeffs, primes, L)
            want = N_direct(primes, L)
            checked += 1
            if got != want:
                ok = False
                print(f"    y = {y}, L = {L}: MISMATCH closed {got} vs direct {want}")
        print(f"  y = {y:>3}: {checked} tight L checked, exact: {ok}"
              + (f", out of range: {skipped}" if skipped else ""))

    print("\nthe hazard condition at the tight block starts, from closed forms only")
    print(f"  {'y':>4} {'tight L in range':>17} {'worst L':>8} {'h/d':>10} {'holds':>7}")
    all_hold = True
    for y in (23, 29, 31, 37, 41, 47, 59, 71, 89, 101, 127, 151, 199):
        primes = odd_primes_upto(y)
        worst = None
        n_in = 0
        for L in TIGHT + [30, 39, 54]:
            out = hazard_condition(L, primes)
            if out is None:
                continue
            n_in += 1
            NL, NL1, h, d, holds = out
            if worst is None or h / d < worst[1]:
                worst = (L, h / d, holds)
        all_hold &= worst[2]
        print(f"  {y:>4} {n_in:>17} {worst[0]:>8} {worst[1]:>10.6f} {str(worst[2]):>7}")
    print(f"  every tight case holds across the range: {all_hold}")
