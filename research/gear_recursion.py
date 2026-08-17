"""Adding one gear: the exact merge transform, and how much the maximum gap can grow.

This is the recursion on the gear set rather than on the run length. It is exact and mechanical - no
expansion, no averaging - and it is built here before any statistics are applied to it.

**The transform.** Let `M` be a gear set with period `P` and exposed set `E ⊂ [0, P)`. Adding a gear `q`
coprime to `P` gives period `Pq`, and by CRT

    E' = { x in [0, Pq) : x mod P in E,  x mod q not in {0, 1} }.

Walking `x` upward is walking `E` around `q` times: lap `l` covers `[lP, (l+1)P)` and contains the points
`e + lP` for `e in E`. Such a point survives when `(e + lP) mod q` avoids `{0, 1}`, that is when

    e mod q  avoids  { -lP mod q,  (1 - lP) mod q }.

So **each lap is `E` with two residue classes mod `q` deleted, and the deleted pair shifts by `-P mod q`
from one lap to the next.** Since `gcd(P, q) = 1` the shift is a generator, so across the `q` laps every
possible phase of gear `q` occurs exactly once. That is the whole content of "adding a gear": `q` copies of
the old pattern, each thinned at a different phase, laid end to end.

**Merging.** Deleting a point merges the two gaps either side of it; deleting `k` consecutive points merges
`k+1` gaps. So every new gap is a sum of consecutive old gaps, and the maximum gap can only grow by
merging.

**The constraint on runs of deletions.** Within one lap the deleted points all lie in two residue classes
`{phi, phi+1}` mod `q`, so any two deleted points differ by `0` or `+-1` mod `q`. Old gaps are at least 3,
so two *consecutive* deleted points differ by at least 3 and by `0` or `+-1` mod `q`, hence by at least
`q - 1`. **Consecutive deletions are at least `q - 1` apart**, which is why long merges are rare and why the
growth of the maximum gap is controlled by `q` rather than by the old maximum.

The script builds the transform, checks it against direct construction, and measures the increment
`F(M + q) - F(M)` against `q`.
"""

import sys
from math import prod
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


def odd_primes_upto(limit):
    return [p for p in range(3, limit + 1)
            if all(p % k for k in range(2, int(p**0.5) + 1))]


def exposed_direct(primes):
    """`E` for the gear set, by direct construction. Gear `q` blocks `= 0, 1 mod q`."""
    P = prod(primes)
    blocked = np.zeros(P, dtype=bool)
    for q in primes:
        blocked[0::q] = True
        blocked[1::q] = True
    return np.flatnonzero(~blocked).astype(np.int64), P


def gaps_of(E, P):
    """Cyclic gap multiset of a sorted exposed set over `[0, P)`."""
    return np.diff(np.concatenate([E, [E[0] + P]]))


def add_gear(E, P, q, want_gaps=True):
    """Apply the transform. Returns `(E', P*q)` or, with `want_gaps`, `(max_gap, gap_histogram)`.

    Laps are processed one at a time so the full `E'` of length about `A*(q-2)` never has to be
    materialised when only the gaps are wanted.
    """
    emod = (E % q).astype(np.int64)
    Pq = P * q
    shift = (-P) % q
    first_kept = None
    last_kept = None
    maxgap = 0
    hist = {}
    prev_tail = None  # last kept absolute position from the previous lap
    for lap in range(q):
        phi = (lap * shift) % q          # = (-lap*P) mod q
        keep = (emod != phi) & (emod != (phi + 1) % q)
        idx = np.flatnonzero(keep)
        if idx.size == 0:
            continue
        pts = E[idx] + lap * P
        if prev_tail is not None:
            g = int(pts[0] - prev_tail)
            maxgap = max(maxgap, g)
            hist[g] = hist.get(g, 0) + 1
        else:
            first_kept = int(pts[0])
        if idx.size > 1:
            d = np.diff(pts)
            m = int(d.max())
            maxgap = max(maxgap, m)
            vals, cnts = np.unique(d, return_counts=True)
            for v, c in zip(vals.tolist(), cnts.tolist()):
                hist[v] = hist.get(v, 0) + c
        prev_tail = int(pts[-1])
        last_kept = prev_tail
    # closing gap around the period
    g = int(first_kept + Pq - last_kept)
    maxgap = max(maxgap, g)
    hist[g] = hist.get(g, 0) + 1
    return maxgap, hist


def chain_max(g, q):
    """`F(M + q)` from the gap word alone, by the exact chain condition.

    A new gap merges `k+1` consecutive old gaps `g_i .. g_{i+k}` when the `k` exposed points between
    them are all deleted in one lap. Those points lie in `{phi, phi+1} mod q`, so taking the first of
    them as the origin, the partial sums of the interior gaps `g_{i+1} .. g_{i+k-1}` must all fall in
    `{0, 1} mod q` (first point on `phi`) or all in `{0, -1} mod q` (first point on `phi+1`). Every
    exposed point is deleted in exactly 2 of the `q` laps, so every chain satisfying that condition is
    realised somewhere and the maximum over chains is exactly the new maximum gap.

    `k = 1` needs no interior gaps and is always available, which is why `F(M + q)` is at least the
    largest sum of two adjacent gaps.
    """
    A = len(g)
    best = 0
    for i in range(A):
        # k = 1: merge g_i and g_{i+1}
        total = int(g[i]) + int(g[(i + 1) % A])
        best = max(best, total)
        # k >= 2: extend while the partial sums stay inside a two-element class
        for allowed in ({0, 1}, {0, q - 1}):
            r = 0
            total = int(g[i])
            j = i + 1
            while True:
                step = int(g[j % A])
                r2 = (r + step) % q
                if r2 not in allowed:
                    break
                total += step
                r = r2
                j += 1
                if j - i > A:
                    break
                best = max(best, total + int(g[j % A]))
    return best


def deletion_spacing_check(E, P, q):
    """Verify that consecutive deleted points within a lap are at least `q - 1` apart."""
    emod = (E % q).astype(np.int64)
    worst = None
    for lap in range(q):
        phi = (lap * ((-P) % q)) % q
        drop = (emod == phi) | (emod == (phi + 1) % q)
        idx = np.flatnonzero(drop)
        if idx.size < 2:
            continue
        pts = E[idx]
        d = np.diff(pts)
        m = int(d.min())
        if worst is None or m < worst:
            worst = m
    return worst


if __name__ == "__main__":
    print("transform against direct construction")
    for y in (7, 11, 13, 17):
        primes = odd_primes_upto(y)
        E, P = exposed_direct(primes)
        nxt = odd_primes_upto(y * 3)
        q = min(p for p in nxt if p > y)
        mg, hist = add_gear(E, P, q)
        E2, P2 = exposed_direct(primes + [q])
        g2 = gaps_of(E2, P2)
        direct_max = int(g2.max())
        direct_hist = {}
        vals, cnts = np.unique(g2, return_counts=True)
        for v, c in zip(vals.tolist(), cnts.tolist()):
            direct_hist[v] = c
        print(f"  gears {primes} + {q}: max gap transform {mg}, direct {direct_max}, "
              f"histograms equal: {hist == direct_hist}")

    print("\nconsecutive deletions are at least q-1 apart")
    for y in (11, 13, 17, 19):
        primes = odd_primes_upto(y)
        E, P = exposed_direct(primes)
        q = min(p for p in odd_primes_upto(y * 3) if p > y)
        worst = deletion_spacing_check(E, P, q)
        print(f"  gears to {y}, adding {q}: minimum spacing between consecutive deletions "
              f"= {worst}, bound q-1 = {q - 1}, holds: {worst is None or worst >= q - 1}")

    print("\nthe increment F(M+q) - F(M), against q")
    print(f"  {'gears to':>9} {'q added':>8} {'F(M)':>6} {'F(M+q)':>8} {'increment':>10} "
          f"{'incr/q':>7} {'sum of q':>9} {'F/sum':>7}")
    # `add_gear` never materialises E', but the *next* step needs E for the enlarged set, and
    # `exposed_direct` there costs a bool array of size P. P passes 3*10^9 once gear 29 is in, so the
    # walk stops rebuilding after the last gear set that fits in memory.
    REBUILD_LIMIT = 111_546_435  # P for gears to 23
    ADD = [7, 11, 13, 17, 19, 23, 29]
    primes = [3, 5]
    E, P = exposed_direct(primes)
    F = int(gaps_of(E, P).max())
    for q in ADD:
        mg, _ = add_gear(E, P, q)
        primes2 = primes + [q]
        s = sum(primes2)
        print(f"  {primes[-1]:>9} {q:>8} {F:>6} {mg:>8} {mg - F:>10} "
              f"{(mg - F) / q:>7.3f} {s:>9} {mg / s:>7.3f}")
        primes, F = primes2, mg
        if q is not ADD[-1]:
            if prod(primes) > REBUILD_LIMIT:
                print(f"    stopping the walk: period {prod(primes)} exceeds the rebuild limit")
                break
            E, P = exposed_direct(primes)
