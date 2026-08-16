"""Next twin without walking candidates: every gear announces where it bites.

The generator of section 20 walks candidate slots and asks every gear whether it blocks
each one, costing `candidates * pi(sqrt(6k))`. That is the wrong way round. Each gear's
distance to its own next tooth is closed form: from a starting slot `k0`, gear `q` first
bites at

    d = min( (u_q - k0) mod q , (-u_q - k0) mod q )

with `u_q = 6^{-1} mod q` from the closed form of section 17b. Two modular reductions per
gear, no candidate ever divided, nothing walked. So set up one cursor per gear, let each
gear mark its own tooth positions across a window, and read off the first unmarked slot.

Cost becomes `pi(sqrt(6k))` for the setup plus `W * sum 2/q` for the marking, instead of
`candidates * pi(sqrt(6k))`. The gears do not get consulted per candidate at all.

What is *not* available in closed form is the joint answer - the first slot no gear bites.
That requires the CRT of every gear at once, whose period is the primorial, exponential in
the bound. So the jump distance for a single gear is free, the jump distance for the whole
set is the problem itself. This module buys the first and pays a window scan for the second.

Window choice. A window of `W` slots is certain to contain a survivor once `W >= F_k(y)`,
the maximum gap of section 18d. `F_k` is known exactly to `y = 43` and grows slowly in
practice, so the module retries with a doubled window rather than assuming a value, which
keeps the result exact regardless.
"""

import sys
from math import isqrt
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from twin_constructor import Gears, tooth


def bulk_gears(limit):
    """Every gear up to `limit`, by the blocked-slot rule applied across a range.

    This is the same rule the machine already uses, run over a whole range at once instead
    of one gap at a time: a known gear places a cursor and advances it, and whatever is
    never blocked is a gear. Only the `1` and `5` slots of the 6-cycle are represented, so
    the storage is one third of the range.

    Bootstrapping uses the trial-division gear list, but only up to `sqrt(limit)`, which is
    negligible next to the range itself.
    """
    m = limit // 6 + 2
    a = np.ones(m, dtype=bool)  # slots 6i + 1
    b = np.ones(m, dtype=bool)  # slots 6i + 5
    a[0] = False                # 6*0 + 1 = 1 is not a gear
    seed = Gears()
    for g in seed.upto(isqrt(limit) + 1):
        # 6 * (q+1)/6 = q+1 = 1 mod q, so for q = 5 mod 6 the inverse is tooth(q);
        # 6 * (q-1)/6 = q-1 = -1 mod q, so for q = 1 mod 6 it is q - tooth(q)
        inv6 = tooth(g) if g % 6 == 5 else g - tooth(g)
        ia = (-inv6) % g        # 6i + 1 = 0 mod g
        ib = (-5 * inv6) % g    # 6i + 5 = 0 mod g
        a[ia::g] = False
        b[ib::g] = False
        if g % 6 == 1:          # a gear must not block itself
            a[(g - 1) // 6] = True
        else:
            b[(g - 5) // 6] = True
    ga = 6 * np.flatnonzero(a) + 1
    gb = 6 * np.flatnonzero(b) + 5
    out = np.concatenate([ga, gb])
    out.sort()
    return out[(out >= 5) & (out <= limit)]


class BulkGears:
    """Gear list backed by `bulk_gears`, grown by doubling the range when needed."""

    def __init__(self, initial=1 << 16):
        self.limit = initial
        self.array = bulk_gears(self.limit)

    def upto(self, limit):
        while limit > self.limit:
            self.limit *= 4
            self.array = bulk_gears(self.limit)
        i = int(np.searchsorted(self.array, limit, side="right"))
        return self.array[:i]


class JumpFinder:
    """Next twin slot found by cursor announcement rather than candidate walking."""

    def __init__(self, bulk=False):
        self.gears = BulkGears() if bulk else Gears()

    def bite_distance(self, q, k0):
        """Closed-form distance from `k0` to the first slot gear `q` blocks."""
        u = tooth(q)
        return min((u - k0) % q, (-u - k0) % q)

    def next_twin(self, after, window=1024):
        """First twin slot strictly above `after`, with the window doubling as needed."""
        k0 = after + 1
        while True:
            top = k0 + window - 1
            bound = isqrt(6 * top + 1)
            gears = self.gears.upto(bound)
            alive = np.ones(window, dtype=bool)
            for q in gears:
                u = tooth(q)
                for t in (u, q - u):
                    start = (t - k0) % q
                    if start < window:
                        alive[start::q] = False
            hits = np.flatnonzero(alive)
            for h in hits:
                k = k0 + int(h)
                # a gear above sqrt(6k+1) has no authority over k; the only way it can
                # matter is by equalling a member outright, impossible once k is large
                if isqrt(6 * k + 1) <= bound:
                    return k, len(gears), window
            window *= 2

    def report(self, after, window=1024):
        k, ngears, used = self.next_twin(after, window)
        return {"after": after, "slot": k, "pair": (6 * k - 1, 6 * k + 1),
                "gears": ngears, "window": used, "jump": k - after}


if __name__ == "__main__":
    jf = JumpFinder()

    print("distance to each gear's next bite, from a sample start - all closed form")
    k0 = 1_000_000_000_001
    print(f"  from k0 = {k0}")
    print(f"  {'gear q':>8} {'k0 mod q':>9} {'u_q':>6} {'bite distance':>14}")
    for q in jf.gears.upto(60):
        print(f"  {q:>8} {k0 % q:>9} {tooth(q):>6} {jf.bite_distance(q, k0):>14}")

    import time

    print("\nnext twin by cursor announcement, with the cost split by phase")
    print(f"  {'after k':>14} {'twin slot':>14} {'jump':>5} {'gears':>7} "
          f"{'grow gears':>11} {'mark+read':>10}")
    for start in (10**6, 10**8, 10**10, 10**12):
        bound = isqrt(6 * (start + 1024) + 1)
        t = time.time()
        jf.gears.upto(bound)          # grow first, so the two costs are separable
        grow = time.time() - t
        t = time.time()
        r = jf.report(start)
        mark = time.time() - t
        print(f"  {r['after']:>14} {r['slot']:>14} {r['jump']:>5} {r['gears']:>7} "
              f"{grow:>10.3f}s {mark:>9.4f}s")
