"""The machine self-blocks, and that gives an exact recursion for the twin count.

Everything here is an exact integer identity, verified rather than estimated.

The self-blocking law. In pair-index space the lower tooth of gear `q` is
`u_q = (q + 1) / 6` when `q = 5 mod 6` and `(q - 1) / 6` when `q = 1 mod 6`. In both
cases the pair sitting at index `u_q` *contains q itself*:

    q = 5 mod 6  =>  q = 6 u_q - 1,  so index u_q is the pair (q, q + 2)
    q = 1 mod 6  =>  q = 6 u_q + 1,  so index u_q is the pair (q - 2, q)

So every gear strikes the pair it belongs to. The strike is correct arithmetic - `q` does
divide a member - but it removes a genuine twin whenever that pair is a twin pair. It is
exactly the exemption the pair-index sieve needs.

Tooth sharing. If `(p, p + 2)` are both gears then `u_p = u_{p+2}`, so the two gears of a
twin pair strike the *same* index. Their teeth coincide at `k = u_p`, roughly `p / 6`,
which is well inside the validity window - whereas a generic pair of gears has its first
tooth coincidence at a CRT position of size up to `q q'`. Twin gears therefore spend an
overlap early, where the window can see it.

The identity. Inside the validity window - `6K + 1 <= y^2`, so that no member below
`6K + 1` can have a prime factor above `y` without being prime - a position survives the
gears up to `y` exactly when both its members are prime and the lower one exceeds `y`.
Writing `T(x)` for the number of twin pairs with lower member at most `x`:

    survivors(y, K) = T(6K + 1) - T(y)          when 6K + 1 <= y^2

Taking `K` maximal gives the recursion `T(y^2) = survivors(y, (y^2 - 1) / 6) + T(y)`,
which unrolls along the square-root tower into a finite exact sum. It terminates in
`O(log log x)` levels, and each level's twins are the next level's self-blocked teeth:
the machine's output at one scale becomes structure inside the machine at the next.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from slip_algebra import gears_upto, teeth, tooth


def self_blocking_failures(limit):
    """Gears whose lower tooth is not the index of their own pair. Should be empty."""
    bad = []
    for q in gears_upto(limit):
        u = tooth(q)
        if q not in (6 * u - 1, 6 * u + 1):
            bad.append((q, u))
    return bad


def shared_teeth(limit):
    """Twin gear pairs below `limit`, with the tooth index they share."""
    gs = set(gears_upto(limit + 2))
    out = []
    for p in gears_upto(limit):
        if p + 2 in gs:
            out.append((p, p + 2, tooth(p), tooth(p) == tooth(p + 2)))
    return out


def survivors(y, K):
    """Positions `1 .. K` left open by every gear up to `y`, self-blocking included."""
    alive = np.ones(K + 1, dtype=bool)
    alive[0] = False
    for q in gears_upto(y):
        for t in teeth(q):
            alive[t::q] = False
    return int(alive.sum())


def twin_count(x):
    """`T(x)`: twin pairs whose lower member is at most `x`, counted from 5 up."""
    gs = set(gears_upto(x + 2))
    return sum(1 for p in gears_upto(x) if p + 2 in gs)


def window_identity(y):
    """Check `survivors(y, K) = T(6K + 1) - T(y)` at the largest window `K`."""
    K = (y * y - 1) // 6
    s = survivors(y, K)
    return {"y": y, "K": K, "survivors": s, "T_top": twin_count(6 * K + 1),
            "T_y": twin_count(y), "holds": s == twin_count(6 * K + 1) - twin_count(y)}


def tower(x):
    """Unroll `T(x)` along the square-root tower as an exact sum of survivor counts.

    Each level contributes the twins it is the first to see: those in `(y, y^2]`. The
    base of the tower is the smallest level, whose twins are counted directly.
    """
    base = 5
    total = twin_count(base)
    parts = [(f"[5, {base}]", base, total)]
    y = base
    while y < x:
        top = min(y * y, x)  # the band this level is the first to see; top <= y^2
        K = (top - 1) // 6
        # survivors already excise T(y) by self-blocking, so bands do not overlap
        s = survivors(y, K)
        total += s
        parts.append((f"({y}, {top}]", y, s))
        y = y * y
    return total, parts


if __name__ == "__main__":
    bad = self_blocking_failures(2000)
    print(f"self-blocking law, gears to 2000: {len(bad)} failures {bad if bad else ''}")

    sh = shared_teeth(120)
    print(f"\ntwin gears share their lower tooth: "
          f"{all(ok for *_, ok in sh)} over {len(sh)} pairs to 120")
    for p, r, u, _ in sh:
        print(f"  gears {p:>3}, {r:>3}  share tooth index {u:>3}  "
              f"= pair ({6 * u - 1}, {6 * u + 1})")

    print("\nwindow identity: survivors(y, K) = T(6K+1) - T(y),  6K+1 = y^2")
    print(f"  {'y':>6} {'K':>8} {'survivors':>10} {'T(6K+1)':>9} {'T(y)':>6} {'holds':>6}")
    for y in (11, 17, 29, 53, 101, 211, 503, 1009):
        r = window_identity(y)
        print(f"  {r['y']:>6} {r['K']:>8} {r['survivors']:>10} {r['T_top']:>9} "
              f"{r['T_y']:>6} {str(r['holds']):>6}")

    print("\nsquare-root tower: T(x) as an exact finite sum of survivor counts")
    for x in (10_000, 1_000_000):
        total, parts = tower(x)
        direct = twin_count(x)
        print(f"  x = {x}")
        for label, y, n in parts:
            print(f"     {label:>18}  gears <= {y:>7}  contributes {n:>7}")
        print(f"     {'sum':>18}  {total:>7}   direct T(x) = {direct}   "
              f"agree: {total == direct}")
