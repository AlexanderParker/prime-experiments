"""Layers between consecutive prime squares, read in the fields construction.

Between g^2 and g'^2 (consecutive primes g < g') the fields that can strike are those of the gears
up to g: the field of every gear below g is already running, and the field of g ENTERS at its
square g^2. A column (n, n+2) in the layer is open to the old fields (gears below g) iff no gear
below g divides n or n+2; among those old-open columns, the new field of g takes the ones where
g divides n or n+2 (its strikes there are g x m with m a survivor in [g, g'^2/g], a short list);
every other old-open column of the layer is a twin (a number below g'^2 with no factor <= g is
prime). So: twins in the layer = old-open columns - the new field's toll.
For consecutive primes g from 5 up to gmax: layer columns, old-open columns, the toll, the twins,
and the list of the new field's kills (g x m); then the minimum of old-open - toll, the layers
where the toll takes the largest share, and how the toll compares with the count of survivors m
in [g, g'^2/g] (its upper bound).
Usage: uv run python layers.py gmax
"""
import sys
import numpy as np
from sympy import primerange, nextprime, isprime


def main():
    gmax = int(sys.argv[1])
    rows = []
    for g in primerange(5, gmax + 1):
        gp = nextprime(g); lo, hi = g * g, gp * gp
        old = [p for p in primerange(5, g)]  # the old gears (2 and 3 are the fold)
        cols = [k for k in range(lo // 6 + 1, hi // 6 + 1) if 6 * k - 1 > lo and 6 * k + 1 < hi]
        old_open = []
        for k in cols:
            a, b = 6 * k - 1, 6 * k + 1
            if all(a % p and b % p for p in old): old_open.append(k)
        toll = [k for k in old_open if (6 * k - 1) % g == 0 or (6 * k + 1) % g == 0]
        twins = [k for k in old_open if k not in toll]
        assert all(isprime(6 * k - 1) and isprime(6 * k + 1) for k in twins)
        bound = sum(1 for m in range(g, hi // g + 1) if m % 6 in (1, 5) and all(m % p for p in old))
        rows.append((g, gp, len(cols), len(old_open), len(toll), len(twins), bound, [ (6*k-1 if (6*k-1)%g==0 else 6*k+1) for k in toll]))
    print("g -> g' | layer columns | old-open | toll of the new field g | twins | survivors m in [g, g'^2/g] (toll bound) | the new field's kills")
    for r in rows[:14]: print(f"{r[0]} -> {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]} | {r[6]} | {r[7]}")
    m = min(rows, key=lambda r: r[5]); print(f"minimum twins in a layer: {m[5]} at {m[0]} -> {m[1]} (old-open {m[3]}, toll {m[4]})")
    worst = max(rows, key=lambda r: r[4] / max(r[3], 1)); print(f"largest toll share: {worst[4]}/{worst[3]} at {worst[0]} -> {worst[1]}")
    share = [r[4] / max(r[3], 1) for r in rows]; print(f"toll share of the old-open columns: mean {np.mean(share):.3f}, max {max(share):.3f}; layers {len(rows)}; toll equals its bound in {sum(1 for r in rows if r[4]==r[6])} layers, below it in {sum(1 for r in rows if r[4]<r[6])}")
    print(f"old-open per layer against layer columns: mean share {np.mean([r[3]/r[2] for r in rows]):.3f}; twins per layer min {min(r[5] for r in rows)}, median {int(np.median([r[5] for r in rows]))}")


if __name__ == "__main__":
    main()
