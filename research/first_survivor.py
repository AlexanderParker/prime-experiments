"""The quantity the reduction actually needs: the first survivor above y.

Section 4 of docs/twin-prime-program.md needs a survivor inside `(y, y^2]`. This
measures where the first one actually is, with the divisor set growing alongside y
so that each y is judged only by the divisors it is allowed to use.

That detail matters: using a fixed large divisor set for every y wrongly kills
midpoints that only a divisor larger than the midpoint itself blocks. For example
6 is a genuine twin midpoint (5, 7) but 5 divides 6 - 1, so with divisors up to 5
already switched on, 6 looks blocked. Survivors coincide with twins only above the
divisor bound, which is exactly what Lemma 2 requires.
"""

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from algorithms.twin_gap import trial_divisors


def sweep(limit=10**7, scan=200_000):
    """For each divisor bound y, report the first survivor midpoint above y."""
    alive = np.ones(limit + 3, dtype=bool)
    rows = []
    for y in trial_divisors(math.isqrt(limit) + 1):
        for r in (1 % y, (y - 1) % y):
            alive[r::y] = False
        seg = alive[y + 1 : min(limit, y + 1 + scan)]
        j = int(np.argmax(seg))
        if not seg[j]:
            raise RuntimeError(f"scan window too small at y = {y}")
        first = y + 1 + j
        rows.append((y, first, first / (y * y)))
    return rows


if __name__ == "__main__":
    rows = sweep()
    print(f"{'y':>6} {'first survivor > y':>19} {'y^2':>14} {'ratio':>8}")
    for y, first, ratio in rows[:6] + rows[-4:]:
        print(f"{y:>6} {first:>19} {y * y:>14} {ratio:>8.4f}")
    tail = [r for r in rows if r[0] >= 5]
    worst = max(tail, key=lambda r: r[2])
    print(f"max ratio for y >= 5: {worst[2]:.4f} at y = {worst[0]} "
          f"(the reduction needs ratio < 1)")
    gaps = [first - y for y, first, _ in tail]
    print(f"first survivor sits at most {max(gaps)} above y across this range")
