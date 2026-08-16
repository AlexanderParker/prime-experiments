"""The blocked-slot algorithm generalised from twins to any even gap.

For gap `h = 2d` measured from an odd base, the two blocked residues in halved
coordinates sit at `{o, o - d}` modulo each odd prime `q` - separated by `d`, not
adjacent. So the twin case (section 3 of docs/twin-prime-program.md) is `d = 1`, gap 4
is `d = 2`, gap 6 is `d = 3`, and so on. When `q` divides `d` the pair collapses to a
single residue, which is exactly why gap-6 pairs are denser than twins.

Two things this script establishes:

1. The class count per period is `prod (q - r_q)` with `r_q = 1` when `q | d` and 2
   otherwise. That is the Hardy-Littlewood factor `prod (q-1)/(q-2)` over `q | h`,
   obtained from the blocking rule alone with no analytic input.

2. The maximum gap is *not* a function of that density. `d = 1, 2, 4` share a density
   exactly, yet their maximum gaps differ. The `d = 1` and `d = 2` patterns are related
   by the dilation `m -> 2m` modulo the odd primorial, which preserves counts but not
   order.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from algorithms.twin_gap import trial_divisors


def pattern(d, y):
    """Survivors of the gap-`2d` blocking for divisors up to `y`, over one period."""
    qs = [q for q in trial_divisors(y) if q > 2]
    period = 1
    for q in qs:
        period *= q
    alive = np.ones(period, dtype=bool)
    for q in qs:
        for r in {0 % q, d % q}:
            alive[r::q] = False
    return np.flatnonzero(alive), period


def max_gap(d, y):
    idx, period = pattern(d, y)
    if idx.size == 0:
        return None, 0, period
    biggest = int(np.diff(idx).max()) if idx.size > 1 else 0
    biggest = max(biggest, int(idx[0]) + period - int(idx[-1]))
    return biggest, idx.size, period


def predicted_classes(d, y):
    """prod (q - r_q), r_q = 1 if q divides d else 2."""
    total = 1
    for q in trial_divisors(y):
        if q > 2:
            total *= q - (1 if d % q == 0 else 2)
    return total


if __name__ == "__main__":
    ys = (11, 13, 17, 19, 23)
    print("maximum gap F(d, y) of the gap-2d pattern")
    header = "  d  gap " + " ".join(f"{'y=' + str(y):>8}" for y in ys)
    print(header)
    for d in range(1, 7):
        row = " ".join(f"{max_gap(d, y)[0]:>8}" for y in ys)
        print(f"{d:>3} {2 * d:>4} {row}")

    print("\nclass counts at y = 23, against prod (q - r_q)")
    for d in range(1, 7):
        _, count, period = max_gap(d, 23)
        ok = "matches" if count == predicted_classes(d, 23) else "MISMATCH"
        print(f"  d={d} (gap {2 * d}): {count} of {period}, density "
              f"{count / period:.5f}  [{ok}]")
