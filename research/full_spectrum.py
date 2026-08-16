"""Full-spectrum exposure: every beat kept, and the exact price of keeping them.

The exposure signal has an exact Fourier representation with the closed-form coefficients of
section 35a, so the next twin can in principle be found spectrally: evaluate

    E(m) = sum over k of Ehat(k) omega^{k m}

for successive `m` and stop at the first `E(m) = 1`. No beat may be dropped - a single
low-amplitude combination tone landing on a slot is a gear blocking a prime, and discarding it
loses the answer. This module builds that evaluator exactly and measures what full fidelity
costs.

Two exact facts decide the cost.

1. `E(m)` is 0 or 1, so a truncated spectrum determines it exactly as soon as the discarded tail
   is bounded by less than 1/2. The relevant quantity is therefore the **L1 norm** of the
   spectrum, not its energy.

2. The L1 norm factorises like the spectrum itself:

       ||Ehat||_1 = prod over q of [ (q-2)/q + (2/q) sum over c != 0 of |cos(2 pi c u_q / q)| ]

   Each factor is bounded below by about `1 + 4/pi - 2/q`, close to 2.27, so the L1 norm grows
   like `2.27^n` in the number of gears. Dropping any fixed fraction of the beats can then move
   the value by far more than 1/2, which is why no truncation is safe.
"""

import sys
from math import cos, pi, prod
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from slip_algebra import gears_upto, tooth


def crt_component(gear_set, q, k):
    P = prod(gear_set)
    return (k * pow((P // q) % q, -1, q)) % q


def ehat(gear_set, k):
    """Closed-form Ehat(k), all gears, no truncation."""
    out = 1.0
    for q in gear_set:
        c = crt_component(gear_set, q, k)
        if c == 0:
            out *= (q - 2) / q
        else:
            out *= -(2.0 / q) * cos(2 * pi * c * tooth(q) / q)
    return out


def exposure_from_spectrum(gear_set, m):
    """E(m) rebuilt from the full spectrum - every frequency, nothing discarded."""
    P = prod(gear_set)
    total = 0.0
    for k in range(P):
        total += ehat(gear_set, k) * np.cos(2 * pi * k * m / P)
    return total


def l1_factor(q):
    """Per-gear contribution to the L1 norm of the spectrum."""
    u = tooth(q)
    s = sum(abs(cos(2 * pi * c * u / q)) for c in range(1, q))
    return (q - 2) / q + (2.0 / q) * s


def l1_norm(gear_set):
    """Exact L1 norm of the spectrum, by the same factorisation."""
    return prod(l1_factor(q) for q in gear_set)


if __name__ == "__main__":
    S = [5, 7, 11]
    P = prod(S)
    print(f"gears {S}, period {P}")

    print("\nfull-spectrum reconstruction against the direct test, no truncation")
    print(f"  {'m':>5} {'E from spectrum':>17} {'E direct':>9} {'agree':>7}")
    worst = 0.0
    for m in range(14):
        got = exposure_from_spectrum(S, m)
        want = 1.0 if all(m % q not in (tooth(q), q - tooth(q)) for q in S) else 0.0
        worst = max(worst, abs(got - want))
        print(f"  {m:>5} {got:>17.12f} {want:>9.1f} {str(abs(got - want) < 1e-9):>7}")
    print(f"  worst deviation over m = 0..13: {worst:.3e}")

    print("\nL1 norm of the spectrum - the quantity that governs safe truncation")
    print(f"  {'gear':>6} {'L1 factor':>12} {'1 + 4/pi - 2/q':>16}")
    for q in gears_upto(30):
        print(f"  {q:>6} {l1_factor(q):>12.6f} {1 + 4 / pi - 2 / q:>16.6f}")

    print("\nL1 norm as gears accumulate, against the 1/2 needed to round safely")
    print(f"  {'gears up to':>12} {'n':>4} {'||Ehat||_1':>16} {'terms in period':>20}")
    for y in (5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 43):
        gs = gears_upto(y)
        print(f"  {y:>12} {len(gs):>4} {l1_norm(gs):>16.3e} {prod(gs):>20}")

    print("\ncost comparison for one slot decision")
    print(f"  {'gears up to':>12} {'n':>7} {'cursor test: n ops':>20} "
          f"{'inclusion-exclusion: 2^n':>26} {'full spectrum: P':>20}")
    for y in (13, 19, 29, 43):
        gs = gears_upto(y)
        n = len(gs)
        print(f"  {y:>12} {n:>7} {n:>20} {2**n:>26} {prod(gs):>20}")
