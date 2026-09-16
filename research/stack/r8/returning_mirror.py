"""Loop, entry 48: the returning mirror.  Idea: end the walk on a column congruent to the ORIGIN
modulo the tail primorial T = 5 * 7 * 11 * ... * m, so the tail gears' openness is inherited
from home (-1 is open to every gear) rather than re-earned.  A move by a multiple of 2T does
that; the walk's own moves are 12 g k, so the total displacement D must satisfy T | D/2, i.e.
2T | D.  The displacement lands in the window iff 2T <= q^2, so T <= q^2 / 2: the tail primorial
must fit the window.
Measured per machine: the largest m with the primorial of 5..m at most q^2 / 2 (how much tail
can be inherited), the gears left over (above m, below q), and whether a column
-1 + 2 T t inside the window is open to those leftover gears for some t (the walk's remaining
task).  This is the descent of 2026-09-15 re-derived as the walk's endgame: reported for
comparison.
usage: uv run python research/stack/r8/returning_mirror.py lo hi
"""
import sys
import numpy as np
from sympy import primerange
from pathlib import Path

def main():
    lo, hi = int(sys.argv[1]), int(sys.argv[2])
    N = hi * hi + 10; sv = np.ones(N + 1, dtype=bool); sv[:2] = False
    for i in range(2, int(N ** 0.5) + 1):
        if sv[i]: sv[i * i::i] = False
    out = [__doc__.strip(), ""]; rows = []
    inh = []; leftn = []; ok = 0; n = 0
    for q in list(primerange(lo, hi + 1)):
        ps = list(primerange(2, q + 1))
        T = 1; m = 1
        for p in ps:
            if 2 * T * p <= q * q: T *= p; m = p
            else: break
        left = [g for g in ps if g > m]
        ts = [t for t in range(1, (q * q - 1) // (2 * T) + 1) if q < 2 * T * t - 1 <= q * q - 2]
        good = [t for t in ts if all((2 * T * t - 1) % g and (2 * T * t + 1) % g for g in left)]
        tw = [t for t in good if sv[2 * T * t - 1] and sv[2 * T * t + 1]]
        n += 1; inh.append(m); leftn.append(len(left)); ok += bool(tw)
        if q in (503, 997, 1499, 1999): rows.append(f"   q = {q}: inherited tail = gears to {m} (T = {T}); leftover gears {len(left)} ({left[:5]}...); columns in reach {len(ts)}; open to the leftover gears {len(good)}; twins {len(tw)}")
    out.insert(2, f"machines {lo}..{hi} ({n}): the inherited tail reaches gear {min(inh)} to {max(inh)}; leftover gears {min(leftn)} to {max(leftn)}; a twin among the inherited columns at {ok}")
    out += rows
    Path(f"research/stack/r8/results_returning_mirror_{lo}_{hi}.txt").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out[2:8]))

if __name__ == "__main__":
    main()
