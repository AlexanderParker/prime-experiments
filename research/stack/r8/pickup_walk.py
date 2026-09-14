"""Loop, iteration 1 (2026-09-15): the pick-up walk.

Principle (owner): the mirror action is the only provable way to carry residues along the line;
build a walk that picks up every gear's residue on the way so the last flip is open by
construction.  The gcd fact says residue -1 can be held for the base only; but a gear's phase
need not be held at -1, it need only be kept OFF its two teeth (0 and gear - 2).  So:

  visit the gears in order (descending from q, or ascending from the first above the base);
  at the step for gear g the mirror is {base, g}; choose direction d and periods k (k = 1..K)
  such that after the move every gear visited so far (and g itself, and the base) has its phase
  off both teeth, and the column stays at or above 0 and at most q^2.
  Greedy: first (k, d) that works; if none works the walk is stuck.
  The landing after the last gear is open to every gear of the machine: a twin if inside the
  window.

Information used: the walk's own phases (column mod gear), never primality of any number.
Measured: machines 11 to 2000, K = 1, 2, 3, 5; descending and ascending; stuck / landed outside /
twin in the window.

usage: uv run python research/stack/r8/pickup_walk.py 2000
"""
import sys
import numpy as np
from sympy import primerange
from pathlib import Path

def walk(q, order, K):
    ps = list(primerange(2, q + 1)); base = []; P = 1
    for p in ps:
        if P * p <= q // 2: P *= p; base.append(p)
        else: break
    gears = [p for p in ps if p not in base]
    if order == 'desc': gears = gears[::-1]
    n = -1; visited = [x for x in base if x >= 5]
    for g in gears:
        visited.append(g); ok = None
        for k in range(1, K + 1):
            for d in (1, -1):
                m = n + 2 * k * P * g * d
                if m < 0 or m > q * q - 2: continue
                if all(m % h not in (0, h - 2) for h in visited): ok = m; break
            if ok is not None: break
        if ok is None: return 'stuck', g, n
        n = ok
    return ('twin' if q < n else 'below window'), None, n

def main():
    Q = int(sys.argv[1]); qs = list(primerange(11, Q + 1))
    N = Q * Q + 10; sv = np.ones(N + 1, dtype=bool); sv[:2] = False
    for i in range(2, int(N ** 0.5) + 1):
        if sv[i]: sv[i * i::i] = False
    out = [__doc__.strip(), ""]
    for order in ('desc', 'asc'):
        for K in (1, 2, 3, 5):
            res = {'twin': 0, 'below window': 0, 'stuck': 0}; stuck_at = []; sample = []
            for q in qs:
                r, g, n = walk(q, order, K)
                if r == 'twin': assert sv[n] and sv[n + 2], (q, n)
                res[r] += 1
                if r == 'stuck': stuck_at.append((q, g))
                if q in (31, 101, 499, 1999): sample.append((q, r, g, n))
            out.append(f"{order}, K = {K}: twin in the window {res['twin']}, landed below the window (open but below q) {res['below window']}, stuck {res['stuck']}; stuck at (q, gear) {stuck_at[:8]}{'...' if len(stuck_at) > 8 else ''}; samples {sample}")
    Path("research/stack/r8/results_pickup_walk.txt").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out[2:]))

if __name__ == "__main__":
    main()
