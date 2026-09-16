"""Loop, entry 45: settle the small gears first and carry them.  Base B = the primorial of the
small gears 2, 3, 5, ..., m (m a parameter): every flip about {B, g} preserves the phases of
every gear up to m, so once the column is open to them it stays open.  Then the walk over the
gears above m (descending) has only those gears in memory, and the free regime needs them all
above twice their number - which they are, until the count reaches half the smallest of them.
Measured per machine: the largest m whose primorial keeps a flip inside the window
(4 B g <= q^2 for the top gear g), the number of gears above m, and whether every step of the
walk is then in the free regime; also the walk's landing.
usage: uv run python research/stack/r8/small_first_carry.py lo hi
"""
import sys
import numpy as np
from sympy import primerange
from pathlib import Path
sys.path.insert(0, "research/stack/r8")
import evolve_walk3 as E

def dist(ph, h): return min(ph, abs(ph - (h - 2)), h - ph if ph > h - 2 else h)

def main():
    lo, hi = int(sys.argv[1]), int(sys.argv[2]); K = 40
    E.QMAX = hi
    N = hi * hi * 4 + 10; sv = np.ones(N + 1, dtype=bool); sv[:2] = False
    for i in range(2, int(N ** 0.5) + 1):
        if sv[i]: sv[i * i::i] = False
    out = [__doc__.strip(), ""]; rows = []; ok = 0; n = 0; free_all = 0
    for q in list(primerange(lo, hi + 1)):
        ps = list(primerange(2, q + 1))
        # largest primorial base whose flip with the top gear still fits: 4 B q <= q^2
        B = 1; m = 1
        for p in ps:
            if 4 * B * p * q <= q * q: B *= p; m = p
            else: break
        above = [g for g in ps if g > m][::-1]
        nA = len(above)
        allfree = all(g > 2 * (i + 1) for i, g in enumerate(above))
        n += 1; free_all += allfree
        # the walk: start from a column open to every gear up to m (home is; and B-flips preserve them)
        visited = []; L = -1; miss = 0
        for i, g in enumerate(above):
            visited.append(g); best = None
            for k in range(1, K + 1):
                for d in (1, -1):
                    x = L + 2 * k * B * g * d
                    if x < 0 or x > q * q - 2: continue
                    on = sum(1 for h in visited if x % h in (0, h - 2))
                    md = min(dist(x % h, h) for h in visited)
                    sc = (-on, md, -k)
                    if best is None or sc > best[0]: best = (sc, x)
            if best is None: miss += 1; break
            if best[0][0] < 0: miss += 1
            L = best[1]
        tw = L is not None and q < L <= q * q - 2 and bool(sv[L] and sv[L + 2])
        ok += tw
        if q in (503, 997, 1999, 2999): rows.append(f"   q = {q}: carried base {B} (gears to {m}), gears above {nA}, every step in the free regime {allfree}; missing keeping moves {miss}; landing {L} {'twin' if tw else 'not a twin'}")
    out.append(f"machines {lo}..{hi} ({n}): every step in the free regime at {free_all}; twin landings {ok}")
    out += rows
    Path(f"research/stack/r8/results_small_first_carry_{lo}_{hi}.txt").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out[2:8]))

if __name__ == "__main__":
    main()
