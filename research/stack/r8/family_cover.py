"""Loop, entry 47: is the candidate family a complete residue system for the small gears?

The family: L + 12 g_i k d, for the first A gears g_i above sqrt q, k = 1..K, d = +-1.
For a small gear h (5, 7, 11, 13, 17): the residues mod h that the family's columns take, and
whether every residue appears (so the family always contains columns off h's two teeth), and
more strongly whether every PAIR of residues (mod h1, mod h2) appears (the family is a complete
system for h1 h2, so no pair of small gears can jointly block it).
Reported per machine: the smallest h whose residues the family misses, and the smallest pair
it misses; and the number of candidates left after the small gears up to 17 have struck.
usage: uv run python research/stack/r8/family_cover.py lo hi A K
"""
import sys
import numpy as np
from sympy import primerange
from pathlib import Path
sys.path.insert(0, "research/stack/r8")
import evolve_walk3 as E

def dist(ph, h): return min(ph, abs(ph - (h - 2)), h - ph if ph > h - 2 else h)

def main():
    lo, hi = int(sys.argv[1]), int(sys.argv[2]); A = int(sys.argv[3]); K = int(sys.argv[4])
    E.QMAX = hi
    N = hi * hi * 4 + 10; sv = np.ones(N + 1, dtype=bool); sv[:2] = False
    for i in range(2, int(N ** 0.5) + 1):
        if sv[i]: sv[i * i::i] = False
    smalls = [5, 7, 11, 13, 17]
    out = [__doc__.strip(), ""]
    miss_single = {}; miss_pair = {}; left = []; n = 0
    for q in list(primerange(lo, hi + 1)):
        m = E.Machine(q, sv); P = 6; seq = [g for g in m.ps if g >= 5][::-1]
        visited = []; L = -1
        for i, g in enumerate(seq[:-2]):
            visited.append(g); nxt = seq[i + 1]
            best = None
            for k in range(1, 41):
                for d in (1, -1):
                    x = L + 2 * k * P * g * d
                    if x < 0 or x > q * q - 2: continue
                    on = sum(1 for h in visited if x % h in (0, h - 2))
                    md = min(dist(x % h, h) for h in visited)
                    fl = 0
                    if on == 0:
                        fl = sum(1 for k2 in range(1, 21) for d2 in (1, -1) if 0 <= x + 2 * k2 * P * nxt * d2 <= q * q - 2 and all((x + 2 * k2 * P * nxt * d2) % h not in (0, h - 2) for h in visited + [nxt]))
                    sc = (-on, fl, md)
                    if best is None or sc > best[0]: best = (sc, x)
            L = best[1]
        gs = [x for x in m.ps if x * x > q][:A]
        fam = []
        for g in gs:
            for k in range(1, K + 1):
                for d in (1, -1):
                    x = L + 2 * k * P * g * d
                    if q < x <= q * q - 2: fam.append(x)
        n += 1
        ms = None
        for h in smalls:
            if len({x % h for x in fam}) < h: ms = h; break
        mp = None
        for i, h1 in enumerate(smalls):
            for h2 in smalls[i + 1:]:
                if len({(x % h1, x % h2) for x in fam}) < h1 * h2: mp = (h1, h2); break
            if mp: break
        miss_single[ms] = miss_single.get(ms, 0) + 1
        miss_pair[mp] = miss_pair.get(mp, 0) + 1
        surv = [x for x in fam if all(x % h not in (0, h - 2) for h in smalls)]
        left.append(len(surv))
    out.insert(2, f"machines {lo}..{hi} ({n}), family = {A} strides x {K} periods x 2 directions = {2*A*K} candidates: smallest gear whose residues the family misses: {miss_single}; smallest pair it misses: {miss_pair}; candidates surviving the gears 5..17: min {min(left)}, mean {sum(left)/n:.1f}")
    Path(f"research/stack/r8/results_family_cover_{lo}_{hi}.txt").write_text("\n".join(out), encoding="utf-8")
    print(out[2])

if __name__ == "__main__":
    main()
