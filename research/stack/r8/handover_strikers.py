"""Loop, entry 62: who strikes the handover's candidates?  The handover column is open to every
gear above the cut, but its candidates are new columns, so ANY gear may strike them.  Measured
at the handover step: of the 2K candidates, how many are struck by a gear above the cut, how
many by a tail gear only, and how many by nothing; and the number of distinct gears that strike
at all, against the number of candidates.
usage: uv run python research/stack/r8/handover_strikers.py lo hi
"""
import sys, math
import numpy as np
from sympy import primerange
from pathlib import Path
sys.path.insert(0, "research/stack/r8")
import evolve_walk3 as E

def dist(ph, h): return min(ph, abs(ph - (h - 2)), h - ph if ph > h - 2 else h)

def main():
    lo, hi = int(sys.argv[1]), int(sys.argv[2])
    E.QMAX = hi
    N = hi * hi * 4 + 10; sv = np.ones(N + 1, dtype=bool); sv[:2] = False
    for i in range(2, int(N ** 0.5) + 1):
        if sv[i]: sv[i * i::i] = False
    tot = dict(cand=0, big=0, tailonly=0, open_=0, distinct=0, n=0, tailgears=0, biggears=0)
    for q in list(primerange(lo, hi + 1)):
        K = max(20, int(math.log(q) ** 2)); m = E.Machine(q, sv); P = 6
        seq = [g for g in m.ps if g >= 5][::-1]; allg = [g for g in m.ps if g >= 5]
        cut = 0
        while cut < len(seq) and seq[cut] > 2 * (cut + 1): cut += 1
        big = set(seq[:cut]); tail = [g for g in allg if g not in big]
        visited = []; L = -1
        for i, g in enumerate(seq[:cut]):
            visited.append(g); nxt = seq[i + 1]
            best = None
            for k in range(1, K + 1):
                for d in (1, -1):
                    x = L + 2 * k * P * g * d
                    if x < 0 or x > q * q - 2: continue
                    on = sum(1 for h in visited if x % h in (0, h - 2))
                    md = min(dist(x % h, h) for h in visited)
                    fl = 0
                    if on == 0:
                        fl = sum(1 for k2 in range(1, min(K, 20) + 1) for d2 in (1, -1) if 0 <= x + 2 * k2 * P * nxt * d2 <= q * q - 2 and all((x + 2 * k2 * P * nxt * d2) % h not in (0, h - 2) for h in visited + [nxt]))
                    sc = (-on, fl, md)
                    if best is None or sc > best[0]: best = (sc, x)
            L = best[1]
        gp = seq[cut] if cut < len(seq) else 7
        cands = [L + 2 * k * P * gp * d for k in range(1, K + 1) for d in (1, -1)]
        cands = [x for x in cands if q < x <= q * q - 2]
        strikers = set(); nbig = 0; ntail = 0; nopen = 0
        for x in cands:
            sb = [h for h in big if x % h in (0, h - 2)]
            st = [h for h in tail if x % h in (0, h - 2)]
            strikers.update(sb); strikers.update(st)
            if sb: nbig += 1
            elif st: ntail += 1
            else: nopen += 1
        tot['n'] += 1; tot['cand'] += len(cands); tot['big'] += nbig; tot['tailonly'] += ntail
        tot['open_'] += nopen; tot['distinct'] += len(strikers); tot['tailgears'] += len(tail); tot['biggears'] += len(big)
    n = tot['n']
    line = (f"machines {lo}..{hi} ({n}), K = (ln q)^2: per handover step, candidates {tot['cand']/n:.1f}; struck by a gear ABOVE the cut {tot['big']/n:.1f}; "
            f"by a tail gear only {tot['tailonly']/n:.1f}; open {tot['open_']/n:.2f}; distinct gears striking {tot['distinct']/n:.1f} "
            f"(the machine has {tot['biggears']/n:.0f} gears above the cut and {tot['tailgears']/n:.0f} in the tail)")
    Path(f"research/stack/r8/results_handover_strikers_{lo}_{hi}.txt").write_text(line, encoding="utf-8")
    print(line)

if __name__ == "__main__":
    main()
