"""Loop, entry 50: the whole walk under the period law K ~ (ln q)^2 / 4.  Mirror {2,3,g},
gears q down to 7, full memory, one-gear lookahead, K scaled with q.  Reported: machines where
a step lacks a keeping move, twin landings, and the free-regime prefix's reach (steps with the
visited gears all above 2n, which the lemma proves with K = n; with K scaled the lemma's own
hypothesis is 2n < the smallest visited gear, unchanged, so the prefix is reported for
comparison, plus the wider prefix K >= 2n where the lemma applies with the scaled K).
usage: uv run python research/stack/r8/walk_scaled.py lo hi
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
    ok = 0; n = 0; fails = []; miss = []; prefixes = []
    for q in list(primerange(lo, hi + 1)):
        K = max(20, int(float(sys.argv[3]) * math.log(q) ** 2)) if len(sys.argv) > 3 else max(4, int(math.log(q) ** 2 / 4))
        m = E.Machine(q, sv); P = 6; seq = [g for g in m.ps if g >= 5][::-1]
        visited = []; L = -1; bad = 0
        free = 0
        while free < len(seq) and seq[free] > 2 * (free + 1): free += 1
        for i, g in enumerate(seq[:-1]):
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
            if best is None: bad += 1; break
            if best[0][0] < 0: bad += 1
            L = best[1]
        n += 1; prefixes.append(free / len(seq))
        if bad: miss.append((q, bad))
        tw = q < L <= q * q - 2 and bool(sv[L] and sv[L + 2])
        ok += tw
        if not tw: fails.append(q)
    line = (f"machines {lo}..{hi} ({n}), K = {sys.argv[3] if len(sys.argv) > 3 else "0.25"} (ln q)^2: "
            f"twin landings {ok}; failures {fails[:10]}; machines with a step lacking a keeping move {len(miss)} {miss[:6]}; free-regime prefix {sum(prefixes)/n*100:.1f}% of the steps on average")
    Path(f"research/stack/r8/results_walk_scaled_{lo}_{hi}.txt").write_text(line, encoding="utf-8")
    print(line)

if __name__ == "__main__":
    main()
