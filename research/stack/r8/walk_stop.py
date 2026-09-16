"""Loop, entry 43: stop the walk at gear 7 (no final flip) and at gear 11 with a wider lookahead.
usage: uv run python research/stack/r8/walk_stop.py lo hi stop_at
"""
import sys
import numpy as np
from sympy import primerange
from pathlib import Path
sys.path.insert(0, "research/stack/r8")
import evolve_walk3 as E

def dist(ph, h): return min(ph, abs(ph - (h - 2)), h - ph if ph > h - 2 else h)

def main():
    lo, hi, stop_at = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]); K = 40
    look = [g for g in (7, 5) if g < stop_at]
    E.QMAX = hi
    N = hi * hi * 4 + 10; sv = np.ones(N + 1, dtype=bool); sv[:2] = False
    for i in range(2, int(N ** 0.5) + 1):
        if sv[i]: sv[i * i::i] = False
    ok = 0; n = 0; fails = []; margins = []
    for q in list(primerange(lo, hi + 1)):
        m = E.Machine(q, sv); P = 6; gears = [g for g in m.ps if g >= 5][::-1]
        seq = [g for g in gears if g >= stop_at]
        visited = []; L = -1
        for i, g in enumerate(seq):
            visited.append(g)
            nxt = seq[i + 1] if i + 1 < len(seq) else (look[0] if look else None)
            full = visited + look
            best = None; margin = 0
            for k in range(1, K + 1):
                for d in (1, -1):
                    x = L + 2 * k * P * g * d
                    if x < 0 or x > q * q - 2: continue
                    on = sum(1 for h in visited if x % h in (0, h - 2))
                    md = min(dist(x % h, h) for h in visited)
                    if i == len(seq) - 1 and all(x % h not in (0, h - 2) for h in full): margin += 1
                    fl = 0
                    if on == 0 and nxt is not None:
                        fl = sum(1 for k2 in range(1, 21) for d2 in (1, -1) if 0 <= x + 2 * k2 * P * nxt * d2 <= q * q - 2 and all((x + 2 * k2 * P * nxt * d2) % h not in (0, h - 2) for h in visited + [nxt]))
                    sc = (-on, fl, md)
                    if best is None or sc > best[0]: best = (sc, x)
            L = best[1]
        n += 1; margins.append(margin)
        if q < L <= q * q - 2 and sv[L] and sv[L + 2]: ok += 1
        else: fails.append(q)
    line = f"stop at gear {stop_at}, lookahead {look}, machines {lo}..{hi} ({n}): twin landings {ok}; failures {fails[:10]}; candidates at the last step open to every gear: min {min(margins)}, mean {sum(margins)/n:.2f}"
    Path(f"research/stack/r8/results_walk_stop_{stop_at}_{lo}_{hi}.txt").write_text(line, encoding="utf-8")
    print(line)

if __name__ == "__main__":
    main()
