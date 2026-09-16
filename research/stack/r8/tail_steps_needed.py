"""Loop, entry 60: how many tail steps does the walk need?  Run the walk with its lookahead to
the cut, then take tail steps one at a time; after each, test whether any candidate of the NEXT
tail step is open to every gear (that is, whether the walk can finish there).  Report the index
of the first tail step that finishes, over the machines.
usage: uv run python research/stack/r8/tail_steps_needed.py lo hi
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
    firsts = []; n = 0; never = []
    for q in list(primerange(lo, hi + 1)):
        K = max(20, int(math.log(q) ** 2)); m = E.Machine(q, sv); P = 6
        seq = [g for g in m.ps if g >= 5][::-1]; allg = [g for g in m.ps if g >= 5]
        cut = 0
        while cut < len(seq) and seq[cut] > 2 * (cut + 1): cut += 1
        visited = []; L = -1; first = None
        for i, g in enumerate(seq):
            visited.append(g); nxt = seq[i + 1] if i + 1 < len(seq) else None
            best = None; finishes = False
            for k in range(1, K + 1):
                for d in (1, -1):
                    x = L + 2 * k * P * g * d
                    if x < 0 or x > q * q - 2: continue
                    if i >= cut and q < x and all(x % h not in (0, h - 2) for h in allg): finishes = True
                    on = sum(1 for h in visited if x % h in (0, h - 2))
                    md = min(dist(x % h, h) for h in visited)
                    fl = 0
                    if on == 0 and nxt is not None:
                        fl = sum(1 for k2 in range(1, min(K, 20) + 1) for d2 in (1, -1) if 0 <= x + 2 * k2 * P * nxt * d2 <= q * q - 2 and all((x + 2 * k2 * P * nxt * d2) % h not in (0, h - 2) for h in visited + [nxt]))
                    sc = (-on, fl, md)
                    if best is None or sc > best[0]: best = (sc, x)
            if i >= cut and finishes and first is None: first = i - cut
            if best is None: break
            L = best[1]
        n += 1
        if first is None: never.append(q)
        else: firsts.append(first)
    from collections import Counter
    c = Counter(firsts)
    line = (f"machines {lo}..{hi} ({n}), K = (ln q)^2: the first tail step at which the walk can finish - "
            f"distribution {sorted(c.items())[:8]}; max {max(firsts)}; machines that never finish in the tail {len(never)} {never[:5]}")
    Path(f"research/stack/r8/results_tail_steps_needed_{lo}_{hi}.txt").write_text(line, encoding="utf-8")
    print(line)

if __name__ == "__main__":
    main()
