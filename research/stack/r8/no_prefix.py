"""Loop, entry 63: does the prefix earn its place?  Compare, per machine:
  (a) the full walk: prefix to the cut, then the handover step - candidates open to every gear;
  (b) no prefix at all: from home (-1), a single step of the same stride and period bound;
  (c) no prefix, but the stride chosen among the first eight gears above sqrt q (the freedom the
      walk would otherwise spend on the prefix).
usage: uv run python research/stack/r8/no_prefix.py lo hi
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
    A = dict(open_=0, zero=0); B = dict(open_=0, zero=0); C = dict(open_=0, zero=0); n = 0
    for q in list(primerange(lo, hi + 1)):
        K = max(20, int(math.log(q) ** 2)); m = E.Machine(q, sv); P = 6
        seq = [g for g in m.ps if g >= 5][::-1]; allg = [g for g in m.ps if g >= 5]
        cut = 0
        while cut < len(seq) and seq[cut] > 2 * (cut + 1): cut += 1
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
        def count(c, g, KK):
            return sum(1 for k in range(1, KK + 1) for d in (1, -1)
                       if q < c + 2 * k * P * g * d <= q * q - 2 and all((c + 2 * k * P * g * d) % h not in (0, h - 2) for h in allg))
        a = count(L, gp, K); b = count(-1, gp, K)
        cs = max(count(-1, g, K) for g in [x for x in m.ps if x * x > q][:8])
        n += 1
        A['open_'] += a; A['zero'] += (a == 0)
        B['open_'] += b; B['zero'] += (b == 0)
        C['open_'] += cs; C['zero'] += (cs == 0)
    line = (f"machines {lo}..{hi} ({n}), K = (ln q)^2: (a) full walk then the handover step: open candidates {A['open_']/n:.2f} on average, none at {A['zero']} machines; "
            f"(b) no prefix, one step from home with the same stride: {B['open_']/n:.2f}, none at {B['zero']}; "
            f"(c) no prefix, best stride of eight: {C['open_']/n:.2f}, none at {C['zero']}")
    Path(f"research/stack/r8/results_no_prefix_{lo}_{hi}.txt").write_text(line, encoding="utf-8")
    print(line)

if __name__ == "__main__":
    main()
