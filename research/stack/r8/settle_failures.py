"""Loop, entry 35: anatomy of the steps that lack a keeping move.  Full-memory settle walk,
descending, K = 15, flex rule, machines to 3000.  At each failing step: the column, the gear,
how many of the thirty candidates fit the range, and for each in-range candidate the visited
gears striking it (so the failure is read as 'few candidates fit' or 'every candidate struck').
usage: uv run python research/stack/r8/settle_failures.py
"""
import sys
import numpy as np
from sympy import primerange
from pathlib import Path
sys.path.insert(0, "research/stack/r8")
import evolve_walk3 as E

def dist(ph, h): return min(ph, abs(ph - (h - 2)), h - ph if ph > h - 2 else h)

def main():
    E.QMAX = 3000; K = 15
    N = E.QMAX * E.QMAX * 4 + 10; sv = np.ones(N + 1, dtype=bool); sv[:2] = False
    for i in range(2, int(N ** 0.5) + 1):
        if sv[i]: sv[i * i::i] = False
    ps = list(primerange(11, E.QMAX + 1)); qs = [p for p in ps if p <= 200] + [p for i, p in enumerate(ps) if p > 200 and i % 3 == 0]
    out = [__doc__.strip(), ""]; few = 0; allstruck = 0
    for q in qs:
        m = E.Machine(q, sv); P = m.B; seq = list(m.above)[::-1]; visited = []; L = -1
        for i, g in enumerate(seq):
            visited.append(g); nxt = seq[i + 1] if i + 1 < len(seq) else None
            cands = []
            for k in range(1, K + 1):
                for d in (1, -1):
                    n = L + 2 * k * P * g * d
                    if 0 <= n <= q * q - 2: cands.append((k, d, n, [h for h in visited if n % h in (0, h - 2)]))
            keeping = [c for c in cands if not c[3]]
            if not keeping:
                if len(cands) <= 3: few += 1
                else: allstruck += 1
                out.append(f"q = {q}, step {i} gear {g}, column {L}: {len(cands)} candidates in range of 30; strikers per candidate: " + "; ".join(f"k{k}{'+' if d > 0 else '-'}:{s}" for k, d, n, s in cands[:8]))
            best = None
            for k, d, n, s in cands:
                on = len(s); md = min(dist(n % h, h) for h in visited)
                fl = 0
                if on == 0 and nxt is not None:
                    fl = sum(1 for k2 in range(1, K + 1) for d2 in (1, -1) if 0 <= n + 2 * k2 * P * nxt * d2 <= q * q - 2 and all((n + 2 * k2 * P * nxt * d2) % h not in (0, h - 2) for h in visited + [nxt]))
                sc = (-on, fl, md, -k)
                if best is None or sc > best[0]: best = (sc, n)
            if best is None: break
            L = best[1]
    out.insert(2, f"failing steps with at most three candidates in range: {few}; with more than three, all struck: {allstruck}")
    Path("research/stack/r8/results_settle_failures.txt").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out[2:20]))

if __name__ == "__main__":
    main()
