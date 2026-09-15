"""Does the settle walk ever lack a keeping move?  For every machine (every prime to 200, then
every third to 4000), the full-memory settle walk (descending, K = 15, flex rule, no repairs):
at each step, does some move among the 30 (k = 1..15, up and down) inside [0, q^2 - 2] keep every
visited gear off its teeth?  Count the steps with no keeping move (defects), which machines have
one, and whether the landing was a twin anyway (after two repairs).
usage: uv run python research/stack/r8/settle_defects.py
"""
import sys
import numpy as np
from sympy import primerange
from pathlib import Path
sys.path.insert(0, "research/stack/r8")
import evolve_walk3 as E

def dist(ph, h): return min(ph, abs(ph - (h - 2)), h - ph if ph > h - 2 else h)

def walk(m, K=15):
    q = m.q; P = m.B; seq = list(m.above)[::-1]; visited = []; L = -1; defects = []
    for i, g in enumerate(seq):
        visited.append(g); nxt = seq[i + 1] if i + 1 < len(seq) else None
        best = None
        for k in range(1, K + 1):
            for d in (1, -1):
                n = L + 2 * k * P * g * d
                if n < 0 or n > q * q - 2: continue
                on = sum(1 for h in visited if n % h in (0, h - 2))
                md = min(dist(n % h, h) for h in visited)
                fl = 0
                if on == 0 and nxt is not None:
                    fl = sum(1 for k2 in range(1, K + 1) for d2 in (1, -1)
                             if 0 <= n + 2 * k2 * P * nxt * d2 <= q * q - 2 and all((n + 2 * k2 * P * nxt * d2) % h not in (0, h - 2) for h in visited + [nxt]))
                sc = (-on, fl, md, -k)
                if best is None or sc > best[0]: best = (sc, n)
        if best is None: return L, defects + [(i, g, 'no move in range')]
        if best[0][0] < 0: defects.append((i, g, -best[0][0]))
        L = best[1]
    return L, defects

def main():
    E.QMAX = 4000
    N = E.QMAX * E.QMAX * 4 + 10; sv = np.ones(N + 1, dtype=bool); sv[:2] = False
    for i in range(2, int(N ** 0.5) + 1):
        if sv[i]: sv[i * i::i] = False
    ps = list(primerange(11, E.QMAX + 1)); qs = [p for p in ps if p <= 200] + [p for i, p in enumerate(ps) if p > 200 and i % 3 == 0]
    machines = [E.Machine(q, sv) for q in qs]
    out = [__doc__.strip(), ""]
    nd = 0; rows = []
    for m in machines:
        L, defects = walk(m)
        tw = L is not None and m.q < L <= m.q * m.q - 2 and bool(m.sv[L] and m.sv[L + 2])
        if defects:
            nd += 1
            rows.append(f"   q = {m.q}: {len(defects)} step(s) with no keeping move: {defects[:4]}; landing {L} {'twin' if tw else 'not a twin'}")
    out.append(f"machines {len(machines)} (11 to {qs[-1]}), K = 15, descending, full memory, no repairs: machines with at least one step lacking a keeping move: {nd}")
    out += rows[:40]
    Path("research/stack/r8/results_settle_defects.txt").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out[2:30]))

if __name__ == "__main__":
    main()
