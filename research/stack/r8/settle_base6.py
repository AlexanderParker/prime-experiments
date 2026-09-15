"""Loop, entry 36: the settle walk with the smallest base.  Mirror {2, 3, g} at every step
(P = 6), every gear from 5 visited descending with full memory, so that at every step at least
q^2 / (12 g) candidates fit the range.  K periods per direction.  Machines to 3000: steps
lacking a keeping move, and landings that are twins (a landing open to every visited gear is
open to every gear of the machine from 5, hence a twin by the landing lemma, since 2 and 3 are
carried from home).
usage: uv run python research/stack/r8/settle_base6.py
"""
import sys
import numpy as np
from sympy import primerange
from pathlib import Path
sys.path.insert(0, "research/stack/r8")
import evolve_walk3 as E

def dist(ph, h): return min(ph, abs(ph - (h - 2)), h - ph if ph > h - 2 else h)

def walk(m, K, flex=True):
    q = m.q; P = 6; seq = [g for g in m.ps if g >= 5][::-1]; visited = []; L = -1; fails = 0; fewfit = 0
    for i, g in enumerate(seq):
        visited.append(g); nxt = seq[i + 1] if i + 1 < len(seq) else None
        best = None; infit = 0
        for k in range(1, K + 1):
            for d in (1, -1):
                n = L + 2 * k * P * g * d
                if n < 0 or n > q * q - 2: continue
                infit += 1
                on = sum(1 for h in visited if n % h in (0, h - 2))
                md = min(dist(n % h, h) for h in visited)
                fl = 0
                if flex and on == 0 and nxt is not None:
                    fl = sum(1 for k2 in range(1, min(K, 20) + 1) for d2 in (1, -1) if 0 <= n + 2 * k2 * P * nxt * d2 <= q * q - 2 and all((n + 2 * k2 * P * nxt * d2) % h not in (0, h - 2) for h in visited + [nxt]))
                sc = (-on, fl, md, -k)
                if best is None or sc > best[0]: best = (sc, n)
        if best is None: return None, fails + 1, fewfit
        if best[0][0] < 0:
            fails += 1
            if infit <= 3: fewfit += 1
        L = best[1]
    return L, fails, fewfit

def main():
    E.QMAX = 1500
    N = E.QMAX * E.QMAX * 4 + 10; sv = np.ones(N + 1, dtype=bool); sv[:2] = False
    for i in range(2, int(N ** 0.5) + 1):
        if sv[i]: sv[i * i::i] = False
    ps = list(primerange(11, E.QMAX + 1)); qs = [p for p in ps if p <= 1500]
    machines = [E.Machine(q, sv) for q in qs]
    out = [__doc__.strip(), ""]
    for K in (40,):
        bad = []; twins = 0; inwin = 0
        for m in machines:
            L, f, ff = walk(m, K)
            if f: bad.append((m.q, f, ff))
            if L is not None and m.q < L <= m.q * m.q - 2:
                inwin += 1
                if m.sv[L] and m.sv[L + 2]: twins += 1
        line = f"base {{2,3}}, K = {K}: machines with a step lacking a keeping move: {len(bad)} of {len(machines)} (q, steps, of which with at most 3 candidates in range): {bad[:12]}{'...' if len(bad) > 12 else ''}; landings in the window {inwin}, twins {twins}"
        out.append(line); print(line, flush=True)
    Path("research/stack/r8/results_settle_base6.txt").write_text("\n".join(out), encoding="utf-8")

if __name__ == "__main__":
    main()
