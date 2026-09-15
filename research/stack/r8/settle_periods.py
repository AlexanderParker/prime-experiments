"""Loop, entry 34: more periods instead of lookahead.  The full-memory settle walk (descending,
flex rule) with K = 15, 30, 60, 120 periods on offer per step: machines (to 3947) with a step
lacking a keeping move, and the landings that are twins.  Also the mirror {base, g, g_prev}
(the previous gear carried in the mirror, so it needs no check) with K = 15.
usage: uv run python research/stack/r8/settle_periods.py
"""
import sys
import numpy as np
from sympy import primerange
from pathlib import Path
sys.path.insert(0, "research/stack/r8")
import evolve_walk3 as E

def dist(ph, h): return min(ph, abs(ph - (h - 2)), h - ph if ph > h - 2 else h)

def walk(m, K, carry_prev=False):
    q = m.q; P = m.B; seq = list(m.above)[::-1]; visited = []; L = -1; fails = 0
    for i, g in enumerate(seq):
        visited.append(g); nxt = seq[i + 1] if i + 1 < len(seq) else None
        M = P * g
        check = visited
        if carry_prev and i > 0 and M * seq[i - 1] * 4 <= q * q: M *= seq[i - 1]; check = [h for h in visited if h != seq[i - 1]]
        best = None
        for k in range(1, K + 1):
            for d in (1, -1):
                n = L + 2 * k * M * d
                if n < 0 or n > q * q - 2: continue
                on = sum(1 for h in check if n % h in (0, h - 2))
                md = min(dist(n % h, h) for h in check) if check else 0
                fl = 0
                if on == 0 and nxt is not None and K <= 30:
                    fl = sum(1 for k2 in range(1, K + 1) for d2 in (1, -1)
                             if 0 <= n + 2 * k2 * P * nxt * d2 <= q * q - 2 and all((n + 2 * k2 * P * nxt * d2) % h not in (0, h - 2) for h in visited + [nxt]))
                sc = (-on, fl, md, -k)
                if best is None or sc > best[0]: best = (sc, n)
        if best is None: return None, fails + 1
        if best[0][0] < 0: fails += 1
        L = best[1]
    return L, fails

def main():
    E.QMAX = 3000
    N = E.QMAX * E.QMAX * 4 + 10; sv = np.ones(N + 1, dtype=bool); sv[:2] = False
    for i in range(2, int(N ** 0.5) + 1):
        if sv[i]: sv[i * i::i] = False
    ps = list(primerange(11, E.QMAX + 1)); qs = [p for p in ps if p <= 200] + [p for i, p in enumerate(ps) if p > 200 and i % 3 == 0]
    machines = [E.Machine(q, sv) for q in qs]
    out = [__doc__.strip(), ""]
    for K, carry in ((30, False), (60, False), (120, False), (15, True)):
        bad = []; twins = 0
        for m in machines:
            L, f = walk(m, K, carry)
            if f: bad.append((m.q, f))
            if L is not None and m.q < L <= m.q * m.q - 2 and m.sv[L] and m.sv[L + 2]: twins += 1
        line = f"K = {K}{', previous gear carried' if carry else ''}: machines with a step lacking a keeping move: {len(bad)} of {len(machines)}: {bad[:12]}{'...' if len(bad) > 12 else ''}; twins {twins}"
        out.append(line); print(line, flush=True)
    Path("research/stack/r8/results_settle_periods.txt").write_text("\n".join(out), encoding="utf-8")

if __name__ == "__main__":
    main()
