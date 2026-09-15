"""Variant of the memory measurement: the mirror at each step is the base times the remembered
gears that fit (product at most q^2 / 60, so that fifteen periods stay inside the window), so
the gears carried in the mirror keep their phase without being checked; only the remembered
gears NOT in the mirror are checked.  Does carrying cut the memory a full streak needs?
usage: uv run python research/stack/r8/settle_memory_carry.py
"""
import sys
import numpy as np
from sympy import primerange
from pathlib import Path
sys.path.insert(0, "research/stack/r8")
import evolve_walk3 as E

def dist(ph, h): return min(ph, abs(ph - (h - 2)), h - ph if ph > h - 2 else h)

def walk(m, w, K=15, R=2):
    q = m.q; B = m.B; seq = list(m.above)[::-1]; visited = []; L = -1
    for i, g in enumerate(seq):
        visited.append(g)
        memory = visited if w == 'all' else visited[-(w + 1):]
        carried = []; M = B * g
        for h in [x for x in memory if x != g]:
            if M * h * 60 <= q * q and L % h not in (0, h - 2): M *= h; carried.append(h)
        check = [h for h in memory if h not in carried]
        best = None
        for k in range(1, K + 1):
            for d in (1, -1):
                n = L + 2 * k * M * d
                if n < 0 or n > q * q - 2: continue
                on = sum(1 for h in check if n % h in (0, h - 2))
                md = min(dist(n % h, h) for h in check) if check else 0
                sc = (-on, md, -k)
                if best is None or sc > best[0]: best = (sc, n)
        if best is None: return None
        L = best[1]
    memory = visited if w == 'all' else visited[-(w + 1):]
    small = [g for g in m.above if g <= m.r] or m.above[:3]
    for _ in range(R):
        if all(L % h not in (0, h - 2) for h in memory): break
        best = None
        for s in small:
            for k in range(1, K + 1):
                for d in (1, -1):
                    n = L + 2 * k * B * s * d
                    if not (q < n <= q * q - 2): continue
                    on = sum(1 for h in memory if n % h in (0, h - 2))
                    sc = (-on, min(dist(n % h, h) for h in memory), -k)
                    if best is None or sc > best[0]: best = (sc, n)
        if best: L = best[1]
    return L

def main():
    E.QMAX = 4000
    N = E.QMAX * E.QMAX * 4 + 10; sv = np.ones(N + 1, dtype=bool); sv[:2] = False
    for i in range(2, int(N ** 0.5) + 1):
        if sv[i]: sv[i * i::i] = False
    ps = list(primerange(11, E.QMAX + 1)); qs = [p for p in ps if p <= 200] + [p for i, p in enumerate(ps) if p > 200 and i % 3 == 0]
    machines = [E.Machine(q, sv) for q in qs]
    out = [__doc__.strip(), "", f"machines {len(machines)}; carrying mirror = base x remembered settled gears that fit (product x 60 <= q^2)"]
    for w in (3, 8, 13, 21, 34, 55, 89, 'all'):
        oks = []; fails = []
        for m in machines:
            L = walk(m, w)
            ok = L is not None and m.q < L <= m.q * m.q - 2 and bool(m.sv[L] and m.sv[L + 2]); oks.append(ok)
            if not ok: fails.append(m.q)
        streak = 0
        for m, ok in zip(machines, oks):
            if m.q < 31: continue
            if ok: streak += 1
            else: break
        line = f"   memory {str(w):>4}: streak from 31 {streak:>3}, total {sum(oks):>3} of {len(machines)}; first failures {fails[:6]}"
        out.append(line); print(line, flush=True)
    Path("research/stack/r8/results_settle_memory_carry.txt").write_text("\n".join(out), encoding="utf-8")

if __name__ == "__main__":
    main()
