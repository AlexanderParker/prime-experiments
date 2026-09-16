"""Loop, entry 46: choose the last stride among several gears.  At the walk's last step the
column L is open to every gear except those the step must re-settle.  The stride is 2 P g
(P the carried base, g the step's gear); different g give different progressions.  Measured
per machine: for each candidate last gear g among the gears above sqrt q (the first eight),
whether StepOpen holds within K = 40 periods, how many of the eighty candidates are open to
every gear, and the smallest k that works.  Reported: the machines where SOME g works (the
walk may choose the stride), against those where a FIXED g works.
usage: uv run python research/stack/r8/last_stride_choice.py lo hi
"""
import sys
import numpy as np
from sympy import primerange
from pathlib import Path
sys.path.insert(0, "research/stack/r8")
import evolve_walk3 as E

def dist(ph, h): return min(ph, abs(ph - (h - 2)), h - ph if ph > h - 2 else h)

def main():
    lo, hi = int(sys.argv[1]), int(sys.argv[2]); K = 40
    E.QMAX = hi
    N = hi * hi * 4 + 10; sv = np.ones(N + 1, dtype=bool); sv[:2] = False
    for i in range(2, int(N ** 0.5) + 1):
        if sv[i]: sv[i * i::i] = False
    out = [__doc__.strip(), ""]
    anyg = 0; n = 0; fixed = {}; mins = []
    for q in list(primerange(lo, hi + 1)):
        m = E.Machine(q, sv); P = 6; seq = [g for g in m.ps if g >= 5][::-1]
        # run the walk down to (but not including) the last two gears, so the column is open to
        # every gear above 7; then try the last stride over several gears
        visited = []; L = -1
        for i, g in enumerate(seq[:-2]):
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
                        fl = sum(1 for k2 in range(1, 21) for d2 in (1, -1) if 0 <= x + 2 * k2 * P * nxt * d2 <= q * q - 2 and all((x + 2 * k2 * P * nxt * d2) % h not in (0, h - 2) for h in visited + [nxt]))
                    sc = (-on, fl, md)
                    if best is None or sc > best[0]: best = (sc, x)
            L = best[1]
        allg = [h for h in m.ps if h >= 5]
        n += 1; works = []
        for gi, g in enumerate([x for x in m.ps if x * x > q][:8]):
            cnt = 0; first = None
            for k in range(1, K + 1):
                for d in (1, -1):
                    x = L + 2 * k * P * g * d
                    if not (q < x <= q * q - 2): continue
                    if all(x % h not in (0, h - 2) for h in allg):
                        cnt += 1
                        if first is None: first = k
            if cnt: works.append((g, cnt, first)); fixed[gi] = fixed.get(gi, 0) + 1
        if works: anyg += 1; mins.append(min(w[2] for w in works))
        if q in (503, 997, 1499, 1999): out.append(f"   q = {q}: last-stride gears that work (gear, open candidates of 80, first k): {works[:6]}")
    out.insert(2, f"machines {lo}..{hi} ({n}): some choice of the last stride gives StepOpen at {anyg}; a fixed stride (the i-th gear above sqrt q) works at {[fixed.get(i, 0) for i in range(8)]}; smallest working k over the choices: max {max(mins) if mins else '-'}")
    Path(f"research/stack/r8/results_last_stride_choice_{lo}_{hi}.txt").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out[2:8]))

if __name__ == "__main__":
    main()
