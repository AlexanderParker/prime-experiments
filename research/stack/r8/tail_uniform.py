"""Loop, entry 52: is the tail's requirement uniform?  StepOpen was stated at the last step; the
tail has many steps.  For each tail step (the gears below the first cut, down to 7) measure the
number of candidates open to every gear VISITED SO FAR (the step's own requirement) and the
number open to EVERY gear of the machine (the stronger, uniform requirement).  If the uniform
count is positive at every tail step, StepOpen may be stated once, at any tail step, and the
walk may stop at the first one that meets it.
usage: uv run python research/stack/r8/tail_uniform.py lo hi
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
    n = 0; allpos = 0; firsts = []; mins = []
    for q in list(primerange(lo, hi + 1)):
        K = max(20, int(math.log(q) ** 2)); m = E.Machine(q, sv); P = 6
        seq = [g for g in m.ps if g >= 5][::-1]; allg = [g for g in m.ps if g >= 5]
        free = 0
        while free < len(seq) and seq[free] > 2 * (free + 1): free += 1
        visited = []; L = -1; uni = []; first = None
        for i, g in enumerate(seq[:-1]):
            visited.append(g); nxt = seq[i + 1]
            best = None; u = 0
            for k in range(1, K + 1):
                for d in (1, -1):
                    x = L + 2 * k * P * g * d
                    if x < 0 or x > q * q - 2: continue
                    on = sum(1 for h in visited if x % h in (0, h - 2))
                    md = min(dist(x % h, h) for h in visited)
                    if i >= free and q < x and all(x % h not in (0, h - 2) for h in allg):
                        u += 1
                        if first is None: first = i - free
                    fl = 0
                    if on == 0:
                        fl = sum(1 for k2 in range(1, min(K, 20) + 1) for d2 in (1, -1) if 0 <= x + 2 * k2 * P * nxt * d2 <= q * q - 2 and all((x + 2 * k2 * P * nxt * d2) % h not in (0, h - 2) for h in visited + [nxt]))
                    sc = (-on, fl, md)
                    if best is None or sc > best[0]: best = (sc, x)
            if i >= free: uni.append(u)
            L = best[1]
        n += 1
        if uni and min(uni) > 0: allpos += 1
        if uni: mins.append(min(uni))
        firsts.append(first if first is not None else -1)
    line = (f"machines {lo}..{hi} ({n}), K = (ln q)^2: tail steps where some candidate is open to EVERY gear: "
            f"all of them at {allpos} machines; the minimum over the tail steps has min {min(mins)} and mean {sum(mins)/len(mins):.2f}; "
            f"the first tail step with such a candidate is step {min(f for f in firsts if f >= 0)} to {max(firsts)} of the tail (0 = the first)")
    Path(f"research/stack/r8/results_tail_uniform_{lo}_{hi}.txt").write_text(line, encoding="utf-8")
    print(line)

if __name__ == "__main__":
    main()
