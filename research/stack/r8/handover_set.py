"""Loop, entry 59: choosing among the handover columns the prefix offers.  At the prefix's last
step the keeping candidates form a set H (the columns the walk could hand over).  For each
column of H, run the first tail step (stride 12 g', g' the first gear below the cut) and count
the candidates open to every gear of the machine.  Reported per machine: |H|, how many columns
of H lead to at least one open candidate, and the best count.
usage: uv run python research/stack/r8/handover_set.py lo hi
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
    n = 0; allgood = 0; shares = []; bests = []; worstmach = []
    for q in list(primerange(lo, hi + 1)):
        K = max(20, int(math.log(q) ** 2)); m = E.Machine(q, sv); P = 6
        seq = [g for g in m.ps if g >= 5][::-1]; allg = [g for g in m.ps if g >= 5]
        cut = 0
        while cut < len(seq) and seq[cut] > 2 * (cut + 1): cut += 1
        visited = []; L = -1; H = []
        for i, g in enumerate(seq[:cut]):
            visited.append(g)
            best = None
            for k in range(1, K + 1):
                for d in (1, -1):
                    x = L + 2 * k * P * g * d
                    if x < 0 or x > q * q - 2: continue
                    on = sum(1 for h in visited if x % h in (0, h - 2))
                    md = min(dist(x % h, h) for h in visited)
                    sc = (-on, md, -k)
                    if best is None or sc > best[0]: best = (sc, x)
                    if i == cut - 1 and on == 0 and q < x: H.append(x)
            L = best[1]
        gp = seq[cut] if cut < len(seq) else 7
        good = 0; best_cnt = 0
        for c in H:
            cnt = 0
            for k in range(1, K + 1):
                for d in (1, -1):
                    x = c + 2 * k * P * gp * d
                    if q < x <= q * q - 2 and all(x % h not in (0, h - 2) for h in allg): cnt += 1
            if cnt: good += 1
            best_cnt = max(best_cnt, cnt)
        n += 1
        if H: shares.append(good / len(H))
        bests.append(best_cnt); allgood += (good == len(H) and len(H) > 0)
        if best_cnt == 0: worstmach.append(q)
    line = (f"machines {lo}..{hi} ({n}), K = (ln q)^2: handover set size {len(H)} at the last machine; "
            f"share of handover columns leading to an open candidate at the first tail step: mean {sum(shares)/len(shares)*100:.1f}%, min {min(shares)*100:.1f}%; "
            f"best open count over the handover set: min {min(bests)}, mean {sum(bests)/n:.1f}; machines where NO handover column works {len(worstmach)} {worstmach[:6]}; "
            f"every handover column works at {allgood} machines")
    Path(f"research/stack/r8/results_handover_set_{lo}_{hi}.txt").write_text(line, encoding="utf-8")
    print(line)

if __name__ == "__main__":
    main()
