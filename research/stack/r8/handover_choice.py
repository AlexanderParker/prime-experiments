"""Loop, entry 58: can the prefix choose the handover column's small residues?  The prefix's
last step has 2K candidates; each is a column with its own residues mod 5, 7, 11.  If the
prefix can reach columns covering every combination of those residues, the walk chooses where
the small gears' teeth sit on the next line, and the tail's covering question is entered at a
place of the walk's choosing.
Measured per machine: at the prefix's last step, the residues (mod 5, mod 7, mod 11) of the
candidates that are open to every visited gear - how many of the 4 x 5 x 9 = 180 admissible
combinations (the residues off each gear's own two teeth) are reached.
usage: uv run python research/stack/r8/handover_choice.py lo hi
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
    shares = []; n = 0; reached_all = 0
    for q in list(primerange(lo, hi + 1)):
        K = max(20, int(math.log(q) ** 2)); m = E.Machine(q, sv); P = 6
        seq = [g for g in m.ps if g >= 5][::-1]
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
                    sc = (-on, md, -k)
                    if best is None or sc > best[0]: best = (sc, x)
            if i == cut - 1:
                # the prefix's last step: collect the residues of every keeping candidate
                combos = set()
                g_last = g
                for k in range(1, K + 1):
                    for d in (1, -1):
                        x = L + 2 * k * P * g_last * d
                        if x < 0 or x > q * q - 2: continue
                        if all(x % h not in (0, h - 2) for h in visited):
                            combos.add((x % 5, x % 7, x % 11))
                adm = [(a, b, c) for a in range(5) for b in range(7) for c in range(11)
                       if a not in (0, 3) and b not in (0, 5) and c not in (0, 9)]
                good = {c for c in combos if c[0] not in (0, 3) and c[1] not in (0, 5) and c[2] not in (0, 9)}
                shares.append(len(good) / len(adm)); reached_all += (len(good) == len(adm))
            L = best[1]
        n += 1
    line = (f"machines {lo}..{hi} ({n}), K = (ln q)^2: at the prefix's last step the keeping candidates cover "
            f"{sum(shares)/len(shares)*100:.1f}% of the 180 admissible residue triples (mod 5, 7, 11) on average, "
            f"min {min(shares)*100:.1f}%, max {max(shares)*100:.1f}%; all 180 reached at {reached_all} machines")
    Path(f"research/stack/r8/results_handover_choice_{lo}_{hi}.txt").write_text(line, encoding="utf-8")
    print(line)

if __name__ == "__main__":
    main()
