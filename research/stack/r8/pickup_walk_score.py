"""Loop, iteration 13: the pick-up walk with a deterministic scoring rule (no search, no
primality; the walk's own phases only).  Descending over the gears above the base; at gear g,
mirror {base, g}; among (periods 1..K, direction) choose the move that keeps the invariant
(every visited gear off its teeth) and maximises the smallest distance of any visited gear's
phase from its nearer tooth (0 or gear - 2); ties by the smaller move.  If no move keeps the
invariant, take the move with the fewest gears on a tooth (the walk carries a defect) and go on.
Reported: machines 11 to 2000: twin in the window, defects at the landing, K = 3, 5, 9.
usage: uv run python research/stack/r8/pickup_walk_score.py 2000
"""
import sys
import numpy as np
from sympy import primerange
from pathlib import Path

def dist(ph, h):
    return min(ph, abs(ph - (h - 2)), h - ph if ph > h - 2 else h)

def main():
    Q = int(sys.argv[1]); qs = list(primerange(11, Q + 1))
    N = Q * Q + 10; sv = np.ones(N + 1, dtype=bool); sv[:2] = False
    for i in range(2, int(N ** 0.5) + 1):
        if sv[i]: sv[i * i::i] = False
    out = [__doc__.strip(), ""]
    for K in (3, 5, 9, 15, 25, 60):
        twin = inwin = clean = 0; ex = []
        for q in qs:
            ps = list(primerange(2, q + 1)); base = []; P = 1
            for p in ps:
                if P * p <= q // 2: P *= p; base.append(p)
                else: break
            gears = [p for p in ps if p not in base][::-1]
            n = -1; visited = [x for x in base if x >= 5]
            for g in gears:
                visited.append(g); best = None
                for k in range(1, K + 1):
                    for d in (1, -1):
                        m = n + 2 * k * P * g * d
                        if m < 0 or m > q * q - 2: continue
                        onteeth = sum(1 for h in visited if m % h in (0, h - 2))
                        score = (-onteeth, min(dist(m % h, h) for h in visited), -k)
                        if best is None or score > best[0]: best = (score, m)
                if best is None: break
                n = best[1]
            ok = q < n <= q * q - 2; inwin += ok
            defects = [h for h in visited if n % h in (0, h - 2)]
            if ok and not defects: clean += 1
            if ok and sv[n] and sv[n + 2]: twin += 1
            if q in (31, 101, 499, 1999): ex.append((q, n, defects[:4]))
        out.append(f"K = {K}: in the window {inwin} of {len(qs)}; landing with no visited gear on a tooth {clean}; twin {twin}; samples {ex}")
    Path("research/stack/r8/results_pickup_walk_score.txt").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out[2:]))

if __name__ == "__main__":
    main()
