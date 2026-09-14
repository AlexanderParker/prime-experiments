"""Loop, iteration 7: landing offsets from the gaps between the top gears (no primality).
P = base product; gaps G1 = q - p, G2 = p - p', G3 = p' - p'' (top four gears).  Landing -1 + 2D.
  R1: D = P G1 G2 / 4          R2: D = P (G1 + G2 + G3)       R3: D = P G1 G2 G3 / 8
  R4: D = P max(G1, G2, G3)    R5: D = P lcm(G1/2, G2/2, G3/2) R6: D = P G1 (number of gears above the base)
  R7: D = P (G1 G2 / 4 + G3 / 2)                                R8: D = P q G1 / 2
Machines 11 to 5000: twin / in window.  usage: uv run python research/stack/r8/blind_gap_rules.py 5000
"""
import sys, math
import numpy as np
from sympy import primerange
from pathlib import Path

def main():
    Q = int(sys.argv[1]); qs = list(primerange(11, Q + 1))
    N = Q * Q + 10; sv = np.ones(N + 1, dtype=bool); sv[:2] = False
    for i in range(2, int(N ** 0.5) + 1):
        if sv[i]: sv[i * i::i] = False
    names = ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8']; cnt = {n: [0, 0] for n in names}
    for q in qs:
        ps = list(primerange(2, q + 1)); base = []; P = 1
        for p in ps:
            if P * p <= q // 2: P *= p; base.append(p)
            else: break
        g = [p for p in ps if p not in base][::-1]
        if len(g) < 4: continue
        G1, G2, G3 = g[0] - g[1], g[1] - g[2], g[2] - g[3]
        Ds = {'R1': P * G1 * G2 // 4, 'R2': P * (G1 + G2 + G3), 'R3': P * G1 * G2 * G3 // 8, 'R4': P * max(G1, G2, G3),
              'R5': P * math.lcm(G1 // 2, G2 // 2, G3 // 2), 'R6': P * G1 * len(g), 'R7': P * (G1 * G2 // 4 + G3 // 2), 'R8': P * q * G1 // 2}
        for n, D in Ds.items():
            E = -1 + 2 * D; inwin = q < E and E + 2 <= q * q
            cnt[n][1] += inwin
            if inwin and sv[E] and sv[E + 2]: cnt[n][0] += 1
    out = [__doc__.strip(), "", f"machines {len(qs)} (11 to {Q}): rule: twin / in window"]
    for n in names: out.append(f"   {n} {cnt[n][0]:>4} / {cnt[n][1]:>4}")
    Path("research/stack/r8/results_blind_gap_rules.txt").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out[2:]))

if __name__ == "__main__":
    main()
