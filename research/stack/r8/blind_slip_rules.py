"""Loop, iteration 5: landing offsets D built from slips of spiral step pairs (no primality).
P = base product, gears above the base g1 > g2 > g3 > ... (g1 = q).  Landing -1 + 2D.
  S(j): D = P (g_j - g_{j+1}), the slip of the consecutive pair j (j = 1..6)
  W:    D = P (q - g) with g the largest gear such that the landing is above q (the first pair
        whose slip clears the window's start)
  A(m): D = P (g1 - g2 + g3 - ... +- g_m), the spiral over the top m gears (m = 2..6)
  T(t): D = t P (q - p), the top pair's slip with t periods (t = 2, 3)
  B:    D = P (q - p) + P (p - p') ... = P (q - p''): consecutive slips summed = the slip of a
        wider pair (S with a gap of two gears)
Machines 11 to 5000: in the window / twin.  usage: uv run python research/stack/r8/blind_slip_rules.py 5000
"""
import sys
import numpy as np
from sympy import primerange
from pathlib import Path

def main():
    Q = int(sys.argv[1]); qs = list(primerange(11, Q + 1))
    N = Q * Q + 10; sv = np.ones(N + 1, dtype=bool); sv[:2] = False
    for i in range(2, int(N ** 0.5) + 1):
        if sv[i]: sv[i * i::i] = False
    names = [f'S({j})' for j in range(1, 7)] + ['W'] + [f'A({m})' for m in range(2, 7)] + ['T(2)', 'T(3)', 'B gap2', 'B gap3']
    cnt = {n: [0, 0] for n in names}
    for q in qs:
        ps = list(primerange(2, q + 1)); base = []; P = 1
        for p in ps:
            if P * p <= q // 2: P *= p; base.append(p)
            else: break
        g = [p for p in ps if p not in base][::-1]
        Ds = {}
        for j in range(1, 7):
            if j < len(g): Ds[f'S({j})'] = P * (g[j - 1] - g[j])
        w = next((x for x in g[1:] if 2 * P * (q - x) - 1 > q), None)
        if w: Ds['W'] = P * (q - w)
        for m in range(2, 7):
            if m <= len(g): Ds[f'A({m})'] = P * sum(x if i % 2 == 0 else -x for i, x in enumerate(g[:m]))
        if len(g) > 1:
            Ds['T(2)'] = 2 * P * (g[0] - g[1]); Ds['T(3)'] = 3 * P * (g[0] - g[1])
        if len(g) > 2: Ds['B gap2'] = P * (g[0] - g[2])
        if len(g) > 3: Ds['B gap3'] = P * (g[0] - g[3])
        for n, D in Ds.items():
            E = -1 + 2 * D; inwin = q < E and E + 2 <= q * q
            cnt[n][1] += inwin
            if inwin and sv[E] and sv[E + 2]: cnt[n][0] += 1
    out = [__doc__.strip(), "", f"machines {len(qs)} (11 to {Q}): rule: twin / in window"]
    for n in names: out.append(f"   {n:<8} {cnt[n][0]:>4} / {cnt[n][1]:>4}")
    Path("research/stack/r8/results_blind_slip_rules.txt").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out[2:]))

if __name__ == "__main__":
    main()
