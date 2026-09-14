"""Loop, iteration 8: offsets from gears with a property below the window, and from q's residues.
Landing -1 + 2D.  P = base product, P_s = the first primorial above q/2, g_min its last gear.
  C1: D = P (c1 - c2), c1 > c2 the two largest gears whose own column (6g-1, 6g+1) is a twin pair
  C2: D = P (t1 - t2), the two largest gears that are twin members
  C3: D = P (q - c1)
  Q1: D = P_s * (q mod g_min)            Q2: D = P_s * ((q - 1) / 2 mod g_min)
  Q3: D = P_s * (count of gears above the base)   Q4: D = P_s * (q mod 6 == 1 ? 2 : 1)
Machines 11 to 5000: twin / in window.  usage: uv run python research/stack/r8/blind_rules8.py 5000
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
    names = ['C1', 'C2', 'C3', 'Q1', 'Q2', 'Q3', 'Q4']; cnt = {n: [0, 0] for n in names}
    for q in qs:
        ps = list(primerange(2, q + 1)); base = []; P = 1
        for p in ps:
            if P * p <= q // 2: P *= p; base.append(p)
            else: break
        rest = [p for p in ps if p not in base]
        Ps = P * rest[0]; gmin = rest[0]
        col = [g for g in rest if sv[6 * g - 1] and sv[6 * g + 1]]
        tm = [g for g in rest if sv[g - 2] or sv[g + 2]]
        Ds = {}
        if len(col) >= 2: Ds['C1'] = P * (col[-1] - col[-2]); Ds['C3'] = P * (q - col[-1]) if q != col[-1] else P * (q - col[-2])
        if len(tm) >= 2: Ds['C2'] = P * (tm[-1] - tm[-2])
        Ds['Q1'] = Ps * (q % gmin or 1); Ds['Q2'] = Ps * (((q - 1) // 2) % gmin or 1); Ds['Q3'] = Ps * len(rest); Ds['Q4'] = Ps * (2 if q % 6 == 1 else 1)
        for n, D in Ds.items():
            E = -1 + 2 * D; inwin = q < E and E + 2 <= q * q
            cnt[n][1] += inwin
            if inwin and sv[E] and sv[E + 2]: cnt[n][0] += 1
    out = [__doc__.strip(), "", f"machines {len(qs)} (11 to {Q}): rule: twin / in window"]
    for n in names: out.append(f"   {n} {cnt[n][0]:>4} / {cnt[n][1]:>4}")
    Path("research/stack/r8/results_blind_rules8.txt").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out[2:]))

if __name__ == "__main__":
    main()
