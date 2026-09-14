"""Loop, iteration 4: every walk from home lands at -1 + 2D (D = the signed sum of the mirror
moves' halves), open by construction exactly to the gears dividing D.  Two flips about axes a1,
a2 give D = a2 - a1, which is the owner's slip: the difference of two mirror sizes (times
periods); gears dividing the difference keep the origin's open phase without sitting in either
mirror.  So a blind walk is a blind choice of D at most q^2/2 with many gear factors.  Rules:
  D1: the largest primorial at most q^2/2 (the descent at t = 1)
  D2: the largest lcm(1..m) at most q^2/2
  D3: the largest m! at most q^2/2 (over 2, since -1 + 2D must be 5 mod 6)
  D4: P_s * (q - p) for the machine's base P_s and the two top gears (difference of the mirrors
      {base, q} and {base, p}, p the prime below q): the slip of the last two spiral steps
  D5: P * (g1 - g2 + g3 - ...) alternating over the gears above the base (the primorial spiral)
  D6: the largest 2^a 3^b at most q^2/2 with -1 + 2D = 5 mod 6 (D a multiple of 3)
Measured: machines 11 to 5000: landing in the window and a twin.
usage: uv run python research/stack/r8/blind_D_rules.py 5000
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
    names = ['D1 primorial', 'D2 lcm', 'D3 factorial', 'D4 slip of top two', 'D5 spiral', 'D6 2^a 3^b']
    cnt = {n: [0, 0] for n in names}; ex = []
    for q in qs:
        ps = list(primerange(2, q + 1)); lim = q * q // 2
        P = 1
        for p in ps:
            if P * p <= lim: P *= p
            else: break
        L = 1; m = 1
        while math.lcm(L, m + 1) <= lim: m += 1; L = math.lcm(L, m)
        F = 2; k = 2
        while F * (k + 1) <= lim: k += 1; F *= k
        base = []; Pb = 1
        for p in ps:
            if Pb * p <= q // 2: Pb *= p; base.append(p)
            else: break
        rest = [p for p in ps if p not in base]
        D4 = Pb * (rest[-1] - rest[-2]) if len(rest) >= 2 else Pb * rest[-1]
        D5 = Pb * sum(g if i % 2 == 0 else -g for i, g in enumerate(rest[::-1]))
        best = 0
        a = 0
        while 2 ** a <= lim:
            b = 1
            while 2 ** a * 3 ** b <= lim:
                best = max(best, 2 ** a * 3 ** b); b += 1
            a += 1
        Ds = {'D1 primorial': P, 'D2 lcm': L, 'D3 factorial': F, 'D4 slip of top two': D4, 'D5 spiral': D5, 'D6 2^a 3^b': best}
        row = []
        for n, D in Ds.items():
            E = -1 + 2 * D; inwin = q < E and E + 2 <= q * q
            cnt[n][1] += inwin; tw = inwin and sv[E] and sv[E + 2]; cnt[n][0] += tw
            row.append(f"{n}: {E}{' T' if tw else ''}")
        if q in (31, 101, 499, 1999, 4999): ex.append(f"   q = {q}: " + "; ".join(row))
    out = [__doc__.strip(), "", f"machines {len(qs)} (11 to {Q}): twin / in window"]
    for n in names: out.append(f"   {n:<20} {cnt[n][0]:>4} / {cnt[n][1]:>4}")
    out += ex
    Path("research/stack/r8/results_blind_D_rules.txt").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out[2:]))

if __name__ == "__main__":
    main()
