"""The spiral with the slip (owner's idea, parked 2026-09-14, run 2026-09-15).

Primorial spiral: base P, gears g_1 > g_2 > ... (descending), step i moves the column by
d_i = +-2 P g_i (alternating, first up).  Slip of a step = how far the move overshoots whole
cycles of another gear: the signed residue of d_i modulo that gear, taken in (-g/2, g/2].
  slip A: against the previous step's gear g_{i-1}
  slip B: against every earlier gear of the spiral (sum of signed residues over j < i)
  slip C: against every gear of the machine outside the base (sum over all)
Summed slips S_A, S_B, S_C.  Candidate landings: E + S, E - S (when S = 0 mod 6, else the
nearest slot), E + 6S, E - 6S.  Measured over the machines: how often each candidate is a twin
in the window, against the plain landing E.

usage: uv run python research/stack/r8/spiral_slip.py 2000
"""
import sys
import numpy as np
from sympy import primerange
from pathlib import Path

def sres(x, g):
    r = x % g
    return r - g if r > g // 2 else r

def main():
    Q = int(sys.argv[1]); qs = list(primerange(11, Q + 1))
    N = 4 * Q * Q + 10; sv = np.ones(N + 1, dtype=bool); sv[:2] = False
    for i in range(2, int(N ** 0.5) + 1):
        if sv[i]: sv[i * i::i] = False
    names = ['E', 'E+SA', 'E-SA', 'E+6SA', 'E-6SA', 'E+SB', 'E-SB', 'E+6SB', 'E-6SB', 'E+SC', 'E-SC', 'E+6SC', 'E-6SC']
    cnt = {n: [0, 0] for n in names}   # twin, in window
    rows = []
    for q in qs:
        ps = list(primerange(2, q + 1)); base = []; P = 1
        for p in ps:
            if P * p <= q // 2: P *= p; base.append(p)
            else: break
        gs = [p for p in ps if p not in base][::-1]
        E = -1; SA = SB = SC = 0; sign = 1
        for i, g in enumerate(gs):
            d = sign * 2 * P * g; E += d
            if i > 0: SA += sres(d, gs[i - 1])
            SB += sum(sres(d, gs[j]) for j in range(i))
            SC += sum(sres(d, h) for h in gs if h != g)
            sign = -sign
        def slot(x):
            # nearest column start (5 mod 6) at or below x
            return x - ((x - 5) % 6)
        cands = {'E': E}
        for nm, S in (('SA', SA), ('SB', SB), ('SC', SC)):
            cands['E+' + nm] = slot(E + S); cands['E-' + nm] = slot(E - S)
            cands['E+6' + nm] = E + 6 * S; cands['E-6' + nm] = E - 6 * S
        for nm, x in cands.items():
            inwin = q < x and x + 2 <= q * q
            cnt[nm][1] += inwin
            if inwin and sv[x] and sv[x + 2]: cnt[nm][0] += 1
        if q in (31, 101, 499, 1999):
            rows.append(f"   q = {q}: E = {E}, S_A = {SA}, S_B = {SB}, S_C = {SC}; " + ", ".join(f"{nm} = {x}{'T' if q < x and x + 2 <= q * q and sv[x] and sv[x + 2] else ''}" for nm, x in cands.items()))
    out = [__doc__.strip(), "", f"machines {len(qs)} (11 to {Q}): candidate: twin / in window"]
    for nm in names: out.append(f"   {nm:<7} {cnt[nm][0]:>4} / {cnt[nm][1]:>4}")
    out += rows
    Path("research/stack/r8/results_spiral_slip.txt").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out[2:]))

if __name__ == "__main__":
    main()
