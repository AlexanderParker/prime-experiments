"""Spirals over gears by the side of their own column, and by residue-free field facts.

A gear g above 3 sits in column (g-1)/6 or (g+1)/6 as its LEFT member (g = 5 mod 6) or RIGHT
member (g = 1 mod 6).  Variants (base kept, descending, first flip up, single mirrors):
  left     gears that are left members of their column (g = 5 mod 6)
  right    gears that are right members (g = 1 mod 6)
  leftsq   left gears at most sqrt q;  rightsq  right gears at most sqrt q
  lefthi   left gears above sqrt q;    righthi  right gears above sqrt q
  twinL    gears that are the LEFT member of a twin prime pair (g + 2 prime)
  twinR    gears that are the RIGHT member of a twin prime pair (g - 2 prime)
Measured: landing in the window, landing itself a twin, passing final step, no-pass machines.

usage: uv run python research/stack/r8/spiral_sides.py 2000
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
    def alt(xs): return sum(x if i % 2 == 0 else -x for i, x in enumerate(xs))
    names = ['left', 'right', 'leftsq', 'rightsq', 'lefthi', 'righthi', 'twinL', 'twinR']
    res = {n: dict(n=0, inwin=0, twinE=0, haspass=0, nopass=[]) for n in names}
    sample = {}
    for q in qs:
        ps = list(primerange(2, q + 1)); base = []; P = 1
        for p in ps:
            if P * p <= q // 2: P *= p; base.append(p)
            else: break
        rest = [p for p in ps if p not in base and p >= 5]; r = int(q ** 0.5); high = [g for g in ps if g > r]
        sets = {'left': [g for g in rest if g % 6 == 5], 'right': [g for g in rest if g % 6 == 1]}
        sets['leftsq'] = [g for g in sets['left'] if g <= r]; sets['rightsq'] = [g for g in sets['right'] if g <= r]
        sets['lefthi'] = [g for g in sets['left'] if g > r]; sets['righthi'] = [g for g in sets['right'] if g > r]
        sets['twinL'] = [g for g in rest if sv[g + 2]]; sets['twinR'] = [g for g in rest if sv[g - 2]]
        for n in names:
            gs = sorted(sets[n], reverse=True)
            if not gs: continue
            E = -1 + 2 * P * alt(gs)
            inwin = q < E and E + 2 <= q * q; twinE = 0 <= E < N - 2 and bool(sv[E] and sv[E + 2])
            pz = [(h, d) for d in (1, -1) for h in high if q < E + 6 * d * h and E + 6 * d * h + 2 <= q * q and sv[E + 6 * d * h] and sv[E + 6 * d * h + 2]]
            R = res[n]; R['n'] += 1; R['inwin'] += inwin; R['twinE'] += twinE; R['haspass'] += bool(pz)
            if not pz: R['nopass'].append(q)
            if q in (499, 1999): sample.setdefault(q, []).append(f"{n:<8} {len(gs):>3} gears  E = {E:>8} {'in' if inwin else 'OUT':<4} {'TWIN' if twinE else '    '} pass {len(pz):>3}" + (f" first {pz[0]}" if pz else ""))
    out = [__doc__.strip(), "", f"{'variant':<9}{'machines':>9}{'in window':>10}{'E twin':>8}{'has pass':>9}  no pass at"]
    for n in names:
        R = res[n]
        if R['n']: out.append(f"{n:<9}{R['n']:>9}{R['inwin']:>10}{R['twinE']:>8}{R['haspass']:>9}  {R['nopass'][:12]}{'...' if len(R['nopass']) > 12 else ''}")
    for q, rows in sample.items():
        out.append(""); out.append(f"q = {q}:"); out += ["   " + x for x in rows]
    Path("research/stack/r8/results_spiral_sides.txt").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out[2:]))

if __name__ == "__main__":
    main()
