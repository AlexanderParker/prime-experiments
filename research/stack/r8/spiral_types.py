"""Approach B: build the spiral from gears of one field type.

The primorial base stays (the lowest gears with product at most q/2, in every mirror).  The
spiral then flips once per gear of the chosen type, descending, alternating up and down, with
mirror {base, g}; landing E' = -1 + 2 P A' where A' is the alternating sum of the chosen gears.
Types tried (gears above the base only):
  all        every gear above the base (the primorial spiral as it stands)
  coltwin    gears whose own column (6g - 1, 6g + 1) is a twin prime pair
  sqin       gears with g^2 inside the window (g > sqrt q)
  sqout      gears with g^2 below the window (g <= sqrt q)
  twinmem    gears that are twin prime members
  solo       gears that are not twin members
For each type and machine: E', inside the window or not, the fields painting E' (smallest gear of
each member, or twin), and the final step from E': how many h above sqrt q pass, the first one.

usage: uv run python research/stack/r8/spiral_types.py 2000
"""
import sys
import numpy as np
from sympy import primerange
from pathlib import Path

TYPES = ['all', 'coltwin', 'sqin', 'sqout', 'twinmem', 'solo']

def main():
    Q = int(sys.argv[1]); qs = list(primerange(11, Q + 1))
    N = Q * Q + 10; sv = np.ones(N + 1, dtype=bool); sv[:2] = False
    for i in range(2, int(N ** 0.5) + 1):
        if sv[i]: sv[i * i::i] = False
    def alt(xs): return sum(x if i % 2 == 0 else -x for i, x in enumerate(xs))
    def smallest(n):
        for g in range(5, int(n ** 0.5) + 1):
            if n % g == 0: return g
        return None
    out = [__doc__.strip(), ""]
    tot = {t: dict(inwin=0, twinE=0, haspass=0, n=0) for t in TYPES}
    sample = {}
    for q in qs:
        ps = list(primerange(2, q + 1)); base = []; P = 1
        for p in ps:
            if P * p <= q // 2: P *= p; base.append(p)
            else: break
        rest = [p for p in ps if p not in base]; r = int(q ** 0.5)
        sel = {
            'all': rest,
            'coltwin': [g for g in rest if sv[6 * g - 1] and sv[6 * g + 1]],
            'sqin': [g for g in rest if g > r],
            'sqout': [g for g in rest if g <= r],
            'twinmem': [g for g in rest if sv[g - 2] or sv[g + 2]],
            'solo': [g for g in rest if not (sv[g - 2] or sv[g + 2])],
        }
        high = [g for g in ps if g > r]
        row = []
        for t in TYPES:
            gs = sel[t][::-1]
            if not gs: row.append(f"{t}: no gears"); continue
            E = -1 + 2 * P * alt(gs)
            inwin = q < E and E + 2 <= q * q
            paint = "twin" if (0 <= E < N - 2 and sv[E] and sv[E + 2]) else (f"{smallest(E) if E > 0 else '-'}|{smallest(E + 2) if E > 0 else '-'}")
            passes = []
            for d in (1, -1):
                for h in high:
                    L = E + 6 * d * h
                    if q < L and L + 2 <= q * q and sv[L] and sv[L + 2]: passes.append((h, d))
            tot[t]['n'] += 1; tot[t]['inwin'] += inwin; tot[t]['twinE'] += (paint == "twin"); tot[t]['haspass'] += bool(passes)
            row.append(f"{t}: {len(gs)} gears, E' = {E}{' in window' if inwin else ' OUT'}, E' painted by {paint}, pass {len(passes)}" + (f" first {passes[0]}" if passes else ""))
        if q in (37, 101, 499, 997, 1999): sample[q] = row
    for q, row in sample.items():
        out.append(f"q = {q}:"); out += ["   " + x for x in row]; out.append("")
    out.append(f"over the {len(qs)} machines 11 to {Q}: per type, machines with E' inside the window / E' itself a twin / a passing final step")
    for t in TYPES:
        out.append(f"   {t:<8} {tot[t]['inwin']:>4} / {tot[t]['twinE']:>4} / {tot[t]['haspass']:>4}   (machines with gears of the type: {tot[t]['n']})")
    Path("research/stack/r8/results_spiral_types.txt").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out[2:]))

if __name__ == "__main__":
    main()
