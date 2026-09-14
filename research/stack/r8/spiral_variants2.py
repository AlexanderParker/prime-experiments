"""More spiral constructions by gear type (owner: keep trying).  Base kept (lowest gears with
product at most q/2, in every mirror).  Variants over a chosen gear set S above the base:
  order      desc (q down) or asc (up from the smallest)
  start      first flip up (+) or down (-)
  pairs      twin-member gears flipped as one mirror {base, p, p+2} (mirror size P p (p+2)), solo
             gears as {base, g}
  k2         every flip with two periods (landing origin +- 4 M instead of 2 M)
Landing E' = -1 + 2 P A' (or with the pair / k2 terms).  Measured per variant over the machines:
E' inside the window, E' itself a twin, a passing final step (some h above sqrt q with E' +- 6h a
twin in the window), and the machines with no pass.

usage: uv run python research/stack/r8/spiral_variants2.py 2000
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
    def landing(P, terms, start):
        # terms: list of mirror cofactors m (mirror = base * m), flips alternate from `start`
        s = 0; sgn = start
        for m in terms: s += sgn * m; sgn = -sgn
        return -1 + 2 * P * s
    variants = []
    for sel in ('sqout', 'sqin', 'all', 'coltwin'):
        for order in ('desc', 'asc'):
            for start in (1, -1):
                for mode in ('single', 'pairs', 'k2'):
                    variants.append((sel, order, start, mode))
    res = {v: dict(n=0, inwin=0, twinE=0, haspass=0, nopass=[]) for v in variants}
    sample = {}
    for q in qs:
        ps = list(primerange(2, q + 1)); base = []; P = 1
        for p in ps:
            if P * p <= q // 2: P *= p; base.append(p)
            else: break
        rest = [p for p in ps if p not in base]; r = int(q ** 0.5); high = [g for g in ps if g > r]
        sets = {'sqout': [g for g in rest if g <= r], 'sqin': [g for g in rest if g > r], 'all': rest,
                'coltwin': [g for g in rest if sv[6 * g - 1] and sv[6 * g + 1]]}
        for v in variants:
            sel, order, start, mode = v
            gs = sets[sel]
            if not gs: continue
            gs = sorted(gs, reverse=(order == 'desc'))
            if mode == 'single': terms = gs
            elif mode == 'k2': terms = [2 * g for g in gs]
            else:
                terms = []; used = set()
                for g in gs:
                    if g in used: continue
                    if g + 2 in gs and g + 2 not in used: terms.append(g * (g + 2)); used |= {g, g + 2}
                    elif g - 2 in gs and g - 2 not in used: terms.append(g * (g - 2)); used |= {g, g - 2}
                    else: terms.append(g); used.add(g)
            E = landing(P, terms, start)
            inwin = q < E and E + 2 <= q * q
            twinE = 0 <= E < N - 2 and bool(sv[E] and sv[E + 2])
            passes = [(h, d) for d in (1, -1) for h in high if q < E + 6 * d * h and E + 6 * d * h + 2 <= q * q and sv[E + 6 * d * h] and sv[E + 6 * d * h + 2]]
            R = res[v]; R['n'] += 1; R['inwin'] += inwin; R['twinE'] += twinE; R['haspass'] += bool(passes)
            if not passes: R['nopass'].append(q)
            if q in (499, 1999): sample.setdefault(q, []).append(f"{sel:<8}{order:<5}{'+' if start > 0 else '-'} {mode:<7} E' = {E:>8} {'in' if inwin else 'OUT':<4} {'TWIN' if twinE else '    '} pass {len(passes):>3}" + (f" first {passes[0]}" if passes else ""))
    out = [__doc__.strip(), "", f"{'set':<8}{'order':<6}{'st':<3}{'mode':<8}{'machines':>9}{'in window':>10}{'E twin':>8}{'has pass':>9}  no pass at"]
    for v in variants:
        R = res[v]
        if R['n'] == 0: continue
        out.append(f"{v[0]:<8}{v[1]:<6}{'+' if v[2] > 0 else '-':<3}{v[3]:<8}{R['n']:>9}{R['inwin']:>10}{R['twinE']:>8}{R['haspass']:>9}  {R['nopass'][:12]}{'...' if len(R['nopass']) > 12 else ''}")
    for q, rows in sample.items():
        out.append(""); out.append(f"q = {q}:"); out += ["   " + x for x in rows]
    Path("research/stack/r8/results_spiral_variants2.txt").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out[2:2 + 1 + len([v for v in variants if res[v]['n']])]))

if __name__ == "__main__":
    main()
