"""Stacked and recursive spiral constructions by gear type (owner: keep trying).

All spirals: descending gears, first flip up, one period, mirror {base, g}.
  sqout>sqin   sqout spiral (gears at most sqrt q above the base) from home, then from its landing
               a sqin spiral (gears above sqrt q) with the same base, first flip up or down
  sqin>sqout   the reverse order
  sub          the SUB-MACHINE's primorial spiral: base_s = lowest gears with product at most
               sqrt(q)/2, spiral over the gears from there up to sqrt q; lands in the sub-window
               (sqrt q, q] when it lands at all
  sub>sqin     from the sub-machine's landing, the sqin spiral with the sub-machine's base
  sub>final    the sub-machine's landing, then the final step alone
For each: landing, in the window (q, q^2] or the sub-window, landing itself a twin, passing
final step (h above sqrt q, E +- 6h a twin in the window), machines with no pass.

usage: uv run python research/stack/r8/spiral_stacks2.py 2000
"""
import sys
import numpy as np
from sympy import primerange
from pathlib import Path

def base_of(ps, bound):
    base = []; P = 1
    for p in ps:
        if P * p <= bound: P *= p; base.append(p)
        else: break
    return base, P

def alt(xs, start=1):
    s = 0; sg = start
    for x in xs: s += sg * x; sg = -sg
    return s

def main():
    Q = int(sys.argv[1]); qs = list(primerange(11, Q + 1))
    N = Q * Q + 10; sv = np.ones(N + 1, dtype=bool); sv[:2] = False
    for i in range(2, int(N ** 0.5) + 1):
        if sv[i]: sv[i * i::i] = False
    names = ['sqout>sqin +', 'sqout>sqin -', 'sqin>sqout +', 'sqin>sqout -', 'sub', 'sub>sqin +', 'sub>sqin -']
    res = {n: dict(n=0, inwin=0, twinE=0, haspass=0, nopass=[]) for n in names}
    sample = {}
    for q in qs:
        ps = list(primerange(2, q + 1)); r = int(q ** 0.5); high = [g for g in ps if g > r]
        base, P = base_of(ps, q // 2)
        rest = [p for p in ps if p not in base]
        sqout = sorted([g for g in rest if g <= r], reverse=True); sqin = sorted([g for g in rest if g > r], reverse=True)
        subps = [p for p in ps if p <= r]; base_s, P_s = base_of(subps, r // 2)
        sub_rest = sorted([p for p in subps if p not in base_s], reverse=True)
        def passes(E):
            return [(h, d) for d in (1, -1) for h in high if q < E + 6 * d * h and E + 6 * d * h + 2 <= q * q and sv[E + 6 * d * h] and sv[E + 6 * d * h + 2]]
        lands = {}
        if sqout and sqin:
            E1 = -1 + 2 * P * alt(sqout)
            lands['sqout>sqin +'] = E1 + 2 * P * alt(sqin, 1); lands['sqout>sqin -'] = E1 + 2 * P * alt(sqin, -1)
            E2 = -1 + 2 * P * alt(sqin)
            lands['sqin>sqout +'] = E2 + 2 * P * alt(sqout, 1); lands['sqin>sqout -'] = E2 + 2 * P * alt(sqout, -1)
        if sub_rest:
            Es = -1 + 2 * P_s * alt(sub_rest)
            lands['sub'] = Es
            lands['sub>sqin +'] = Es + 2 * P_s * alt(sqin, 1); lands['sub>sqin -'] = Es + 2 * P_s * alt(sqin, -1)
        for n, E in lands.items():
            inwin = (q < E and E + 2 <= q * q) if n != 'sub' else (r < E and E + 2 <= q)
            twinE = 0 <= E < N - 2 and bool(sv[E] and sv[E + 2])
            pz = passes(E)
            R = res[n]; R['n'] += 1; R['inwin'] += inwin; R['twinE'] += twinE; R['haspass'] += bool(pz)
            if not pz: R['nopass'].append(q)
            if q in (101, 499, 1999): sample.setdefault(q, []).append(f"{n:<13} E = {E:>8} {'in' if inwin else 'OUT':<4} {'TWIN' if twinE else '    '} pass {len(pz):>3}" + (f" first {pz[0]}" if pz else ""))
    out = [__doc__.strip(), "", f"{'variant':<14}{'machines':>9}{'in window':>10}{'E twin':>8}{'has pass':>9}  no pass at"]
    for n in names:
        R = res[n]
        if R['n']: out.append(f"{n:<14}{R['n']:>9}{R['inwin']:>10}{R['twinE']:>8}{R['haspass']:>9}  {R['nopass'][:12]}{'...' if len(R['nopass']) > 12 else ''}")
    for q, rows in sample.items():
        out.append(""); out.append(f"q = {q} (sub-machine base {base_of([p for p in primerange(2, int(q**0.5)+1)], int(q**0.5)//2)[0]}):"); out += ["   " + x for x in rows]
    Path("research/stack/r8/results_spiral_stacks2.txt").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out[2:]))

if __name__ == "__main__":
    main()
