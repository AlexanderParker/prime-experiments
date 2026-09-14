"""Two more level constructions (owner: both).

Levels as in spiral_groups_levels.py: b_0 = q, b_1 = floor sqrt q, ...; base_k = lowest gears
with product at most b_k / 2, always holding 2 and 3; gears_k = primes in (b_{k+1}, b_k] outside
base_k; each level's spiral from the previous landing, descending, first flip up.

A. The final step one level down.  E_1 = the landing after level 1 (in the sub-window
   (sqrt q, q]).  Final step there with a sub-machine gear h in (b_2, b_1] (h at most sqrt q):
   landing E_1 +- 6h, a twin inside (b_1, b_1^2] = the sub-machine's window.  This is the window
   statement for the machine b_1 run by the same walk.
B. Level 0 with the sqout set instead of sqin: from E_1, the level-0 spiral over the gears at
   most sqrt q outside the machine's base (sqout) with the machine's base, then the final step
   with h above sqrt q.
Also C: from E_1 directly, the final step with h above sqrt q (no level-0 spiral).

usage: uv run python research/stack/r8/spiral_levels2.py 2000
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

def alt(xs): return sum(x if i % 2 == 0 else -x for i, x in enumerate(xs))

def main():
    Q = int(sys.argv[1]); qs = list(primerange(11, Q + 1))
    N = Q * Q + 10; sv = np.ones(N + 1, dtype=bool); sv[:2] = False
    for i in range(2, int(N ** 0.5) + 1):
        if sv[i]: sv[i * i::i] = False
    names = ['A sub final', 'B sqout at 0', 'C E1 final']
    res = {n: dict(n=0, inwin=0, twinE=0, haspass=0, nopass=[]) for n in names}
    sample = {}
    for q in qs:
        ps = list(primerange(2, q + 1)); r = int(q ** 0.5)
        if r < 5: continue
        bounds = [q]
        while int(bounds[-1] ** 0.5) >= 5: bounds.append(int(bounds[-1] ** 0.5))
        levels = []
        for k, b in enumerate(bounds):
            lo = bounds[k + 1] if k + 1 < len(bounds) else 1
            lps = [p for p in ps if p <= b]; bk, Pk = base_of(lps, max(b // 2, 6))
            gk = sorted([p for p in lps if p > lo and p not in bk], reverse=True)
            levels.append((k, b, lo, bk, Pk, gk))
        # landing after level 1
        E = -1
        for k, b, lo, bk, Pk, gk in reversed(levels):
            if k == 0: break
            if gk: E = E + 2 * Pk * alt(gk)
        E1 = E; b1 = bounds[1]; b2 = bounds[2] if len(bounds) > 2 else 1
        subh = [p for p in ps if b2 < p <= b1]
        # A: final step in the sub-machine's window with h <= sqrt q
        pzA = [(h, d) for d in (1, -1) for h in subh if b1 < E1 + 6 * d * h and E1 + 6 * d * h + 2 <= b1 * b1 and sv[E1 + 6 * d * h] and sv[E1 + 6 * d * h + 2]]
        inA = b1 < E1 and E1 + 2 <= b1 * b1
        RA = res['A sub final']; RA['n'] += 1; RA['inwin'] += inA; RA['twinE'] += bool(sv[E1] and sv[E1 + 2]) if 0 <= E1 < N - 2 else 0; RA['haspass'] += bool(pzA)
        if not pzA: RA['nopass'].append(q)
        # B: level 0 with sqout
        base0, P0 = base_of(ps, q // 2); high = [p for p in ps if p > r]
        sqout = sorted([p for p in ps if p not in base0 and 5 <= p <= r], reverse=True)
        EB = E1 + 2 * P0 * alt(sqout) if sqout else None
        def passes(E): return [(h, d) for d in (1, -1) for h in high if q < E + 6 * d * h and E + 6 * d * h + 2 <= q * q and sv[E + 6 * d * h] and sv[E + 6 * d * h + 2]]
        if EB is not None:
            pzB = passes(EB); inB = q < EB and EB + 2 <= q * q
            RB = res['B sqout at 0']; RB['n'] += 1; RB['inwin'] += inB; RB['twinE'] += bool(sv[EB] and sv[EB + 2]) if 0 <= EB < N - 2 else 0; RB['haspass'] += bool(pzB)
            if not pzB: RB['nopass'].append(q)
        else: pzB = None; inB = None
        # C: from E1 directly, final step with h > sqrt q
        pzC = passes(E1); inC = q < E1 and E1 + 2 <= q * q
        RC = res['C E1 final']; RC['n'] += 1; RC['inwin'] += inC; RC['twinE'] += bool(sv[E1] and sv[E1 + 2]) if 0 <= E1 < N - 2 else 0; RC['haspass'] += bool(pzC)
        if not pzC: RC['nopass'].append(q)
        if q in (101, 499, 997, 1999):
            sample[q] = [f"E1 = {E1} (b1 = {b1}, sub-window ({b1}, {b1 * b1}], sub gears for the final step {subh})",
                         f"A: passes {len(pzA)}" + (f" first {pzA[0]} -> ({E1 + 6 * pzA[0][1] * pzA[0][0]}, {E1 + 6 * pzA[0][1] * pzA[0][0] + 2})" if pzA else ""),
                         f"B: EB = {EB} {'in' if inB else 'OUT'}, passes {len(pzB) if pzB is not None else '-'}" + (f" first {pzB[0]}" if pzB else ""),
                         f"C: from E1 with h > sqrt q: in window {inC}, passes {len(pzC)}" + (f" first {pzC[0]}" if pzC else "")]
    out = [__doc__.strip(), "", f"{'variant':<14}{'machines':>9}{'in window':>10}{'E twin':>8}{'has pass':>9}  no pass at"]
    for n in names:
        R = res[n]
        if R['n']: out.append(f"{n:<14}{R['n']:>9}{R['inwin']:>10}{R['twinE']:>8}{R['haspass']:>9}  {R['nopass'][:14]}{'...' if len(R['nopass']) > 14 else ''}")
    for q, rows in sample.items():
        out.append(""); out.append(f"q = {q}:"); out += ["   " + x for x in rows]
    Path("research/stack/r8/results_spiral_levels2.txt").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out[2:]))

if __name__ == "__main__":
    main()
