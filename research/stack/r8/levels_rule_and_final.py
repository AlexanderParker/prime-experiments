"""(1) The level rule with the edge fixed: levels continue only while b_k >= 7, so the deepest
level always has the gears 5 and 7 over the base {2, 3} and lands at 23 inside (7, 49].  Measured
over the machines: the level-1 landing in the sub-window, the final step one level down (A), the
direct final flip from E_1 with h > sqrt q (C).
(2) The final flip from the level-1 landing E_1 <= q, in the fields: for q = 499 and 1999, every
gear h > sqrt q with 6h > q - E_1 (so the landing enters the window), the landing (E_1 + 6h,
E_1 + 6h + 2), and the field that paints it: the smallest gear dividing a member (row of the
multiples field), the member, the products:j order; or PASS.

usage: uv run python research/stack/r8/levels_rule_and_final.py 2000
"""
import sys
import numpy as np
from sympy import primerange, factorint
from pathlib import Path

def base_of(ps, bound):
    base = []; P = 1
    for p in ps:
        if P * p <= bound: P *= p; base.append(p)
        else: break
    return base, P

def alt(xs): return sum(x if i % 2 == 0 else -x for i, x in enumerate(xs))

def levels_landing(q, ps):
    bounds = [q]
    while int(bounds[-1] ** 0.5) >= 11: bounds.append(int(bounds[-1] ** 0.5))
    E = -1; lands = []
    for k in range(len(bounds) - 1, -1, -1):
        b = bounds[k]; lo = bounds[k + 1] if k + 1 < len(bounds) else 1
        lps = [p for p in ps if p <= b]; bk, Pk = base_of(lps, max(b // 2, 6))
        gk = sorted([p for p in lps if p > lo and p not in bk], reverse=True)
        if gk: E = E + 2 * Pk * alt(gk)
        lands.append((k, b, bk, gk, E))
    return bounds, lands

def main():
    Q = int(sys.argv[1]); qs = list(primerange(11, Q + 1))
    N = Q * Q + 10; sv = np.ones(N + 1, dtype=bool); sv[:2] = False
    for i in range(2, int(N ** 0.5) + 1):
        if sv[i]: sv[i * i::i] = False
    out = [__doc__.strip(), ""]
    c = dict(n=0, sub=0, in1=0, twin1=0, A=0, C=0, top=0); noA = []; noC = []; notop = []
    for q in qs:
        ps = list(primerange(2, q + 1)); r = int(q ** 0.5); high = [p for p in ps if p > r]
        bounds, lands = levels_landing(q, ps)
        c['n'] += 1
        E0 = lands[-1][4]
        pz0 = [(h, d) for d in (1, -1) for h in high if q < E0 + 6 * d * h and E0 + 6 * d * h + 2 <= q * q and sv[E0 + 6 * d * h] and sv[E0 + 6 * d * h + 2]]
        c['top'] += bool(pz0)
        if not pz0: notop.append(q)
        if len(bounds) < 2: continue
        c['sub'] += 1; b1 = bounds[1]; E1 = [L for k, b, bk, gk, L in lands if k == 1][0]
        c['in1'] += (b1 < E1 and E1 + 2 <= b1 * b1); c['twin1'] += bool(sv[E1] and sv[E1 + 2])
        b2 = bounds[2] if len(bounds) > 2 else 1
        subh = [p for p in ps if b2 < p <= b1 and p >= 5]
        pzA = [(h, d) for d in (1, -1) for h in subh if b1 < E1 + 6 * d * h and E1 + 6 * d * h + 2 <= b1 * b1 and sv[E1 + 6 * d * h] and sv[E1 + 6 * d * h + 2]]
        pzC = [(h, d) for d in (1, -1) for h in high if q < E1 + 6 * d * h and E1 + 6 * d * h + 2 <= q * q and sv[E1 + 6 * d * h] and sv[E1 + 6 * d * h + 2]]
        c['A'] += bool(pzA); c['C'] += bool(pzC)
        if not pzA: noA.append(q)
        if not pzC: noC.append(q)
    out.append(f"(1) rule b_k >= 11, machines 11 to {Q}: {c['n']} machines; top-level final flip passes at {c['top']} (fails {notop}); "
               f"{c['sub']} with a sub-machine: E_1 in the sub-window {c['in1']}, E_1 a twin {c['twin1']}, A (final step one level down) {c['A']} (fails {noA}), C (final flip from E_1 with h > sqrt q) {c['C']} (fails {noC})")
    for q in (499, 1999):
        ps = list(primerange(2, q + 1)); r = int(q ** 0.5); high = [p for p in ps if p > r]
        bounds, lands = levels_landing(q, ps); E1 = [L for k, b, bk, gk, L in lands if k == 1][0]
        out.append(""); out.append(f"(2) q = {q}: levels {[(k, b, bk, len(gk), L) for k, b, bk, gk, L in lands]}; E_1 = {E1}; window ({q}, {q*q}]; the flip enters the window iff 6h > {q - E1}, i.e. h > {(q - E1) / 6:.1f}")
        taken = {}
        for h in high:
            L = E1 + 6 * h
            if not (q < L and L + 2 <= q * q): continue
            f1, f2 = factorint(L), factorint(L + 2)
            if f1.get(L) == 1 and f2.get(L + 2) == 1:
                out.append(f"   h = {h:>4}: landing ({L}, {L + 2})  PASS"); continue
            cands = [(min(p for p in f if p >= 5), m, f) for m, f in ((L, f1), (L + 2, f2)) if any(p >= 5 and (len(f) > 1 or f[p] > 1) for p in f)]
            g, m, f = min(cands)
            fs = "*".join(f"{p}^{e}" if e > 1 else str(p) for p, e in sorted(f.items())); j = sum(f.values())
            out.append(f"   h = {h:>4}: landing ({L}, {L + 2})  painted in row {g} (multiples), member {m} = {fs}, products:{j}" + (" own" if g == h else ""))
    Path("research/stack/r8/results_levels_rule_and_final.txt").write_text("\n".join(out), encoding="utf-8")
    print(out[2]); print("\n".join(l for l in out[3:] if 'PASS' in l or l.startswith('(2)') or l.startswith('   h') and int(l.split()[2].rstrip(':')) < 120))

if __name__ == "__main__":
    main()
