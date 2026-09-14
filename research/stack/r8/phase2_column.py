"""Phase 2 and the base of the spiral: the landing is column h modulo the base.

The primorial spiral lands at E = -1 + 2 P A, so E = -1 (mod P) where P is the product of the
base gears.  The final step {3, h} lands at L = E + 6 d h, hence modulo every base gear g:
  up   (d = +1): L = 6h - 1,  L + 2 = 6h + 1      -- the landing IS column h
  down (d = -1): L = -(6h + 1), L + 2 = -(6h - 1) -- the landing is the reflection of column h
So a base gear g takes the high gear h iff g strikes column h = (6h - 1, 6h + 1), in either
direction: the base gears' forbidden classes of h are +-6^{-1} (mod g), independent of E and d.
The non-base gears of the sub-machine (P's next prime .. sqrt q) forbid classes that depend on E;
the high gears forbid their own two classes.

This script checks the identity at every machine and tables, per machine, the gear roles and
which stage takes each high gear.  usage: uv run python research/stack/r8/phase2_column.py 2000
"""
import sys
import numpy as np
from sympy import primerange
from pathlib import Path

def main():
    Q = int(sys.argv[1])
    qs = list(primerange(11, Q + 1))
    N = Q * Q + 10
    sieve = np.ones(N + 1, dtype=bool); sieve[:2] = False
    for i in range(2, int(N ** 0.5) + 1):
        if sieve[i]: sieve[i * i::i] = False
    def alt(xs): return sum(g if i % 2 == 0 else -g for i, g in enumerate(xs))
    out = [__doc__.strip(), ""]
    out.append(f"{'q':>5} {'base':<16} {'sub non-base':<22} {'reach':>5} {'col open':>8} {'sub':>4} {'pass':>4} {'col twin':>8} {'ct pass':>7}  first pass")
    mism = 0; tot = dict(reach=0, colopen=0, sub=0, ps=0, coltwin=0, ctpass=0)
    detail = {}
    for q in qs:
        primes = list(primerange(2, q + 1)); base = []; P = 1
        for p in primes:
            if P * p <= q // 2: P *= p; base.append(p)
            else: break
        E = -1 + 2 * P * alt([p for p in primes if p not in base][::-1])
        assert (E + 1) % P == 0
        gears = [p for p in primes if p >= 5]
        r = int(q ** 0.5)
        bg = [g for g in base if g >= 5]
        sub_nb = [g for g in gears if g <= r and g not in base]
        high = [g for g in gears if g > r]
        c = dict(reach=0, colopen=0, sub=0, ps=0, coltwin=0, ctpass=0); first = None
        rows = []
        for d in (1, -1):
            for h in high:
                L = E + 6 * d * h
                if not (q < L and L + 2 <= q * q): continue
                c['reach'] += 1
                def takes(g): return L % g == 0 or (L + 2) % g == 0
                def strikes_col(g): return (6 * h - 1) % g == 0 or (6 * h + 1) % g == 0
                for g in bg:
                    if takes(g) != strikes_col(g): mism += 1
                coltwin = bool(sieve[6 * h - 1] and sieve[6 * h + 1])
                if coltwin: c['coltwin'] += 1
                taker = next((g for g in bg if takes(g)), None); stage = 'base'
                if taker is None:
                    c['colopen'] += 1
                    taker = next((g for g in sub_nb if takes(g)), None); stage = 'sub'
                if taker is None:
                    c['sub'] += 1
                    taker = next((g for g in high if takes(g)), None); stage = 'high'
                if taker is None:
                    stage = 'pass'; c['ps'] += 1
                    if coltwin: c['ctpass'] += 1
                    if first is None: first = (h, '+' if d > 0 else '-')
                rows.append((h, d, coltwin, stage, taker))
        for k in tot: tot[k] += c[k]
        detail[q] = (base, sub_nb, E, rows)
        out.append(f"{q:>5} {str(base):<16} {str(sub_nb):<22} {c['reach']:>5} {c['colopen']:>8} {c['sub']:>4} {c['ps']:>4} {c['coltwin']:>8} {c['ctpass']:>7}  {first}")
    out.append("")
    out.append(f"identity mismatches (base gear takes h vs base gear strikes column h): {mism}")
    out.append(f"totals: reach {tot['reach']}, column open to base {tot['colopen']}, left by sub-machine {tot['sub']}, pass {tot['ps']}; "
               f"h with column h a twin prime pair {tot['coltwin']}, of which pass {tot['ctpass']}")
    for q in (101, 499):
        base, sub_nb, E, rows = detail[q]
        out.append(""); out.append(f"q = {q}: base {base}, sub non-base {sub_nb}, high gears above {int(q**0.5)}; E = {E}")
        for h, d, ct, stage, taker in rows:
            out.append(f"   h = {h} {'+' if d > 0 else '-'}: column h {'twin' if ct else 'struck'}; {stage}" + (f" (taken by {taker})" if taker else ""))
    Path("research/stack/r8/results_phase2_column.txt").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out[-3 - len(detail[499][3]) - len(detail[101][3]) - 3:]))

if __name__ == "__main__":
    main()
