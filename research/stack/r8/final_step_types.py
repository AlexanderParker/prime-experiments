"""Approach A: exclude the dangerous field types at the final step, then pick the axis.

From the primorial spiral's landing E the final flip about {3, h} lands at E + 6dh (d = +-1).
Exclusions by type, none of them using E's residues on the gear:
  1. base gears (E = -1 mod g, g >= 5): they bite the landing iff they bite column h itself
     (kernel landing_open_base_iff), so drop every h whose own column (6h - 1, 6h + 1) is painted
     by a base gear;
  2. the axis gear h itself: it bites iff h | E or h | E + 2 (kernel own_landing_iff); drop those.
What remains is then charged to the two E-placed types: a gear above sqrt q (products:2 or :3
only) or a gear at most sqrt q outside the base (products:2 and up); or it passes (twin).

usage: uv run python research/stack/r8/final_step_types.py 2000
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
    out = [__doc__.strip(), "", f"{'q':>5} {'E':>8} {'cand':>5} {'aft1':>5} {'aft12':>5} {'pass':>4} {'by>sqrt':>7} {'by<=sqrt':>8}  first pass  (col-twin h among aft12: n, pass)"]
    tot = dict(c=0, a1=0, a12=0, ps=0, hi=0, lo=0, ct=0, ctp=0); nopass = []
    for q in qs:
        ps = list(primerange(2, q + 1)); base = []; P = 1
        for p in ps:
            if P * p <= q // 2: P *= p; base.append(p)
            else: break
        E = -1 + 2 * P * alt([p for p in ps if p not in base][::-1])
        r = int(q ** 0.5); bg = [g for g in base if g >= 5]
        band = [g for g in ps if 5 <= g <= r and g not in base]; high = [g for g in ps if g > r]
        c = dict(c=0, a1=0, a12=0, ps=0, hi=0, lo=0, ct=0, ctp=0); first = None
        for d in (1, -1):
            for h in high:
                L = E + 6 * d * h
                if not (q < L and L + 2 <= q * q): continue
                c['c'] += 1
                if any((6 * h - 1) % g == 0 or (6 * h + 1) % g == 0 for g in bg): continue
                c['a1'] += 1
                if E % h == 0 or (E + 2) % h == 0: continue
                c['a12'] += 1
                coltwin = bool(sv[6 * h - 1] and sv[6 * h + 1])
                if coltwin: c['ct'] += 1
                taker = next((g for g in band if L % g == 0 or (L + 2) % g == 0), None)
                if taker is not None: c['lo'] += 1; continue
                taker = next((g for g in high if g != h and (L % g == 0 or (L + 2) % g == 0)), None)
                if taker is not None: c['hi'] += 1; continue
                assert sv[L] and sv[L + 2]
                c['ps'] += 1
                if coltwin: c['ctp'] += 1
                if first is None: first = (h, '+' if d > 0 else '-')
        for k in tot: tot[k] += c[k]
        if c['ps'] == 0: nopass.append(q)
        out.append(f"{q:>5} {E:>8} {c['c']:>5} {c['a1']:>5} {c['a12']:>5} {c['ps']:>4} {c['hi']:>7} {c['lo']:>8}  {str(first):<11} ({c['ct']}, {c['ctp']})")
    out.append("")
    out.append(f"totals: candidates {tot['c']}, after exclusion 1 (column h painted by a base gear) {tot['a1']}, after 2 (h divides E or E+2) {tot['a12']}, pass {tot['ps']}; "
               f"taken by a gear above sqrt q {tot['hi']}, by a gear at most sqrt q outside the base {tot['lo']}")
    out.append(f"h whose own column is a twin, among the survivors of 1 and 2: {tot['ct']}, of which pass {tot['ctp']}")
    out.append(f"machines with no passing h: {nopass}")
    Path("research/stack/r8/results_final_step_types.txt").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out[3:12])); print("..."); print("\n".join(out[-4:]))

if __name__ == "__main__":
    main()
