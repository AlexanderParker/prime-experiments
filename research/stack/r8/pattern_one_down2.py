"""The three-column pattern located by the sub-machine's own flips.

pattern_one_down.py showed the machine sqrt(7q)'s walk lands above q at most machines (its
window runs to 7q), so its flips cannot reach the h-range ((q - E_1)/6, q].  Here the locator is
the sub-machine itself: from E_1 (its level landing, at most q) its flips n = E_1 +- 6h'' with
h'' a gear at most sqrt q (above the fourth root) give candidates n; accept n when n is a gear of
q in the range and (E_1 + 6n, E_1 + 6n + 2) is a twin.  Also tried: n = E_1 +- 6h'' for every
gear h'' of the sub-machine including the base, and n = E_1 +- 12h'' (mirror {2, 3, h''}).

usage: uv run python research/stack/r8/pattern_one_down2.py 2000
"""
import sys
import numpy as np
from sympy import primerange
from pathlib import Path
sys.path.insert(0, "research/stack/r8")
from levels_rule_and_final import levels_landing

def main():
    Q = int(sys.argv[1]); qs = list(primerange(11, Q + 1))
    N = 8 * Q + 10; sv = np.ones(N + 1, dtype=bool); sv[:2] = False
    for i in range(2, int(N ** 0.5) + 1):
        if sv[i]: sv[i * i::i] = False
    out = [__doc__.strip(), ""]
    variants = {'6h, h above 4th root': (6, True), '6h, every gear of the sub-machine': (6, False), '12h, every gear': (12, False)}
    res = {v: dict(ok=0, fails=[]) for v in variants}; n_with = 0; rows = []
    for q in qs:
        ps = list(primerange(2, q + 1)); bounds, lands = levels_landing(q, ps)
        if len(bounds) < 2: continue
        n_with += 1; b1 = bounds[1]
        E1 = [L for k, b, bk, gk, L in lands if k == 1][0]
        for v, (step, above) in variants.items():
            hs = [p for p in ps if p <= b1 and p >= 5 and (not above or p > int(b1 ** 0.5))]
            found = None
            for d in (1, -1):
                for h2 in hs:
                    n = E1 + step * d * h2
                    if not (n > (q - E1) / 6 and n <= q and n > 1 and sv[n]): continue
                    L = E1 + 6 * n
                    if q < L and L + 2 <= q * q and sv[L] and sv[L + 2]: found = (h2, d, n, L); break
                if found: break
            if found: res[v]['ok'] += 1
            else: res[v]['fails'].append(q)
            if q in (499, 997, 1999):
                rows.append(f"q = {q}, {v}: E_1 = {E1}, sub gears {hs}; " + (f"h'' = {found[0]} {'up' if found[1] > 0 else 'down'} -> n = {found[2]} -> ({found[3]}, {found[3] + 2})" if found else "none"))
    out.append(f"machines with a sub-machine: {n_with}")
    for v in variants:
        out.append(f"   {v}: located at {res[v]['ok']}; fails {res[v]['fails'][:20]}{'...' if len(res[v]['fails']) > 20 else ''}")
    out += rows
    Path("research/stack/r8/results_pattern_one_down2.txt").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out[2:]))

if __name__ == "__main__":
    main()
