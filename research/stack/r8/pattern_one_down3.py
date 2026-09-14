"""Locating the final flip's gear h by the sub-machine's flips, more mirrors (owner: keep going).

From E_1 (the sub-machine's level landing, at most q) candidates n = E_1 + 2 k M d, with M the
mirror size: {3, h''} (M = 3h''), {2, 3, h''} (M = 6h''), {base_0, h''} (M = P_0 h'', P_0 the
machine's base product), {base_1, h''} (M = P_1 h''), for k = 1 and k = 1, 2, 3; h'' every gear of
the sub-machine from 5.  Accept n when n is a gear of q in ((q - E_1)/6, q] and (E_1 + 6n,
E_1 + 6n + 2) is a twin.  Also from E_0 (the top landing, in the window): candidates n = E_0 + 2kMd
must be gears h of q with (E_0 + 6h) a twin.  And the union: any mirror of the list.

usage: uv run python research/stack/r8/pattern_one_down3.py 2000
"""
import sys
import numpy as np
from sympy import primerange
from pathlib import Path
sys.path.insert(0, "research/stack/r8")
from levels_rule_and_final import levels_landing, base_of

def main():
    Q = int(sys.argv[1]); qs = list(primerange(11, Q + 1))
    N = Q * Q + 10; sv = np.ones(N + 1, dtype=bool); sv[:2] = False
    for i in range(2, int(N ** 0.5) + 1):
        if sv[i]: sv[i * i::i] = False
    out = [__doc__.strip(), ""]
    names = ['{3,h} k1', '{3,h} k1-3', '{2,3,h} k1', '{2,3,h} k1-3', '{base0,h} k1', '{base0,h} k1-3', '{base1,h} k1', '{base1,h} k1-3', 'any of these']
    res = {(src, n): dict(ok=0, fails=[]) for src in ('E1', 'E0') for n in names}
    n_with = 0; rows = []
    for q in qs:
        ps = list(primerange(2, q + 1)); bounds, lands = levels_landing(q, ps)
        if len(bounds) < 2: continue
        n_with += 1; b1 = bounds[1]
        E1 = [L for k, b, bk, gk, L in lands if k == 1][0]; E0 = lands[-1][4]
        base0, P0 = base_of(ps, max(q // 2, 6)); base1, P1 = base_of([p for p in ps if p <= b1], max(b1 // 2, 6))
        hs = [p for p in ps if 5 <= p <= b1]
        for src, E in (('E1', E1), ('E0', E0)):
            def good(n):
                if not (1 < n <= q and sv[n]): return False
                L = E + 6 * n
                return q < L and L + 2 <= q * q and sv[L] and sv[L + 2]
            anyok = False; first_any = None
            for nm in names[:-1]:
                mir, ks = nm.split(' '); ks = [1] if ks == 'k1' else [1, 2, 3]
                M = {'{3,h}': 3, '{2,3,h}': 6, '{base0,h}': P0, '{base1,h}': P1}[mir]
                found = None
                for k in ks:
                    for d in (1, -1):
                        for h2 in hs:
                            n = E + 2 * k * M * d * h2
                            if good(n): found = (h2, k, d, n); break
                        if found: break
                    if found: break
                R = res[(src, nm)]
                if found: R['ok'] += 1; anyok = True; first_any = first_any or (nm, found)
                else: R['fails'].append(q)
            R = res[(src, 'any of these')]
            if anyok: R['ok'] += 1
            else: R['fails'].append(q)
            if q in (499, 1999): rows.append(f"q = {q} from {src} = {E}: first located by {first_any}")
    out.append(f"machines with a sub-machine: {n_with}")
    for src in ('E1', 'E0'):
        out.append(f"from {src}:")
        for nm in names:
            R = res[(src, nm)]; out.append(f"   {nm:<16} located at {R['ok']:>3}; fails {R['fails'][:16]}{'...' if len(R['fails']) > 16 else ''}")
    out += rows
    Path("research/stack/r8/results_pattern_one_down3.txt").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out[2:]))

if __name__ == "__main__":
    main()
