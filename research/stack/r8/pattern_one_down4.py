"""Locating the final flip's gear h: two chained flips, and flips from the deeper landing.

(a) Two flips from E_1 with sub-machine gears h1, h2 (mirrors {3, h}, one period each, either
    direction): n = E_1 + 6 d1 h1 + 6 d2 h2.  Accept when n is a gear of q in ((q - E_1)/6, q]
    and (E_1 + 6n, E_1 + 6n + 2) is a twin.  Also with {2, 3, h} (12 h steps).
(b) From the deeper landing E_2 (the level-2 landing, 23 when it exists) with level-1 gears:
    n = E_2 + 2 k M d h'', k = 1..6, mirrors {3, h''} and {2, 3, h''}.
(c) From E_1 with the top machine's own small gears above sqrt q as h'' (first three high
    gears), one flip {3, h''}, k = 1..3.

usage: uv run python research/stack/r8/pattern_one_down4.py 2000
"""
import sys
import numpy as np
from sympy import primerange
from pathlib import Path
sys.path.insert(0, "research/stack/r8")
from levels_rule_and_final import levels_landing

def main():
    Q = int(sys.argv[1]); qs = list(primerange(11, Q + 1))
    N = Q * Q + 10; sv = np.ones(N + 1, dtype=bool); sv[:2] = False
    for i in range(2, int(N ** 0.5) + 1):
        if sv[i]: sv[i * i::i] = False
    names = ['a: two flips {3,h}', 'a: two flips {2,3,h}', 'b: from E_2, {3,h} k1-6', 'b: from E_2, {2,3,h} k1-6', 'c: from E_1, three high gears k1-3', 'any']
    res = {n: dict(ok=0, fails=[]) for n in names}; n_with = 0; rows = []
    for q in qs:
        ps = list(primerange(2, q + 1)); bounds, lands = levels_landing(q, ps)
        if len(bounds) < 2: continue
        n_with += 1; b1 = bounds[1]; r = int(q ** 0.5)
        E1 = [L for k, b, bk, gk, L in lands if k == 1][0]
        E2 = [L for k, b, bk, gk, L in lands if k == 2]; E2 = E2[0] if E2 else None
        hs = [p for p in ps if 5 <= p <= b1]; high3 = [p for p in ps if p > r][:3]
        def good(n):
            if not (1 < n <= q and n > (q - E1) / 6 and sv[n]): return False
            L = E1 + 6 * n
            return q < L and L + 2 <= q * q and sv[L] and sv[L + 2]
        found = {}
        for nm, step in (('a: two flips {3,h}', 6), ('a: two flips {2,3,h}', 12)):
            f = None
            for h1 in hs:
                for d1 in (1, -1):
                    for h2 in hs:
                        for d2 in (1, -1):
                            n = E1 + step * d1 * h1 + step * d2 * h2
                            if good(n): f = (h1, d1, h2, d2, n); break
                        if f: break
                    if f: break
                if f: break
            found[nm] = f
        for nm, step in (('b: from E_2, {3,h} k1-6', 6), ('b: from E_2, {2,3,h} k1-6', 12)):
            f = None
            if E2 is not None:
                for k in range(1, 7):
                    for d in (1, -1):
                        for h2 in hs:
                            n = E2 + step * k * d * h2
                            if good(n): f = (h2, k, d, n); break
                        if f: break
                    if f: break
            found[nm] = f
        f = None
        for k in (1, 2, 3):
            for d in (1, -1):
                for h2 in high3:
                    n = E1 + 6 * k * d * h2
                    if good(n): f = (h2, k, d, n); break
                if f: break
            if f: break
        found['c: from E_1, three high gears k1-3'] = f
        found['any'] = next((v for v in found.values() if v), None)
        for nm in names:
            if found[nm]: res[nm]['ok'] += 1
            else: res[nm]['fails'].append(q)
        if q in (127, 499, 1999): rows.append(f"q = {q}: E_1 = {E1}, E_2 = {E2}; " + "; ".join(f"{nm}: {found[nm]}" for nm in names[:-1]))
    out = [__doc__.strip(), "", f"machines with a sub-machine: {n_with}"]
    for nm in names:
        R = res[nm]; out.append(f"   {nm:<36} located at {R['ok']:>3}; fails {R['fails'][:14]}{'...' if len(R['fails']) > 14 else ''}")
    out += rows
    Path("research/stack/r8/results_pattern_one_down4.txt").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out[2:]))

if __name__ == "__main__":
    main()
