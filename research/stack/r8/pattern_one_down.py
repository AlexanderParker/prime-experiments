"""The three-column pattern one level down, by the same walk.

Top machine q, level-1 landing E_1 <= q.  The final flip needs a gear h in ((q - E_1)/6, q] with
(E_1 + 6h, E_1 + 6h + 2) a twin.  The deciding rows are the gears up to sqrt(7q); let m be the
largest prime at most sqrt(7q).  Run the machine m's own walk (levels, rule b_k >= 11; for m
below 121 just its top spiral over its base) to its landing E_m, then its final flips
n = E_m +- 6h' with h' a gear of m above sqrt m.  Accept n when n is a gear of q in the range and
(E_1 + 6n, E_1 + 6n + 2) is a twin: then the machine m's walk has located the top machine's h.
Reported: per machine whether some n is accepted, the first, and the machines where none is.

usage: uv run python research/stack/r8/pattern_one_down.py 2000
"""
import sys
import numpy as np
from sympy import primerange, prevprime
from pathlib import Path
sys.path.insert(0, "research/stack/r8")
from levels_rule_and_final import levels_landing

def main():
    Q = int(sys.argv[1]); qs = list(primerange(11, Q + 1))
    N = 8 * Q + 10; sv = np.ones(N + 1, dtype=bool); sv[:2] = False
    for i in range(2, int(N ** 0.5) + 1):
        if sv[i]: sv[i * i::i] = False
    out = [__doc__.strip(), ""]
    ok = 0; n_with = 0; fails = []; rows = []
    for q in qs:
        ps = list(primerange(2, q + 1)); bounds, lands = levels_landing(q, ps)
        if len(bounds) < 2: continue
        n_with += 1
        E1 = [L for k, b, bk, gk, L in lands if k == 1][0]
        m = prevprime(int((7 * q) ** 0.5) + 1)
        mps = list(primerange(2, m + 1)); mb, ml = levels_landing(m, mps); Em = ml[-1][4]
        rm = int(m ** 0.5); hs = [p for p in mps if p > rm and p >= 5]
        found = None; tried = 0
        for d in (1, -1):
            for h2 in hs:
                n = Em + 6 * d * h2
                if not (n > (q - E1) / 6 and n <= q and n > 1 and sv[n]): continue
                tried += 1
                L = E1 + 6 * n
                if q < L and L + 2 <= q * q and sv[L] and sv[L + 2]:
                    found = (h2, d, n, L); break
            if found: break
        if found: ok += 1
        else: fails.append(q)
        if q in (499, 997, 1999) or (not found and len(fails) <= 6):
            rows.append(f"q = {q}: E_1 = {E1}, m = {m}, E_m = {Em}, gears of m above sqrt m: {hs}; candidates n that are gears of q in range: {tried}; " +
                        (f"accepted h' = {found[0]} {'up' if found[1] > 0 else 'down'} -> n = {found[2]} -> landing ({found[3]}, {found[3] + 2})" if found else "none accepted"))
    out.append(f"machines with a sub-machine: {n_with}; the machine m's walk locates a valid h for q at {ok}; fails at {fails[:30]}{'...' if len(fails) > 30 else ''}")
    out += rows
    Path("research/stack/r8/results_pattern_one_down.txt").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out[2:]))

if __name__ == "__main__":
    main()
