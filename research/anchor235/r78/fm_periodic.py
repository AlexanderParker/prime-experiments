"""fm_periodic.py -- P7: the periodic control S_H, H = {+-1} mod 30 (and the r77 case mod 12 re-run as a gate).

S_H = {n : n mod m in H}; closed under multiplication; irreducible = no factorisation a * b with a, b in S_H, 1 < a <= b.
Columns (m j - 1, m j + 1). Cuts: c_1 = least irreducible, c_{k+1} = g_k^2 with g_k the least irreducible >= c_k.
Step 8 on S_H: a column with both members irreducible in every section. Output: results/periodic.json"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from fm_common import spf_table

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)


def divisors_from(fs):
    ds = [1]
    for p in fs:
        ds = ds + [d * p for d in ds]
    return sorted(set(ds))


def run(m, H, N, spf):
    inH = np.zeros(m, dtype=bool)
    inH[list(H)] = True
    inS = inH[np.arange(N + 1) % m]
    inS[0] = False
    # irreducible in S_H: n in S_H, n > 1, and no divisor d with 1 < d <= sqrt(n), d in S_H, n/d in S_H
    def irr(n):
        if not inS[n] or n == 1:
            return False
        fs = []
        x = n
        while x > 1:
            p = int(spf[x])
            fs.append(p)
            x //= p
        for d in divisors_from(fs):
            if 1 < d and d * d <= n and inS[d] and inS[n // d]:
                return False
        return True
    g1 = next(n for n in range(2, N) if irr(n))
    c2 = g1 * g1
    g2 = next(n for n in range(c2, N) if irr(n))
    c3 = g2 * g2
    g3 = next(n for n in range(c3, N) if irr(n)) if c3 < N else None
    secs = []
    for lo, hi in [(c2, c3), (c3, min(g3 * g3, N) if g3 else N)]:
        if lo >= N:
            break
        twins = []
        for j in range(lo // m + 1, (hi - 1) // m + 1):
            l, r = m * j - 1, m * j + 1
            if l > lo and r < hi and irr(l) and irr(r):
                twins.append((l, r))
        kinds = {}
        for l, r in twins:
            def om(x):
                c = 0
                while x > 1:
                    x //= int(spf[x])
                    c += 1
                return c
            k = (om(l), om(r))
            kinds[str(k)] = kinds.get(str(k), 0) + 1
        secs.append(dict(lo=lo, hi=hi, prefix_only=hi >= N, twin_irreducible_columns=len(twins), first=twins[:5], kinds=kinds))
    return dict(m=m, H=sorted(H), g=[g1, g2, g3], cuts=[c2, c3, g3 * g3 if g3 else None], sections=secs)


def main():
    N = 4_000_000
    spf = spf_table(N)
    out = {}
    r12 = run(12, {1, 11}, N, spf)
    print("mod 12:", json.dumps(r12))
    assert r12["sections"][0]["twin_irreducible_columns"] == 558, r12
    out["mod12"] = r12
    r30 = run(30, {1, 29}, N, spf)
    print("mod 30:", json.dumps(r30))
    out["mod30"] = r30
    with open(os.path.join(RES, "periodic.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
