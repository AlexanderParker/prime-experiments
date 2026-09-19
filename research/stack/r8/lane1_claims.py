"""Tests of the fresh math lane's three claims (tree node R5.f.i), every prime 37 <= p <= PMAX.

Claim 1 (square-scale transfer): for a twin column c of the stretch (P = 6c-1, P+2 = 6c+1), the
columns of P^2, P(P+2), (P+2)^2 are 6c^2-2c (upper member P^2, lower P^2-2), 6c^2 (lower P(P+2),
upper 36c^2+1), 6c^2+2c (upper (P+2)^2, lower (P+2)^2-2).  Claim: some twin of every stretch has
one of the three partners P^2-2, 36c^2+1, (P+2)^2-2 free of every prime factor <= p.  (The lane
wrote P^2+2 and (P+2)^2+2; the partner members in those columns are P^2-2 and (P+2)^2-2; both
readings are tested.)
Claim 2 (single plug): every stretch has a base-open column (no gear <= B = q^(2/3)) struck by
exactly one top gear; every top-gear cofactor is prime.
Claim 3 (near pairs): every stretch has two base-open columns at distance < B/6; and no top gear
plugs two base-open columns at distance < (g-2)/6.

usage: uv run python lane1_claims.py [PMAX]
"""
import sys, math
import numpy as np

PMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 20000

def primes_upto(n):
    s = np.ones(n + 1, dtype=bool); s[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]: s[i*i::i] = False
    return np.nonzero(s)[0]

P = primes_upto(PMAX + 200)
plist = [int(x) for x in P if 37 <= x <= PMAX]
gears_all = [int(g) for g in P if g >= 5]

def rough(n, bound):
    for g in gears_all:
        if g > bound: return True
        if n % g == 0: return False
    return True

def is_prime(n):
    if n < 2: return False
    for g in P:
        g = int(g)
        if g * g > n: return True
        if n % g == 0: return False
    return True  # n < (max P)^2 always holds here

fail1 = []; fail1b = []; fail2 = []; fail3 = []; sep_viol = 0; comp_cof = 0
c1_first = []  # index of the first clearing twin
for p in plist:
    q = int(P[np.searchsorted(P, p) + 1])
    c0 = (p * p + 1) // 6 + 1; c1 = (q * q - 1) // 6 - 1
    L = c1 - c0 + 1
    B = int(round(q ** (2.0 / 3.0)))
    base = np.zeros(L, dtype=bool)
    plugs = np.zeros(L, dtype=np.int32)
    plug_gear = {}   # column -> list of (g, member)
    for h in P:
        h = int(h)
        if h < 5: continue
        if h > p: break
        u = pow(6, -1, h)
        for tooth, side in ((u, -1), (h - u, +1)):
            idx = np.arange((tooth - c0) % h, L, h)
            if h <= B:
                base[idx] = True
            else:
                plugs[idx] += 1
                for i in idx:
                    plug_gear.setdefault(int(i), []).append((h, side))
    bopen = np.nonzero(~base)[0]
    twins = np.nonzero((~base) & (plugs == 0))[0]
    # Claim 2
    pc = plugs[bopen]
    if not (pc == 1).any(): fail2.append(p)
    for i in bopen:
        for (g, side) in plug_gear.get(int(i), []):
            m = 6 * (i + c0) + side
            if not is_prime(m // g): comp_cof += 1
    # Claim 3
    if len(bopen) < 2 or np.diff(bopen).min() >= B / 6: fail3.append(p)
    for i in bopen:
        for (g, s1) in plug_gear.get(int(i), []):
            for j in bopen[(bopen > i) & (bopen < i + (g - 2) / 6)]:
                if any(g2 == g for (g2, _) in plug_gear.get(int(j), [])): sep_viol += 1
    # Claim 1
    if len(twins):
        ok = False; okb = False; first = None
        for n, i in enumerate(twins):
            c = int(i + c0); Pm = 6 * c - 1
            targets = (Pm * Pm - 2, 36 * c * c + 1, (Pm + 2) ** 2 - 2)
            targets_b = (Pm * Pm + 2, 36 * c * c + 1, (Pm + 2) ** 2 + 2)
            if not ok and any(rough(t, p) for t in targets): ok = True; first = n
            if not okb and any(rough(t, p) for t in targets_b): okb = True
            if ok and okb: break
        if not ok: fail1.append(p)
        if not okb: fail1b.append(p)
        if first is not None: c1_first.append(first)

print(f"stretches tested: {len(plist)} (p = 37..{plist[-1]})")
print(f"Claim 1 (partners P^2-2, 36c^2+1, (P+2)^2-2 p-rough for some twin): failures {fail1}")
print(f"        first clearing twin index: mean {np.mean(c1_first):.2f}, max {max(c1_first)}")
print(f"Claim 1 as written (P^2+2, 36c^2+1, (P+2)^2+2): failures {fail1b}")
print(f"Claim 2 (a base-open column with exactly one plug): failures {fail2}; composite cofactors: {comp_cof}")
print(f"Claim 3 (two base-open columns closer than B/6): failures {fail3}; separation-law violations: {sep_viol}")
