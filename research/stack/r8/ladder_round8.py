"""Round 8 of the twin ladder (tree node R5.f.ix): the reuse recount and the index bound's supply side.

Claim 1 (recount). Observed reuse among the first N base-open offsets, by the statistic the
expectation computes: sum_g C(k_g, 2) over gears g > x (distinct prime factors per member), with
the histogram of k_g and the split by case (g | d, 3d-1, 3d+1 for the pairs it serves); compare
with the exact pair expectation (ladder_round7.py's model).  Prediction: sum C(k_g,2) matches the
expectation within two Poisson sigma; sum (k_g - 1) is below it by about N/(3x).
Claim 2 (supply). For every twin: F(x) (exact where held, 0.7 x ln x above 61) and the pigeonhole
guarantee G = floor((4c-1)/(F+1)); flags i <= ceil(4 ln s) and i <= G; the least s above which the
second never fails (predicted about 6.8 x 10^4) and the failures below it.

usage: uv run python ladder_round8.py [PMAX_INDEX] [PFULL]
"""
import sys, math, collections
import numpy as np
from sympy import isprime, primefactors, jacobi_symbol

PMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 10**6
PFULL = int(sys.argv[2]) if len(sys.argv) > 2 else 10**5
FEXACT = {5: 1, 7: 4, 11: 6, 13: 10, 17: 17, 19: 24, 23: 33, 29: 42, 31: 57, 37: 87, 41: 90, 43: 102,
          47: 117, 53: 144, 59: 160, 61: 179}

def primes_upto(n):
    s = np.ones(n + 1, dtype=bool); s[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]: s[i*i::i] = False
    return np.nonzero(s)[0]

P = primes_upto(PMAX + 2)
ps = set(int(x) for x in P)
gears = [int(g) for g in P if g >= 5]
twin_lowers = [int(x) for x in P if x >= 5 and (int(x) + 2) in ps]

def base_open_sorted(Pm):
    s = Pm + 1; c = s // 6; s2 = s * s; jmax = 2 * c - 1; n = 2 * jmax + 1
    x = math.isqrt(s)
    struck = np.zeros(n, dtype=bool)
    for g in gears:
        if g > x: break
        u = pow(6, -1, g)
        for jr in ((u * (1 - s2)) % g, (u * (-1 - s2)) % g):
            struck[(jr + jmax) % g::g] = True
    bo = np.nonzero(~struck)[0] - jmax
    return s, s2, x, c, bo[np.argsort(np.abs(bo), kind='stable')]

def F_est(x):
    below = [g for g in FEXACT if g <= x]
    if x <= 61: return FEXACT[max(below)] if below else 1
    return int(0.7 * x * math.log(x))

def w(g, t):
    t %= g
    if t == 0: return 1 / (g - 2)
    if t == 1: return 0.0
    return 2 / (g - 2) if jacobi_symbol(t, g) == 1 else 0.0

# Claim 1
N = 32
sum_pairs = 0; sum_km1 = 0; hist = collections.Counter(); expct = 0.0
case_obs = collections.Counter(); case_exp = collections.Counter()
for Pm in twin_lowers:
    if Pm > PFULL: break
    s, s2, x, c, bo = base_open_sorted(Pm)
    first = [int(j) for j in bo[:N]]
    hits = collections.defaultdict(set)   # gear -> set of offset indices
    for i, j in enumerate(first):
        for m in (s2 + 6 * j - 1, s2 + 6 * j + 1):
            if not isprime(m):
                for g in primefactors(m):
                    if g > x: hits[g].add(i)
    for g, idxs in hits.items():
        k = len(idxs); hist[k] += 1
        if k >= 2:
            sum_pairs += k * (k - 1) // 2; sum_km1 += k - 1
            for a in sorted(idxs):
                for b in sorted(idxs):
                    if a < b:
                        d = first[a] - first[b]
                        if d % g == 0: case_obs['d'] += 1
                        elif (3 * d - 1) % g == 0: case_obs['3d-1'] += 1
                        elif (3 * d + 1) % g == 0: case_obs['3d+1'] += 1
                        else: case_obs['none'] += 1
    for a in range(len(first)):
        for b in range(a + 1, len(first)):
            d = first[a] - first[b]
            for num, case in ((d, 'd'), (3 * d - 1, '3d-1'), (3 * d + 1, '3d+1')):
                if num == 0: continue
                for g in primefactors(abs(num)):
                    if g <= x: continue
                    if case == 'd': val = w(g, 1 - 6 * first[a]) + w(g, -1 - 6 * first[a])
                    elif case == '3d-1': val = w(g, 1 - 6 * first[a])
                    else: val = w(g, -1 - 6 * first[a])
                    expct += val; case_exp[case] += val
print(f"Claim 1 (recount, first {N}, twins to {PFULL}): sum C(k,2) = {sum_pairs}, sum (k-1) = {sum_km1}, pair expectation {expct:.1f} (Poisson sigma {math.sqrt(expct):.0f})")
print("   multiplicity histogram k: " + ", ".join(f"{k}: {hist[k]}" for k in sorted(hist)))
print("   observed pairs by case: " + ", ".join(f"{k}: {v}" for k, v in sorted(case_obs.items())) + " | expected: " + ", ".join(f"{k}: {v:.0f}" for k, v in sorted(case_exp.items())))

# Claim 2
fails_G = []; fails_4ln = []; run_law_max = (0.0, None); last_fail_G = None
for Pm in twin_lowers:
    s, s2, x, c, bo = base_open_sorted(Pm)
    hit = None
    for i, j in enumerate(bo):
        j = int(j)
        if isprime(s2 + 6 * j - 1) and isprime(s2 + 6 * j + 1): hit = i; break
    Fx = F_est(x); G = (4 * c - 1) // (Fx + 1)
    if hit > math.ceil(4 * math.log(s)): fails_4ln.append((Pm, hit))
    if hit >= G:
        fails_G.append((Pm, s, hit, G)); last_fail_G = s
print(f"Claim 2 (supply), twins to {PMAX}: i > ceil(4 ln s): {fails_4ln[:8]} (count {len(fails_4ln)})")
print(f"   i >= pigeonhole guarantee G: count {len(fails_G)}, last failing s = {last_fail_G}; first few {fails_G[:8]}")
