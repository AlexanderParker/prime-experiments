"""Round 7 of the twin ladder (tree node R5.f.viii): exact reuse expectation, the steered
universal list, fully plugged blocks by pattern.

Claim 1 (exact reuse). A gear g > x strikes two base-open offsets j_i, j_k (d = j_i - j_k) only if
g | d, g | 3d-1 or g | 3d+1; the common phase is t = 1 - 6 j_i (case 3d-1), t = -1 - 6 j_i (case
3d+1), both when g | d; over the twins T = s^2 mod g has weight 1/(g-2) at 0 and 2/(g-2) at each
nonzero square != 1.  E = sum over pairs and eligible g of w_g(t).  Observed reuse is counted two
ways: by least prime factors only (the earlier count) and by ALL prime factors > x of both
members (the model's object).
Claim 2 (universal list). For twins with s = 0 mod Q (Q = 35, 385) the base-open offsets at the
gears dividing Q are exactly j != +-u_g mod g; the first-twin index among the x-rough offsets has
mean 4.0 (not below 3.6).
Claim 3 (no impossible block). For N = 2, 3, 4 tabulate the difference patterns of the first N
base-open offsets, occurrences and fully plugged occurrences; any pattern with >= 100
occurrences never fully plugged would be an obstruction.

usage: uv run python ladder_round7.py [PMAX_STEER] [PFULL]
"""
import sys, math, collections
import numpy as np
from sympy import isprime, primefactors, factorint, jacobi_symbol

PMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 10**6
PFULL = int(sys.argv[2]) if len(sys.argv) > 2 else 10**5

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
    return s, s2, x, bo[np.argsort(np.abs(bo), kind='stable')]

def w(g, t):
    t %= g
    if t == 0: return 1 / (g - 2)
    if t == 1: return 0.0
    return 2 / (g - 2) if jacobi_symbol(t, g) == 1 else 0.0

# Claims 1 and 3 (P <= PFULL)
obs_lpf = {16: 0, 32: 0}; obs_all = {16: 0, 32: 0}; expct = {16: 0.0, 32: 0.0}
by_case = collections.Counter()
patterns = {2: collections.Counter(), 3: collections.Counter(), 4: collections.Counter()}
full = {2: collections.Counter(), 3: collections.Counter(), 4: collections.Counter()}
for Pm in twin_lowers:
    if Pm > PFULL: break
    s, s2, x, bo = base_open_sorted(Pm)
    first = [int(j) for j in bo[:32]]
    lpf = []; allf = []; plugged = []
    for i, j in enumerate(first):
        fs = set(); comp = False
        for m in (s2 + 6 * j - 1, s2 + 6 * j + 1):
            if not isprime(m):
                comp = True
                pf = [g for g in primefactors(m) if g > x]
                lpf.append((i, min(pf)))
                for g in pf: fs.add(g); allf.append((i, g))
        plugged.append(comp)
    for N in (16, 32):
        gs = [g for (i, g) in lpf if i < N]; obs_lpf[N] += len(gs) - len(set(gs))
        gs2 = [g for (i, g) in allf if i < N]; obs_all[N] += len(gs2) - len(set(gs2))
        E = 0.0
        for a in range(min(N, len(first))):
            for b in range(a + 1, min(N, len(first))):
                d = first[a] - first[b]
                for num, case in ((d, 'd'), (3 * d - 1, '3d-1'), (3 * d + 1, '3d+1')):
                    if num == 0: continue
                    for g in primefactors(abs(num)):
                        if g <= x: continue
                        if case == 'd':
                            val = w(g, 1 - 6 * first[a]) + w(g, -1 - 6 * first[a])
                        elif case == '3d-1':
                            val = w(g, 1 - 6 * first[a])
                        else:
                            val = w(g, -1 - 6 * first[a])
                        E += val
                        if N == 32: by_case[case] += val
        expct[N] += E
    for N in (2, 3, 4):
        if len(first) >= N:
            pat = tuple(first[k] - first[0] for k in range(1, N))
            patterns[N][pat] += 1
            if all(plugged[:N]): full[N][pat] += 1

print(f"Claim 1 (exact reuse), twins to {PFULL}:")
for N in (16, 32):
    print(f"   first {N}: observed reuse by least factors {obs_lpf[N]}, by all factors > x {obs_all[N]}; exact expectation {expct[N]:.1f}")
print("   expectation by case (first 32): " + ", ".join(f"{k}: {v:.1f}" for k, v in by_case.items()))
print("Claim 3 (blocks):")
for N in (2, 3, 4):
    never = [(pat, n) for pat, n in patterns[N].items() if n >= 100 and full[N][pat] == 0]
    tot = sum(patterns[N].values()); fully = sum(full[N].values())
    print(f"   N={N}: patterns {len(patterns[N])}, occurrences {tot}, fully plugged {fully} ({fully/tot:.3f}); patterns with >= 100 occurrences never fully plugged: {never[:5]}")

# Claim 2 (steered, to PMAX)
print("Claim 2 (steered universal list):")
for Q in (35, 385):
    idxs = []; bad = 0
    for Pm in twin_lowers:
        s = Pm + 1
        if s % Q: continue
        s, s2, x, bo = base_open_sorted(Pm)
        # universal list check at the gears dividing Q
        for j in bo[:200]:
            for g in (5, 7, 11):
                if Q % g == 0:
                    u = pow(6, -1, g)
                    if int(j) % g in (u, (g - u) % g): bad += 1
        for i, j in enumerate(bo):
            j = int(j)
            if isprime(s2 + 6 * j - 1) and isprime(s2 + 6 * j + 1): idxs.append(i); break
    idxs = np.array(idxs)
    print(f"   Q={Q}: steered twins {len(idxs)}; universal-list violations {bad}; first-twin index mean {idxs.mean():.2f} (se {idxs.std()/math.sqrt(len(idxs)):.2f}), max {idxs.max()}")
