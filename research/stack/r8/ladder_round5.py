"""Round 5 of the twin ladder (tree node R5.f.vi): the two halves of the sieve-form rung.

Window of the twin P = 6c-1: offsets |j| <= 2c-1, members s^2 + 6j -+ 1, s = P + 1.  Base gears:
5..x, x = floor(sqrt s).  Base-open offsets: struck by no base gear.  Top gears: (x, P].

Claim A (base half). (i) proved: no F(x)+1 consecutive offsets all base-struck. (ii) the base-open
count within |j| <= F(x) is >= 0.3 * 2F(x) * W(x), W(x) = prod_{5<=g<=x}(1-2/g); the N-th base-open
offset by distance has |j| <= 3N/(2W(x)) for N = 8, 16, 32. (iii) the longest base-struck run in
the window is <= 0.7 x ln x.
Claim B (top half). Span S_N = 2 D_N of the first N base-open offsets; if S_N < (x-1)/3 every top
gear plugs at most one of them (min-gap); no top gear plugs two of the first N while the span
condition holds; the first twin among the base-open offsets is within the first 24.

usage: uv run python ladder_round5.py [PMAX]
"""
import sys, math
import numpy as np
from sympy import isprime, primefactors

PMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 10**5
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

def F_est(x):
    below = [g for g in FEXACT if g <= x]
    if x <= 61: return FEXACT[max(below)] if below else 1
    return int(0.7 * x * math.log(x))

worst_run = (0.0, None); min_cnt_ratio = (9.0, None); DN_ratio = {8: 0.0, 16: 0.0, 32: 0.0}
span_ok = {8: 0, 16: 0, 32: 0}; span_n = {8: 0, 16: 0, 32: 0}; double = {8: 0, 16: 0, 32: 0}
first_idx = []; distinct_used = {8: [], 16: [], 32: []}
for Pm in twin_lowers:
    s = Pm + 1; c = s // 6; s2 = s * s; jmax = 2 * c - 1; n = 2 * jmax + 1
    x = math.isqrt(s)
    struck = np.zeros(n, dtype=bool)
    W = 1.0
    for g in gears:
        if g > x: break
        W *= (1 - 2 / g)
        u = pow(6, -1, g)
        for jr in ((u * (1 - s2)) % g, (u * (-1 - s2)) % g):
            struck[(jr + jmax) % g::g] = True
    # A(iii): longest base-struck run
    run = 0; best = 0
    for v in struck:
        if v: run += 1; best = max(best, run)
        else: run = 0
    r = best / (0.7 * x * math.log(x)) if x >= 5 else 0.0
    if r > worst_run[0]: worst_run = (r, Pm)
    bopen = np.nonzero(~struck)[0] - jmax
    order = np.argsort(np.abs(bopen), kind='stable')
    bo = bopen[order]
    # A(ii): count within |j| <= F(x)
    Fx = F_est(x)
    cnt = int((np.abs(bo) <= Fx).sum())
    heur = 0.3 * 2 * Fx * W
    cr = cnt / heur if heur > 0 else 9.0
    if cr < min_cnt_ratio[0]: min_cnt_ratio = (cr, Pm)
    for N in (8, 16, 32):
        if len(bo) >= N:
            DN = abs(int(bo[N - 1]))
            DN_ratio[N] = max(DN_ratio[N], DN * 2 * W / (3 * N))
    # B: plugs on the first N base-open offsets
    firstN = bo[:32]
    plugs = []  # (offset index, gear)
    tw_index = None
    for i, j in enumerate(firstN):
        j = int(j); lo = s2 + 6 * j - 1; hi = s2 + 6 * j + 1
        plo = isprime(lo); phi = isprime(hi)
        if plo and phi and tw_index is None: tw_index = i
        for m, isp in ((lo, plo), (hi, phi)):
            if not isp:
                g = min(primefactors(m))
                plugs.append((i, g))
    if tw_index is None:
        # search beyond 32
        for i in range(32, len(bo)):
            j = int(bo[i])
            if isprime(s2 + 6 * j - 1) and isprime(s2 + 6 * j + 1): tw_index = i; break
    first_idx.append(tw_index if tw_index is not None else -1)
    for N in (8, 16, 32):
        if len(bo) < N: continue
        span = 2 * abs(int(bo[N - 1]))
        span_n[N] += 1
        if span < (x - 1) / 3:
            span_ok[N] += 1
            gs = [g for (i, g) in plugs if i < N]
            if len(gs) != len(set(gs)): double[N] += 1
            distinct_used[N].append(len(set(gs)))

fi = np.array(first_idx)
print(f"twin lowers: {len(twin_lowers)} (P <= {PMAX})")
print(f"A(iii) longest base-struck run / (0.7 x ln x): max {worst_run[0]:.3f} at P = {worst_run[1]}")
print(f"A(ii)  base-open count within |j| <= F(x) against 0.3 * 2F * W: min ratio {min_cnt_ratio[0]:.2f} at P = {min_cnt_ratio[1]}")
print(f"       D_N * 2W / (3N): max " + ", ".join(f"N={N}: {DN_ratio[N]:.2f}" for N in (8, 16, 32)))
print(f"B      first twin among base-open offsets: index mean {fi[fi>=0].mean():.2f}, max {fi.max()}, beyond 24: {(fi > 23).sum()} twins, absent: {(fi<0).sum()}")
for N in (8, 16, 32):
    du = distinct_used[N]
    print(f"       N={N}: span < (x-1)/3 at {span_ok[N]}/{span_n[N]} twins; a top gear plugging two of the first N under the span condition: {double[N]}; distinct gears used mean {np.mean(du) if du else float('nan'):.1f}")
