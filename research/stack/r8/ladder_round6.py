"""Round 6 of the twin ladder (tree node R5.f.vii): the index law, the eligibility filter, the
bands and reuse.

Centre s = P+1, offsets |j| <= 2c-1, members s^2 + 6j -+ 1 = s^2 - A, s^2 - B with A = 1 - 6j,
B = -1 - 6j.  Base gears 5..x, x = floor(sqrt s); base-open offsets ordered by |j|.

Claim 1 (index law): the first twin among the base-open offsets has index i(P) <= ceil(4 ln P);
report max i, max i/ln P, and the decade maxima.
Claim 2 (eligibility): every prime factor g > x of a member s^2 - A satisfies (A | g) = 1 or A = 0
mod g (square phase); zero exceptions; the fraction of gears in (x, x^2] barred from an offset -
(A|g) = (B|g) = -1 - is about 1/4 for non-square A.
Claim 3 (bands and reuse): for the first 32 base-open offsets, how many each band (x, 2x],
(x, 4x], (x, x^1.5], (x, x^2] plugs; reuse of a plugging gear among the first 16 and 32 against the
independent prediction sum_{x<g<3 S_N} (2/g)^2 C(N, 2).

usage: uv run python ladder_round6.py [PMAX_INDEX] [PMAX_FULL]
"""
import sys, math
import numpy as np
from sympy import isprime, primefactors, jacobi_symbol

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

# Claim 1
idx = []; ratio_max = (0.0, None); dec_max = {}
for Pm in twin_lowers:
    s, s2, x, bo = base_open_sorted(Pm)
    hit = None
    for i, j in enumerate(bo):
        j = int(j)
        if isprime(s2 + 6 * j - 1) and isprime(s2 + 6 * j + 1): hit = i; break
    idx.append(hit)
    r = hit / math.log(Pm)
    if r > ratio_max[0]: ratio_max = (r, Pm, hit)
    d = int(math.log10(Pm)); dec_max[d] = max(dec_max.get(d, 0), hit)
idx = np.array(idx)
print(f"Claim 1 (index law), twins {len(twin_lowers)} to {PMAX}: max index {idx.max()}, max i/ln P = {ratio_max[0]:.3f} at P = {ratio_max[1]} (i = {ratio_max[2]}); mean index {idx.mean():.2f}")
print("   decade maxima: " + ", ".join(f"10^{d}: {m}" for d, m in sorted(dec_max.items())))
print(f"   twins with i > 4 ln P: {sum(1 for Pm, i in zip(twin_lowers, idx) if i > 4 * math.log(Pm))}")

# Claims 2 and 3 on P <= PFULL, first 32 base-open offsets
exceptions = 0; barred = []; band_counts = {"2x": [], "4x": [], "x^1.5": [], "x^2": []}
reuse = {16: 0, 32: 0}; reuse_pred = {16: 0.0, 32: 0.0}; ntw = 0
for Pm in twin_lowers:
    if Pm > PFULL: break
    s, s2, x, bo = base_open_sorted(Pm)
    first = [int(j) for j in bo[:32]]
    plugs = []  # (offset index, least prime factor > x)
    for i, j in enumerate(first):
        A = 1 - 6 * j; B = -1 - 6 * j
        for m, Aval in ((s2 - A, A), (s2 - B, B)):   # m = s^2 - A
            if not isprime(m):
                g = min(primefactors(m))
                plugs.append((i, g))
                if Aval % g != 0 and jacobi_symbol(Aval % g, g) != 1: exceptions += 1
        # barred fraction over gears in (x, x^2] for this offset (sample: gears up to min(x^2, 3000))
        if i < 4:
            cnt = 0; tot = 0
            for g in gears:
                if g <= x: continue
                if g > min(x * x, 3000): break
                tot += 1
                if A % g and B % g and jacobi_symbol(A % g, g) == -1 and jacobi_symbol(B % g, g) == -1: cnt += 1
            if tot: barred.append(cnt / tot)
    ntw += 1
    for name, y in (("2x", 2 * x), ("4x", 4 * x), ("x^1.5", int(x ** 1.5)), ("x^2", x * x)):
        band_counts[name].append(sum(1 for (i, g) in plugs if x < g <= y))
    for N in (16, 32):
        gs = [g for (i, g) in plugs if i < N]
        reuse[N] += len(gs) - len(set(gs))
        SN = 2 * abs(first[N - 1]) if len(first) >= N else 0
        pred = 0.0
        for g in gears:
            if g <= x: continue
            if g >= 3 * SN: break
            pred += (2 / g) ** 2
        reuse_pred[N] += pred * (N * (N - 1) / 2)
print(f"Claim 2 (eligibility), twins {ntw} to {PFULL}: exceptions to the square-phase law among all plugs of the first 32 base-open offsets: {exceptions}; barred fraction mean {np.mean(barred):.3f} (predicted 0.25)")
print("Claim 3 (bands): mean plugs of the first 32 base-open offsets by gears in " + ", ".join(f"(x,{k}]: {np.mean(v):.1f}" for k, v in band_counts.items()))
for N in (16, 32):
    print(f"   reuse of a plugging gear among the first {N}: observed {reuse[N]}, independent prediction {reuse_pred[N]:.1f}")
