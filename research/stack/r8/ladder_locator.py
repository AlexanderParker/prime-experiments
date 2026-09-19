"""The rung in centre coordinates (tree node R5.f.iii, lane round 3).

s = P + 1 = 6c; centre member P(P+2) = s^2 - 1; offset j has members s^2 + 6j -+ 1; the new
centre is s' = s^2 + 6j.  Gear g <= P strikes offset j iff s^2 + 6j = +-1 (mod g).  Since column c
is a twin, s != +-1 mod g for every gear g < P, so s^2 != 1; g strikes the centre iff g | s^2 + 1.
Inheritance: if j = 0 (mod g) and g does not divide s^2 + 1 then g misses offset j.

Claim 1 (locator): with y maximal such that prod_{5<=g<=y} g <= c/50, D = {g <= y : g | s^2+1},
M = prod of the gears <= y outside D, L = {j : |j| <= 2c-1, j = 0 mod M, j != 0, 2u mod g for g in D}
contains a twin centre for every twin lower P <= PMAX.  Also reported at fixed y = 7 and y = 13.
Claim 2 (clean rung): for P <= P2MAX the stretch holds a twin centre s' with no gear <= 13
dividing s'^2 + 1; report the largest clean y available.
Claim 3 (exact count): T(c) / (T0(c) prod_{sqrt P < g <= P} (1 - 2/g)) >= 1/2, T0 the offsets
surviving the gears <= sqrt P, T the twins.

usage: uv run python ladder_locator.py [PMAX] [P2MAX]
"""
import sys, math
import numpy as np
from sympy import isprime

PMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 10**6
P2MAX = int(sys.argv[2]) if len(sys.argv) > 2 else 10**5

def primes_upto(n):
    s = np.ones(n + 1, dtype=bool); s[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]: s[i*i::i] = False
    return np.nonzero(s)[0]

P = primes_upto(PMAX + 2)
ps = set(int(x) for x in P)
gears = [int(g) for g in P if g >= 5]
twin_lowers = [int(x) for x in P if x >= 5 and (int(x) + 2) in ps]

def locator_run(Pm, y):
    s = Pm + 1; c = s // 6; s2 = s * s
    gy = [g for g in gears if g <= y]
    D = [g for g in gy if (s2 + 1) % g == 0]
    M = 1
    for g in gy:
        if g not in D: M *= g
    jmax = 2 * c - 1
    cands = []
    kmax = jmax // M
    for k in range(-kmax, kmax + 1):
        j = k * M
        ok = True
        for g in D:
            u = pow(6, -1, g)
            if j % g == 0 or j % g == (2 * u) % g: ok = False; break
        if ok: cands.append(j)
    cands.sort(key=abs)
    for idx, j in enumerate(cands):
        if isprime(s2 + 6 * j - 1) and isprime(s2 + 6 * j + 1):
            return len(cands), idx, j, len(D)
    return len(cands), None, None, len(D)

# Claim 1
print("Claim 1 (locator):")
for mode in ("rule", 7, 13):
    fails = []; vac = 0; sizes = []; idxs = []
    for Pm in twin_lowers:
        c = (Pm + 1) // 6
        if mode == "rule":
            y = 0; prod = 1
            for g in gears:
                if prod * g <= c / 50: prod *= g; y = g
                else: break
        else: y = mode
        n, idx, j, nD = locator_run(Pm, y)
        if n == 0: vac += 1; continue
        sizes.append(n)
        if idx is None: fails.append((Pm, y, n, nD))
        else: idxs.append(idx)
    print(f"  y = {mode}: twins {len(twin_lowers)}, vacuous (empty L) {vac}, failures {len(fails)} {fails[:12]}")
    if sizes: print(f"     |L| min/median/max {min(sizes)}/{int(np.median(sizes))}/{max(sizes)}; first-success index mean {np.mean(idxs):.2f} max {max(idxs)}")

# Claims 2 and 3 on the smaller range with a full sieve of each stretch
print("\nClaims 2 and 3 (full sieve of each twin's stretch):")
fail2 = []; ymax_list = []; ratios = []; worst = (9.0, None)
small = [Pm for Pm in twin_lowers if Pm <= P2MAX]
for Pm in small:
    s = Pm + 1; c = s // 6; s2 = s * s; jmax = 2 * c - 1
    n = 2 * jmax + 1
    lowmem = np.arange(-jmax, jmax + 1, dtype=np.int64) * 6 + (s2 - 1)   # lower members
    struck_small = np.zeros(n, dtype=bool); struck_all = np.zeros(n, dtype=bool)
    sq = math.isqrt(Pm)
    for g in gears:
        if g > Pm: break
        r1 = (-(s2 - 1)) % g   # 6j = 1 - s^2  -> lower member divisible
        # lower member s2 + 6j - 1 = 0 mod g  <=> 6j = 1 - s2 ;  upper: 6j = -1 - s2
        u = pow(6, -1, g)
        jl = (u * (1 - s2)) % g; ju = (u * (-1 - s2)) % g
        for jr in (jl, ju):
            start = (jr - (-jmax)) % g
            struck_all[start::g] = True
            if g <= sq: struck_small[start::g] = True
    T0 = int((~struck_small).sum()); T = int((~struck_all).sum())
    prod = 1.0
    for g in gears:
        if g > Pm: break
        if g > sq: prod *= (1 - 2 / g)
    ratio = T / (T0 * prod) if T0 else float('nan')
    ratios.append(ratio)
    if ratio < worst[0]: worst = (ratio, Pm)
    # claim 2: twin centres s' = s2 + 6j, clean y
    twins_j = np.nonzero(~struck_all)[0] - jmax
    best_y = 0
    for j in twins_j:
        sp = s2 + 6 * int(j)
        v = sp * sp + 1
        yy = 0
        for g in gears:
            if v % g == 0: break
            yy = g
            if g > 200: break
        best_y = max(best_y, yy)
    ymax_list.append(best_y)
    if best_y < 13: fail2.append((Pm, best_y))
print(f"  Claim 2: twins {len(small)}; no clean rung at y = 13: {fail2[:12]} (count {len(fail2)}); largest clean y: min {min(ymax_list)}, median {int(np.median(ymax_list))}")
print(f"  Claim 3: ratio T / (T0 prod(1-2/g)): min {worst[0]:.3f} at P = {worst[1]}; mean {np.nanmean(ratios):.3f}; 1st pct {np.nanpercentile(ratios, 1):.3f}")
