"""Plug share by base level: share(B) = 1 - T(s)/U_B(s), U_B = base-open columns of the stretch at
level B, T = twins. Derived: share(B) ~ 1 - 0.79 ln^2 B / ln^2 s (twin density per column 6 x 1.32/(4 ln^2 s) over
base-open density prod_{g<=B}(1-2/g) ~ 2.5/ln^2 B). Verify at three twin centres for
B = s^0.234, s^0.35, s^0.5, s^0.65, s^0.8; report measured share and the derived value with the exact
product prod(1-2/g) instead of its asymptotic.
"""
import numpy as np, sys, math
S_LIST = [1302, 2082, 2970]
N = (max(S_LIST) + 1) ** 2 + 2
spf = np.zeros(N, dtype=np.int32)
for p in range(2, int(N ** 0.5) + 1):
    if spf[p] == 0:
        blk = spf[p*p::p]; blk[blk == 0] = p; spf[p*p::p] = blk
idx = np.nonzero(spf == 0)[0]; spf[idx] = idx; spf[0] = spf[1] = 1
primes = [p for p in range(5, 4000) if spf[p] == p]
print("   s   exponent    B   base-open  twins  share  derived(exact product)  derived(asymptotic)")
for s in S_LIST:
    c = s // 6; js = np.arange(-(2 * c - 1), 2 * c)
    lo = s * s + 6 * js - 1; hi = lo + 2
    twin = int(((spf[lo] == lo) & (spf[hi] == hi)).sum())
    for e in [0.234, 0.35, 0.5, 0.65, 0.8]:
        B = int(s ** e)
        rough = (spf[lo] > B) & (spf[hi] > B)
        U = int(rough.sum())
        prodB = math.prod(1 - 2 / g for g in primes if g <= B)
        derived_exact = 1 - (6 * 1.3203 / (4 * math.log(s) ** 2)) / prodB
        derived_asym = 1 - 0.79 * math.log(B) ** 2 / math.log(s) ** 2
        print(f"{s:5d}   {e:.3f}   {B:4d}   {U:9d}  {twin:5d}  {1 - twin/U:.3f}        {derived_exact:.3f}                 {derived_asym:.3f}")
