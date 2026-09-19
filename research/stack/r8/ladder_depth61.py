"""Fixed-depth rung (tree node R5.f.xi): base-open at depth 61, where the record F(61) = 179 is exact,
so the supply lemma is unconditional: the window of 4c - 1 columns holds >= (4c - 1)/180 base-open
offsets.  Measures the first-twin index among the 61-rough offsets and its law in ln s.
usage: uv run python ladder_depth61.py [PMAX]
"""
import sys, math
import numpy as np
from sympy import isprime
PMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 200000
def primes_upto(n):
    s = np.ones(n + 1, dtype=bool); s[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]: s[i*i::i] = False
    return np.nonzero(s)[0]
P = primes_upto(PMAX + 2); ps = set(int(x) for x in P)
gears61 = [int(g) for g in P if 5 <= g <= 61]
tl = [int(x) for x in P if x >= 67 and (int(x) + 2) in ps]
idx = []; ratio = []; dec = {}
for Pm in tl:
    s = Pm + 1; c = s // 6; s2 = s * s; jmax = 2 * c - 1; n = 2 * jmax + 1
    struck = np.zeros(n, dtype=bool)
    for g in gears61:
        u = pow(6, -1, g)
        for jr in ((u * (1 - s2)) % g, (u * (-1 - s2)) % g):
            struck[(jr + jmax) % g::g] = True
    bo = np.nonzero(~struck)[0] - jmax
    bo = bo[np.argsort(np.abs(bo), kind='stable')]
    G = (4 * c - 1) // 180
    for i, j in enumerate(bo):
        j = int(j)
        if isprime(s2 + 6 * j - 1) and isprime(s2 + 6 * j + 1):
            idx.append(i); ratio.append(i / math.log(s) ** 2); d = int(math.log10(Pm)); dec.setdefault(d, []).append(i); break
idx = np.array(idx); ratio = np.array(ratio)
print(f"twins {len(tl)} (67 <= P <= {PMAX}); depth 61 (F = 179 exact): first-twin index among 61-rough offsets mean {idx.mean():.2f}, max {idx.max()}; i/(ln s)^2 mean {ratio.mean():.4f} max {ratio.max():.4f}")
for d in sorted(dec): v = np.array(dec[d]); print(f"   P ~ 10^{d}: n={len(v)} mean index {v.mean():.2f} max {v.max()}  mean i/(ln s)^2 {(v/ (math.log(10**d)**2)).mean():.4f}")
