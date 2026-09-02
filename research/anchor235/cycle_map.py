import sys; sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), ".."))
import numpy as np
from math import prod, log
from collections import Counter
from word_tree_r29 import spf_sieve
X = 100_000_000
spf = spf_sieve(X + 40); isp = spf == np.arange(len(spf)); isp[:2] = False
J = X // 30; j = np.arange(J); base = 30 * j
RES = (11, 13, 17, 19, 29, 31)
full = np.flatnonzero(np.all(np.stack([isp[base + r] for r in RES], 1), 1))
OPEN30 = {1, 11, 13, 17, 19, 29}
primes = [int(p) for p in np.flatnonzero(isp[:20000]) if p >= 7]
def forb(q):  # cycle indices (mod q) hit by gear q: cycle of q*m for the six open m
    return sorted({((q * m - 11) // 30) % q for m in range(1, 30) if (q * m) % 30 in OPEN30})
print("forbidden cycle residues j mod q per gear (cycle j = numbers 30j+11 .. 30j+31, three twin slots):")
for q in primes[:8]:
    f = forb(q); print(f"  q={q:>2}: j mod {q} must avoid {f}  -> allowed {q - len(f)} of {q}")
# verify: every full survivor avoids forbidden residues of every gear q <= sqrt(30j+31); and the converse over the period of 7,11,13
ok = all(all((jj % q) not in forb(q) for q in primes if q * q <= 30 * jj + 31) for jj in full)
print(f"all {len(full)} survivors satisfy the residue rule for every gear up to sqrt: {ok}")
# candidates after first gears
for upto in (7, 11, 13, 17, 19, 23):
    gs = [q for q in primes if q <= upto]
    cand = np.ones(J, dtype=bool)
    for q in gs:
        cand &= ~np.isin(j % q, forb(q))
    print(f"  cycles allowed by gears up to {upto}: {int(cand.sum())} of {J} (fraction {cand.mean():.4f} = prod (1 - f/q) {prod(1 - len(forb(q))/q for q in gs):.4f}); survivors among them {len(full)}")
# where they sit: residues of survivors mod 7, 11, 13
for q in (7, 11, 13):
    print(f"  survivors j mod {q}: {dict(sorted(Counter((full % q).tolist()).items()))}")
d = np.diff(full)
print(f"spacings between consecutive surviving cycles (in cycles): min {d.min()}, median {np.median(d):.0f}, max {d.max()}")
print("  spacing mod 7: " + str(dict(sorted(Counter((d % 7).tolist()).items()))) + f"   (residue 0 forced when both survivors share j mod 7)")
print("  spacings divisible by 7: " + f"{int((d % 7 == 0).sum())} of {len(d)}; by 77: {int((d % 77 == 0).sum())}; by 1001: {int((d % 1001 == 0).sum())}")
print("  first survivors and spacings: " + ", ".join(f"{int(a)}(+{int(b)})" for a, b in zip(full[:15], d[:15])))
# growth: count per block vs sieve density sum over cycles of prod_{7<=q<=sqrt(30j)} (1-6/q) (5/7 for q=7)
blocks = [(0, 10**6), (10**6, 10**7), (10**7, 3*10**7), (3*10**7, 6*10**7), (6*10**7, 10**8)]
print("growth (numbers block: survivors, sum of sieve density, ratio):")
for a, b in blocks:
    lo, hi = a // 30, b // 30
    cnt = int(((full >= lo) & (full < hi)).sum())
    # density at cycle j: prod over gears q <= sqrt(30j)
    js = np.arange(max(lo, 1), hi, 997)
    dens = []
    for jj in js:
        gs = [q for q in primes if q * q <= 30 * jj]
        dens.append(prod(1 - len(forb(q)) / q for q in gs))
    est = float(np.mean(dens)) * (hi - lo)
    print(f"  [{a:>9}, {b:>9}): {cnt:>4}  density-sum {est:8.2f}  ratio {cnt / est:.2f}")
bad = [int(jj) for jj in full if not all((jj % q) not in forb(q) for q in primes if q * q <= 30 * jj + 31)]
print(f"survivors breaking the residue rule: {bad}  (the gear's own prime: 11 and 13 sit in cycle 0)")
