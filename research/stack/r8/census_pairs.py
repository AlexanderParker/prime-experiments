"""Residue-mod-5 pair test on the census: (j of rung s' of s, nearest-rung offset of s'), 2441 pairs."""
import os, math
import numpy as np
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__))
D = np.genfromtxt(os.path.join(HERE, "rungs_0_3000.csv"), delimiter=",", names=True)
s = D["s"].astype(np.int64); j = D["j"].astype(np.int64); jn = D["j_nearest"].astype(np.int64)
m = s > 12  # drop the exceptional parents 6 and 12 (s = +-1 mod 5, 7, 11, 13)
j = j[m]; jn = jn[m]; s = s[m]
n = len(j)
res = np.zeros((5, 5), dtype=int)
for u, v in zip(j % 5, jn % 5):
    res[u, v] += 1
print(f"census pairs n {n}; 5x5 table rows j mod 5, cols nearest-child-offset mod 5:")
for r in range(5):
    print("   ", res[r].tolist())
c, p, _, _ = stats.chi2_contingency(res)
print(f"  full table chi2 {c:.1f} p {p:.3g}")
A = res[np.ix_([0, 1], [0, 2, 3])]
B = res[np.ix_([2, 3, 4], [1, 3, 4])]
ca, pa, _, _ = stats.chi2_contingency(A)
cb, pb, _, _ = stats.chi2_contingency(B)
print(f"  block A rows{{0,1}}xcols{{0,2,3}}: chi2 {ca:.3f} df 2 p {pa:.3f};  block B rows{{2,3,4}}xcols{{1,3,4}}: chi2 {cb:.3f} df 4 p {pb:.3f}; outside blocks {res.sum()-A.sum()-B.sum()}")
for r in (3, 4):
    row = res[r, [1, 3, 4]]; tot = row.sum(); diag = res[r, r]
    print(f"  row {r}: allowed cols counts {row.tolist()}  diagonal {diag} of {tot}, expected {tot/3:.1f}, z {(diag - tot/3)/math.sqrt(tot*(1/3)*(2/3)):+.2f}")
sg = np.zeros((2, 2), dtype=int)
for u, v in zip(j, jn):
    sg[int(u > 0), int(v > 0)] += 1
c1, p1, _, _ = stats.chi2_contingency(sg)
print(f"  sign table {sg.tolist()} chi2 {c1:.3f} p {p1:.3f}")
for g in (5, 7, 11, 13):
    tab = {}
    for sv, jv, jnv in zip(s, j, jn):
        spv = sv * sv + 6 * jv
        key = (spv * spv) % g
        tab.setdefault(key, np.zeros(g, dtype=int))[jnv % g] += 1
    inv = pow(6, -1, g)
    bad = 0
    for key in sorted(tab):
        row = tab[key]
        forb = {(-(key - 1)) % g * inv % g, (-(key + 1)) % g * inv % g}
        bad += sum(int(row[b]) for b in forb)
        allowed = [b for b in range(g) if b not in forb]
        sub = row[allowed]
        chi = ((sub - sub.mean()) ** 2 / sub.mean()).sum()
        print(f"  g={g} child s'^2 mod g = {key:2d}: nearest-offset classes {row.tolist()} forbidden {sorted(forb)} hits {sum(int(row[b]) for b in forb)}; uniform over allowed chi2 {chi:.2f} df {len(allowed)-1} p {1-stats.chi2.cdf(chi, len(allowed)-1):.3f}")
    print(f"  g={g}: total hits in forbidden classes {bad}")
