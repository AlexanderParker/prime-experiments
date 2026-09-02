import numpy as np
OPEN30 = {1, 11, 13, 17, 19, 29}
TYPE = {11: "A lower", 13: "A upper", 17: "B lower", 19: "B upper", 29: "C lower", 1: "C upper"}   # A=11|13 B=17|19 C=29|31
def mset(q): return [m for m in range(1, 30) if (q * m) % 30 in OPEN30]
print("(1) slot types hit by each gear per run of 30q (A = 11|13, B = 17|19, C = 29|31):")
for q in (7, 11, 13, 37, 41, 43, 47):
    print(f"  q={q:>2}: " + ", ".join(f"{q}x{m}={q*m} ({TYPE[(q*m)%30]})" for m in mset(q)))
print("  forced: q*m runs over all six residues {1,11,13,17,19,29} exactly once, so every gear hits each slot type twice per run - once on the lower number, once on the upper. No gear favours a type; the class only sets WHERE in the run.")
print()
print("(2) two gears on one slot (k = slot, numbers 6k-1 | 6k+1); per q*q' slots:")
def u(q): return pow(6, -1, q)
for q, r in ((37, 41), (37, 43), (41, 43), (7, 11), (997, 1009)):
    N = 3 * q * r
    k = np.arange(N)
    hq_lo, hq_hi = (k % q) == u(q), (k % q) == (q - u(q))     # gear q on 6k-1 / 6k+1
    hr_lo, hr_hi = (k % r) == u(r), (k % r) == (r - u(r))
    same = ((hq_lo & hr_lo) | (hq_hi & hr_hi)).sum() / 3        # same number, product q*r*m
    cross = ((hq_lo & hr_hi) | (hq_hi & hr_lo)).sum() / 3       # one number each: double kill
    both = ((hq_lo | hq_hi) & (hr_lo | hr_hi))
    ao = np.isin(k % 5, (0, 2, 3))
    first = k[both & ao][:3]
    print(f"  ({q},{r}): same-number collisions {same:.0f}, double-kill collisions {cross:.0f} per {q*r} slots; on anchor-open slots {(both & ao).sum()/3:.1f} per {q*r} (= 4*3/5); "
          f"first anchor-open collisions k={first.tolist()} -> numbers " + ", ".join(f"{6*x-1}|{6*x+1}" for x in first))
print("  forced: k = +-u mod q and k = +-u' mod q' is 4 residue classes mod q*q' - 2 where both gears sit on the same number (multiples of q*q'), 2 where they take one number each. Nothing else. Three gears share a slot only mod q*q'*q''.")
print()
print("(3) overlap budget on anchor-open slots, gears 7..Q: sum of tooth densities vs blocked fraction")
from math import prod
ps = [7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
for Q in (13, 23, 47):
    gs = [p for p in ps if p <= Q]
    s = sum(2 / g for g in gs); b = 1 - prod(1 - 2 / g for g in gs)
    print(f"  gears up to {Q}: teeth density sum {s:.3f}, blocked fraction {b:.3f}, wasted on overlap {s - b:.3f}")
