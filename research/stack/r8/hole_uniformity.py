"""Statement (a) of draft 5i at small scale: for machine B (gears 5..B), h_B(L) = the least number of
holes in any window of length L over one period, against the mean delta L (delta = hole density).
Windows L = B^2/6, B^2/3, B^2 (the scales of E4, of the stretch at level B, and of the full square).
Pre-registered: the worst window holds at least a quarter of the mean at every scale tested and the
ratio worst/mean rises with L (uniformity improves with scale).
"""
import numpy as np
from math import prod

def inv(a, m): return pow(a, -1, m)
def teeth(g):
    c = inv(6, g); return sorted({c % g, (-c) % g})
def paint(gears, P):
    painted = np.zeros(P, dtype=np.int8)
    for g in gears:
        for t in teeth(g): painted[t::g] = 1
    return painted

primes = [5, 7, 11, 13, 17, 19, 23]
print("  B   P        density   L   mean-holes  worst  worst/mean")
for k in range(3, 8):
    gears = primes[:k]; B = gears[-1]; P = prod(gears)
    hole = (1 - paint(gears, P)).astype(np.int32); delta = hole.sum() / P
    Lmax = B * B
    pref = np.concatenate([[0], np.cumsum(np.concatenate([hole, hole[:Lmax]]))])
    for L in [B * B // 6, B * B // 3, B * B]:
        worst = int((pref[L:L + P] - pref[:P]).min())
        print(f"{B:3d} {P:9d}  {delta:.4f}  {L:4d}   {delta*L:8.2f}  {worst:5d}   {worst/(delta*L):.3f}")
