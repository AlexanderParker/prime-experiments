"""The thin-band bound (draft section 5, E4e): F(q) <= max{ L : 2 sum_{top} ceil(L/g) >= h_B(L) },
base B = gears 5..B held exactly (h_B(L) = the least number of base holes in any window of length
L, from one period), top = gears in (B, q] bounded by the union bound only.
Verification of the derived numbers: base 17 -> q = 19, 23; base 23 -> q = 29, 31.
Pre-registered: the bound is finite whenever 2 sum_{top} 1/g < base hole density; it exceeds the
true record by a factor 1.5-2.5 at these sizes; it is vacuous once the top band is thick.
"""
from math import prod, ceil
import numpy as np

def inv(a, m): return pow(a, -1, m)
def teeth(g):
    c = inv(6, g); return sorted({c % g, (-c) % g})
def paint(gears, P):
    painted = np.zeros(P, dtype=np.int8)
    for g in gears:
        for t in teeth(g):
            painted[t::g] = 1
    return painted

def h_B(painted, Lmax):
    """least number of holes in any cyclic window of length L, L = 1..Lmax (numpy prefix sums)"""
    hole = (1 - painted).astype(np.int32)
    P = len(hole)
    pref = np.concatenate([[0], np.cumsum(np.concatenate([hole, hole[:Lmax]]))])
    out = {}
    for L in range(1, Lmax + 1):
        out[L] = int((pref[L:L + P] - pref[:P]).min())
    return out

cases = [(11, [13], {13: 10}), (13, [17], {17: 17}), (17, [19], {19: 24}), (19, [23], {23: 33}), (23, [29], {29: 42})]
for B, qs, truth in cases:
    base = [g for g in [5, 7, 11, 13, 17, 19, 23] if g <= B]
    P = prod(base); pat = paint(base, P)
    density = 1 - pat.sum() / P
    hb = h_B(pat, 160)
    for q in qs:
        top = [g for g in [7, 11, 13, 17, 19, 23, 29, 31] if B < g <= q]
        cap = 2 * sum(1 / g for g in top)
        bound = max((L for L in range(1, 161) if 2 * sum(ceil(L / g) for g in top) >= hb[L]), default=None)
        print(f"base 5..{B} (hole density {density:.4f}), top {top} (2 sum 1/g = {cap:.4f}): "
              f"F({q}) <= {bound}; true F({q}) = {truth[q]}; h_B at 20,30,40,50,60: {[hb[L] for L in (20,30,40,50,60)]}", flush=True)
