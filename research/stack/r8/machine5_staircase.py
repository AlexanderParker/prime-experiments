"""Staircase rule (node c.iii follow-up). Gear 13 hits holes 1..10 mod 13 once each and misses
holes 11, 12, 0 mod 13. For a gear g not 7 or 13, one class hits holes in a progression with step
7^{-1} mod g, and 7^{-1} = +-1 or +-2 mod g only for g | 6, 8, 13, 15 - so for g >= 11, g != 13, a
class hits at most one of any three consecutive holes, and the gear at most two laps of their 15.
PRE-REGISTERED (mu floor, Q4): a chain of 13 or more filled holes contains three consecutive
13-free holes, needing at least 8 gears other than 7, 13; so no chain of 13 holes for q < 41
(available: 11, 17, 19, 23, 29, 31, 37 = 7 gears at q = 37). Two consecutive 13-free holes (chain
of 12) need at least 5 such gears: q >= 29. Test: longest chain in the first 2e8 laps at
q = 31, 37 (partial period, refutation check only); verify the step claim for g <= 61.
"""
import numpy as np
for g in [11, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61]:
    s = pow(7, -1, g); assert min(s, g - s) >= 3, g
print("step 7^-1 mod g is at least 3 from 0 for every gear 11..61 except 13 (step 2): verified")
L = 200_000_000
for q, gears in [(31, [11, 13, 17, 19, 23, 29, 31]), (37, [11, 13, 17, 19, 23, 29, 31, 37])]:
    struck = np.zeros(L, dtype=bool)
    for g in gears:
        a = pow(30, -1, g); struck[a::g] = True; struck[(-a) % g::g] = True
    H = struck[5:5 + 7 * (L // 7 - 1)].reshape(-1, 7)[:, :5]
    filled = H.all(axis=1)
    f = np.concatenate(([0], filled.astype(np.int8), [0])); e = np.flatnonzero(np.diff(f)); ch = e[1::2] - e[::2]
    h0 = e[::2][ch.argmax()]
    print(f"q={q}: longest chain of filled holes in first {L} laps = {ch.max()} at hole {h0}; "
          f"13-free holes in it: {[h for h in range(h0, h0 + ch.max()) if h % 13 in (11, 12, 0)]}")
    del struck, H
