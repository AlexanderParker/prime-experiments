"""The share form with the freedom of the turn: for every Q <= 1859 (the certified ladder to F(59) = 161 covers
y_m(Q) <= 59 for m = 1), the least turn m whose share certificate F(y_m(Q)) <= c_m(Q) holds, over ALL m with
y_m(Q) <= 59; and the Q at which NO turn is share-certified although a twin exists in (Q, Q^2].
usage: uv run python research/valves/r2/share_free.py
"""
import math
import numpy as np
F = {0: 1, 5: 2, 7: 5, 11: 7, 13: 11, 17: 18, 19: 25, 23: 34, 29: 43, 31: 58, 37: 88, 41: 91, 43: 103, 47: 118, 53: 145, 59: 161}
rungs = sorted(F)
N = 61 * 61 + 10
s = np.ones(N + 1, dtype=bool); s[:2] = False
for i in range(2, int(N ** 0.5) + 1):
    if s[i]:
        s[i * i::i] = False
tw = s[:-2] & s[2:]
none = []; least = {}
for Q in range(3, 1860):
    best = None
    for m in range(1, Q):
        top = (m + 1) * Q + 2
        if top >= 61 * 61:
            break
        ym = max(y for y in rungs if y * y <= top)
        klo = (m * Q + 7) // 6; khi = ((m + 1) * Q + 1) // 6; cm = khi - klo + 1
        if F[ym] <= cm:
            best = m; break
    least[Q] = best
    if best is None:
        none.append(Q)
from collections import Counter
print(f"Q in [3, 1859]: least share-certified turn m: distribution {sorted(Counter(v for v in least.values() if v).items())}")
print(f"Q with NO share-certified turn (all m with y_m <= 59): {none}")
print("existence in the valve at those Q (a twin in (Q, 2Q] already): " + ", ".join(f"Q={Q}: {int(tw[Q + 1:2 * Q + 1].sum())} twins in (Q, 2Q]" for Q in none))
# the share form at rung 7 in detail
for Q in none:
    row = []
    for m in range(1, 6):
        top = (m + 1) * Q + 2; ym = max(y for y in rungs if y * y <= top)
        klo = (m * Q + 7) // 6; khi = ((m + 1) * Q + 1) // 6
        row.append(f"m={m}: y_m={ym} F={F[ym]} c_m={khi - klo + 1}")
    print(f"  Q={Q}: " + "; ".join(row))
