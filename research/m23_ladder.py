"""Round 21 (mechanic): machine-23 qualifying ladder for Formalist's 23->29 step.
Direct full-period scan (period 5*7*11*13*17*19*23 = 37,182,145 slots), cyclic.
Outputs: F_j(23) j=1..8, Q_j(23; a=10) j=3..8, and the no_big_run analogue:
max run of consecutive gaps all >= 10 (Formalist's Machine19Q recipe needs it).
Asserts against r17 census values (C13 table)."""
import numpy as np
from math import prod

gears = [5, 7, 11, 13, 17, 19, 23]
P = prod(gears)
ex = np.zeros(P, bool)
for g in gears:
    u = pow(6, -1, g)
    ex[u % g::g] = True
    ex[(-u) % g::g] = True
op = np.flatnonzero(~ex).astype(np.int64)
n = len(op)
# cyclic gaps: wrap the first opening around the period
ops = np.concatenate([op, op[:9] + P])
d = np.diff(ops).astype(np.int64)
gaps = d[:n]  # n cyclic gaps
F = int(gaps.max())
print(f"machine 23: period {P:,}, openings {n:,}, F = {F}")
assert F == 34

# F_j ladder j=1..8 (cyclic: window sums over the wrapped array)
csum = np.concatenate([[0], np.cumsum(d)])
Fj = []
for j in range(1, 9):
    w = csum[j:] - csum[:-j]
    Fj.append(int(w[:n].max()))
print("F_j (j=1..8):", Fj)
assert Fj[:6] == [34, 39, 50, 58, 65, 77], Fj

# Q_j(23; a=10): max sum of j consecutive gaps whose j-2 MIDDLE gaps all >= 10
a = 10
big = (d >= a)
Qj = {}
for j in range(3, 9):
    # middles: positions i+1 .. i+j-2 all big
    mid_ok = np.ones(n, bool)
    for m in range(1, j - 1):
        mid_ok &= big[m:m + n]
    w = (csum[j:] - csum[:-j])[:n]
    vals = w[mid_ok]
    Qj[j] = int(vals.max()) if len(vals) else 0
print("Q_j(23; a=10) j=3..8:", [Qj[j] for j in range(3, 9)])
assert Qj[3] == 43 and Qj[4] == 50 and Qj[5] == 55 and Qj[6] == 60 and Qj[7] == 0 and Qj[8] == 0, Qj  # r17 C13 row was wrong; addresses verified

# no_big_run: longest run of consecutive gaps all >= 10 (cyclic)
runs = np.zeros(n + 9, np.int64)
cur = 0; best = 0
bigall = (np.concatenate([gaps, gaps[:9]]) >= a)
for b in bigall:
    cur = cur + 1 if b else 0
    best = max(best, cur)
print(f"max run of consecutive gaps >= {a}: {best}")
# count of length-3 runs of big gaps (the Q_5 suppliers)
b3 = big[:n] & big[1:n+1] & big[2:n+2]
print(f"# length-3 big-gap runs (cyclic positions): {int(b3.sum())}")
# criterion vs q'=29: max_j Q_j <= F + 29?
mq = max(Qj.values())
print(f"max_j Q_j = {mq}  vs  F + 29 = {F + 29}  margin {F + 29 - mq}")
print("ALL ASSERTS PASSED")
