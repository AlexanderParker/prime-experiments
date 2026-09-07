"""The earliest turn an ember-free opening of length L can hold a twin slot, exactly from the
smooth vectors: an ember-free opening of length L >= 2 in turn t is a block of L + 2 consecutive
integers each s_i x P_i with P_i > Q, so t >= max s_i; the smallest possible max s_i over all
blocks of L + 2 consecutive integers (searched over starts n <= NMAX; the extremal residue
patterns repeat with a small period) is the onset threshold, computed separately for blocks that
hold a slot pair (two members coprime to q# at distance 2, both at positions 0..L-1 and 2..L+1)
and for blocks that hold none.  For L = 1 the object is the pair {n, n+2} alone.

usage: uv run python research/valves/r4/onset_threshold.py q [NMAX]
"""
import sys, math
import numpy as np

q = int(sys.argv[1]); NMAX = int(sys.argv[2]) if len(sys.argv) > 2 else 20_000_000
engine = [p for p in [2, 3, 5, 7, 11, 13] if p <= q]; qsharp = math.prod(engine)
qprime = {5: 7, 7: 11, 11: 13, 13: 17}[q]
m = np.arange(NMAX + 16, dtype=np.int64)
rest = m.copy(); rest[0] = 1
for p in engine:
    while True:
        d = rest % p == 0
        if not d.any():
            break
        rest[d] //= p
s = m // np.maximum(rest, 1); s[0] = 1
coprime = rest == m               # s == 1
print(f"q = {q}, q# = {qsharp}, q' = {qprime}, search n <= {NMAX}")
for L in range(1, qprime - 2):
    if L == 1:
        mx = np.maximum(s[1:NMAX], s[3:NMAX + 2])
        slot = coprime[1:NMAX] & coprime[3:NMAX + 2]
    else:
        k = L + 2
        mx = s[1:NMAX].copy()
        for i in range(1, k):
            mx = np.maximum(mx, s[1 + i:NMAX + i])
        slot = np.zeros(NMAX - 1, dtype=bool)
        for i in range(L):
            slot |= coprime[1 + i:NMAX + i] & coprime[3 + i:NMAX + 2 + i]
    a = int(mx[slot].min()) if slot.any() else None
    b = int(mx[~slot].min()) if (~slot).any() else None
    na = int(np.flatnonzero(slot & (mx == a))[0]) + 1 if a is not None else None
    nb = int(np.flatnonzero(~slot & (mx == b))[0]) + 1 if b is not None else None
    va = ([int(s[na]), int(s[na + 2])] if L == 1 else s[na:na + L + 2].tolist()) if na else None
    vb = ([int(s[nb]), int(s[nb + 2])] if L == 1 else s[nb:nb + L + 2].tolist()) if nb else None
    print(f"L = {L}: onset turn of a slot-holding opening >= {a} (block at n = {na}, vector {va}); of a slot-free opening >= {b} (n = {nb}, vector {vb})")
