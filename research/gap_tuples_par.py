"""Round 23 (mechanic, for Constructor): parallel worker for the realised gap
m-tuple dictionary (see research/gap_tuples.py for the single-process version).

Each worker owns the slot range [LO, HI) and emits every m-tuple whose FIRST
opening lies in its range, reading a little past HI so boundary tuples are
complete.  The union over workers is therefore exact and disjointly attributed.

usage:
  worker : uv run python research/gap_tuples_par.py Y M LO HI OUT.npy
  merge  : uv run python research/gap_tuples_par.py Y M merge OUT.csv f1.npy f2.npy ...
"""
import sys, time
from math import prod
import numpy as np

Y = int(sys.argv[1])
M = int(sys.argv[2])
BITS = 7
SEG = 40_000_000
OVER = 400          # slots read past HI: comfortably > M+1 openings


def primes_upto(n):
    s = np.ones(n + 1, bool); s[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]: s[i * i::i] = False
    return [int(p) for p in np.flatnonzero(s)]


GEARS = [p for p in primes_upto(Y) if p >= 5]
P = prod(GEARS)
NOPEN = prod(q - 2 for q in GEARS)


def openings(lo, hi):
    ex = np.zeros(hi - lo, bool)
    for g in GEARS:
        u = pow(6, -1, g)
        ex[(u - lo) % g::g] = True
        ex[((-u) - lo) % g::g] = True
    return np.flatnonzero(~ex).astype(np.int64) + lo


def keys_of(op):
    if len(op) < M + 1:
        return np.empty(0, np.int64)
    g = np.diff(op)
    n = len(g) - M + 1
    k = np.zeros(n, np.int64)
    for t in range(M):
        k |= g[t:n + t] << (BITS * t)
    return k


if sys.argv[3] == 'merge':
    out = sys.argv[4]
    seen = np.zeros(1 << (M * BITS), bool)
    nop = 0
    mx = 0
    for f in sys.argv[5:]:
        d = np.load(f, allow_pickle=False)
        seen[d[2:]] = True
        nop += int(d[0]); mx = max(mx, int(d[1]))
    assert nop == NOPEN, (nop, NOPEN, "opening count != closed form")
    ks = np.flatnonzero(seen)
    tup = np.stack([(ks >> (BITS * t)) & ((1 << BITS) - 1) for t in range(M)], 1)
    print(f"machine {Y}: openings {nop:,} (= closed form, asserted), "
          f"F({Y}) = {mx}")
    print(f"  realised {M}-tuples: {len(ks):,}")
    vals = sorted(set(int(v) for v in tup.ravel()))
    print(f"  distinct gap values appearing: {len(vals)} "
          f"(min {vals[0]}, max {vals[-1]})")
    with open(out, 'w') as fh:
        fh.write(",".join(f"g{t+1}" for t in range(M)) + "\n")
        for row in tup:
            fh.write(",".join(str(int(x)) for x in row) + "\n")
    print(f"  wrote {out}")
    for m2 in range(1, M):
        sub = np.unique(tup[:, :m2], axis=0)
        print(f"  induced {m2}-tuple dictionary: {len(sub):,}")
    sys.exit()

LO, HI, OUT = int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
t0 = time.time()
seen = np.zeros(1 << (M * BITS), bool)
nop = 0
mx = 0
tail = np.empty(0, np.int64)
lo = LO
while lo < HI:
    hi = min(lo + SEG, HI)
    op = openings(lo, hi)
    nop += len(op)
    if len(op):
        mx = max(mx, int(np.diff(np.concatenate([tail, op])).max())
                 if len(tail) or len(op) > 1 else mx)
    op = np.concatenate([tail, op])
    k = keys_of(op)
    if len(k):
        seen[k] = True
    tail = op[-M:].copy() if len(op) >= M else op.copy()
    lo = hi
# read past HI so boundary tuples are complete (wrapping at P)
ext = []
need = HI
while len(ext) < M + 1 and need < HI + 20 * OVER:
    seg = openings(need % P, min(need % P + OVER, P))
    ext.extend(int(x) + (need - need % P) for x in seg)
    need += OVER
op = np.concatenate([tail, np.array(ext[:M + 1], np.int64)])
k = keys_of(op)
if len(k):
    seen[k] = True
ks = np.flatnonzero(seen).astype(np.int64)
np.save(OUT, np.concatenate([np.array([nop, mx], np.int64), ks]))
print(f"[{LO:,},{HI:,}) openings {nop:,} maxgap {mx} tuples {len(ks):,} "
      f"in {time.time()-t0:.0f}s -> {OUT}", flush=True)
