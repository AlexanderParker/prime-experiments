"""Round 23 (mechanic, for Constructor): the REALISED GAP m-TUPLE DICTIONARY.

Constructor's abstraction A_m keeps the last m-1 gap VALUES as state; its exact
certificate input is the set of realised m-tuples of consecutive gaps of the
machine.  This computes that set at full period by a segmented sieve, with the
tuples packed into a single integer key so the dedup is a scatter write rather
than a sort.

Key packing: gaps at these machines are < 128, so a 4-tuple packs into 28 bits
(7 bits per gap) and the "seen" set is a 2^28 boolean array (268 MB).

Validation: machine 31's dictionary is computed first and its size/contents can
be compared with Constructor's own; the opening count is asserted against the
closed form prod_{5<=q<=y} (q-2), and the maximal gap against the known F(y).

usage: uv run python research/gap_tuples.py Y M [SEG]
   e.g. research/gap_tuples.py 31 4        (validation run, ~4 min)
        research/gap_tuples.py 37 4        (the deliverable)
Writes research/data/gap_tuples_{Y}_{M}.csv (one tuple per line) and prints a
summary.
"""
import sys, time
from math import prod
import numpy as np

Y = int(sys.argv[1])
M = int(sys.argv[2]) if len(sys.argv) > 2 else 4
SEG = int(sys.argv[3]) if len(sys.argv) > 3 else 200_000_000
BITS = 7
F_KNOWN = {13: 11, 17: 18, 19: 25, 23: 34, 29: 43, 31: 58, 37: 88, 41: 91}


def primes_upto(n):
    s = np.ones(n + 1, bool); s[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]: s[i * i::i] = False
    return [int(p) for p in np.flatnonzero(s)]


GEARS = [p for p in primes_upto(Y) if p >= 5]
P = prod(GEARS)
NOPEN = prod(q - 2 for q in GEARS)
print(f"machine {Y}: gears {GEARS}", flush=True)
print(f"  period P = {P:,}; openings (closed form) = {NOPEN:,}; "
      f"m = {M}, key = {M*BITS} bits", flush=True)

seen = np.zeros(1 << (M * BITS), bool)
t0 = time.time()
tail = np.empty(0, np.int64)
first = None
nop = 0
maxgap = 0
lo = 0
while lo < P:
    hi = min(lo + SEG, P)
    ex = np.zeros(hi - lo, bool)
    for g in GEARS:
        u = pow(6, -1, g)
        ex[(u - lo) % g::g] = True
        ex[((-u) - lo) % g::g] = True
    op = np.flatnonzero(~ex).astype(np.int64) + lo
    nop += len(op)
    if first is None:
        first = op[:M + 1].copy()
    op = np.concatenate([tail, op])
    gaps = np.diff(op)
    if len(gaps):
        maxgap = max(maxgap, int(gaps.max()))
    if len(gaps) >= M:
        key = np.zeros(len(gaps) - M + 1, np.int64)
        for t in range(M):
            key |= gaps[t:len(gaps) - M + 1 + t] << (BITS * t)
        seen[key] = True
    tail = op[-(M):].copy() if len(op) >= M else op.copy()
    if (hi // SEG) % 20 == 0 or hi == P:
        print(f"  {hi:,}/{P:,}  openings {nop:,}  maxgap {maxgap}  "
              f"distinct {int(seen.sum()):,}  t={time.time()-t0:.0f}s",
              flush=True)
    lo = hi

# cyclic wrap
op = np.concatenate([tail, first + P])
gaps = np.diff(op)
if len(gaps) >= M:
    key = np.zeros(len(gaps) - M + 1, np.int64)
    for t in range(M):
        key |= gaps[t:len(gaps) - M + 1 + t] << (BITS * t)
    seen[key] = True
maxgap = max(maxgap, int(gaps.max()))

assert nop == NOPEN, (nop, NOPEN, "opening count != closed form")
if Y in F_KNOWN:
    assert maxgap == F_KNOWN[Y], (maxgap, F_KNOWN[Y], "max gap != known F")
keys = np.flatnonzero(seen)
print(f"\nDONE in {time.time()-t0:.0f}s. openings {nop:,} (= closed form, "
      f"asserted); F({Y}) = {maxgap} (asserted)", flush=True)
print(f"  realised {M}-tuples: {len(keys):,}")
tup = np.stack([(keys >> (BITS * t)) & ((1 << BITS) - 1) for t in range(M)], 1)
vals = sorted(set(int(v) for v in tup.ravel()))
print(f"  distinct gap values in tuples: {len(vals)}  min {vals[0]} max {vals[-1]}")
out = f"research/data/gap_tuples_{Y}_{M}.csv"
with open(out, "w") as f:
    f.write(",".join(f"g{t+1}" for t in range(M)) + "\n")
    for row in tup:
        f.write(",".join(str(int(x)) for x in row) + "\n")
print(f"  wrote {out}")
for m2 in range(1, M):
    sub = np.unique(tup[:, :m2], axis=0)
    print(f"  induced {m2}-tuple dictionary: {len(sub):,}")
