"""Round 24 (mechanic): memory-lean realised gap m-tuple dictionary.

Same object as research/gap_tuples.py, but the "seen" set is a PYTHON SET of
packed int keys instead of a 2^(7m) boolean array (268 MB at m = 4).  The
dictionaries have ~1e5-1e6 entries, so the set costs ~10-80 MB and the tool
runs on a box with a few hundred MB free - which is what killed the round-23
six-worker machine-37 run (0.4 GB free of 15.6 GB, other lanes' jobs).

Asserts the opening count against prod_{5<=q<=y}(q-2) and the maximal gap
against the known F(y) exactly as the round-23 tool does.

usage: uv run python research/gap_tuples_lean.py Y M [SEG] [OUT.csv]
"""
import sys
import time
from math import prod
import numpy as np

Y = int(sys.argv[1])
M = int(sys.argv[2]) if len(sys.argv) > 2 else 4
SEG = int(sys.argv[3]) if len(sys.argv) > 3 else 20_000_000
OUT = (sys.argv[4] if len(sys.argv) > 4
       else f"research/data/gap_tuples_{Y}_{M}.csv")
BITS = 7
F_KNOWN = {13: 11, 17: 18, 19: 25, 23: 34, 29: 43, 31: 58, 37: 88, 41: 91}


def primes_upto(n):
    s = np.ones(n + 1, bool)
    s[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]:
            s[i * i::i] = False
    return [int(p) for p in np.flatnonzero(s)]


GEARS = [p for p in primes_upto(Y) if p >= 5]
P = prod(GEARS)
NOPEN = prod(q - 2 for q in GEARS)
print(f"machine {Y}: gears {GEARS}", flush=True)
print(f"  period {P:,}, openings (closed form) {NOPEN:,}, m={M}", flush=True)

seen = set()
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
    if first is None and len(op):
        first = op[:M + 1].copy()
    op = np.concatenate([tail, op])
    gaps = np.diff(op)
    if len(gaps):
        maxgap = max(maxgap, int(gaps.max()))
    if len(gaps) >= M:
        n = len(gaps) - M + 1
        key = np.zeros(n, np.int64)
        for t in range(M):
            key |= gaps[t:n + t] << (BITS * t)
        seen.update(np.unique(key).tolist())
    tail = op[-M:].copy() if len(op) >= M else op.copy()
    if hi == P or (hi // SEG) % 100 == 0:
        print(f"  {hi:,}/{P:,}  openings {nop:,}  maxgap {maxgap}  "
              f"distinct {len(seen):,}  t={time.time()-t0:.0f}s", flush=True)
    lo = hi

op = np.concatenate([tail, first + P])          # cyclic wrap
gaps = np.diff(op)
if len(gaps) >= M:
    n = len(gaps) - M + 1
    key = np.zeros(n, np.int64)
    for t in range(M):
        key |= gaps[t:n + t] << (BITS * t)
    seen.update(np.unique(key).tolist())
maxgap = max(maxgap, int(gaps.max()))

assert nop == NOPEN, (nop, NOPEN, "opening count != closed form")
if Y in F_KNOWN:
    assert maxgap == F_KNOWN[Y], (maxgap, F_KNOWN[Y], "max gap != known F")
keys = np.array(sorted(seen), dtype=np.int64)
tup = np.stack([(keys >> (BITS * t)) & ((1 << BITS) - 1)
                for t in range(M)], 1)
print(f"\nDONE {time.time()-t0:.0f}s. openings {nop:,} (asserted); "
      f"F({Y}) = {maxgap} (asserted); realised {M}-tuples {len(keys):,}")
with open(OUT, "w") as f:
    f.write(",".join(f"g{t+1}" for t in range(M)) + "\n")
    for row in tup:
        f.write(",".join(str(int(x)) for x in row) + "\n")
print(f"  wrote {OUT}")
for m2 in range(1, M):
    sub = set()
    for t in range(M - m2 + 1):
        sub |= set(map(tuple, tup[:, t:t + m2].tolist()))
    print(f"  induced {m2}-tuple dictionary (all contiguous windows): "
          f"{len(sub):,}")
