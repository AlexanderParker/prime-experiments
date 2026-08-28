"""Round 24 (mechanic): memory-lean PARALLEL realised gap m-tuple dictionary.

Round 23's gap_tuples_par.py packs a tuple into 7 bits per gap, so its "seen"
array is 2^(7m) bools = 268 MB at m = 4, and six workers need 1.6 GB.  On a
box with 0.4-1.0 GB free (the other lanes' jobs) six workers die SILENTLY -
which is exactly what happened to the round-23 machine-37 run and again on the
first round-24 attempt.

The fix is arithmetic, not scheduling: gaps are bounded by F(y), so pack in
BASE F+1 instead of base 128.  At machine 37 that is 89^4 = 62.7M bools =
63 MB instead of 268 MB, a 4.3x cut, and the scatter write is unchanged.

usage:
  worker : uv run python research/gap_tuples_lean_par.py Y M LO HI OUT.npy [SEG]
  merge  : uv run python research/gap_tuples_lean_par.py Y M merge OUT.csv f1.npy ...
Each worker emits every m-tuple whose FIRST opening lies in [LO, HI) and reads
past HI so boundary tuples are complete; the union is exact and disjoint.
"""
import sys
import time
from math import prod
import numpy as np

Y = int(sys.argv[1])
M = int(sys.argv[2])
SEGD = 20_000_000
OVER = 400
F_KNOWN = {13: 11, 17: 18, 19: 25, 23: 34, 29: 43, 31: 58, 37: 88, 41: 91}
BASE = F_KNOWN[Y] + 1


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
NKEY = BASE ** M


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
        k = k * BASE + g[t:n + t]
    return k


if sys.argv[3] == 'merge':
    out = sys.argv[4]
    seen = np.zeros(NKEY, bool)
    nop = 0
    mx = 0
    for f in sys.argv[5:]:
        d = np.load(f, allow_pickle=False)
        seen[d[2:]] = True
        nop += int(d[0])
        mx = max(mx, int(d[1]))
    assert nop == NOPEN, (nop, NOPEN, "opening count != closed form")
    assert mx == F_KNOWN[Y], (mx, F_KNOWN[Y], "max gap != known F")
    ks = np.flatnonzero(seen)
    tup = np.stack([(ks // BASE ** (M - 1 - t)) % BASE for t in range(M)], 1)
    print(f"machine {Y}: openings {nop:,} (= closed form, asserted), "
          f"F({Y}) = {mx} (asserted)")
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
        sub = set()
        for t in range(M - m2 + 1):
            sub |= set(map(tuple, tup[:, t:t + m2].tolist()))
        print(f"  induced {m2}-tuple dictionary (contiguous windows): "
              f"{len(sub):,}")
    sys.exit()

LO, HI, OUT = int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
SEG = int(sys.argv[6]) if len(sys.argv) > 6 else SEGD
t0 = time.time()
seen = np.zeros(NKEY, bool)
nop = 0
mx = 0
tail = np.empty(0, np.int64)
lo = LO
while lo < HI:
    hi = min(lo + SEG, HI)
    # ROUND-24: retry on MemoryError.  The box's commit charge is saturated
    # by other lanes' jobs (a 10.5 GB Lean kernel among them) and ~27 MB
    # allocations fail TRANSIENTLY; five of six workers died that way on
    # the first launch.  A segment is a pure function of [lo, hi), so
    # sleep-and-retry is exact.
    for attempt in range(1000):
        try:
            op = openings(lo, hi)
            cat = np.concatenate([tail, op])
            k = keys_of(cat)
            break
        except MemoryError:
            if attempt % 20 == 0:
                print(f"  MemoryError at {lo:,} attempt {attempt}; "
                      f"waiting", flush=True)
            time.sleep(15)
    else:
        raise MemoryError(f"segment {lo} failed 1000 attempts")
    nop += len(op)
    if len(cat) > 1:
        mx = max(mx, int(np.diff(cat).max()))
    if len(k):
        seen[k] = True
    tail = cat[-M:].copy() if len(cat) >= M else cat.copy()
    lo = hi
    if (lo // SEG) % 200 == 0:
        print(f"  {lo:,}/{HI:,} openings {nop:,} distinct "
              f"{int(seen.sum()):,} t={time.time()-t0:.0f}s", flush=True)
# read past HI so the boundary tuples are complete (wrapping at P)
ext = []
need = HI
while len(ext) < M + 1 and need < HI + 20 * OVER:
    seg = openings(need % P, min(need % P + OVER, P))
    ext.extend(int(x) + (need - need % P) for x in seg)
    need += OVER
cat = np.concatenate([tail, np.array(ext[:M + 1], np.int64)])
k = keys_of(cat)
if len(k):
    seen[k] = True
if len(cat) > 1:
    mx = max(mx, int(np.diff(cat).max()))
ks = np.flatnonzero(seen).astype(np.int64)
np.save(OUT, np.concatenate([np.array([nop, mx], np.int64), ks]))
print(f"worker [{LO}, {HI}) done: openings {nop:,} maxgap {mx} "
      f"distinct {len(ks):,}  {time.time()-t0:.0f}s", flush=True)
