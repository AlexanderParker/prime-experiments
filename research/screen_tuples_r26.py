"""Round 26 (mechanic): PHASE-SATURATION SCREEN over a gap-tuple dictionary.

The obstruction of docs/novel/phase-saturation-arity.md is not special to kill
words - it applies to ANY prescribed pattern of openings.  A gap m-tuple
w = (g_1..g_m) is realised at machine M only if every gear q <= M has a phase
that avoids all m+1 exposed offsets X = {0, g_1, g_1+g_2, ...}:

    FREE_q(X) = Z_q \\ ( (X mod q) u ((X - s_q) mod q) )  must be NON-EMPTY,
    s_q = -2 * 6^{-1} (mod q).

Since |FREE_q(X)| >= q - 2(m+1), only gears q < 2(m+1) can ever fire: for a
4-tuple (5 exposed points) that is gears 5 and 7 ONLY, so the whole screen is
two lookups per tuple and the pass is seconds over millions of rows.

WHY IT MATTERS HERE.  A dict_transfer SUPERSET is sound but inflated, and
Constructor reports the m41 arity-4 superset is inflated enough to stall the
chain (12/12 sampled superset-YES tuples CRT-refuted).  This screen removes
tuples that are provably unrealised, so it TIGHTENS a superset while keeping it
a superset - exactly the operation a certificate input can absorb.

usage: <venv>/python research/screen_tuples_r26.py Y IN.csv [OUT.csv]
"""
import sys

import numpy as np


def primes_upto(n):
    return [p for p in range(2, n + 1)
            if all(p % d for d in range(2, int(p ** 0.5) + 1))]


Y = int(sys.argv[1])
IN = sys.argv[2]
OUT = sys.argv[3] if len(sys.argv) > 3 else None
GEARS = [p for p in primes_upto(Y) if p >= 5]

rows = np.loadtxt(IN, delimiter=',', skiprows=1, dtype=np.int64)
if rows.ndim == 1:
    rows = rows[None, :]
m = rows.shape[1]
n = len(rows)
# exposed offsets: 0 and the prefix sums
X = np.zeros((n, m + 1), np.int64)
np.cumsum(rows, axis=1, out=X[:, 1:])

alive = np.ones(n, bool)
report = []
for q in GEARS:
    if q >= 2 * (m + 1):
        break                      # |FREE_q| >= q - 2(m+1) > 0: cannot fire
    s = (-2 * pow(6, -1, q)) % q
    # bad[i, a] = 1 if phase a of gear q blocks an exposed slot of tuple i
    bad = np.zeros((n, q), bool)
    R = X % q
    for t in range(m + 1):
        bad[np.arange(n), R[:, t]] = True
        bad[np.arange(n), (R[:, t] - s) % q] = True
    dead = bad.all(axis=1)
    report.append((q, int((dead & alive).sum())))
    alive &= ~dead

print(f"machine {Y}: {n:,} {m}-tuples in {IN}")
for q, k in report:
    print(f"  gear {q:2d} has no admissible phase for {k:,} of them "
          f"-> ZERO BY THEOREM")
print(f"  gears >= {2*(m+1)} can never fire (|FREE_q| >= q - {2*(m+1)})")
print(f"  SURVIVORS: {int(alive.sum()):,}  "
      f"({100.0*alive.sum()/n:.2f}% of the input; "
      f"{n - int(alive.sum()):,} removed by arithmetic alone)")
for m2 in range(1, m):
    sub = set()
    for t in range(m - m2 + 1):
        sub |= set(map(tuple, rows[alive][:, t:t + m2].tolist()))
    sub0 = set()
    for t in range(m - m2 + 1):
        sub0 |= set(map(tuple, rows[:, t:t + m2].tolist()))
    print(f"  induced {m2}-tuple dictionary: {len(sub0):,} -> {len(sub):,}")
if OUT:
    with open(OUT, 'w') as fh:
        fh.write(",".join(f"g{t+1}" for t in range(m)) + "\n")
        for r in rows[alive]:
            fh.write(",".join(str(int(x)) for x in r) + "\n")
    print(f"  wrote {OUT}")
