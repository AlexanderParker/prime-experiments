"""Focused memo-counter probe: correctness on cheap cases + speed on the
expensive m31 tuples the DFS could not afford."""
import sys
import os
import time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from qualrun_zerocert import pattern_count, pattern_count_memo, primes

gears13 = primes(5, 13)
for X, Y in (([0, 6], range(1, 6)), ([0, 6, 17],
             [t for t in range(1, 17) if t != 6]),
             ([0, 4, 8], [1, 2, 3, 5, 6, 7])):
    A, _ = pattern_count(gears13, list(X), list(Y))
    B, _ = pattern_count_memo(gears13, list(X), list(Y))
    assert A == B, (X, A, B)
print("m13 correctness 3/3", flush=True)

gears19 = primes(5, 19)
tup = (15, 23, 23)
X = [0]
for v in tup:
    X.append(X[-1] + v)
Y = [t for t in range(1, sum(tup)) if t not in set(X)]
t0 = time.time()
B, nb = pattern_count_memo(gears19, X, Y)
print(f"m19 {tup}: MEMO count {B} states {nb:,} {time.time()-t0:.1f}s "
      f"(DFS: 1.37M nodes; exact 0)", flush=True)
assert B == 0

gears31 = primes(5, 31)
for tup in ((25, 49), (25, 25, 49)):
    X = [0]
    for v in tup:
        X.append(X[-1] + v)
    Y = [t for t in range(1, sum(tup)) if t not in set(X)]
    t0 = time.time()
    B, nb = pattern_count_memo(gears31, X, Y)
    print(f"m31 {tup}: MEMO count {B} states {nb:,} {time.time()-t0:.1f}s",
          flush=True)
print("done", flush=True)
