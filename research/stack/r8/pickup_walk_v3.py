"""Loop, iteration 15: the flex pick-up walk with two repair flips, checked further.
Same as entry 14 'desc, flex, repair 2', with 2 and 3 forced into the base (so every landing is a
left member), run on every machine 11 to 5000 and on a sample of larger machines to 20000.
usage: uv run python research/stack/r8/pickup_walk_v3.py 5000 20000
"""
import sys
import numpy as np
from sympy import primerange
from pathlib import Path
sys.path.insert(0, "research/stack/r8")
from pickup_walk_v2 import run

def main():
    Q1, Q2 = int(sys.argv[1]), int(sys.argv[2])
    qs = list(primerange(11, Q1 + 1)) + list(primerange(Q1 + 1, Q2 + 1))[::60]
    N = Q2 * Q2 + 10; sv = np.ones(N + 1, dtype=bool); sv[:2] = False
    for i in range(2, int(N ** 0.5) + 1):
        if sv[i]: sv[i * i::i] = False
    ok = 0; fails = []; sample = []
    for q in qs:
        n, good = run(q, 'desc', 'flex', 2)
        if good: assert sv[n] and sv[n + 2]; ok += 1
        else: fails.append(q)
        if q in (499, 1999, 4999) or q > 15000: sample.append((q, n, good))
    out = [__doc__.strip(), "", f"machines {len(qs)} ({qs[0]} to {qs[-1]}, every prime to {Q1} then every 20th): twin at {ok}; fails {fails}", f"samples {sample[:8]}"]
    Path("research/stack/r8/results_pickup_walk_v3.txt").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out[2:]))

if __name__ == "__main__":
    main()
