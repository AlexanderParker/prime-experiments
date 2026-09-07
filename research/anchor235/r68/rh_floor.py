"""rh_floor.py -- the arc floor (docs/proofs/21 Theorem 3, a certificate for the real teeth that
"fails for random separations") re-tested with the arc-1 case separated: for every pair
g < h <= 97 and every separation pair (s_g, s_h) with 1 <= s_g <= (g-1)/2, 1 <= s_h <= (h-1)/2
(every arc combination, exhaustively, not sampled), count the L in [2, max(a_g, a_h)] with
c(g, h; L) > 0, split by whether min(a_g, a_h) = 1.

    uv run python research/anchor235/r68/rh_floor.py [HMAX]
"""
import json
import os
import sys
from itertools import combinations

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rh_core import OUT, PRIMES, pair_pattern, window_counts

HMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 97


def main():
    gears = [p for p in PRIMES if p <= HMAX]
    tot = {"arc1": [0, 0], "arc>=2": [0, 0]}     # [instances, exceptions]
    exc_L = {}
    pairs = 0
    for g, h in combinations(gears, 2):
        pairs += 1
        for ag in range(1, (g - 1) // 2 + 1):
            for ah in range(1, (h - 1) // 2 + 1):
                amax = max(ag, ah)
                if amax < 2:
                    continue
                U, A, B = pair_pattern(g, ag, h, ah)
                key = "arc1" if min(ag, ah) == 1 else "arc>=2"
                for L in range(2, amax + 1):
                    c = int(window_counts(A, L).max() + window_counts(B, L).max() - window_counts(U, L).max())
                    tot[key][0] += 1
                    if c > 0:
                        tot[key][1] += 1
                        exc_L[(key, L)] = exc_L.get((key, L), 0) + 1
    print(f"pairs g < h <= {HMAX}: {pairs}; every arc pair (a_g, a_h) exhaustively")
    for k, (n, e) in tot.items():
        print(f"  {k}: {e} exceptions in {n} instances")
    print("  exceptions by (case, L):", {f"{k[0]} L={k[1]}": v for k, v in sorted(exc_L.items())})
    with open(os.path.join(OUT, "floor.json"), "w") as f:
        json.dump({"HMAX": HMAX, "pairs": pairs, "totals": tot,
                   "exceptions_by_L": {f"{k[0]} L={k[1]}": v for k, v in exc_L.items()}}, f, indent=1)


if __name__ == "__main__":
    main()
