"""Round 27 (mechanic) probe: can Constructor's cover CSP decide the two
machine-53 kill words that pysat has been grinding on for over an hour?

Rule 20: when a SAT descent stalls, buy the verdict from a different vehicle.
The two words are 3-chains at 53 -> 59 whose span (157) sits at or below
F_2(53) = 159, so no span scan refutes them.
"""
import sys
import time
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crt_dict import decide_cover, gears_of, Budget      # noqa: E402

qs = gears_of(53)
for w in [(39, 118), (59, 98)]:
    X = [0]
    for g in w:
        X.append(X[-1] + g)
    Y = [t for t in range(1, X[-1]) if t not in set(X)]
    t0 = time.time()
    try:
        ok, _, nodes = decide_cover(qs, X, Y, node_budget=80_000_000)
        print(w, "REALISED" if ok else "ZERO", "nodes", nodes,
              "%.1fs" % (time.time() - t0), flush=True)
    except Budget:
        print(w, "NODE BUDGET EXCEEDED after %.1fs" % (time.time() - t0),
              flush=True)
