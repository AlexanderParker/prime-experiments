"""Recursion depth of the layered walk: number of times the top gear hits a lower-word landing
during one walk (0 = lower walk already closed). Also hop-size distribution given a hit."""
import sys
sys.path.insert(0, __import__("os").path.dirname(__file__))
import numpy as np
from gp_walk import word, walk, PR
from collections import Counter
for q in [11, 13, 17, 19, 23]:
    gears = [g for g in PR if g <= q]
    low = gears[:-1]; g = gears[-1]; u = pow(6, -1, g)
    w, P = word(gears); wl, Pl = word(low)
    W = walk(w, P); Wl = walk(wl, Pl)
    s = np.arange(P)
    depth = np.zeros(P, dtype=int); hop = np.zeros(P, dtype=int)
    x = s + Wl[s % Pl]
    hit = ((x % g) == u) | ((x % g) == g - u)
    cur = x.copy(); d = 0
    while hit.any():
        d += 1
        depth[hit] += 1
        cur = np.where(hit, cur + 1 + Wl[(cur + 1) % Pl], cur)
        hit = hit & (((cur % g) == u) | ((cur % g) == g - u))
    H = W - Wl[s % Pl]
    assert np.array_equal(cur - x, H)
    dist = Counter(depth.tolist()); tot = P
    hops = Counter(H[depth == 1].tolist())
    print(f"{'+'.join(map(str, gears))}: depth " + " ".join(f"{k}:{v/tot:.4f}" for k, v in sorted(dist.items())) +
          f"  | hop given one hit: " + " ".join(f"{k}:{v/sum(hops.values()):.3f}" for k, v in sorted(hops.items())[:6]) + f" max {max(hops)}")
