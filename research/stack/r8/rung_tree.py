"""The rung tree from (5, 7), enumerated level by level (tree node R5.f.xv).

Depth 0 = {6}; the children of a twin centre s are the twin centres s' = s^2 + 6j, |j| <= 2c - 1
(both s' - 1 and s' + 1 prime).  Enumerates depths 0..DMAX fully (children found by a numpy
segmented sieve of the stretch), prints each level's size, min and max, and the child-count
distribution; at the last level prints the predicted child counts 1.32 s/(ln s)^2 for a sample.

usage: uv run python rung_tree.py [DMAX]
"""
import sys, math
import numpy as np

DMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 3

def primes_upto(n):
    s = np.ones(n + 1, dtype=bool); s[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]: s[i*i::i] = False
    return np.nonzero(s)[0]

def children(s, small):
    """twin centres strictly inside ((s-1)^2, (s+1)^2): sieve the odd numbers there."""
    lo, hi = (s - 1) ** 2 + 1, (s + 1) ** 2 - 1
    n = hi - lo + 1
    comp = np.zeros(n, dtype=bool)
    for p in small:
        p = int(p)
        if p * p > hi: break
        start = (-lo) % p
        if lo + start == p: start += p          # never mark p itself
        comp[start::p] = True
    prime = ~comp
    # twin centres s' = 6c' with s'-1, s'+1 prime: s' in (lo, hi), s' = 0 mod 6
    out = []
    first6 = lo + ((-lo) % 6)
    for sp in range(first6, hi, 6):
        if lo <= sp - 1 and sp + 1 <= hi and prime[sp - 1 - lo] and prime[sp + 1 - lo]:
            out.append(sp)
    return out

level = [6]
print("depth 0: 1 node: [6]")
for d in range(1, DMAX + 1):
    top = max(level)
    small = primes_upto(int(math.isqrt((top + 1) ** 2)) + 10)
    nxt = []; counts = []
    for s in level:
        ch = children(s, small)
        counts.append(len(ch)); nxt.extend(ch)
    nxt.sort()
    pred = [1.32 * s / math.log(s) ** 2 for s in level]
    print(f"depth {d}: {len(nxt)} nodes; min {nxt[0]}, max {nxt[-1]} ({len(str(nxt[-1]))} digits); child counts of depth {d-1}: "
          + (str(counts) if len(counts) <= 12 else f"min {min(counts)}, mean {np.mean(counts):.1f}, max {max(counts)}")
          + f"; predicted 1.32 s/(ln s)^2: " + (", ".join(f"{p:.1f}" for p in pred) if len(pred) <= 12 else f"mean {np.mean(pred):.1f}"), flush=True)
    if len(nxt) == 0:
        print("THE TREE IS FINITE - level empty"); break
    level = nxt
