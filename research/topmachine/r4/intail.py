"""In use, how much of the machine is core?  A certified lower bound on the wheel record
F_top(G) for the in-use gear set, by explicitly constructing a cover (L16).

A cover of [0, L) - one phase per gear - proves F_top >= L.  Gears above L + 1 are then tail,
everything else core.  Greedy: gears smallest first, each taking the phase that covers the most
still-uncovered cells; the construction is a certificate, so the bound is proved.

usage: uv run python research/topmachine/r4/intail.py
"""

import json
import os
from math import isqrt

import numpy as np

from zone import primes_upto

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
OUT = []


def say(s=""):
    print(s)
    OUT.append(str(s))


def greedy_cover(gears, L):
    """Try to cover [0, L) with one phase per gear.  Returns (covered_count, used_gears)."""
    cov = np.zeros(L, dtype=bool)
    used = 0
    nunc = L
    for g in gears:
        if nunc == 0:
            break
        idx = np.flatnonzero(~cov)
        u = np.bincount(idx % g, minlength=g)
        s = u + np.roll(u, 2)          # cells hit by phase a: a and a - 2 (mod g)
        a = int(np.argmax(s))
        cov[a::g] = True
        cov[(a - 2) % g::g] = True
        nunc = L - int(np.count_nonzero(cov))
        used += 1
    return int(np.count_nonzero(cov)), used


def largest_covered(gears, lo, hi):
    """Exponential, then binary, search for the largest L the greedy cover reaches."""
    while True:
        c, _ = greedy_cover(gears, hi)
        if c < hi:
            break
        lo, hi = hi, hi * 2
    best = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        c, _ = greedy_cover(gears, mid)
        if c == mid:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def main():
    say("# In use, the machine is nearly all core (a certified cover)")
    say()
    say("| q | N | Q | m | certified F_top >= | tail gears (g > F+1) | tail fraction | "
        "2m - (m mod 2) | certified / 2m |")
    say("|---|---|---|---|---|---|---|---|---|")
    rows = []
    for q in (5, 7, 11, 13, 17, 19, 23, 29, 37):
        for N in (10 ** 6, 10 ** 7, 10 ** 8):
            Q = isqrt(N)
            gears = [int(p) for p in primes_upto(Q) if p > q]
            m = len(gears)
            L = largest_covered(gears, 1, 20 * m)
            tail = len([g for g in gears if g > L + 1])
            rows.append({"q": q, "N": N, "Q": Q, "m": m, "F_lower": L, "tail": tail})
            say(f"| {q} | 1e{len(str(N))-1} | {Q:,} | {m} | **{L:,}** | {tail} | "
                f"{tail / m:.4f} | {2*m - (m % 2)} | {L / (2*m):.2f} |")
    say()
    json.dump(rows, open(os.path.join(RES, "intail.json"), "w"), indent=1)
    with open(os.path.join(RES, "intail.out"), "w") as f:
        f.write("\n".join(OUT))


if __name__ == "__main__":
    main()
