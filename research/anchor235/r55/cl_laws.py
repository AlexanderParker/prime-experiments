"""r55 cl_laws - item 6: the statements that hold without exception, with their counts.

L1  the period law            c(g,h;L+gh) = c(g,h;L) + 4                (also for random seps)
L2  the arc floor             c(g,h;L) = 0 for every L <= max(a_g,a_h)  (onset >= max(a)+1)
L3  the shared-arc law        a_g = a_h = a  =>  c(g,h;a+1) >= 1        (every shared-arc
                              configuration of every pair, not only the real one)
L4  the period floor          c(g,h;L) >= 4 floor(L/gh)                 (corollary of L1)
"""
import os
import random

import numpy as np

from cl_core import (RESULTS, PRIMES, arc, dump, pair_c_at, pair_profile, real_sep,
                     say_factory)

LINES = []
say = say_factory(LINES)
GEARS = [p for p in PRIMES if 5 <= p <= 97]
SEED = 550905


def main():
    os.makedirs(RESULTS, exist_ok=True)
    rng = random.Random(SEED)
    say("=" * 100)
    say("ITEM 6 - what holds without exception, with the count")
    say("=" * 100)

    # ---- L1 with RANDOM separations (the real ones are done in cl_pairs.py)
    n1 = e1 = 0
    for _ in range(120):
        g, h = rng.sample([p for p in GEARS if p <= 43], 2)
        g, h = min(g, h), max(g, h)
        sg, sh = rng.randrange(1, g), rng.randrange(1, h)
        P = g * h
        c = pair_profile(g, sg, h, sh, 2 * P)["c"]
        n1 += P
        e1 += int(np.count_nonzero(c[1:P + 1] + 4 != c[1 + P:2 * P + 1]))
    say(f"  L1  c(L+gh) = c(L)+4 with RANDOM separations: {n1} instances, {e1} exceptions")

    # ---- L2 the arc floor, real and random
    n2 = e2 = 0
    for i, g in enumerate(GEARS):
        for h in GEARS[i + 1:]:
            sg, sh = real_sep(g), real_sep(h)
            a = max(arc(g, sg), arc(h, sh))
            c = pair_profile(g, sg, h, sh, a)["c"]
            n2 += a - 1
            e2 += int(np.count_nonzero(c[2:a + 1] != 0))
    say("  (L = 1 is excluded throughout: c(g,h;1) = 1 for every pair, both gears wanting the")
    say("   single column - a triviality, not a collision)")
    say(f"  L2  c(g,h;L) = 0 for 2 <= L <= max(a_g,a_h), REAL separations: "
        f"{n2} instances over {len(GEARS) * (len(GEARS) - 1) // 2} pairs, {e2} exceptions")
    n2r = e2r = npr = epr = 0
    for i, g in enumerate(GEARS):
        for h in GEARS[i + 1:]:
            for _ in range(3):
                sg, sh = rng.randrange(1, g), rng.randrange(1, h)
                a = max(arc(g, sg), arc(h, sh))
                c = pair_profile(g, sg, h, sh, a)["c"]
                n2r += a - 1
                e2r += int(np.count_nonzero(c[2:a + 1] != 0))
                npr += 1
                epr += int(np.count_nonzero(c[2:a + 1] != 0) > 0)
    say(f"  L2  same with RANDOM separations: {n2r} instances, {e2r} exceptions "
        f"({epr} of {npr} draws carry at least one)")

    # ---- L3 the shared-arc law: EVERY shared-arc configuration of every pair
    n3 = e3 = 0
    dist = {}
    for i, g in enumerate(GEARS):
        for h in GEARS[i + 1:]:
            for a in range(1, (g - 1) // 2 + 1):
                for sg in {a, g - a}:
                    for sh in {a, h - a}:
                        if arc(g, sg) != a or arc(h, sh) != a:
                            continue
                        v = pair_c_at(g, sg, h, sh, a + 1)
                        n3 += 1
                        if v < 1:
                            e3 += 1
                        dist[v] = dist.get(v, 0) + 1
    say(f"  L3  a_g = a_h = a  =>  c(g,h;a+1) >= 1: {n3} shared-arc configurations over all "
        f"{len(GEARS) * (len(GEARS) - 1) // 2} pairs, {e3} exceptions")
    say(f"      distribution of c(g,h;a+1): {dict(sorted(dist.items()))}")
    say("      (the real machine's twin pairs are the case a_g = a_h with s = 3^{-1}: file 20")
    say("       Lemma 1 gives a_g = a_{g+2} = (g+1)/3, so every twin pair is an instance)")

    # ---- L4 the period floor
    n4 = e4 = 0
    for i, g in enumerate(GEARS[:12]):
        for h in GEARS[:12][i + 1:]:
            P = g * h
            c = pair_profile(g, real_sep(g), h, real_sep(h), min(4 * P, 6000))["c"]
            for L in range(1, len(c)):
                n4 += 1
                if c[L] < 4 * (L // P):
                    e4 += 1
    say(f"  L4  c(g,h;L) >= 4 floor(L/gh): {n4} instances, {e4} exceptions")
    dump(LINES, "cl_laws.txt")


if __name__ == "__main__":
    main()
