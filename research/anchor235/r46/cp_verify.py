"""Branch 2f.i - independent verification of the fully compatible chain violators.

Recomputes F(M), F_2(M), Q*_J(M) and F(M + q') by a DIRECT sieve of M + q' (prover B's
attainment gate G2) for the members cp_compat.py flagged as coherent (I = 0) chain violators.
Reuses research/proof/chain_family_r32.py for the family definition and the machinery.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "proof"))
from chain_family_r32 import (direct_F_new, gaps_of, gears_of, next_prime,  # noqa: E402
                              open_mask, qstar_table, letter_a, real_tooth, summarize)
from math import prod  # noqa: E402

CASES = [
    (11, [1, 1, 5], 1, "29/18"),
    (17, [1, 3, 4, 4, 4], 4, "8/1"),
    (17, [2, 3, 3, 3, 3], 3, "6/1"),
    (17, [1, 1, 2, 2, 3], 3, "real 1/3 (control)"),
    (11, [1, 1, 2], 2, "real 1/3 (control)"),
]

for y, teeth, v1, tag in CASES:
    gears = gears_of(y)
    q1 = next_prime(y)
    P = prod(gears)
    g = gaps_of(open_mask(gears, teeth, P))
    a = letter_a(q1, v1)
    F, F2, tab = qstar_table(g, q1, a)
    s = summarize(F, F2, tab, q1)
    Fnew = direct_F_new(gears, teeth, q1, v1)
    chain = s["chain"]
    print("m%-2d teeth=%s v'=%d (%s)  a=%d b=%d" % (y, teeth, v1, tag, a, q1 - a))
    print("    seps=%s  F=%d F2=%d budget=F+q'=%d  max_J Q*_J=%d  DIRECT F(M+q')=%d  "
          "attainment max(F2,chain)=%d  BUDGET %s"
          % ([(2 * v) % q for q, v in zip(gears, teeth)], F, F2, F + q1, chain, Fnew,
             max(F2, chain), "VIOLATED by %d" % (Fnew - F - q1) if Fnew > F + q1 else "ok"))
    for J in sorted(tab):
        r = tab[J]
        print("      J=%d Q*=%2d word=%s gL=%d gR=%d %s" %
              (J, r["Q"], r["word"], r["gL"], r["gR"], "literal" if r["literal"] else "padded"))
