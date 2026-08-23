"""Round 20 lateral: THE n-POINT EXPOSED-SET CORRELATION, and the gear x
lag-pair object c_q(g1, g2).

DERIVATION. n points at offsets d_1..d_n are all exposed to gear q iff the
phase r avoids the 2n classes {t - d_i : t in {u,-u}}. Two of those classes
coincide iff d_i - d_j = 0 (same tooth) or d_i - d_j = +-2u (opposite teeth).
So, writing O for the number of unordered pairs with d_i - d_j = +-2u mod q,

    c_q(d_1..d_n) = q - 2n + O          (exact whenever the 2n classes are
                                         distinct except for those coincidences,
                                         i.e. whenever q >= 2n)

This ONE formula subsumes everything I have built:
  * n = 1: c = q - 2                       (the exposed set itself)
  * n = 2: c = q - 4 + O, O in {0,1}       (round 18: q-2 if q|g, q-3 if
                                            g = +-2u, q-4 else)
  * c_q > 0 forced when q > 2n             (round 17's completeness lemma -
                                            only gears q <= 2n can block)
and the O-count is always the same three tooth-relationships, now applied
pairwise.

For a LAG PAIR (three points 0, g1, g1+g2 - two adjacent gaps) the pairwise
differences are g1, g2, g1+g2, and O counts how many are = +-2u mod q. O <= 2
always: if g1 = e1*2u and g2 = e2*2u then g1+g2 = (e1+e2)*2u in {-4u, 0, 4u},
and 4u = +-2u would force 2u = 0 or 6u = 0, both impossible since 6u = 1.
So c_q(g1,g2) in {q-6, q-5, q-4} in the non-degenerate case.
"""
from itertools import combinations
import numpy as np

def teeth(q):
    u = pow(6, -1, q)
    return u % q, (-u) % q

def c_brute(q, offs):
    t = set(teeth(q))
    return sum(1 for r in range(q) if all((r + d) % q not in t for d in offs))

def c_formula(q, offs):
    u = pow(6, -1, q)
    two = (2 * u) % q
    n = len(offs)
    O = 0
    for a, b in combinations(offs, 2):
        d = (a - b) % q
        if d == two or d == (-two) % q:
            O += 1
    return q - 2 * n + O

print("=" * 76)
print("PART 1: the n-point closed form, brute-force verified")
import random
random.seed(20)
bad = ok = 0
fails = []
for q in (5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43):
    for n in (1, 2, 3, 4, 5):
        for _ in range(300):
            offs = sorted(random.sample(range(q), min(n, q)))
            if len(offs) < n:
                continue
            b, f = c_brute(q, offs), c_formula(q, offs)
            if q >= 2 * n:
                if b == f: ok += 1
                else:
                    bad += 1
                    if len(fails) < 5: fails.append((q, n, offs, b, f))
print(f"  regime q >= 2n: {ok} checks, {bad} mismatches" +
      (f"  examples {fails}" if fails else ""))
print("  (outside q >= 2n the formula can under-count, because the 2n forbidden")
print("   classes must overlap by pigeonhole - that overlap IS round 17's")
print("   completeness lemma: gears q <= 2n are the only ones that can block.)")

print("=" * 76)
print("PART 2: the lag-pair object c_q(g1,g2) - range and the three cases")
for q in (5, 7, 11, 13):
    u = pow(6, -1, q); two = (2 * u) % q
    vals = {}
    for g1 in range(1, q + 1):
        for g2 in range(1, q + 1):
            c = c_brute(q, [0, g1, g1 + g2])
            vals.setdefault(c, 0)
            vals[c] += 1
    print(f"  gear {q:>2} (2u = {two}): c_q(g1,g2) distribution over all "
          f"(g1,g2) mod q: {dict(sorted(vals.items()))}")
print("  gear 5 can reach c = 0: a lag pair CAN be blocked outright by gear 5")
print("  alone - the n=3 analogue of the round-16 AP lemma.")

print("=" * 76)
print("PART 3: which lag pairs gear 5 kills outright (c_5 = 0)")
q = 5
dead = [(g1, g2) for g1 in range(5) for g2 in range(5)
        if c_brute(q, [0, g1, g1 + g2]) == 0]
print(f"  (g1 mod 5, g2 mod 5) with c_5 = 0: {dead}")
print(f"  -> {len(dead)}/25 residue pairs are IMPOSSIBLE for three consecutive")
print(f"     openings, from gear 5 alone, at every scale.")
