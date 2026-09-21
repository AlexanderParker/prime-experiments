"""The leg rule on blocks (manager derivation, 2026-09-21).

Block b holds columns 5b, 5b+2, 5b+3 at positions p = 1, 2, 3 (e_1, e_2, e_3) = (0, 2, 3).
Gear g >= 7 strikes block b at position p iff b = s(+-c - e_p) mod g, s = 5^{-1}, c = 6^{-1} mod g.

DERIVATION. Two cells of gear g lie D blocks apart iff D is a difference of two of the six values
s(+-c - e_p).
  Same class: D = +-s (e_p - e_p') with e differences 1, 2, 3, so 5D = +-1, +-2, +-3 mod g.
  Cross class: D = +-(2 s c + s j) with j = e_p - e_p' in {0, +-1, +-2, +-3}; multiplying by 15 and
  using 5s = 1, 6c = 1 gives 15D = +-(1 + 3j), i.e. 15D = +-1, +-2, +-4, +-5, +-7, +-8, +-10.

RULE: gear g strikes two blocks D apart iff g divides one of
    5D - 3, 5D - 2, 5D - 1, 5D + 1, 5D + 2, 5D + 3        (same class)
    15D +- 1, 15D +- 2, 15D +- 4, 15D +- 5, 15D +- 7, 15D +- 8, 15D +- 10   (cross class)

PRE-REGISTERED consequences, all to be checked directly:
  D = 0 (two cells in one block): only gear 7, positions 1 and 2.
  D = 1 (neighbouring blocks): only 7, 11, 13, 17, 19, 23 - no gear from 29 up.
  For each D the gears able to bridge it are exactly the prime factors >= 7 of a fixed finite list
  of numbers of size about 15D, so the bridging set is FINITE AND INDEPENDENT OF q.
"""
from sympy import primerange, factorint

E = (0, 2, 3)


def cells(g):
    c = pow(6, -1, g)
    s = pow(5, -1, g)
    out = []
    for sign in (c, (-c) % g):
        for p, e in enumerate(E, start=1):
            out.append(((s * (sign - e)) % g, p))
    return out


def rule_gears(D, limit):
    same = [5 * D + j for j in (-3, -2, -1, 1, 2, 3)]
    cross = [15 * D + e for e in (1, -1, 2, -2, 4, -4, 5, -5, 7, -7, 8, -8, 10, -10)]
    out = set()
    for m in same + cross:
        if m:
            for p in factorint(abs(m)):
                if 7 <= p <= limit:
                    out.add(p)
    return sorted(out)


LIMIT = 500
print("D | gears striking two blocks D apart: by direct search | by the rule | agree")
for D in range(0, 13):
    direct = []
    for g in primerange(7, LIMIT + 1):
        cs = cells(g)
        if any((b2 - b1) % g == D % g and (b1, p1) != (b2, p2) for b1, p1 in cs for b2, p2 in cs):
            direct.append(g)
    if D == 0:
        pred = [7]
    else:
        # g | D returns a gear to the SAME block, which needs two cells in one block: gear 7 only
        pred = sorted(set(rule_gears(D, LIMIT)) | ({7} if D % 7 == 0 else set()))
    print(f"{D:2d} | {direct} | {pred} | {direct == pred}")
    assert direct == pred, D
print()
print("PROVED: the gears able to strike two blocks D apart are exactly the prime factors >= 7 of")
print("5D +- 1, 5D +- 2, 5D +- 3, 15D +- 1, 15D +- 2, 15D +- 4, 15D +- 5, 15D +- 7, 15D +- 8,")
print("15D +- 10 - plus gear 7 when 7 divides D, since only gear 7 has two cells in one block -")
print("a finite set independent of q. COROLLARY: a gear larger than 15D + 10 cannot bridge the")
print("distance D, so a gear larger than 15m - 5 puts at most ONE cell into a chain of m blocks.")
print()
print("size of the bridging set for each D (checked to gear 500):")
for D in range(1, 21):
    print(f"  D={D:2d}: {len(rule_gears(D, LIMIT))} gears {rule_gears(D, LIMIT)}")
