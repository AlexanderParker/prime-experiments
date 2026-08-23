"""Round 20 lateral: the interior DISJUNCTION - the exact object, and why the
construct prunes it.

Exact expansion (endpoints exposed, interior all killed):
    density(gap exactly g) = sum over T subset of interior of (-1)^|T| D({0,g} u T)
with D(S) = prod_q c_q(S)/q. So the correct object is an ALTERNATING SUM OF
EXPOSED-SET CORRELATIONS - every term in closed form from the construct.

THE PRUNING. c_5(S) = |{r : S + r contained in {0,2,3}}|, so ANY point set
occupying 4 or more distinct residues mod 5 contributes EXACTLY ZERO. Most
subsets do. Same for gear 7 at 6 residues. So the tree is far thinner than
C(g-1,k) suggests, and the thinning is governed by the same small-gear
structure as everything else.
"""
from itertools import combinations
from math import comb
from split_gap_law import primes

def c_q(q, offs):
    u = pow(6, -1, q); t = {u % q, (-u) % q}
    return sum(1 for r in range(q) if all((r + d) % q not in t for d in offs))

print("=" * 78)
print("PART E: how much of the inclusion-exclusion tree survives?")
print(f"  {'g':>3} {'k':>3} {'C(g-1,k)':>10} {'c5>0':>9} {'c5,c7>0':>9} "
      f"{'survive':>9} {'pruned':>8}")
for g in (12, 16, 20):
    interior = list(range(1, g))
    for k in (1, 2, 3, 4, 5):
        if k > len(interior):
            continue
        tot = comb(len(interior), k)
        n5 = n57 = 0
        for T in combinations(interior, k):
            S = [0] + list(T) + [g]
            if c_q(5, S) == 0:
                continue
            n5 += 1
            if c_q(7, S) > 0:
                n57 += 1
        print(f"  {g:>3} {k:>3} {tot:>10} {n5:>9} {n57:>9} "
              f"{n57/tot:>9.4f} {1-n57/tot:>8.2%}")
print()
print("  So the exact object's term count is not C(g-1,k) but a small fraction")
print("  of it, and the surviving subsets are exactly those whose point set")
print("  fits inside 3 residues mod 5 and 5 mod 7 - a condition read straight")
print("  off the construct. The disjunction does not factorise, but its")
print("  EXPANSION is built from factorising pieces, and the construct's zero")
print("  set is what makes the expansion sparse.")
