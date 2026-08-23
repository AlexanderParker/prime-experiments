"""Round 20 lateral: the EXACT inclusion-exclusion object, and the cheapest
honest approximation to it.

The endpoint condition is a CONJUNCTION and factorises by CRT - that is why
c_q has a closed form. The interior condition ("every interior slot killed by
SOME gear") is a DISJUNCTION and does not. But the disjunction has an exact
expansion, and its building block is precisely the n-point correlation:

  #(gap of exactly g) / period
      = sum over subsets T of the interior of (-1)^|T| * D({0, g} u T),
        where D(S) = prod_q c_q(S)/q  is the exposure density of the point set S.

So THE CORRECT OBJECT IS AN ALTERNATING SUM OF EXPOSED-SET CORRELATIONS -
built entirely out of the construct, one term per subset of the interior.

CHEAPEST HONEST APPROXIMATION: truncate. Bonferroni's inequalities make the
truncation RIGOROUS - stopping after an even number of correction terms gives
an upper bound, after an odd number a lower bound. No heuristic. The only
question is how deep you must go before the bounds are useful, which is what
this file measures against the exact full-period counts.
"""
from itertools import combinations
from math import prod
import numpy as np
from split_gap_law import primes

def c_q(q, offs):
    u = pow(6, -1, q); t = {u % q, (-u) % q}
    return sum(1 for r in range(q) if all((r + d) % q not in t for d in offs))

def D(gears, S):
    v = 1.0
    for q in gears:
        v *= c_q(q, S) / q
    return v

def bonferroni(gears, g, kmax):
    """partial sums of the alternating expansion, depth 0..kmax"""
    interior = list(range(1, g))
    out = []
    tot = 0.0
    for k in range(kmax + 1):
        term = 0.0
        for T in combinations(interior, k):
            term += D(gears, [0] + list(T) + [g])
        tot += (-1) ** k * term
        out.append(tot)
    return out

def exact_counts(y, cap=30, chunk=40_000_000):
    gears = primes(5, y)
    P = prod(gears)
    cnt = np.zeros(cap + 1, np.int64)
    carry = None; a = 0
    while a < P:
        S = min(chunk, P - a)
        killed = np.zeros(S, bool)
        for q in gears:
            u = pow(6, -1, q)
            for t in (u, q - u):
                killed[(t - a) % q::q] = True
        o = np.flatnonzero(~killed).astype(np.int64) + a
        if carry is not None:
            o = np.concatenate(([carry], o))
        d = np.diff(o)
        cnt += np.bincount(d[d <= cap], minlength=cap + 1)
        carry = int(o[-1]); a += S
    return cnt, gears, P

y = 19
cnt, gears, P = exact_counts(y)
print("=" * 78)
print(f"PART 5: Bonferroni depth needed, machine {y} (period {P})")
print(f"  {'g':>3} {'exact':>8} " + " ".join(f"k={k:<9}" for k in range(0, 7)))
for g in (8, 12, 16, 20, 24):
    b = bonferroni(gears, g, 6)
    row = " ".join(f"{x*P:>10.0f}" for x in b)
    print(f"  {g:>3} {int(cnt[g]):>8} {row}")
print("  (k even = rigorous UPPER bound, k odd = rigorous LOWER bound;")
print("   the exact count must lie between consecutive partial sums.)")
print()
print("  Reading: the bounds do not bracket usefully at small depth - the odd")
print("  (lower) partial sums are hugely negative until the alternating series")
print("  settles, and the depth needed grows with g. That is Brun's problem in")
print("  the machine's own language, and it is the honest limit of the cheap")
print("  approximation: the construct gives every term in closed form, but the")
print("  number of terms needed is what the sieve fight has always been about.")
