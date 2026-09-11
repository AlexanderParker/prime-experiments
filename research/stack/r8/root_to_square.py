"""The root and its square on one wheel: does the single-tooth gap after r bound the two-tooth
gap after r^2, for EVERY root r, not only the prime?

Wheel W = {5..g} (gears), period P = product. Root r: a number in S coprime to the wheel (an
opening of the single-tooth wheel). d(r) = the distance to the next single-tooth opening
above r (for the real prime g this is the prime gap g' - g, since below g'^2 openings are
primes). L(r) = d (2r + d) / 6 = the number of columns between r^2 and (r + d)^2. B(r) = the
number of columns from the column of r^2 to the first two-tooth open column above it (the
first column whose two members are both coprime to the wheel), i.e. the first blind offset.
The finer statement at a prime g says B(g) <= L(g). Question: does B(r) <= L(r) hold for every
root r in a full period of roots? Reported: the count of roots, the count of failures, the
failures' r, d, L, B (first few), and the same restricted to r that are prime.
Usage: uv run python root_to_square.py g [rmax]
"""
import sys
import numpy as np
from math import prod
from sympy import primerange, isprime


def main():
    g = int(sys.argv[1]); gears = list(primerange(5, g + 1)); P = prod(gears)
    rmax = int(sys.argv[2]) if len(sys.argv) > 2 else 6 * P + g
    # single-tooth: number n in S is open iff coprime to P; two-tooth: column k open iff 6k-1 and 6k+1 both coprime
    # precompute column openness mod P (period P in columns)
    ks = np.arange(P)
    open_col = np.ones(P, dtype=bool)
    for h in gears:
        open_col &= ((6 * ks - 1) % h != 0) & ((6 * ks + 1) % h != 0)
    # next open column offset from each column (cyclic)
    oc2 = np.concatenate([open_col, open_col])
    nxt = np.zeros(2 * P, dtype=np.int64); last = None
    for i in range(2 * P - 1, -1, -1):
        if oc2[i]: last = i
        nxt[i] = (last - i) if last is not None else 10 ** 9
    def first_blind(col):  # first open column strictly above col, as an offset
        c = col % P
        return nxt[c + 1] + 1
    roots = [r for r in range(g, rmax) if r % 6 in (1, 5) and all(r % h for h in gears)]
    fails = []; prime_fails = []
    for idx, r in enumerate(roots[:-1]):
        d = roots[idx + 1] - r
        L = d * (2 * r + d) // 6
        col = (r * r - 1) // 6
        B = first_blind(col)
        if B > L:
            fails.append((r, d, L, B))
            if isprime(r): prime_fails.append((r, d, L, B))
    print(f"wheel {{5..{g}}}: {len(roots)-1} roots r in [{g}, {rmax}); B(r) > L(r) at {len(fails)} roots; among prime roots: {len(prime_fails)}")
    print("first failures (r, gap d, L, B):", fails[:12])
    if prime_fails: print("prime failures:", prime_fails[:12])


if __name__ == "__main__":
    main()
