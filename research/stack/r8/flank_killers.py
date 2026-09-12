"""Location re-look through the fields construction: who strikes the flanks of a twin.

For every column k (slot 6k-1, 6k+1) in [n0, n0+nn) record the smallest gear striking the column
on the left flank (column k-1) and on the right flank (column k+1), read row by row as in the
fields view (the row of the smallest gear that divides either member). Compare the flank-pair
distribution at twin columns with the distribution at columns open to the gears <= B but not
twins, and at all columns. A location rule would show as a flank pair over-represented at twins
beyond the wheel's own rate; the wheel's rate is the same pair frequency on all columns.
Also: how often is the toll of the new field zero per layer (companion to layers.py).
Usage: uv run python flank_killers.py n0 nn
"""
import sys
from collections import Counter
from sympy import primerange, isprime, nextprime, factorint


def smallest_gear(n):
    return min(p for p in factorint(n) if p >= 5) if any(p >= 5 for p in factorint(n)) else None


def main():
    n0, nn = int(sys.argv[1]), int(sys.argv[2])
    k0, k1 = n0 // 6 + 1, (n0 + nn) // 6
    def strike(k):
        a, b = 6 * k - 1, 6 * k + 1
        s = [smallest_gear(x) for x in (a, b) if not isprime(x)]
        s = [x for x in s if x]
        return min(s) if s else 0  # 0 = open column
    S = {k: strike(k) for k in range(k0 - 1, k1 + 2)}
    twins = Counter(); allc = Counter(); opens_nontwin = Counter()
    for k in range(k0, k1 + 1):
        pair = (S[k - 1], S[k + 1])
        allc[pair] += 1
        if S[k] == 0:
            if isprime(6 * k - 1) and isprime(6 * k + 1): twins[pair] += 1
            else: opens_nontwin[pair] += 1
    T, A = sum(twins.values()), sum(allc.values())
    print(f"columns {A}, twin columns {T}, open-non-twin {sum(opens_nontwin.values())}")
    print("flank pair (left striker, right striker) | share at twins | share at all columns | ratio")
    for pair, c in twins.most_common(12):
        print(f"{pair} | {c/T:.3f} | {allc[pair]/A:.3f} | {(c/T)/(allc[pair]/A):.2f}")
    both_open = twins[(0, 0)] / T if T else 0
    print(f"twins with both flanks open: {both_open:.3f} against {allc[(0,0)]/A:.3f} on all columns")
    # toll-zero layers
    z = 0; n = 0
    for g in primerange(5, 1000):
        gp = nextprime(g); lo, hi = g * g, gp * gp
        old = list(primerange(5, g)); toll = 0
        for k in range(lo // 6 + 1, hi // 6 + 1):
            a, b = 6 * k - 1, 6 * k + 1
            if a <= lo or b >= hi: continue
            if all(a % p and b % p for p in old) and (a % g == 0 or b % g == 0): toll += 1
        n += 1; z += (toll == 0)
    print(f"layers g^2..g'^2 for g < 1000: {n}; toll zero in {z}")


if __name__ == "__main__":
    main()
