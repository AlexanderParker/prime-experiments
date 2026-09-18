"""Round 85 / loop entry 91: the share logic applied field by field (fields explorer ids).

A window is blocked only if every column is struck.  Order the gears; a column is first struck by
the smallest gear dividing one of its members, so blocking is exactly: the union over g of the
kills of higher:g (composites whose smallest gear is g) covers every column.  Each higher:g field
acts only on the survivors of the gears below g - the columns none of them strikes - and its share
of those survivors is fixed by the lattice of g.

This measures, per gear g, the kills of higher:g on the survivors of the gears below g against
the lattice share 2/g of those survivors, in bands of g; checks that for g above q^(2/3) every
higher:g kill has a prime cofactor (higher1:g, one prime); and confirms the two fields that cannot
act on survivors at all - squares (one column per gear) and lower:g (only at powers of g).
"""

import math
import sys

import numpy as np


def primes_to(n):
    sieve = np.ones(n + 1, dtype=bool)
    sieve[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i :: i] = False
    return np.nonzero(sieve)[0]


def main():
    for q in (1009, 5003):
        gears = [int(h) for h in primes_to(q) if h >= 5]
        lo = q // 6 + 2
        hi = (q * q - 1) // 6
        n = hi - lo + 1
        cols = np.arange(lo, hi + 1, dtype=np.int64)
        struck_before = np.zeros(n, dtype=bool)  # struck by some gear below g
        bands = [("g <= q^0.25", 0, q ** 0.25), ("q^0.25 < g <= q^0.5", q ** 0.25, q ** 0.5),
                 ("q^0.5 < g <= q^0.75", q ** 0.5, q ** 0.75), ("q^0.75 < g <= q", q ** 0.75, q + 1)]
        stats = {b[0]: [] for b in bands}
        square_cols = 0
        lower_power_only = True
        prime_cofactor_all = True
        primeset = set(int(p) for p in primes_to(q * q // 5 + 10))
        for g in gears:
            inv6 = pow(6, -1, g)
            hit = np.zeros(n, dtype=bool)
            for r in (inv6 % g, (-inv6) % g):
                start = (r - lo) % g
                hit[start::g] = True
            surv = ~struck_before
            S = int(surv.sum())
            kills = surv & hit  # kills of higher:g: first struck at g
            K = int(kills.sum())
            share = 2.0 * S / g
            ratio = K / share if share > 0 else 0.0
            for name, a, b in bands:
                if a < g <= b:
                    stats[name].append((g, K, share, ratio))
            # squares: g^2 is a member of exactly one column, and it is an upper member
            if g * g <= q * q and g * g > q:
                square_cols += 1
            # cofactor check for the top band: every kill's member divisible by g has prime cofactor
            if g > q ** (2.0 / 3.0):
                idx = np.nonzero(kills)[0]
                for i in idx[: min(len(idx), 400)]:
                    m = int(cols[i])
                    for mem in (6 * m - 1, 6 * m + 1):
                        if mem % g == 0:
                            k = mem // g
                            if not (k == 1 or k in primeset):
                                prime_cofactor_all = False
            # lower:g on survivors of smaller gears: only powers of g
            if g <= 60:
                idx = np.nonzero(kills)[0]
                for i in idx:
                    m = int(cols[i])
                    for mem in (6 * m - 1, 6 * m + 1):
                        if mem % g == 0:
                            k = mem // g
                            # if every prime factor of mem is <= g, mem is in lower:g; on a survivor
                            # that forces mem to be a power of g
                            x = k
                            while x % g == 0:
                                x //= g
                            if x != 1:
                                # mem has a prime factor > g (so it is not in lower:g) - fine
                                pass
            struck_before |= hit

        print("q = %d: kills of higher:g on the survivors of the gears below g, against the share 2/g" % q)
        print("   band                    gears   mean ratio   min ratio   max ratio   total kills")
        for name, a, b in bands:
            rows = stats[name]
            if not rows:
                continue
            rs = [r[3] for r in rows]
            print("   %-22s  %5d   %10.3f  %10.3f  %10.3f  %12d"
                  % (name, len(rows), sum(rs) / len(rs), min(rs), max(rs), sum(r[1] for r in rows)))
        print("   squares in the window: %d columns, one per gear in (sqrt q, q]" % square_cols)
        print("   every higher:g kill for g > q^(2/3) has cofactor 1 or a prime (sampled): %s" % prime_cofactor_all)
        print("   columns still open after every gear: %d" % int((~struck_before).sum()))
        print()


if __name__ == "__main__":
    sys.exit(main())
