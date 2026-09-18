"""Round 93 / loop entry 99: the twins in each stretch a new gear opens.

For consecutive gears p < q the machine q extends the window from p^2 to q^2.  By
`new_gear_only_square` (proofs/StretchRule.lean) the new gear strikes nothing in (p^2, q^2] that
the gears up to p had left open, except the column of its own square; so the twins of machine q
in that stretch are exactly the open columns of the gears up to p there.  A twin-free window
would need a twin-free stretch of this kind somewhere inside it.  This counts the twins in every
consecutive-gear stretch up to a bound and reports the sparsest.
"""

import sys

from sympy import isprime, primerange


def main():
    bound = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    ps = list(primerange(5, bound))
    rows = []
    for p, q in zip(ps, ps[1:]):
        lo = p * p // 6 + 1
        hi = (q * q - 1) // 6
        tw = 0
        first = None
        for m in range(lo, hi + 1):
            a = 6 * m - 1
            if a <= p * p:
                continue
            if isprime(a) and isprime(a + 2):
                tw += 1
                if first is None:
                    first = a
        rows.append((tw, p, q, hi - lo + 1, first))
    rows.sort()
    print("twins inside each stretch (p^2, q^2] for consecutive gears p < q below %d (%d stretches)" % (bound, len(rows)))
    print("   fewest twins in a stretch:")
    for tw, p, q, cols, first in rows[:8]:
        print("     p=%5d q=%5d  columns %6d  twins %4d  first twin at %s" % (p, q, cols, tw, first))
    print("   stretches with no twin at all: %d" % sum(1 for w in rows if w[0] == 0))
    print("   twins per column, smallest ratio: %.4f (p = %d)" % min((w[0] / w[3], w[1]) for w in rows))


if __name__ == "__main__":
    sys.exit(main())
