"""xm_family_check.py -- second, independent measurement of xm_family.py's killer counts: a plain
per-member loop over the tooth family with the section sieved column by column in Python
integers (no bitmasks, no numpy), for the cuts given on the command line (default 17 and 29).
Also prints the kill map of the first killer at the smallest cut with one.
"""
import itertools
import sys

PRIMES = [5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59]
NEXT = {PRIMES[i]: PRIMES[i + 1] for i in range(len(PRIMES) - 1)}


def gears_of(p):
    return [g for g in PRIMES if g <= p]


def main():
    ps = [int(v) for v in sys.argv[1:]] or [17, 29]
    for p in ps:
        pn = NEXT[p]
        a = (p * p - 1) // 6
        b = (pn * pn - 1) // 6
        cols = list(range(a + 1, b))
        gears = gears_of(p)
        ranges = [range(1, (g - 1) // 2 + 1) for g in gears]
        killers = 0
        first = None
        total = 0
        for vs in itertools.product(*ranges):
            total += 1
            ok = True
            for k in cols:
                struck = False
                for g, v in zip(gears, vs):
                    r = k % g
                    if r == v or r == g - v:
                        struck = True
                        break
                if not struck:
                    ok = False
                    break
            if ok:
                killers += 1
                if first is None:
                    first = vs
        print(f"p = {p}: family {total:,d}, killers {killers:,d} ({100 * killers / total:.3f} %), "
              f"first killer {list(first) if first else None}")
        if first is not None:
            print(f"  kill map of the first killer on columns {a + 1}..{b - 1}:")
            for k in cols:
                who = [g for g, v in zip(gears, first) if (k % g) in (v, g - v)]
                print(f"    column {k} = ({6 * k - 1}, {6 * k + 1}): struck by {who}")


if __name__ == "__main__":
    main()
