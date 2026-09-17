"""Round 67 / loop entry 72: the enriching chain - each landing carries one more gear.

Along a multiplicative chain the carried gears only ever grow (`mult_carried_monotone`), and a
gear dividing the landing can never strike its multiples (`landing_gear_never_strikes`).  So the
chain has a natural greedy form: at each step multiply by the smallest gear the landing does not
yet carry, times whatever small factor is needed to land on a twin centre again.

Part (a) measures the mechanism: landings that already carry small gears reach the next landing
with a smaller multiplier.
Part (b) runs the enriching chain and reports the multiplier used, the gears carried, and how
much of the allowance (j + 3 <= t) was spent.
"""

import sys

from sympy import factorint, isprime, primerange


def twins_from(x, count):
    out = []
    c = x - (x % 6)
    while len(out) < count:
        if c >= 12 and isprime(c - 1) and isprime(c + 1):
            out.append(c)
        c += 6
    return out


def smallest_j(t, cap=4000):
    j = 2
    while j <= cap:
        if isprime(t * j - 1) and isprime(t * j + 1):
            return j
        j += 1
    return None


def main():
    print("(a) the multiplier against the gears the landing already carries, 400 landings near 1e6")
    ts = twins_from(10 ** 6, 400)
    groups = {}
    for t in ts:
        key = tuple(p for p in (5, 7, 11) if t % p == 0)
        groups.setdefault(key, []).append(smallest_j(t))
    print("    gears in the landing     n     mean j   median   min   max")
    for k in sorted(groups, key=lambda z: (len(z), z)):
        v = sorted(groups[k])
        n = len(v)
        print(
            "    %-20s %5d  %7.1f  %7d  %4d  %4d"
            % (str(k) if k else "(none)", n, sum(v) / n, v[n // 2], v[0], v[-1])
        )

    print()
    print("(b) the enriching chain from 12: multiply by the smallest missing gear times a small m")
    print("  step   new gear   multiplier j   m = j / p   landing digits   gears carried   j as share of the allowance")
    t = 12
    for step in range(1, 40):
        have = set(factorint(t))
        p = next(q for q in primerange(2, 10 ** 6) if q not in have)
        j = None
        for m in range(1, 20000):
            cand = p * m
            if cand + 3 > t:
                break
            if isprime(t * cand - 1) and isprime(t * cand + 1):
                j = cand
                break
        if j is None:
            print("  step %d: no enriching multiplier for gear %d within the cap" % (step, p))
            break
        t *= j
        print(
            "  %4d   %8d   %12d   %9d   %14d   %13d   %.2g"
            % (step, p, j, j // p, len(str(t)), len(factorint(t)), j / (t // j))
        )
        if len(str(t)) > 60:
            break


if __name__ == "__main__":
    sys.exit(main())
