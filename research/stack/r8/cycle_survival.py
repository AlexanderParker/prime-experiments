"""The machine's openings across cycles. For machine q the columns of the window (q, q^2] open to
the machine's gears are exactly the twins of cycle 1 (a survivor below q^2 is prime), and they
sit at the same offsets in every cycle (the machine's rows are periodic with q#). In cycle c each
such offset is either a twin again or killed by a field higher:g with g > q (no other field can
touch a machine-open column). Per machine and cycle: how many of the openings are twins again,
and which gears (smallest factor) ate the rest.
Usage: uv run python cycle_survival.py qmax cycles
"""
import sys
from collections import Counter
from sympy import primerange, isprime, factorint


def main():
    qmax, C = int(sys.argv[1]), int(sys.argv[2])
    for q in primerange(5, qmax + 1):
        P = 1
        for r in primerange(2, q + 1): P *= r
        tw = [n for n in range(q + 1, q * q) if n % 6 == 5 and isprime(n) and isprime(n + 2)]
        line = []; eaters = Counter()
        for c in range(1, C + 1):
            s = (c - 1) * P; alive = 0
            for n in tw:
                a, b = n + s, n + s + 2
                if isprime(a) and isprime(b): alive += 1
                else:
                    for x in (a, b):
                        if not isprime(x): eaters[min(factorint(x))] += 1
            line.append(alive)
        print(f"machine {q}: openings {len(tw)}; twins again per cycle {line}; eaters (smallest gear, count) {sorted(eaters.items())[:10]}")


if __name__ == "__main__":
    main()
