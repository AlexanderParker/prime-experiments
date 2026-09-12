"""Walks whose axis carries many gears (owner: both; 2026-09-13). Effective axis A = k P_m,
P_m the product of the first m gears (2, 3, 5, ...); landing (2 k P_m - 1, 2 k P_m + 1),
certified for those m gears by the mirror; the remaining gears up to q must miss it, which the
rule arranges by the roots (k = -+(2 P_m)^-1 mod h are the teeth of h on k). Trade-off: more
gears carried, fewer multiples below q^2, fewer chances for the roots.
Per machine q: for each m with 2 P_m + 1 <= q^2, does a k exist (landing in the window,
avoiding the teeth of the remaining gears)? Report the largest m that works, the number of
gears the roots handled at that m, the number of multiples available, and the landing position.
Usage: uv run python carry_many.py qmax
"""
import sys
from sympy import primerange, isprime


def main():
    qmax = int(sys.argv[1]); primes = list(primerange(2, qmax + 1))
    qs = [q for q in primes if q >= 11]
    print("q | gears in machine | largest m carried with a landing | multiples of 2 P_m below q^2 at that m | k used | gears left to the roots | landing 2kP_m / q^2 | twin")
    rows = []
    for q in qs:
        gears = [p for p in primes if 5 <= p <= q]
        best = None
        P = 1
        for m, p in enumerate(primes, 1):
            if p > q: break
            P *= p
            if 2 * P + 1 > q * q: break
            rest = [h for h in gears if h > p]
            avail = (q * q - 1) // (2 * P)
            k = 1; hit = None
            while 2 * k * P + 1 <= q * q:
                n = 2 * k * P - 1
                if n > q and all(n % h and (n + 2) % h for h in rest): hit = k; break
                k += 1
            if hit is not None: best = (m, avail, hit, len(rest), 2 * hit * P / (q * q), isprime(2 * hit * P - 1) and isprime(2 * hit * P + 1))
        rows.append((q, len(gears), best))
    for q, ng, b in rows:
        if q in (11, 13, 31, 101, 211, 401, 1009, 2003, 3001, 4001, 4999) or b is None:
            print(f"{q} | {ng} | " + (" | ".join(str(x) for x in b) if b else "none"))
    from collections import Counter
    c = Counter(b[0] if b else 0 for _, _, b in rows)
    print("largest carried m over machines (m: count):", sorted(c.items()))
    print("machines where the largest-m landing is not a twin:", sum(1 for _, _, b in rows if b and not b[5]))


if __name__ == "__main__":
    main()
