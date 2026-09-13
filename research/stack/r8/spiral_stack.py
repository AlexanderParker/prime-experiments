"""Stacked spirals (owner, 2026-09-13): run the spiral with base {2} (mirrors {2, g}), then from
its endpoint the spiral with base {2, 3} (mirrors {2, 3, g}), then base {2, 3, 5}, ... Each
stage's flips move by 2 P_B g d (P_B the product of the base), so the stage moves the column by
2 P_B A_B, A_B the alternating sum of the gears above the base, descending. Stages are added
while the base's product stays below q (later stages would leave the window).
Also the reverse stacking (largest base first). Per machine: the final endpoint (E/q), whether
it is a twin inside the window, the distance to the nearest twin, and the gears carried to the
end (the divisors of E + 1, since E + 1 = sum of the stage moves), against the single spiral.
Usage: uv run python spiral_stack.py qmax
"""
import sys, statistics
from sympy import primerange, isprime, factorint


def twin(n): return isprime(n) and isprime(n + 2)


def nearest(E, q):
    for s in range(0, 200000, 2):
        for v in (E + s, E - s):
            if v % 6 == 5 and twin(v) and q < v and v + 2 <= q * q: return s
    return None


def stage_move(base_gears, gears_above):
    P = 1
    for b in base_gears: P *= b
    A = sum(g if i % 2 == 0 else -g for i, g in enumerate(gears_above[::-1]))
    return 2 * P * A


def main():
    qmax = int(sys.argv[1]); qs = list(primerange(11, qmax + 1))
    res = {'single {2,g}': [], 'stack growing base': [], 'stack shrinking base': []}
    for q in qs:
        primes = list(primerange(2, q + 1))
        bases = []
        P = 1
        for i, b in enumerate(primes):
            P *= b
            if P >= q: break
            bases.append(primes[:i + 1])
        moves = [stage_move(B, [g for g in primes if g not in B]) for B in bases]
        for name, ms in (('single {2,g}', moves[:1]), ('stack growing base', moves), ('stack shrinking base', moves[::-1])):
            E = -1 + sum(ms)
            carried = [h for h in factorint(abs(E + 1)) if 5 <= h <= q] if E + 1 else []
            res[name].append((q, E / q, twin(E) and q < E <= q * q, nearest(E, q), len(carried), len(bases)))
    print(f"machines 11..{qmax} ({len(qs)}): stacking | stages (mean) | E/q mean (min..max) | E a twin in the window | nearest twin distance median / max | carried gears mean")
    for name, rows in res.items():
        pos = [r[1] for r in rows]; ds = [r[3] for r in rows if r[3] is not None]
        print(f"   {name} | {statistics.mean(r[5] for r in rows):.1f} | {statistics.mean(pos):.2f} ({min(pos):.2f}..{max(pos):.2f}) | {sum(r[2] for r in rows)} / {len(rows)} | {statistics.median(ds)} / {max(ds)} | {statistics.mean(r[4] for r in rows):.2f}")
    print("examples (q, E/q, twin, distance, carried) for the growing stack:", [(r[0], round(r[1], 2), r[2], r[3], r[4]) for r in res['stack growing base'][:8]])


if __name__ == "__main__":
    main()
