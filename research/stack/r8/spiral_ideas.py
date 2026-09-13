"""Spiral ideas, tested separately (owner, 2026-09-13): (1) the order of the gears (descending,
ascending, by the tooth 6^-1 mod g, by tooth + gear, by the first strike above q = the striking
order, random); (2) the gear set (all gears; only the gears below sqrt q); (3) the base ({2}
pairing, {2, 3} pairing). Endpoint = -1 + 2 P_base * A where A is the alternating sum of the
spiral's gears in the chosen order, P_base = 2 or 6. Fast primality by a sieve.
Reported per variant, machines 11 .. qmax: E/q mean (min..max), endpoints inside the window,
endpoints that are twins, median distance to the nearest twin, gears carried (divisors of A
among the gears) mean.
Usage: uv run python spiral_ideas.py qmax
"""
import sys, random, statistics
import numpy as np
from sympy import primerange, factorint


def main():
    qmax = int(sys.argv[1]); random.seed(3)
    N = qmax * qmax + 10
    sieve = np.ones(N + 1, dtype=bool); sieve[:2] = False
    for i in range(2, int(N ** 0.5) + 1):
        if sieve[i]: sieve[i * i::i] = False
    def twin(n): return 0 <= n < N - 2 and sieve[n] and sieve[n + 2]
    def near(E, q, cap=4000):
        for s in range(0, cap, 2):
            for v in (E + s, E - s):
                if v % 6 == 5 and q < v and v + 2 <= q * q and twin(v): return s
        return cap
    def alt(order): return sum(g if i % 2 == 0 else -g for i, g in enumerate(order))
    qs = list(primerange(11, qmax + 1)); stats = {}
    def rec(name, q, A, mult):
        E = -1 + mult * A; d = stats.setdefault(name, {'pos': [], 'twin': 0, 'dist': [], 'carry': [], 'n': 0, 'inwin': 0})
        d['n'] += 1; d['pos'].append(E / q); inw = q < E <= q * q; d['inwin'] += inw; d['twin'] += (inw and twin(E)); d['dist'].append(near(E, q))
        d['carry'].append(len([h for h in factorint(abs(A)) if 5 <= h <= q]) if A else 0)
    for q in qs:
        g = list(primerange(3, q + 1)); g5 = [h for h in g if h >= 5]
        def fs_num(h):
            if h <= 3: return 0        # 3 never strikes a twin slot: put it first
            m = (q + 1 + h - 1) // h
            while (m * h) % 6 not in (1, 5): m += 1
            return m * h
        # (1) orders, all gears, {2, g}
        rec('(1) order: descending', q, alt(g[::-1]), 4)
        rec('(1) order: ascending', q, alt(g), 4)
        rec('(1) order: by tooth 6^-1 mod g', q, alt(sorted(g, key=lambda h: pow(6, -1, h) if h > 3 else 0)), 4)
        rec('(1) order: by tooth + gear', q, alt(sorted(g, key=lambda h: (pow(6, -1, h) if h > 3 else 0) + h)), 4)
        rec('(1) order: striking order (first strike above q)', q, alt(sorted(g, key=fs_num)), 4)
        for r in range(5):
            gg = g[:]; random.shuffle(gg); rec('(1) order: random (5 per machine)', q, alt(gg), 4)
        # (2) gear set: below sqrt q only
        sub = [h for h in g if h * h <= q]; sub5 = [h for h in sub if h >= 5]
        rec('(2) set: gears below sqrt q, {2,g}, descending', q, alt(sub[::-1]), 4)
        rec('(2) set: gears below sqrt q, striking order', q, alt(sorted(sub, key=fs_num)), 4)
        # (3) base {2,3}
        rec('(3) base {2,3,g}: all gears >= 5, descending', q, alt(g5[::-1]), 12)
        rec('(3) base {2,3,g}: gears below sqrt q', q, alt(sub5[::-1]), 12)
        rec('(3) base {2,3,g}: striking order', q, alt(sorted(g5, key=fs_num)), 12)
    print(f"machines 11..{qmax} ({len(qs)}): variant | E/q mean (min..max) | inside the window | E a twin | nearest twin distance median | gears carried mean")
    for name, d in stats.items():
        print(f"   {name} | {statistics.mean(d['pos']):.2f} ({min(d['pos']):.2f}..{max(d['pos']):.2f}) | {d['inwin']} / {d['n']} | {d['twin']} | {statistics.median(d['dist'])} | {statistics.mean(d['carry']):.2f}")


if __name__ == "__main__":
    main()
