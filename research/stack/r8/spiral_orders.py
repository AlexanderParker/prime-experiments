"""Spiral orders (owner: order the gears by inverse-modulo remainder plus the gear, the striking
order; go nuts). The spiral from home with the mirror {2, g} (or {2, 3, g}) over the gears in a
chosen order, directions alternating: endpoint -1 + 4 A (or -1 + 12 A), A the alternating sum
in that order. The order decides the signs, hence A, hence which gears the endpoint carries
(the divisors of A) and where it lands.
Orders tried: descending (the baseline); ascending; by the tooth u = 6^-1 mod g (the inverse
modulo remainder) ascending; by u + g; by the first column above q that g strikes (the striking
order); by the first strike above the zone start on the m-line; 20 random orders (control);
and the max-carry order: signs chosen (balanced) so that A is a multiple of the largest
primorial of gears below q/2 (a subset-sum with signs), so the endpoint carries those gears.
Per order, machines 7 .. qmax: endpoint position E/q, E a twin in the window (count), distance to
the nearest twin (median), gears carried (mean count, mean product), and the strike rate of the
smallest three uncarried gears at E against 2/h.
Usage: uv run python spiral_orders.py qmax
"""
import sys, random, statistics
from sympy import primerange, isprime, factorint


def twin(n): return isprime(n) and isprime(n + 2)


def alt(order): return sum(g if i % 2 == 0 else -g for i, g in enumerate(order))


def nearest_twin(E, q, step):
    for s in range(0, 100000, step):
        for v in (E + s, E - s):
            if v % 6 == 5 and twin(v) and q < v and v + 2 <= q * q: return s
    return None


def max_carry_signs(gears, target_mod):
    """balanced signs (+/- counts differing by at most one) with sum = 0 mod target_mod and sum near q/2:
    greedy: start descending-alternating, then swap sign pairs to fix the residue (small search)"""
    n = len(gears); signs = [1 if i % 2 == 0 else -1 for i in range(n)]
    A = sum(s * g for s, g in zip(signs, gears)); best = (abs(A % target_mod), signs[:])
    for _ in range(4000):
        i, j = random.sample(range(n), 2)
        if signs[i] == signs[j]: continue
        signs[i], signs[j] = signs[j], signs[i]
        A = sum(s * g for s, g in zip(signs, gears))
        if A % target_mod == 0 and A > 0: return signs
    return None


def main():
    qmax = int(sys.argv[1]); qs = list(primerange(7, qmax + 1)); random.seed(3)
    def orders(q):
        g = list(primerange(3, q + 1))
        k0 = (q + 1) // 12 + 1
        def first_strike_col(h):   # first column k >= k0 on the m-line struck by h
            k = k0
            while (12 * k - 1) % h and (12 * k + 1) % h: k += 1
            return k
        def first_strike_num(h):   # first slot number above q struck by h
            n = q + 1
            while not (n % 6 in (1, 5) and n % h == 0): n += 1
            return n
        yield 'descending', g[::-1]
        yield 'ascending', g[:]
        yield 'by tooth 6^-1 mod g', sorted(g, key=lambda h: pow(6, -1, h) if h > 3 else 0)
        yield 'by tooth + gear', sorted(g, key=lambda h: (pow(6, -1, h) if h > 3 else 0) + h)
        yield 'by first strike above q', sorted(g, key=first_strike_num)
        yield 'by first strike on the m-line above the zone start', sorted(g, key=first_strike_col)
        for r in range(20):
            gg = g[:]; random.shuffle(gg); yield f'random {r}', gg
    stats = {}
    for q in qs:
        for name, order in orders(q):
            A = alt(order); E = -1 + 4 * A
            key = name if not name.startswith('random') else 'random (20 orders)'
            d = stats.setdefault(key, {'pos': [], 'twin': 0, 'dist': [], 'carry': [], 'prod': [], 'n': 0})
            d['n'] += 1; d['pos'].append(E / q); d['twin'] += (twin(E) and q < E <= q * q)
            ds = nearest_twin(E, q, 2); d['dist'].append(ds)
            gears = [h for h in factorint(abs(A)) if 5 <= h <= q] if A else []
            d['carry'].append(len(gears)); d['prod'].append(1 if not gears else eval('*'.join(map(str, gears))))
        # max-carry order
        g = list(primerange(3, q + 1)); P = 1; primorial_gears = []
        for h in primerange(5, q + 1):
            if P * h <= q // 2: P *= h; primorial_gears.append(h)
            else: break
        signs = max_carry_signs(g, P) if P > 1 else None
        d = stats.setdefault('max-carry (A a multiple of the largest primorial below q/2)', {'pos': [], 'twin': 0, 'dist': [], 'carry': [], 'prod': [], 'n': 0})
        if signs:
            A = sum(s * h for s, h in zip(signs, g)); E = -1 + 4 * A
            d['n'] += 1; d['pos'].append(E / q); d['twin'] += (twin(E) and q < E <= q * q); d['dist'].append(nearest_twin(E, q, 2))
            gears = [h for h in factorint(abs(A)) if 5 <= h <= q]; d['carry'].append(len(gears)); d['prod'].append(eval('*'.join(map(str, gears))) if gears else 1)
    print(f"machines 7..{qmax} ({len(qs)}): order | endpoint E/q mean (min..max) | E a twin in the window | nearest twin distance median | gears carried mean | product of carried gears mean")
    for name, d in stats.items():
        ds = [x for x in d['dist'] if x is not None]
        print(f"   {name} | {statistics.mean(d['pos']):.2f} ({min(d['pos']):.2f}..{max(d['pos']):.2f}) | {d['twin']} / {d['n']} | {statistics.median(ds) if ds else None} | {statistics.mean(d['carry']):.2f} | {statistics.mean(d['prod']):.1f}")


if __name__ == "__main__":
    main()
