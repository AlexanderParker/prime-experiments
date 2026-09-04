"""R2.a.i.a.1.a item 6 - the compulsory prefix as a q-free lever.

N-C3: every optimal adversarial cover at d >= 385 uses all of 11,13,17,19,23,29,31.  At a real q
those seven gears have phases fixed by q^2 mod 11.13.17.19.23.29.31 = 3,234,846,615.  So ask: at a
failure, how much of the island set do the seven smallest gears actually take, against the best and
the mean over all phases?  If a failure forces them near their maximum, the failure condition is a
condition on q modulo 3.2e9 - a finite, checkable modulus, and no density is involved.
"""
import numpy as np
from math import prod
SMALL = [11, 13, 17, 19, 23, 29, 31]

def arc(q):
    return (q + 1) // 3 if q % 6 == 5 else (2 * q + 1) // 3

def islands(d):
    return np.array([i for i in range(1, d) if i % 35 in (5, 10, 12, 17)], dtype=np.int64)

def cover_count(isl, gears, phases):
    hit = np.zeros(len(isl), dtype=bool)
    for g, r in zip(gears, phases):
        u = pow(6, -1, g)
        a = ((2 - r) * u) % g
        b = ((-r) * u) % g
        rem = isl % g
        hit |= (rem == a) | (rem == b)
    return int(hit.sum())

def best_and_mean(isl, gears, trials=200000, rng=None):
    """max and mean of the union size over phase vectors (exhaustive if small, else sampled)."""
    opts = []
    for g in gears:
        opts.append([(x * x) % g for x in range(1, (g + 1) // 2)])
    tot = prod(len(o) for o in opts)
    if tot <= trials:
        best, s, n = 0, 0, 0
        import itertools
        for ph in itertools.product(*opts):
            c = cover_count(isl, gears, ph)
            best = max(best, c); s += c; n += 1
        return best, s / n, n, True
    best, s = 0, 0
    for _ in range(trials):
        ph = [o[rng.integers(len(o))] for o in opts]
        c = cover_count(isl, gears, ph)
        best = max(best, c); s += c
    return best, s / trials, trials, False

rng = np.random.default_rng(11)
print("prod of the seven smallest gears = %d" % prod(SMALL))
print("   q      d     m     real take by 11..31   best over phases   mean over phases   percentile")
for q in [173, 341, 353, 461, 683, 1151, 1487, 1649, 2849,
          1499, 2003, 2399, 2801, 3251, 4001, 5003]:
    d = arc(q); isl = islands(d); m = len(isl)
    ph = [(q * q) % g for g in SMALL]
    real = cover_count(isl, SMALL, ph)
    best, mean, n, ex = best_and_mean(isl, SMALL, 20000, rng)
    # percentile of the real take among sampled phase vectors
    cnt = 0
    for _ in range(4000):
        p = [((x * x) % g) for g, x in zip(SMALL, [int(rng.integers(1, (g + 1) // 2)) for g in SMALL])]
        if cover_count(isl, SMALL, p) <= real:
            cnt += 1
    fail = q in (173, 341, 353, 461, 683, 1151, 1487, 1649, 2849)
    print("  %-6d %-5d %-5d %-21s %-18d %-18.2f %.3f  %s"
          % (q, d, m, "%d (%.3f m)" % (real, real / m), best, mean, cnt / 4000,
             "FAIL" if fail else ""))
