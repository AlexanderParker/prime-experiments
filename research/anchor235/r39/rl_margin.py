"""R2.a.i.a - part 5: the margin of the island frame, and the exact failure list.

For B = 7 (the frame that works) and B = 13 (the frame that does not), per q:
  * the exact list of primes whose islands in [1, d) are ALL struck (no free island);
  * the minimum, median and maximum number of FREE islands by q band;
  * the smallest q above which the free count never returns to 0 in the sweep;
  * the free fraction (free islands / islands) against prod_{B < g <= q} (1 - 2/g).

Writes results/rl_margin.txt.
Usage: uv run python research/anchor235/r39/rl_margin.py
"""
import os
from math import isqrt

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
LOG = open(os.path.join(OUT, "rl_margin.txt"), "w")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.write(s + "\n")


QMAX = 20000


def sieve(n):
    fl = bytearray([1]) * (n + 1)
    fl[0:2] = b"\x00\x00"
    for i in range(2, isqrt(n) + 1):
        if fl[i]:
            fl[i * i:: i] = bytearray(len(range(i * i, n + 1, i)))
    return fl


FL = sieve(QMAX + 10)
GEARS = [p for p in range(5, QMAX + 1) if FL[p]]
UU = [pow(6, -1, g) for g in GEARS]
NG = len(GEARS)
DMAX = (2 * QMAX + 1) // 3 + 2

BS = (7, 13)
MASK = {}
for B in BS:
    res = np.load(os.path.join(OUT, "rl_isl_%d.npy" % B))
    mod = int(open(os.path.join(OUT, "rl_isl_%d_mod.txt" % B)).read())
    m = np.zeros(DMAX, dtype=bool)
    rs = np.array(sorted(int(v) for v in res), dtype=np.int64)
    for base in range(0, DMAX, mod):
        idx = rs + base
        idx = idx[idx < DMAX]
        m[idx] = True
    MASK[B] = m

rec = {B: [] for B in BS}
for qi, q in enumerate(GEARS):
    nq = qi + 1
    qq = q * q
    d = (2 * pow(6, -1, q)) % q
    D = int(d)
    if D < 2:
        for B in BS:
            rec[B].append((q, 0, 0))
        continue
    cnt = np.zeros(D, dtype=np.int16)
    for j in range(nq):
        g = GEARS[j]
        u = UU[j]
        r = qq % g
        a = ((2 - r) * u) % g
        b = ((-r) * u) % g
        if a < D:
            cnt[a::g] += 1
        if b < D:
            cnt[b::g] += 1
    op = cnt == 0
    op[0] = False
    for B in BS:
        isl = MASK[B][:D].copy()
        isl[0] = False
        rec[B].append((q, int(isl.sum()), int((isl & op).sum())))

for B in BS:
    say("")
    say("=== B = %d ===" % B)
    r = rec[B]
    noisl = [q for (q, n, f) in r if n == 0]
    nofree = [q for (q, n, f) in r if n > 0 and f == 0]
    say("primes with no island at all in [1, d): %d  %s" % (len(noisl), noisl))
    say("primes with islands but none free: %d" % len(nofree))
    say("   the full list: %s" % nofree)
    if nofree:
        top = max(nofree)
        above = [q for (q, n, f) in r if q > top]
        say("   largest such prime %d; primes above it in the sweep: %d, all with a free island"
            % (top, len(above)))
    say("free-island count by q band:")
    say("     band          walks   min   median   mean    max    islands(med)   free/islands")
    for lo, hi in [(5, 100), (100, 1000), (1000, 5000), (5000, 10000), (10000, 20000)]:
        sub = [(q, n, f) for (q, n, f) in r if lo <= q < hi and n > 0]
        if not sub:
            continue
        fs = np.array([f for (_, _, f) in sub])
        ns = np.array([n for (_, n, _) in sub])
        say("  %6d-%-6d  %6d  %4d  %6d  %6.2f  %5d   %10d     %.4f"
            % (lo, hi, len(sub), fs.min(), int(np.median(fs)), fs.mean(), fs.max(),
               int(np.median(ns)), fs.sum() / ns.sum()))

say("")
say("=== the free fraction against the independent-gear product prod_{B<g<=q} (1 - 2/g) ===")
say("   q         B=7 measured   B=7 product    B=13 measured  B=13 product")
prod = {}
for B in BS:
    p = 1.0
    d = {}
    for g in GEARS:
        if g > B:
            p *= (1 - 2.0 / g)
        d[g] = p
    prod[B] = d
for lo, hi in [(1000, 5000), (5000, 10000), (10000, 20000)]:
    line = "%6d-%-6d" % (lo, hi)
    for B in BS:
        sub = [(q, n, f) for (q, n, f) in rec[B] if lo <= q < hi and n > 0]
        fs = sum(f for (_, _, f) in sub)
        ns = sum(n for (_, n, _) in sub)
        qm = int(np.median([q for (q, _, _) in sub]))
        gq = max(g for g in GEARS if g <= qm)
        line += "   %.5f       %.5f " % (fs / ns, prod[B][gq])
    say(line)
LOG.close()
