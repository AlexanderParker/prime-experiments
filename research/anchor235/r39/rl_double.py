"""R2.a.i.a - part 4: the doubling, and the exact counting constants.

(a) THE DOUBLING.  Gear g strikes offset i iff q^2 = 2 - 6i or -6i (mod g).  Because the phase
enters only as the SQUARE q^2, an admissible target contributes exactly TWO classes of q mod g
(its two square roots), never one and never three.  So the number of classes r = q mod g in
(Z/g)^* at which g strikes offset i is exactly 2 chi_g(i), and the strike rate over q is exactly
2 chi_g(i)/(g-1).  Checked exhaustively for every gear <= GEXH and every offset class.
Averaged over the offsets: sum_i chi_g(i) = g - 1, so the mean strike rate of gear g over ALL
offsets is exactly 2/g - the machine's own rate - while the rate is 0 on the |Bar(g)| ~ g/4
barred classes.

(b) The empirical rate over the real primes q <= QMAX, gear by gear (a rate, prime
equidistribution; reported and stopped).

(c) The exact counting constants for item 6: sum_{B < g <= q} 2/g, the expected number of large
gears striking one island, against the measured strikes/island.  Because the island set is a
union of residue classes mod P_B and every gear g > B is coprime to P_B, the offsets of an
island class are equidistributed mod g: a large gear strikes islands at exactly the same rate
2/g as it strikes all offsets.  The frame divides targets and strikes by the same rho_B.

Writes results/rl_double.txt.
Usage: uv run python research/anchor235/r39/rl_double.py
"""
import os
from math import isqrt

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
LOG = open(os.path.join(OUT, "rl_double.txt"), "w")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.write(s + "\n")


QMAX = 20000
GEXH = 500


def sieve(n):
    fl = bytearray([1]) * (n + 1)
    fl[0:2] = b"\x00\x00"
    for i in range(2, isqrt(n) + 1):
        if fl[i]:
            fl[i * i:: i] = bytearray(len(range(i * i, n + 1, i)))
    return fl


FL = sieve(QMAX + 10)
GEARS = [p for p in range(5, QMAX + 1) if FL[p]]
say("gears 5..%d: %d" % (QMAX, len(GEARS)))

# ---------------------------------------------------------------- (a) exhaustive doubling check
say("")
say("=== 4a. the doubling: #{r in (Z/g)* : g strikes offset i} = 2 chi_g(i), exhaustive ===")
bad = 0
cells = 0
oddcells = 0
for g in GEARS:
    if g > GEXH:
        break
    qr = np.zeros(g, dtype=bool)
    for t in range(1, (g + 1) // 2):
        qr[(t * t) % g] = True
    sq = {}
    for r in range(1, g):
        sq.setdefault((r * r) % g, []).append(r)
    for i in range(g):
        x = (-6 * i) % g
        chi = int(qr[x]) + int(qr[(x + 2) % g])
        hits = set()
        for t in (x, (x + 2) % g):
            if t in sq:
                hits |= set(sq[t])
        cells += 1
        if len(hits) != 2 * chi:
            bad += 1
        if len(hits) % 2:
            oddcells += 1
say("gears 5..%d, all offset classes: %d (gear, offset) cells checked" % (GEXH, cells))
say("cells where the count differs from 2 chi_g(i): %d" % bad)
say("cells with an ODD number of striking q-classes: %d (the square phase forbids it)" % oddcells)
chk = 0
for g in GEARS[:60]:
    qr = np.zeros(g, dtype=bool)
    for t in range(1, (g + 1) // 2):
        qr[(t * t) % g] = True
    s = sum(int(qr[(-6 * i) % g]) + int(qr[((-6 * i) + 2) % g]) for i in range(g))
    if s != g - 1:
        chk += 1
say("gears 5..%d with sum_i chi_g(i) != g - 1: %d  (so the mean strike rate is exactly 2/g)"
    % (GEARS[59], chk))

# ---------------------------------------------------------------- (b) the empirical rate
say("")
say("=== 4b. the empirical strike rate over the real primes q <= %d (a rate) ===" % QMAX)
say("gear   offset classes   max |measured - 2chi/(g-1)|   at offset class   strikes counted")
qs = np.array(GEARS, dtype=np.int64)
for g in (5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
    qr = np.zeros(g, dtype=bool)
    for t in range(1, (g + 1) // 2):
        qr[(t * t) % g] = True
    r = (qs * qs) % g
    worst = 0.0
    wat = -1
    tot = 0
    for i in range(g):
        x = (-6 * i) % g
        chi = int(qr[x]) + int(qr[(x + 2) % g])
        hit = int(((r == x) | (r == (x + 2) % g)).sum())
        tot += hit
        meas = hit / len(qs)
        dev = abs(meas - 2.0 * chi / (g - 1))
        if dev > worst:
            worst, wat = dev, i
    say("%4d   %10d       %.5f                      %8d          %8d"
        % (g, g, worst, wat, tot))

# ---------------------------------------------------------------- (c) counting constants
say("")
say("=== 4c. exact counting constants: sum_{B < g <= q} 2/g ===")
say("   q        B=7      B=11     B=13     B=17     (measured strikes/island in the band)")
pref = {}
run = 0.0
for g in GEARS:
    run += 2.0 / g
    pref[g] = run
for qtest in (100, 500, 1000, 5000, 10000, 15000, 19997):
    line = "%7d " % qtest
    for B in (7, 11, 13, 17):
        s = pref[max(g for g in GEARS if g <= qtest)] - pref[B]
        line += "  %6.3f " % s
    say(line)
say("")
say("the exact identity behind item 6: the island set for bound B is a union of residue classes")
say("mod P_B; every gear g > B is coprime to P_B, so its two strike classes mod g meet the island")
say("classes in the same proportion as they meet all offsets.  Hence")
say("   strikes on islands / islands  =  sum_{B < g <= q} 2/g  +  boundary,")
say("independent of B except through the lower limit: the frame divides both sides by rho_B.")

nq = sum(1 for g in GEARS if g > 1487)
say("")
say("primes q in (1487, 20000]: %d of %d" % (nq, len(GEARS)))
LOG.close()
