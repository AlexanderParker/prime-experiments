"""R2.a.i.a - part 6: does a large gear strike the ISLANDS at the same rate as it strikes all
offsets?

The mechanism claim behind item 6 is a CRT identity: the island set S_B is a union of residue
classes mod P_B, and every gear g > B is coprime to P_B, so g's two strike classes mod g meet
the island classes in exactly the proportion in which they meet all offsets.  The landscape
therefore gives the large gears no discount: their rate on islands is the machine's own 2/g.

Measured directly, over every prime q = 5..20000 and the B = 13 islands of [1, d):
  strikes by gear g on islands / (number of islands seen, weighted by how often g is in range),
against 2/g.

Writes results/rl_rate.txt.
Usage: uv run python research/anchor235/r39/rl_rate.py
"""
import os
from math import isqrt

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
LOG = open(os.path.join(OUT, "rl_rate.txt"), "w")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.write(s + "\n")


QMAX = 20000
B = 13


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

res = np.load(os.path.join(OUT, "rl_isl_%d.npy" % B))
mod = int(open(os.path.join(OUT, "rl_isl_%d_mod.txt" % B)).read())
mask = np.zeros(DMAX, dtype=bool)
rs = np.array(sorted(int(v) for v in res), dtype=np.int64)
for base in range(0, DMAX, mod):
    idx = rs + base
    idx = idx[idx < DMAX]
    mask[idx] = True
islpos = np.flatnonzero(mask)

TRACK = [g for g in GEARS if 13 < g <= 199]
hits = {g: 0 for g in TRACK}
seen = {g: 0 for g in TRACK}
tot_isl = 0
for qi, q in enumerate(GEARS):
    qq = q * q
    d = (2 * pow(6, -1, q)) % q
    D = int(d)
    if D < 2:
        continue
    ip = islpos[(islpos >= 1) & (islpos < D)]
    if len(ip) == 0:
        continue
    tot_isl += len(ip)
    for g in TRACK:
        if g > q:
            break
        u = pow(6, -1, g)
        r = qq % g
        a = ((2 - r) * u) % g
        b = ((-r) * u) % g
        m = ip % g
        hits[g] += int(((m == a) | (m == b)).sum())
        seen[g] += len(ip)

say("B = %d islands seen over the sweep: %d" % (B, tot_isl))
say("")
say("gear   strikes on islands   islands in range   measured rate   2/g       ratio")
for g in TRACK:
    if seen[g] == 0:
        continue
    say("%5d   %16d   %16d   %.6f      %.6f  %.4f"
        % (g, hits[g], seen[g], hits[g] / seen[g], 2.0 / g,
           (hits[g] / seen[g]) / (2.0 / g)))
rat = [(hits[g] / seen[g]) / (2.0 / g) for g in TRACK if seen[g]]
say("")
say("ratio measured/(2/g) over the %d gears tracked: min %.4f, mean %.4f, max %.4f"
    % (len(rat), min(rat), float(np.mean(rat)), max(rat)))
LOG.close()
