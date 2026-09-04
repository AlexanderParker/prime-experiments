"""R2.a.i.a.1 - item 3 and item 6: what it costs to cover the islands of [1, d).

PART A (real phases).  For a prime q the minimum blocking set of ALL islands in [1, d) inside
(7, q] does not exist as soon as one island is free (N-R4).  The non-vacuous quantity is the
minimum blocking set of the STRUCK islands: the fewest gears of (7, q] whose strikes account for
every island that is struck.  Exact by ILP (HiGHS, proved optimal).

PART B (free phases - the adversary).  Give every gear g > 7 its full freedom: it may take any
nonzero quadratic residue r for q^2 (mod g), which puts its two strike classes at (2 - r) u_g and
-r u_g, i.e. two classes mod g at the fixed separation d_g = 2 u_g.  K(d) = the fewest gears that
can then cover EVERY island of [1, d).  K(d) is q-free: it is the number of gears that would have
to cooperate to defeat the witness at arc length d.  Upper bound: exact ILP over gears <= GMAX
(achievable, so a genuine upper bound).  Lower bound: the counting bound from the largest
achievable set sizes over ALL gears.  Where they meet, K(d) is exact.

Writes results/iw_cover.txt.
Usage: uv run python research/anchor235/r40/iw_cover.py
"""
import os
from math import isqrt

import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import coo_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
LOG = open(os.path.join(OUT, "iw_cover.txt"), "w")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.write(s + "\n")


def sieve(n):
    fl = bytearray([1]) * (n + 1)
    fl[0:2] = b"\x00\x00"
    for i in range(2, isqrt(n) + 1):
        if fl[i]:
            fl[i * i:: i] = bytearray(len(range(i * i, n + 1, i)))
    return fl


NMAX = 40000
FL = sieve(NMAX)
PR = [p for p in range(5, NMAX + 1) if FL[p]]


def islands(d):
    return [i for i in range(1, d) if i % 35 in (5, 10, 12, 17)]


def ilp_cover(nelem, sets):
    """sets: list of (label, tuple of element indices). Exact minimum cover, HiGHS."""
    rows, cols = [], []
    for j, (lab, s) in enumerate(sets):
        for e in s:
            rows.append(e)
            cols.append(j)
    n = len(sets)
    A = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(nelem, n))
    res = milp(c=np.ones(n), constraints=LinearConstraint(A, lb=1, ub=np.inf),
               integrality=np.ones(n), bounds=Bounds(0, 1))
    if not res.success:
        return None, None
    pick = [sets[j][0] for j in range(n) if res.x[j] > 0.5]
    return len(pick), pick


# ------------------------------------------------------------------ PART A
say("=== PART A: minimum blocking set of the STRUCK islands, real phases, exact ILP ===")
say("(the free islands cannot be covered by any gear of the machine, so the blocking set of ALL")
say(" islands does not exist for any prime q > 1487 - that is N-R4 restated)")
say("")
say("  q      d      islands  free   struck   MBS   MBS/struck   gears used <= 100")
sample = [p for p in PR if p <= 200]
sample += [p for p in PR if 200 < p <= 20000][::37]
for q in sample:
    d = (2 * pow(6, -1, q)) % q
    if d < 2:
        continue
    isl = islands(d)
    if not isl:
        continue
    qq = q * q
    cover = {}
    for g in PR:
        if g > q:
            break
        s = tuple(k for k, i in enumerate(isl)
                  if (qq - (2 - 6 * i)) % g == 0 or (qq + 6 * i) % g == 0)
        if s:
            cover[g] = s
    hit = set()
    for s in cover.values():
        hit.update(s)
    free = len(isl) - len(hit)
    if not hit:
        continue
    remap = {e: k for k, e in enumerate(sorted(hit))}
    sets = [(g, tuple(remap[e] for e in s)) for g, s in cover.items()]
    k, pick = ilp_cover(len(hit), sets)
    say("  %-6d %-6d %-8d %-6d %-8d %-5d %-12.4f %d"
        % (q, d, len(isl), free, len(hit), k, k / len(hit),
           sum(1 for g in pick if g <= 100)))

# ------------------------------------------------------------------ PART B
say("")
say("=== PART B: K(d), the adversarial cover number (q-free) ===")
say("gear g free to pick any nonzero QR r for q^2 mod g -> two classes at separation d_g = 2/6")
say("")
say("   d     islands m   GMAX   candidate sets   K(d) upper (ILP)   counting lower   gears used")
for d in (35, 70, 140, 280, 560, 1120, 2240, 3500):
    isl = islands(d)
    m = len(isl)
    if m == 0:
        continue
    GMAX = min(max(2 * d, 200), 20000)
    idx = {i: k for k, i in enumerate(isl)}
    seen = {}
    csize = []          # max achievable set size per gear, for the counting bound
    for g in PR:
        if g <= 7:
            continue
        if g > GMAX:
            break
        buck = {}
        for i in isl:
            buck.setdefault(i % g, []).append(idx[i])
        u = pow(6, -1, g)
        best = 0
        qrs = set((t * t) % g for t in range(1, (g + 1) // 2))
        for r in qrs:
            a = ((2 - r) * u) % g
            b = ((-r) * u) % g
            s = buck.get(a, []) + buck.get(b, [])
            if not s:
                continue
            fs = frozenset(s)
            best = max(best, len(fs))
            if fs not in seen:
                seen[fs] = g
        csize.append((g, best))
    # dominance reduction: drop any set that is a proper subset of another
    allsets = sorted(seen.items(), key=lambda kv: -len(kv[0]))
    bit = [0] * m
    for j, (fs, g) in enumerate(allsets):
        for e in fs:
            bit[e] |= 1 << j
    keep = []
    for j, (fs, g) in enumerate(allsets):
        mask = None
        for e in fs:
            mask = bit[e] if mask is None else (mask & bit[e])
        mask &= ~(1 << j)
        # any superset must appear EARLIER (sorted by decreasing size) and be strictly larger
        dominated = False
        while mask:
            lb = mask & -mask
            jj = lb.bit_length() - 1
            if len(allsets[jj][0]) > len(fs):
                dominated = True
                break
            mask ^= lb
        if not dominated:
            keep.append((g, tuple(fs)))
    k, pick = ilp_cover(m, keep)
    # counting lower bound over ALL gears: sizes from gears <= GMAX, then 2 for every larger gear
    sizes = sorted([c for _, c in csize], reverse=True)
    tot = 0
    lb = 0
    for c in sizes:
        if tot >= m:
            break
        tot += c
        lb += 1
    while tot < m:
        tot += 2
        lb += 1
    say("  %-6d %-11d %-6d %-16d %-18d %-16d %s"
        % (d, m, GMAX, len(keep), k, lb, sorted(set(pick))[:14]))

say("")
say("=== PART B2: the CRT modulus a failure forces ===")
say("a cover with gears S pins q modulo prod(S); the smallest K(d) gears give the smallest")
say("possible modulus.  prod of the first k gears above 7:")
run = 1
ks = []
for g in PR:
    if g <= 7:
        continue
    run *= g
    ks.append((len(ks) + 1, g, run))
    if len(ks) >= 30:
        break
for k, g, run in ks:
    if k in (3, 5, 8, 10, 12, 15, 18, 20, 25, 30):
        say("   k = %2d (top gear %5d): prod = %.4g" % (k, g, run))
LOG.close()
