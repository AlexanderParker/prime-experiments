"""R2.a.i.a.1 - item 2: the failures of the island witness, taken apart.

For every q coprime to 6 at which the B = 7 witness fails (every island of [1, d) struck), report

  * d and the arc (short  d = (q+1)/3  when q = 5 mod 6;  long  d = (2q+1)/3  when q = 1 mod 6),
  * the islands in [1, d) and, for each, every gear that strikes it and the root s with q = +-s,
  * the EXACT minimum number of gears whose strikes cover all the islands (set cover by ILP,
    HiGHS, proved optimal), and one optimal cover,
  * the residues of q that made the failure possible (q mod 11, 13, 17, 19, ...).

Writes results/iw_failures.txt.
Usage: uv run python research/anchor235/r40/iw_failures.py
"""
import os
from math import isqrt

import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import coo_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
LOG = open(os.path.join(OUT, "iw_failures.txt"), "w")


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


NMAX = 3000
FL = sieve(NMAX)
PRIMES = [p for p in range(5, NMAX + 1) if FL[p]]

FAIL_PRIME = [17, 23, 29, 41, 53, 73, 113, 137, 173, 197, 233, 263, 353, 461, 683, 1151, 1487]
FAIL_COMP = [121, 247, 341, 1649]
NOISLAND = [5, 7, 11]


def islands(d):
    return [i for i in range(1, d) if i % 35 in (5, 10, 12, 17)]


def strikers(q, d):
    """{offset: [(gear, root s such that q = +-s mod g and s^2 = -6i or 2-6i)]}"""
    qq = q * q
    st = {}
    for i in islands(d):
        lst = []
        for g in PRIMES:
            if g > q:
                break
            u = pow(6, -1, g)
            if (qq - (2 - 6 * i)) % g == 0 or (qq + 6 * i) % g == 0:
                lst.append(g)
        st[i] = lst
    return st


def exact_cover(universe, sets):
    """minimum number of sets covering the universe; ILP, proved optimal by HiGHS.
    sets: list of (label, frozenset)."""
    idx = {e: k for k, e in enumerate(universe)}
    rows, cols = [], []
    keep = []
    for j, (lab, s) in enumerate(sets):
        if not s:
            continue
        keep.append(j)
        for e in s:
            rows.append(idx[e])
            cols.append(len(keep) - 1)
    if not keep:
        return None, None
    n = len(keep)
    A = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(universe), n))
    res = milp(c=np.ones(n), constraints=LinearConstraint(A, lb=1, ub=np.inf),
               integrality=np.ones(n), bounds=Bounds(0, 1))
    if not res.success:
        return None, None
    pick = [sets[keep[j]][0] for j in range(n) if res.x[j] > 0.5]
    return len(pick), sorted(pick)


say("=== item 2: the failures of the B = 7 island witness, exactly ===")
say("")
say(" q      prime  q mod 6  arc     d     islands  min cover  optimal cover")
rowsout = []
for q in sorted(FAIL_PRIME + FAIL_COMP):
    d = (2 * pow(6, -1, q)) % q
    isl = islands(d)
    st = strikers(q, d)
    gears = sorted({g for v in st.values() for g in v})
    sets = [(g, frozenset(i for i in isl if g in st[i])) for g in gears]
    k, cov = exact_cover(isl, sets)
    arc = "short" if q % 6 == 5 else "long"
    say(" %-6d %-6s %d       %-6s %-5d %-8d %-10d %s"
        % (q, "yes" if FL[q] else "no", q % 6, arc, d, len(isl), k, cov))
    rowsout.append((q, d, len(isl), k, cov, st, isl))

say("")
say("short arc (q = 5 mod 6) among the 17 prime failures: %d of 17; the long-arc one: %s"
    % (sum(1 for q in FAIL_PRIME if q % 6 == 5),
       [q for q in FAIL_PRIME if q % 6 == 1]))
say("largest number of islands at a failure: %d" % max(r[2] for r in rowsout))
say("largest exact minimum cover: %d" % max(r[3] for r in rowsout))

say("")
say("=== the islands and their strikers, failure by failure ===")
for q, d, n, k, cov, st, isl in rowsout:
    say("")
    say("q = %d  (d = %d, %d islands, min cover %d = %s)" % (q, d, n, k, cov))
    for i in isl:
        gs = st[i]
        say("   i = %-5d (i mod 35 = %2d): struck by %s"
            % (i, i % 35, gs if len(gs) <= 12 else str(gs[:12]) + " ... (%d)" % len(gs)))

say("")
say("=== which residues of q made it possible (the 17 prime failures) ===")
for g in (11, 13, 17, 19, 23, 29, 31):
    from collections import Counter
    c = Counter(q % g for q in FAIL_PRIME)
    say(" q mod %2d: %s   (largest class %d of 17)"
        % (g, dict(sorted(c.items())), max(c.values())))

say("")
say("=== how many gears are SOLE strikers of an island at each failure ===")
say(" q      islands  islands with exactly one striker   those gears")
for q, d, n, k, cov, st, isl in rowsout:
    sole = [(i, st[i][0]) for i in isl if len(st[i]) == 1]
    say(" %-6d %-8d %-33d %s" % (q, n, len(sole), sorted({g for _, g in sole})))

say("")
say("=== the smallest striker of each island, pooled over the 17 prime failures ===")
from collections import Counter
c = Counter()
for q, d, n, k, cov, st, isl in rowsout:
    if q not in FAIL_PRIME:
        continue
    for i in isl:
        c[st[i][0]] += 1
tot = sum(c.values())
say("islands: %d;  smallest-striker distribution:" % tot)
for g, v in c.most_common(12):
    say("   gear %-5d %5d  %.4f   (2/g = %.4f)" % (g, v, v / tot, 2 / g))
LOG.close()
