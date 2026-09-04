"""R2.a.i.a.1 - item 3/6, part B: K(d), the adversarial cover number, EXACT.

Give every gear g > 7 its full freedom: it may take any nonzero quadratic residue r for q^2 (mod g)
- that is exactly the freedom a real q has - which puts its two strike classes at (2 - r) u_g and
-r u_g, two classes mod g at the fixed separation d_g = 2 u_g.  Each gear may be used at ONE phase
(a real q gives it one).  K(d) = the fewest gears that can then strike every island of [1, d).
K(d) is q-free.  It is the number of gears that would have to cooperate to defeat the witness at
arc length d.

EXACTNESS.  Write a = (2 - r) u_g, b = -r u_g, so 6(a - b) = 2 (mod g).
  * a covered set of size >= 3 needs two islands in one class mod g, so g < d;
  * a covered set of size 2 from the two different classes needs g | 3(i - j) - 1 for two islands
    i, j of [1, d), and |3(i - j) - 1| <= 3d, so g <= 3d;
  * a covered set of size 1 is achievable for every island i at infinitely many gears g > 3d
    (any g > 3d with -6i or 2 - 6i a nonzero QR mod g strikes i and nothing else).
So enumerating every gear 11 <= g <= 3d + 2 at every nonzero-QR phase, plus one singleton per
island, enumerates EVERY set the adversary can play.  The ILP over that list, with the constraint
that a gear is used at most once, is therefore the exact K(d), certified optimal by HiGHS.

Writes results/iw_adv.txt.
Usage: uv run python research/anchor235/r40/iw_adv.py [--DS 35,70,...]
"""
import argparse
import os
from math import isqrt, log

import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import coo_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
LOG = open(os.path.join(OUT, "iw_adv.txt"), "w")


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


ap = argparse.ArgumentParser()
ap.add_argument("--DS", type=str, default="35,70,140,280,560,1120,2240,3500,5000,7000")
args = ap.parse_args()
DS = [int(v) for v in args.DS.split(",")]

NMAX = 3 * max(DS) + 10
FL = sieve(NMAX)
PR = [p for p in range(11, NMAX + 1) if FL[p]]
say("gears enumerated up to %d (%d gears)" % (NMAX, len(PR)))
say("")
say("   d      m=islands   candidate sets   K(d)         status        countingLB    cover (gears)")

res_rows = []
for d in DS:
    isl = [i for i in range(1, d) if i % 35 in (5, 10, 12, 17)]
    m = len(isl)
    if m == 0:
        continue
    idx = {i: k for k, i in enumerate(isl)}
    GMAX = 3 * d + 2
    cand = []            # (gear, frozenset)
    csize = []
    for g in PR:
        if g > GMAX:
            break
        buck = {}
        for i in isl:
            buck.setdefault(i % g, []).append(idx[i])
        u = pow(6, -1, g)
        seen_g = set()
        best = 0
        for t in range(1, (g + 1) // 2):
            r = (t * t) % g
            a = ((2 - r) * u) % g
            b = ((-r) * u) % g
            s = buck.get(a, []) + buck.get(b, [])
            if not s:
                continue
            fs = frozenset(s)
            best = max(best, len(fs))
            if len(fs) >= 2 and fs not in seen_g:
                seen_g.add(fs)
                cand.append((g, fs))
        csize.append(best if best else 0)
    # a size-1 set from a real gear is interchangeable with the generic singleton below,
    # which is achievable at infinitely many gears > 3d, so only sets of size >= 2 are kept
    # from real gears.  One generic singleton per island completes the enumeration.
    for k in range(m):
        cand.append((-(k + 1), frozenset([k])))
    # dominance: a set contained in a strictly larger set is never needed (cheap instances only)
    if len(cand) <= 20000:
        order = sorted(range(len(cand)), key=lambda j: -len(cand[j][1]))
        bit = [0] * m
        for pos, j in enumerate(order):
            for e in cand[j][1]:
                bit[e] |= 1 << pos
        keep = []
        for pos, j in enumerate(order):
            fs = cand[j][1]
            mask = None
            for e in fs:
                mask = bit[e] if mask is None else (mask & bit[e])
            mask &= ~(1 << pos)
            dominated = False
            while mask:
                lb = mask & -mask
                pp = lb.bit_length() - 1
                if len(cand[order[pp]][1]) > len(fs):
                    dominated = True
                    break
                mask ^= lb
            if not dominated:
                keep.append(cand[j])
    else:
        keep = cand
    n = len(keep)
    rows, cols = [], []
    for j, (g, fs) in enumerate(keep):
        for e in fs:
            rows.append(e)
            cols.append(j)
    A = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(m, n))
    cons = [LinearConstraint(A, lb=1, ub=np.inf)]
    # one phase per gear
    bygear = {}
    for j, (g, fs) in enumerate(keep):
        if g > 0:
            bygear.setdefault(g, []).append(j)
    rr, cc = [], []
    nr = 0
    for g, js in bygear.items():
        if len(js) > 1:
            for j in js:
                rr.append(nr)
                cc.append(j)
            nr += 1
    if nr:
        B = coo_matrix((np.ones(len(rr)), (rr, cc)), shape=(nr, n))
        cons.append(LinearConstraint(B, lb=-np.inf, ub=1))
    out = milp(c=np.ones(n), constraints=cons, integrality=np.ones(n), bounds=Bounds(0, 1),
               options={"time_limit": 900.0, "mip_rel_gap": 0.0})
    pick = sorted(keep[j][0] for j in range(n) if out.x[j] > 0.5)
    K = len(pick)
    dual = int(np.ceil(out.mip_dual_bound - 1e-9)) if out.mip_dual_bound is not None else -1
    proved = "exact" if dual >= K else "UB (LB %d)" % dual
    sizes = sorted(csize, reverse=True)
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
    say("  %-6d %-11d %-16d %-12d %-13s %-13d %s"
        % (d, m, n, K, proved, lb, [g for g in pick if g > 0]))
    res_rows.append((d, m, K, lb))

say("")
say("growth of K(d):")
say("   d       m      K(d)   K/ln(d)   K/log2(d)   counting LB")
for d, m, K, lb in res_rows:
    say("  %-7d %-6d %-6d %-9.3f %-11.3f %d" % (d, m, K, K / log(d), K / log(d, ) * log(2), lb))
LOG.close()
