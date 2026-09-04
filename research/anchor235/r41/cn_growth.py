"""R2.a.i.a.1.a item 1 - the growth law of K(d), exact where the ILP closes.

K(d) = fewest gears g > 7, each used at ONE reachable phase (q^2 = r mod g, r a nonzero quadratic
residue), whose strike classes  i = (2-r)u_g, -r u_g (mod g), u_g = 6^{-1} mod g,  together hit
every island of [1, d)  (islands: i = 5, 10, 12, 17 mod 35).

EXACT ENUMERATION (parent's argument, reused): a gear can cover >= 3 islands only if g < d; it can
cover 2 islands i, j only if g | 3(i - j) - 1, so g <= 3d; and a singleton is available for every
island at infinitely many gears above 3d.  So gears 11 <= g <= 3d + 2 at every nonzero-QR phase,
plus one generic singleton per island, is the complete play list.

MODEL.  Binary x_{g,t} per (gear, phase-class), binary s_i per island singleton.  Cover each
island; at most one phase per gear.  Only SAME-GEAR dominance is applied (a phase of g whose set is
contained in another phase of the same g is never needed) - cross-gear dominance is NOT sound under
the one-phase-per-gear rule and is not used.  HiGHS, mip_rel_gap 0; the reported LB is HiGHS's dual
bound, so "exact" means proved optimal.

--free runs the same question with the separation FREE (a gear may take ANY two classes mod g) -
the tooth-counterfactual family's freedom - to test whether K(d) is that family's ladder.

Usage: uv run python research/anchor235/r41/cn_growth.py --DS 35,70,... [--tl 900] [--free]
"""
import argparse
import os
import time
from math import isqrt, log

import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import coo_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)


def sieve(n):
    fl = bytearray([1]) * (n + 1)
    fl[0:2] = b"\x00\x00"
    for i in range(2, isqrt(n) + 1):
        if fl[i]:
            fl[i * i:: i] = bytearray(len(range(i * i, n + 1, i)))
    return fl


def islands(d):
    return [i for i in range(1, d) if i % 35 in (5, 10, 12, 17)]


def gear_sets(g, buck, free=False, isl=None):
    out = set()
    if free:
        # every 2-island set is available from a free-separation gear above d (added as an
        # unlimited pseudo-gear), so only sets of size >= 3 need a real gear here: at least one of
        # the two classes must hold two islands.
        ks = sorted(buck)
        big = [k for k in ks if len(buck[k]) >= 2]
        for k in big:
            if len(buck[k]) >= 3:
                out.add(frozenset(buck[k]))
            for k2 in ks:
                if k2 == k:
                    continue
                s = frozenset(buck[k] + buck[k2])
                if len(s) >= 3:
                    out.add(s)
    else:
        u = pow(6, -1, g)
        # only phases whose classes contain an island matter, and each is r = -6i or 2-6i for an
        # island i: O(m) phases per gear instead of O(g), with no loss (a phase with no island is
        # never worth playing).
        h = (g - 1) // 2
        rs = set()
        for i in isl:
            for r in (((-6 * i) % g), ((2 - 6 * i) % g)):
                if r and pow(r, h, g) == 1:
                    rs.add(r)
        for r in rs:
            s = buck.get(((2 - r) * u) % g, []) + buck.get(((-r) * u) % g, [])
            if len(s) >= 2:
                out.add(frozenset(s))
    return out


def maximal_only(S):
    """Keep only the inclusion-maximal sets of a family (used per gear: sound)."""
    L = sorted(S, key=lambda s: -len(s))
    keep = []
    for s in L:
        if not any(s < t for t in keep):
            keep.append(s)
    return keep


def build(d, PR, free=False):
    isl = islands(d)
    m = len(isl)
    idx = {i: k for k, i in enumerate(isl)}
    # with a FREE separation a gear above d can cover at most 2 islands, and every PAIR is then
    # available at infinitely many gears (added below), so only gears g <= d need enumerating.
    GMAX = d if free else 3 * d + 2
    cand = []          # (gear, frozenset); gear 0 = generic singleton
    maxsize = []
    for g in PR:
        if g > GMAX:
            break
        buck = {}
        for i in isl:
            buck.setdefault(i % g, []).append(idx[i])
        S = gear_sets(g, buck, free=free, isl=isl)
        if not S:
            maxsize.append(1)
            continue
        maxsize.append(max(len(s) for s in S))
        for s in maximal_only(S):
            cand.append((g, s))
    if free:
        # with free separation a gear above d covers ANY pair of islands; one such pseudo-gear per
        # pair (unlimited supply above 3d, so no clique constraint)
        for a in range(m):
            for b in range(a + 1, m):
                cand.append((0, frozenset((a, b))))
    for k in range(m):
        cand.append((0, frozenset((k,))))
    return isl, m, cand, maxsize


def solve(m, cand, tl, clique=True):
    n = len(cand)
    rows, cols = [], []
    for j, (g, s) in enumerate(cand):
        for e in s:
            rows.append(e)
            cols.append(j)
    A = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(m, n))
    cons = [LinearConstraint(A, lb=1, ub=np.inf)]
    bygear = {}
    if clique:
        for j, (g, s) in enumerate(cand):
            if g > 0:
                bygear.setdefault(g, []).append(j)
    rr, cc, nr = [], [], 0
    for g, js in bygear.items():
        if len(js) > 1:
            for j in js:
                rr.append(nr)
                cc.append(j)
            nr += 1
    if nr:
        cons.append(LinearConstraint(coo_matrix((np.ones(len(rr)), (rr, cc)), shape=(nr, n)),
                                     lb=-np.inf, ub=1))
    out = milp(c=np.ones(n), constraints=cons, integrality=np.ones(n), bounds=Bounds(0, 1),
               options={"time_limit": float(tl), "mip_rel_gap": 0.0, "presolve": True})
    if out.x is None:
        return None, -1, None
    pick = [j for j in range(n) if out.x[j] > 0.5]
    lb = out.mip_dual_bound
    return pick, (int(np.ceil(lb - 1e-6)) if lb is not None else -1), out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--DS", type=str, default="35,70,140,280,560,1120")
    ap.add_argument("--tl", type=float, default=900.0)
    ap.add_argument("--free", action="store_true")
    ap.add_argument("--multi", action="store_true")
    ap.add_argument("--tag", type=str, default="growth")
    args = ap.parse_args()
    DS = [int(v) for v in args.DS.split(",")]
    LOG = open(os.path.join(OUT, "cn_%s.txt" % args.tag), "w")

    def say(*a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        LOG.write(s + "\n")
        LOG.flush()

    NMAX = 3 * max(DS) + 10
    FL = sieve(NMAX)
    PR = [p for p in range(11, NMAX + 1) if FL[p]]
    say("# K(d)%s%s : gears to %d (%d gears), time limit %.0f s"
        % (" FREE separation" if args.free else "",
           " MULTI-PHASE (no one-phase-per-gear rule)" if args.multi else "",
           NMAX, len(PR), args.tl))
    say("#  d      m    cand   K      LB    status     countLB  budget/m  secs")
    rows = []
    for d in DS:
        t0 = time.time()
        isl, m, cand, maxsize = build(d, PR, free=args.free)
        pick, lb, out = solve(m, cand, args.tl, clique=not args.multi)
        if pick is None:
            say("  %-6d %-4d %-6d ILP FAILED" % (d, m, len(cand)))
            continue
        K = len(pick)
        status = "exact" if lb >= K else "UB(LB %d)" % lb
        sizes = sorted(maxsize, reverse=True)
        tot, clb = 0, 0
        for c in sizes:
            if tot >= m:
                break
            tot += c
            clb += 1
        gears = sorted(cand[j][0] for j in pick if cand[j][0] > 0)
        budget = sum(len(cand[j][1]) for j in pick)
        say("  %-6d %-4d %-6d %-6d %-5d %-10s %-8d %-9.3f %.1f"
            % (d, m, len(cand), K, lb, status, clb, budget / m, time.time() - t0))
        say("      cover: %s + %d singleton(s)" % (gears, K - len(gears)))
        say("      sizes: %s" % sorted((len(cand[j][1]) for j in pick), reverse=True))
        rows.append((d, m, K, lb, clb))
    say("")
    say("#  d      m     K    K*(ln d)^3/d   pi(2.66 sqrt d)-4   countLB")
    for d, m, K, lb, clb in rows:
        x = 2.66 * (d ** 0.5)
        pix = sum(1 for p in range(2, int(x) + 1) if FL[p]) - 4
        say("   %-6d %-5d %-4d %-14.3f %-19d %d" % (d, m, K, K * log(d) ** 3 / d, pix, clb))
    LOG.close()


main()
