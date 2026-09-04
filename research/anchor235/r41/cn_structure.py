"""R2.a.i.a.1.a items 2 and 5 - the anatomy of an optimal cover, and the covering classes.

For each d: solve K(d) exactly, then
  (a) report which gears the optimum uses, how many islands each takes, the overlap (islands
      struck more than once), and the budget sum|S_j|/m;
  (b) enumerate ALL optimal GEAR SETS by no-good cuts (forbid a found gear set, re-solve, stop
      when the optimum rises above K) - the uniqueness question;
  (c) count the covering residue classes: a cover with phases r_j forces q^2 = r_j (mod g_j), and
      q^2 = r has exactly 2 solutions q mod g when r is a nonzero QR, so the (cover, phase) pair is
      realised by exactly 2^K classes of q mod P = prod g_j.  Verified by brute force mod P for
      small covers.  For the gear SET, the number of realising phase vectors is counted exactly by
      backtracking where feasible.

Usage: uv run python research/anchor235/r41/cn_structure.py --DS 35,70,140,280,560,1120
"""
import argparse
import os
import time
from math import isqrt, prod, log10

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


def maximal_only(S):
    L = sorted(S, key=lambda s: -len(s))
    keep = []
    for s in L:
        if not any(s < t for t in keep):
            keep.append(s)
    return keep


def build(d, PR):
    isl = islands(d)
    m = len(isl)
    idx = {i: k for k, i in enumerate(isl)}
    cand = []
    for g in PR:
        if g > 3 * d + 2:
            break
        buck = {}
        for i in isl:
            buck.setdefault(i % g, []).append(idx[i])
        u = pow(6, -1, g)
        S = set()
        for t in range(1, (g + 1) // 2):
            r = (t * t) % g
            s = buck.get(((2 - r) * u) % g, []) + buck.get(((-r) * u) % g, [])
            if len(s) >= 2:
                S.add(frozenset(s))
        for s in maximal_only(S):
            cand.append((g, s))
    for k in range(m):
        cand.append((0, frozenset((k,))))
    return isl, m, cand


def solve(m, cand, tl, nogoods):
    n = len(cand)
    rows, cols = [], []
    for j, (g, s) in enumerate(cand):
        for e in s:
            rows.append(e)
            cols.append(j)
    cons = [LinearConstraint(coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(m, n)),
                             lb=1, ub=np.inf)]
    bygear = {}
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
    for gs in nogoods:
        js = [j for j in range(n) if cand[j][0] in gs]
        rr2 = np.zeros(len(js))
        cons.append(LinearConstraint(coo_matrix((np.ones(len(js)), (rr2, js)), shape=(1, n)),
                                     lb=-np.inf, ub=len(gs) - 1))
    out = milp(c=np.ones(n), constraints=cons, integrality=np.ones(n), bounds=Bounds(0, 1),
               options={"time_limit": float(tl), "mip_rel_gap": 0.0, "presolve": True})
    if out.x is None:
        return None, -1
    return [j for j in range(n) if out.x[j] > 0.5], out.mip_dual_bound


def count_phase_vectors(d, gears, cap=6e8):
    """Exact number of phase vectors (one nonzero QR per gear) whose union covers all islands.
    Backtracking with a bitmask; returns None if the search space exceeds cap."""
    isl = islands(d)
    m = len(isl)
    idx = {i: k for k, i in enumerate(isl)}
    full = (1 << m) - 1
    per = []
    for g in gears:
        buck = {}
        for i in isl:
            buck.setdefault(i % g, []).append(idx[i])
        u = pow(6, -1, g)
        masks = []
        for t in range(1, (g + 1) // 2):
            r = (t * t) % g
            s = buck.get(((2 - r) * u) % g, []) + buck.get(((-r) * u) % g, [])
            mk = 0
            for e in s:
                mk |= 1 << e
            masks.append(mk)
        per.append(masks)
    if prod(len(p) for p in per) > cap:
        return None
    # order gears by decreasing best coverage, then DFS with a reachability prune
    order = sorted(range(len(per)), key=lambda j: -max(bin(x).count("1") for x in per[j]))
    per = [per[j] for j in order]
    suffix = [0] * (len(per) + 1)
    for j in range(len(per) - 1, -1, -1):
        suffix[j] = suffix[j + 1] + max(bin(x).count("1") for x in per[j])
    total = 0

    def rec(j, cov):
        nonlocal total
        need = bin(full & ~cov).count("1")
        if need == 0:
            t = 1
            for k in range(j, len(per)):
                t *= len(per[k])
            total += t
            return
        if j == len(per) or suffix[j] < need:
            return
        for mk in per[j]:
            rec(j + 1, cov | mk)
    rec(0, 0)
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--DS", type=str, default="35,70,140,280,560,1120")
    ap.add_argument("--tl", type=float, default=600.0)
    ap.add_argument("--maxsets", type=int, default=40)
    ap.add_argument("--tag", type=str, default="structure")
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

    for d in DS:
        t0 = time.time()
        isl, m, cand = build(d, PR)
        pick, lb = solve(m, cand, args.tl, [])
        K = len(pick)
        say("=" * 100)
        say("d = %d   islands m = %d   K(d) = %d   (LB %.3f)" % (d, m, K, lb))
        # (a) anatomy of one optimum
        cnt = np.zeros(m, dtype=int)
        rowinfo = []
        for j in pick:
            g, s = cand[j]
            for e in s:
                cnt[e] += 1
            rowinfo.append((g, len(s), sorted(isl[e] for e in s)))
        rowinfo.sort(key=lambda r: r[0])
        budget = sum(r[1] for r in rowinfo)
        say("  budget sum|S| = %d = %.3f m ; islands struck once %d, twice %d, 3+ %d"
            % (budget, budget / m, int((cnt == 1).sum()), int((cnt == 2).sum()),
               int((cnt >= 3).sum())))
        say("  gear : islands taken")
        for g, sz, off in rowinfo:
            say("   %-6s %-3d  %s" % (g if g else "gen", sz, off if len(off) <= 14 else
                                      str(off[:14]) + "..."))
        gears = sorted(g for g, _, _ in rowinfo if g > 0)
        smalls = [p for p in PR if p <= max(gears)]
        pref = 0
        for p in smalls:
            if p in gears:
                pref += 1
            else:
                break
        say("  gears: %s" % gears)
        say("  consecutive prefix from 11: %d of %d gears (%.2f); largest gear %d, sqrt d = %.1f"
            % (pref, K, pref / K, max(gears), d ** 0.5))
        missing = [p for p in smalls if p not in gears]
        say("  small gears NOT used below the largest used gear: %s" % missing)
        # (c) covering classes
        P = prod(gears)
        say("  P = prod gears = %.4g ; classes of q mod P realising this cover = 2^%d = %d ;"
            " density %.4g" % (P, K, 2 ** K, 2 ** K / P))
        for qq in (int(3 * d), int(1.5 * d)):
            say("     P / q = %.4g at q = %d (arc %s)   P / q^2 = %.4g"
                % (P / qq, qq, "short" if qq == 3 * d else "long", P / qq ** 2))
        npv = count_phase_vectors(d, gears)
        if npv is not None:
            say("  phase vectors of this GEAR SET that cover everything: %d ; total q-classes"
                " mod P = %d x 2^%d = %.4g" % (npv, npv, K, npv * 2 ** K))
        else:
            say("  phase vectors of this gear set: search space too large (>= 6e8), not counted")
        # (b) how many optimal gear sets
        nogoods = [set(gears)]
        sets_found = [tuple(gears)]
        gear_freq = {g: 1 for g in gears}
        while len(sets_found) < args.maxsets:
            pk, _ = solve(m, cand, args.tl, nogoods)
            if pk is None or len(pk) > K:
                break
            gs = sorted(g for g in (cand[j][0] for j in pk) if g > 0)
            if len(pk) != K:
                break
            sets_found.append(tuple(gs))
            nogoods.append(set(gs))
            for g in gs:
                gear_freq[g] = gear_freq.get(g, 0) + 1
        exhausted = len(sets_found) < args.maxsets
        say("  distinct optimal GEAR SETS found: %d%s"
            % (len(sets_found), " (all of them)" if exhausted else " (cap reached, more exist)"))
        always = sorted(g for g, c in gear_freq.items() if c == len(sets_found))
        say("  gears in EVERY optimal set found: %s" % always)
        say("  gear frequency: %s" % sorted(gear_freq.items()))
        say("  time %.1f s" % (time.time() - t0))
    LOG.close()


main()
